"""Pipeline orchestration: run context, stage registry, and the runner.

Stages are plain functions ``fn(ctx: RunContext) -> StageResult`` registered
in :data:`STAGES`. The orchestrator executes them in order, times each one,
catches failures (optionally continuing when ``required=False``), and records
everything into the :class:`JobReport`.

Phase 1 ships the registry with *stub* stages so the CLI, job layout and
report schema are proven end-to-end before the real implementations land in
Phases 3-6 (each replaces its stub in place).
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from core.config import Config
from core.logging import ProgressHub, get_logger, setup_logging
from core.paths import IMAGE_SUFFIXES, JobPaths, VIDEO_SUFFIXES
from core.report import JobReport

log = get_logger("pipeline")

# Valid pipeline modes (must match config.yaml `modes:` keys).
MODES = ("auto", "generative", "capture", "depth_relief")


class StageError(Exception):
    """Raised by a stage to signal a fatal failure of that stage."""


@dataclass
class StageResult:
    status: str = "success"  # success | failed | skipped
    warnings: list[str] = field(default_factory=list)
    fallback_used: Optional[str] = None
    artifacts: dict[str, Any] = field(default_factory=dict)


StageFn = Callable[["RunContext"], StageResult]


@dataclass
class RunContext:
    """Everything a stage needs: config, job paths, report, progress hub."""

    cfg: Config
    job: JobPaths
    report: JobReport
    progress: ProgressHub
    mode: str
    preset: str
    cli: dict[str, Any] = field(default_factory=dict)
    _shared: dict[str, Any] = field(default_factory=dict)

    # -- shared scratch space between stages -----------------------------------
    def shared(self, key: str, default: Any = None) -> Any:
        return self._shared.get(key, default)

    def set_shared(self, key: str, value: Any) -> None:
        self._shared[key] = value

    def warn(self, message: str) -> None:
        log.warning(message)
        self.report.add_warning(message)


@dataclass
class Stage:
    name: str
    fn: StageFn
    required: bool = True  # False = failure is recorded but pipeline continues


# ---------------------------------------------------------------------------
# Stub stages — replaced phase by phase
# ---------------------------------------------------------------------------


def _stub(ctx: RunContext, phase: str, note: str) -> StageResult:
    log.info("STUB %s: %s", ctx.mode, note)
    ctx.progress.emit("pipeline", "progress", f"[stub] {note}", 0.0)
    return StageResult(status="skipped", warnings=[f"{note} (implemented in {phase})"])


def stage_ingest(ctx: RunContext) -> StageResult:
    """Phase 3: input ingestion & preprocessing."""
    from scripts.preprocess.ingest import run_ingest

    return run_ingest(ctx)


def stage_generate_mesh(ctx: RunContext) -> StageResult:
    """Phase 4: mesh generation backend chain with fallbacks."""
    from scripts.generate.generate_mesh import run_generate

    return run_generate(ctx)


def stage_print_prep(ctx: RunContext) -> StageResult:
    """Phase 5: Blender print-proofing (headless portable Blender)."""
    from core.blender_runner import run_blender_script

    raw = ctx.job.raw_obj_path
    if not raw.is_file():
        raise StageError(f"print_prep: raw mesh missing: {raw}")

    preset = ctx.cfg.preset(ctx.preset)
    print_cfg = ctx.cfg.print_cfg
    script_dir = ctx.cfg.path.parent / "scripts" / "blender"
    final_dir = ctx.job.final_dir
    final_dir.mkdir(parents=True, exist_ok=True)

    # Detail-preserving repair: clean raw meshes (Hunyuan3D, poisson) skip
    # the voxel remesh; genuinely broken meshes still get it as a repair.
    needs_repair = _mesh_needs_repair(raw)
    log.info("print_prep: raw mesh needs_repair=%s", needs_repair)

    stats_json = ctx.job.temp_dir / "mesh_stats.json"
    run_blender_script(
        ctx.cfg,
        script_dir / "auto_print_prep.py",
        script_args=[
            "--input", str(raw),
            "--output-dir", str(final_dir),
            "--target-height", str(print_cfg.get("target_height_mm", 120.0)),
            "--base-thickness", str(print_cfg.get("base_thickness_mm", 4.0)),
            "--voxel-size", str(preset["voxel_size_mm"]),
            "--decimate-ratio", str(preset["decimate_ratio"]),
            "--input-watertight", "false" if needs_repair else "true",
            "--min-triangles", str(preset.get("min_triangles", 25_000)),
            "--max-triangles", str(preset.get("max_triangles", 2_000_000)),
        ],
        timeout=900,
    )
    stl = ctx.job.stl_path
    if not stl.is_file():
        raise StageError(f"print_prep: Blender finished but {stl} was not produced")

    run_blender_script(
        ctx.cfg,
        script_dir / "mesh_stats.py",
        script_args=["--input", str(stl), "--output", str(stats_json)],
        timeout=300,
    )
    stats = {}
    if stats_json.is_file():
        import json

        stats = json.loads(stats_json.read_text(encoding="utf-8"))

    glb = ctx.job.glb_path
    ctx.report.mesh = {
        "raw_mesh": str(raw),
        "stl": str(stl) if stl.is_file() else None,
        "glb": str(glb) if glb.is_file() else None,
        "stats": stats,
    }
    warnings = []
    if stats.get("watertight") is False:
        warnings.append(f"mesh has {stats.get('non_manifold_edges')} non-manifold edges")
    if stats.get("base_flat") is False:
        warnings.append("base may not be perfectly flat")
    for w in warnings:
        ctx.warn(w)

    return StageResult(
        status="success",
        warnings=warnings,
        artifacts={
            "stl": str(stl),
            "glb": str(glb) if glb.is_file() else None,
            "stats": stats,
            "input_watertight": not needs_repair,
            "target_height_mm": print_cfg.get("target_height_mm", 120.0),
        },
    )


def _mesh_needs_repair(path: Path) -> bool:
    """Decide whether the raw mesh needs the lossy voxel-remesh repair.

    Strict watertightness is too harsh: generative outputs often carry a
    handful of non-manifold edges or a tiny floater while being 99.99%
    clean, and remeshing those erases facial detail. Only genuinely broken
    meshes (large holes, many bad edges, or a dominant floater fraction)
    get the remesh. Fail-safe: any load/check error reports needs-repair.
    """
    try:
        import trimesh
        from collections import Counter

        mesh = trimesh.load(path, force="mesh")
        if mesh.is_watertight:
            return False
        counts = Counter(map(tuple, mesh.edges_sorted.tolist()))
        boundary = sum(1 for n in counts.values() if n == 1)
        nonmanifold = sum(1 for n in counts.values() if n > 2)
        try:
            parts = mesh.split(only_watertight=False)
            largest = max((len(p.faces) for p in parts), default=0)
            total = len(mesh.faces)
            floater_fraction = 1.0 - (largest / total) if total else 1.0
        except Exception:  # noqa: BLE001 - split can fail on weird meshes
            floater_fraction = 0.0
        needs = boundary > 200 or nonmanifold > 20 or floater_fraction > 0.05
        log.info("print_prep: mesh defects boundary=%d nonmanifold=%d "
                 "floaters=%.1f%% -> needs_repair=%s",
                 boundary, nonmanifold, floater_fraction * 100, needs)
        return needs
    except Exception as exc:  # noqa: BLE001 - never block prep on the check
        log.warning("repair check failed (%s); assuming broken mesh", exc)
        return True


def stage_validate(ctx: RunContext) -> StageResult:
    """Phase 6: printability validation of the final STL."""
    stl = ctx.job.stl_path
    if not stl.is_file():
        raise StageError(f"validate: STL missing: {stl}")

    from scripts.validate.check_mesh import validate_stl

    checks = validate_stl(ctx.cfg, stl)
    ctx.report.mesh.setdefault("stats", {})
    ctx.report.mesh["validation"] = checks

    warnings = []
    if not checks.get("printable", True):
        warnings.append(f"mesh failed validation: {', '.join(checks.get('failures', []))}")
    if checks.get("repair_succeeded"):
        warnings.append("mesh was auto-repaired (bust_repaired.stl)")
    for w in warnings:
        ctx.warn(w)
    ctx.progress.emit("validate", "finished",
                      f"printable={checks.get('printable')}", 1.0)

    return StageResult(
        status="success",
        warnings=warnings,
        artifacts={"validation": checks},
    )


STAGES: list[Stage] = [
    Stage("ingest", stage_ingest),
    Stage("generate_mesh", stage_generate_mesh),
    Stage("print_prep", stage_print_prep),
    Stage("validate", stage_validate),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_pipeline(
    cfg: Config,
    job: JobPaths,
    mode: str,
    preset: str,
    cli: Optional[dict[str, Any]] = None,
    progress: Optional[ProgressHub] = None,
    stages: Optional[list[Stage]] = None,
) -> JobReport:
    """Execute the stage list against a job workspace; returns the report.

    Always terminates: stage exceptions become ``failed`` stage records, and
    the report is saved regardless of outcome.
    """
    t0 = time.monotonic()
    setup_logging(cfg.logging_level, log_dir=cfg.resolve_path("paths.logs", "./logs"))
    progress = progress or ProgressHub(log_path=job.log_path.with_suffix(".events.jsonl"))
    progress.emit("pipeline", "started", f"job {job.job_id} mode={mode} preset={preset}")

    report = JobReport.new(
        job_id=job.job_id,
        input_path=job.input_path,
        input_type=_input_type(job.input_path),
        config_snapshot={
            "mode": mode,
            "preset": preset,
            "quality_preset": cfg.preset(preset),
            "mode_chain": cfg.mode_chain(mode),
        },
    )
    ctx = RunContext(
        cfg=cfg, job=job, report=report, progress=progress, mode=mode, preset=preset,
        cli=cli or {},
    )
    job.ensure_dirs()

    stage_list = stages or STAGES
    if cli and cli.get("resume") and job.report_path.is_file():
        previous = _load_report_stage_statuses(job.report_path)
        if previous:
            skipped = [s for s in stage_list if s.name in previous and previous[s.name] == "success"]
            if skipped:
                log.info("resume: skipping already-successful stages: %s",
                         [s.name for s in skipped])
                for s in skipped:
                    rec = report.stage(s.name)
                    rec.status = "skipped"
                    rec.warnings = ["resumed from previous report"]
                    progress.emit(s.name, "skipped", "resumed from previous report")
            stage_list = [s for s in stage_list if s not in skipped]

    for stage in stage_list:
        rec = report.stage(stage.name)
        progress.emit(stage.name, "started", f"stage '{stage.name}' started")
        ts = time.monotonic()
        try:
            result = stage.fn(ctx)
            rec.status = result.status
            rec.duration_s = round(time.monotonic() - ts, 3)
            rec.warnings.extend(result.warnings)
            rec.fallback_used = result.fallback_used
            rec.artifacts.update(result.artifacts)
            if result.fallback_used:
                report.fallbacks_used.append(result.fallback_used)
            progress.emit(stage.name, "finished", f"stage '{stage.name}' -> {result.status}")
        except StageError as exc:
            rec.status = "failed"
            rec.duration_s = round(time.monotonic() - ts, 3)
            rec.error = str(exc)
            log.error("Stage '%s' failed: %s", stage.name, exc)
            progress.emit(stage.name, "failed", str(exc))
            if stage.required:
                break
        except Exception as exc:  # noqa: BLE001 - report and continue/fail
            rec.status = "failed"
            rec.duration_s = round(time.monotonic() - ts, 3)
            rec.error = f"{type(exc).__name__}: {exc}"
            log.error("Stage '%s' crashed: %s", stage.name, traceback.format_exc())
            progress.emit(stage.name, "failed", rec.error)
            if stage.required:
                break

    success = all(s.status in ("success", "skipped") for s in report.stages)
    report.finalize(total_duration_s=time.monotonic() - t0, success=success)
    report.save(job.report_path)
    progress.emit("pipeline", "finished", f"success={success} report={job.report_path}")
    return report


# ---------------------------------------------------------------------------
# Input classification
# ---------------------------------------------------------------------------


def _input_type(input_path: Path) -> str:
    if input_path.is_dir():
        return "images"
    if input_path.suffix.lower() in VIDEO_SUFFIXES:
        return "video"
    if input_path.suffix.lower() in IMAGE_SUFFIXES:
        return "single_image"
    return "unknown"


def classify_input(input_path: Path) -> str:
    """Public classifier used by the CLI and GUI for validation."""
    return _input_type(input_path)


def _load_report_stage_statuses(report_path: Path) -> dict[str, str]:
    """Read {stage_name: status} from a previous run's report (for --resume)."""
    import json

    try:
        with open(report_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return {s["name"]: s["status"] for s in data.get("stages", [])}
    except (OSError, ValueError, KeyError):
        return {}
