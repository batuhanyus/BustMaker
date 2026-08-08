"""Input ingestion stage: raw input -> preprocessed RGBA frame set.

Handles all three input types (video / image folder / single image):

1. video   -> ffmpeg frame extraction (even spacing) -> keyframe selection
2. images  -> EXIF-normalized copies -> keyframe selection (blur/exposure)
3. single  -> orientation-normalized copy

Then background removal (rembg) produces RGBA PNGs in ``preprocessed/``,
binary masks in ``debug/masks/`` (when requested), and ``preprocessed/metadata.json``
with per-frame quality + rejection reasons. Downstream stages consume
``ctx.shared("preprocessed_frames")``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.config import Config
from core.ffmpeg_wrapper import extract_frames, probe
from core.logging import get_logger
from core.pipeline import RunContext, StageResult
from core.paths import IMAGE_SUFFIXES

log = get_logger("ingest")


@dataclass
class IngestSummary:
    input_type: str
    source_frames: int
    accepted_frames: int
    rejected: dict  # reason -> count
    masked_frames: int
    mask_failures: int
    mask_quality_min: Optional[float]
    mask_quality_mean: Optional[float]
    frames: list[Path]  # preprocessed RGBA paths
    warnings: list[str]


def run_ingest(ctx: RunContext) -> StageResult:
    cfg = ctx.cfg
    preset = cfg.preset(ctx.preset)
    job = ctx.job

    try:
        from scripts.preprocess.select_keyframes import select_keyframes
    except ImportError:  # pragma: no cover - dev aid
        raise RuntimeError("select_keyframes not importable") from None

    ctx.progress.emit("ingest", "started", "classifying input")
    itype = _classify(job.input_path)
    summary = IngestSummary(
        input_type=itype, source_frames=0, accepted_frames=0, rejected={},
        masked_frames=0, mask_failures=0, mask_quality_min=None,
        mask_quality_mean=None, frames=[], warnings=[],
    )

    max_frames = preset["video_max_frames"]
    blur_thr = float(cfg.get("video.blur_threshold", 60.0))
    exp_range = tuple(cfg.get("video.exposure_ok_range", [25, 235]))

    # ---- collect candidate frames -------------------------------------------
    if itype == "video":
        info = probe(cfg, job.input_path)
        log.info("video %s: %.1fs, %.0f fps, %dx%d",
                 job.input_path.name, info.duration_s, info.fps, info.width, info.height)
        if info.duration_s <= 0:
            raise RuntimeError(f"cannot read video: {job.input_path}")
        extracted = extract_frames(
            cfg, job.input_path, job.temp_dir / "extracted",
            interval_sec=float(cfg.get("video.extract_interval_sec", 1.0)),
            max_frames=max_frames * 3,
        )
        summary.source_frames = len(extracted)
        selection = select_keyframes(extracted, max_frames, blur_thr, exp_range)
    elif itype == "images":
        imgs = _list_images(job.input_path)
        if not imgs:
            raise RuntimeError(f"no images found in {job.input_path}")
        summary.source_frames = len(imgs)
        # normalize orientation into temp (video frames need no EXIF handling)
        normalized = []
        for i, img in enumerate(imgs):
            dst = job.temp_dir / "normalized" / f"photo_{i:05d}.jpg"
            _exif_normalize_copy(img, dst)
            normalized.append(dst)
        selection = select_keyframes(normalized, max_frames, blur_thr, exp_range)
    else:  # single_image
        summary.source_frames = 1
        dst = job.temp_dir / "normalized" / "photo_00000.jpg"
        _exif_normalize_copy(job.input_path, dst)
        selection = select_keyframes([dst], 1, blur_thr, exp_range)
        if not selection.accepted:
            # single image: accept anyway, with a warning (best effort input)
            selection.accepted = selection.rejected
            selection.rejected = []
            summary.warnings.append("single image failed quality checks; proceeding anyway")

    accepted = selection.accepted
    summary.accepted_frames = len(accepted)
    summary.rejected = selection.summary()["rejected_reasons"]
    log.info("ingest: %d accepted / %d rejected", len(accepted), len(selection.rejected))

    if not accepted:
        raise RuntimeError("no usable frames after quality filtering")

    # ---- background removal ---------------------------------------------------
    ctx.progress.emit("ingest", "progress", f"matting {len(accepted)} frames", 0.5)
    try:
        from adapters.background.rembg_adapter import remove_background
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"rembg adapter unavailable: {exc}") from None

    model = str(cfg.get("background_removal.model", "u2net"))
    keep_debug = bool(ctx.cli.get("debug") or cfg.get("report.include_debug", False))
    mask_dir = job.debug_dir / "masks" if keep_debug else None
    qualities: list[float] = []
    failures = 0

    for i, verdict in enumerate(accepted):
        out = job.preprocessed_dir / f"frame_{i:05d}.png"
        try:
            res = remove_background(cfg, verdict.path, out, model=model, mask_dir=mask_dir)
            qualities.append(res.quality)
        except Exception as exc:  # noqa: BLE001 - tolerate per-frame matting failure
            failures += 1
            log.warning("matting failed for %s: %s", verdict.path.name, exc)
            shutil.copy2(verdict.path, out)  # keep RGB, no alpha
        ctx.progress.emit("ingest", "progress", f"matting {i + 1}/{len(accepted)}", 0.5 + 0.5 * (i + 1) / len(accepted))

    summary.masked_frames = len(qualities)
    summary.mask_failures = failures
    if qualities:
        summary.mask_quality_min = round(min(qualities), 3)
        summary.mask_quality_mean = round(sum(qualities) / len(qualities), 3)
    if failures:
        summary.warnings.append(f"{failures}/{len(accepted)} frames fell back to unmasked RGB")

    frames = sorted(job.preprocessed_dir.glob("frame_*.png"))
    summary.frames = frames
    _write_metadata(job.preprocessed_dir / "metadata.json", summary, selection)

    ctx.set_shared("preprocessed_frames", frames)
    ctx.set_shared("input_type", itype)
    ctx.set_shared("ingest_summary", summary)

    for w in summary.warnings:
        ctx.warn(w)

    return StageResult(
        status="success",
        artifacts={
            "input_type": itype,
            "source_frames": summary.source_frames,
            "accepted_frames": summary.accepted_frames,
            "masked_frames": summary.masked_frames,
            "mask_failures": summary.mask_failures,
            "mask_quality_mean": summary.mask_quality_mean,
            "mask_quality_min": summary.mask_quality_min,
            "rejected": summary.rejected,
            "preprocessed_dir": str(job.preprocessed_dir),
        },
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _classify(path: Path) -> str:
    from core.pipeline import classify_input
    return classify_input(path)


def _list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _exif_normalize_copy(src: Path, dst: Path) -> None:
    from PIL import Image, ImageOps

    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "PA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            # JPEG cannot store alpha: composite onto a white background.
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dst, quality=95)  # re-encode; orientation baked in


def _write_metadata(path: Path, summary: IngestSummary, selection) -> None:
    payload = {
        "input_type": summary.input_type,
        "source_frames": summary.source_frames,
        "accepted_frames": summary.accepted_frames,
        "rejected": summary.rejected,
        "masked_frames": summary.masked_frames,
        "mask_failures": summary.mask_failures,
        "mask_quality": {
            "min": summary.mask_quality_min,
            "mean": summary.mask_quality_mean,
        },
        "warnings": summary.warnings,
        "frames": [
            {"index": i, "file": p.name, "masked": i < summary.masked_frames}
            for i, p in enumerate(summary.frames)
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
