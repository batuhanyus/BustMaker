"""Project path resolution and job directory management.

Everything derives from the project root (the directory containing
``core/``), so the pipeline works from any CWD: paths in ``config.yaml`` are
resolved relative to the config file, and vendor/model/tool discovery is
project-anchored. Portable tools are resolved in this order:

1. explicit config value
2. ``vendor/<tool>/`` inside the project
3. system PATH (``shutil.which``)
"""

from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.config import Config

# Accepted input suffixes (lowercase, with dot)
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}


def project_root() -> Path:
    """Absolute path of the repository root (parent of the ``core`` package)."""
    return Path(__file__).resolve().parent.parent


def _is_exe_candidate(p: Path) -> bool:
    return p.is_file() and (p.suffix.lower() == ".exe" or not p.suffix)


def find_blender(cfg: Config) -> Optional[Path]:
    """Locate the portable Blender executable.

    Order: ``paths.blender_dir`` -> ``vendor/blender`` -> PATH.
    Returns None when Blender is not installed yet (setup script must run).
    """
    return _find_tool(cfg, "blender_dir", "blender")


def find_ffmpeg(cfg: Config) -> Optional[Path]:
    """Locate ffmpeg: ``paths.ffmpeg_path`` -> ``vendor/ffmpeg`` -> PATH."""
    return _find_tool(cfg, "ffmpeg_path", "ffmpeg")


def find_colmap(cfg: Config) -> Optional[Path]:
    """Locate COLMAP: ``paths.colmap_path`` -> ``vendor/colmap`` -> PATH."""
    return _find_tool(cfg, "colmap_path", "colmap")


def _find_tool(cfg: Config, cfg_key: str, name: str) -> Optional[Path]:
    # 1. explicit config value
    raw = cfg.get(f"paths.{cfg_key}", None)
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = cfg.path.parent / p
        if _is_exe_candidate(p):
            return p.resolve()
        # allow config pointing at a directory containing the binary
        for cand in (p / (name + ".exe"), p / name):
            if _is_exe_candidate(cand):
                return cand.resolve()

    # 2. project vendor dir
    vendor = cfg.path.parent / "vendor" / name
    for cand in (vendor / (name + ".exe"), vendor / name):
        if _is_exe_candidate(cand):
            return cand.resolve()

    # 3. PATH
    found = shutil.which(name)
    return Path(found).resolve() if found else None


@dataclass
class JobPaths:
    """All filesystem locations for one pipeline run."""

    job_id: str
    input_path: Path
    job_dir: Path
    preprocessed_dir: Path
    raw_mesh_dir: Path
    final_dir: Path
    debug_dir: Path
    temp_dir: Path
    log_path: Path
    report_path: Path
    cfg: Config = field(repr=False)

    # -- final artifacts ------------------------------------------------------
    @property
    def stl_path(self) -> Path:
        return self.final_dir / str(self.cfg.get("print.stl_name", "bust.stl"))

    @property
    def glb_path(self) -> Path:
        return self.final_dir / str(self.cfg.get("print.glb_name", "preview.glb"))

    # -- raw mesh (written by mesh generation backends) -----------------------
    @property
    def raw_obj_path(self) -> Path:
        return self.raw_mesh_dir / "raw_mesh.obj"

    @property
    def raw_glb_path(self) -> Path:
        return self.raw_mesh_dir / "raw_mesh.glb"

    # -- scaffolding ------------------------------------------------------------
    def ensure_dirs(self) -> None:
        for d in (
            self.preprocessed_dir,
            self.raw_mesh_dir,
            self.final_dir,
            self.debug_dir,
            self.temp_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, cfg: Config, input_path: Path, output_dir: Optional[Path] = None) -> "JobPaths":
        """Build a fresh job workspace. ``output_dir`` defaults to
        ``<cfg paths.output>/<input-name>/``; a timestamp suffix keeps repeated
        runs from clobbering each other unless an explicit output dir is given."""
        input_path = input_path.resolve()
        if output_dir is None:
            base = cfg.resolve_path("paths.output", "./output") / _job_slug(input_path)
            output_dir = base / time.strftime("%Y%m%d-%H%M%S")
        else:
            output_dir = Path(output_dir).resolve()

        job_id = uuid.uuid4().hex[:12]
        return cls(
            job_id=job_id,
            input_path=input_path,
            job_dir=output_dir,
            preprocessed_dir=output_dir / "preprocessed",
            raw_mesh_dir=output_dir / "raw_mesh",
            final_dir=output_dir / "final",
            debug_dir=output_dir / "debug",
            temp_dir=cfg.resolve_path("paths.temp", "./temp") / job_id,
            log_path=cfg.resolve_path("paths.logs", "./logs") / f"{time.strftime('%Y%m%d-%H%M%S')}-{job_id}.log",
            report_path=output_dir / "report.json",
            cfg=cfg,
        )


def _job_slug(input_path: Path) -> str:
    name = input_path.name if input_path.is_dir() else input_path.stem
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "job"
