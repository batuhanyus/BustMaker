"""Mesh backend interface.

Every mesh-generation backend (generative, capture, depth-relief) implements
:class:`MeshBackend` and returns a :class:`BackendResult`. The orchestrator in
``scripts/generate/`` walks the mode's strategy chain and tries backends in
order, recording every failure in the job report — the pipeline never fails
silently.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.config import Config
from core.pipeline import RunContext


@dataclass
class BackendResult:
    success: bool
    mesh_path: Optional[Path] = None      # OBJ/PLY output
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None           # failure reason (recorded, never silent)
    settings: dict = field(default_factory=dict)  # resolution/fp16/backend details
    duration_s: float = 0.0


class MeshBackend(ABC):
    """One mesh-generation backend (e.g. TripoSR, Stable Fast 3D, COLMAP,
    depth-relief). ``name`` is used in reports and fallback tracking."""

    name: str = "backend"

    @abstractmethod
    def generate(self, ctx: RunContext, input_frames: list[Path], out_path: Path) -> BackendResult:
        """Produce a raw mesh at ``out_path`` (OBJ) from preprocessed RGBA frames."""

    def run(self, ctx: RunContext, input_frames: list[Path], out_path: Path) -> BackendResult:
        t0 = time.monotonic()
        try:
            result = self.generate(ctx, input_frames, out_path)
        except Exception as exc:  # noqa: BLE001 - backends must never crash the chain
            result = BackendResult(success=False, error=f"{type(exc).__name__}: {exc}")
        result.duration_s = round(time.monotonic() - t0, 2)
        return result


def is_available(cfg: Config, backend_name: str) -> bool:
    """Cheap presence check used to skip obviously unavailable backends
    (missing model weights, missing binaries) before loading anything."""
    models_dir = cfg.resolve_path("paths.models", "./models")
    vendor_dir = cfg.resolve_path("paths.vendor", "./vendor")

    if backend_name == "triposr":
        return (models_dir / "triposr" / "model.ckpt").is_file()
    if backend_name == "stable_fast_3d":
        return (models_dir / "stable_fast_3d" / "model.safetensors").is_file()
    if backend_name == "depth_relief":
        return (models_dir / "depth_anything_v2" / "model.safetensors").is_file()
    if backend_name == "colmap":
        from core.paths import find_colmap
        return find_colmap(cfg) is not None
    return False
