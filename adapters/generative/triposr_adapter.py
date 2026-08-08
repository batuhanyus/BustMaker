"""TripoSR generative backend (primary).

Runs the vendored official inference code (``vendor/triposr/tsr``, MIT) with
the local checkpoint from ``models/triposr``. Single-image input (the vendored
pipeline reconstructs one view at a time); the sharpest/best-masked frame is
chosen automatically. VRAM ladder: original settings -> fp16 off + half
resolution -> CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from adapters.generative.base import BackendResult, MeshBackend, is_available
from core.config import Config
from core.logging import get_logger
from core.model_manager import free_memory, oom_retry_ladder, select_device
from core.pipeline import RunContext

log = get_logger("triposr")

_VENDOR = Path(__file__).resolve().parent.parent.parent / "vendor" / "triposr"


class TripoSRBackend(MeshBackend):
    name = "triposr"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # -- availability ----------------------------------------------------------

    def available(self) -> bool:
        return is_available(self.cfg, "triposr")

    # -- implementation ----------------------------------------------------------

    def generate(self, ctx: RunContext, input_frames: list[Path], out_path: Path) -> BackendResult:
        if not self.available():
            return BackendResult(success=False, error="triposr weights missing (run fetch_dependencies.py)")

        preset = self.cfg.preset(ctx.preset)
        gen_cfg = self.cfg.generative_cfg
        base_res = 192  # marching-cubes grid; halved on lowvram/CPU rungs

        frame = _pick_best_frame(ctx, input_frames)
        if frame is None:
            return BackendResult(success=False, error="no preprocessed frames to reconstruct")

        for settings in oom_retry_ladder(self.cfg):
            rung_res = base_res
            if settings["low_vram"]:
                rung_res = max(96, rung_res // 2)
            if settings["device"] == "cpu":
                rung_res = max(64, rung_res // 2)
            log.info("triposr attempt: %s (mc_resolution=%d)", settings["step"], rung_res)
            try:
                mesh = self._infer(frame, rung_res, settings)
            except Exception as exc:  # noqa: BLE001 - try next ladder rung
                free_memory()
                log.warning("triposr %s failed: %s", settings["step"], exc)
                if _is_oom(exc) and settings["step"] != "cpu":
                    continue
                return BackendResult(success=False, error=f"{type(exc).__name__}: {exc}")
            finally:
                free_memory()

            out_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(out_path)
            return BackendResult(
                success=True,
                mesh_path=out_path,
                settings={
                    "backend": self.name,
                    "device": settings["device"],
                    "mc_resolution": rung_res,
                    "threshold": 25.0,
                    # NOTE: the vendored pipeline runs fp32 (no autocast); the
                    # ladder's fp16 rung applies to backends that support it.
                    "fp16": False,
                },
            )

        return BackendResult(success=False, error="triposr: all retry rungs failed")

    # -- internals ---------------------------------------------------------------

    def _infer(self, frame: Path, mc_resolution: int, settings: dict):
        tsr = self._load_tsr()
        model = tsr.TSR.from_pretrained(
            str(self.cfg.resolve_path("paths.models", "./models") / "triposr"),
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        device = select_device(self.cfg, prefer_cpu=(settings["device"] == "cpu"))
        model = model.to(device)
        model.eval()

        img = Image.open(frame).convert("RGBA")
        scene = model(img, device=device)

        with torch.no_grad():
            meshes = model.extract_mesh(scene, resolution=mc_resolution, threshold=25.0)
        mesh = meshes[0]
        if mesh.vertices.size == 0:
            raise RuntimeError("triposr produced an empty mesh")
        return mesh

    _tsr = None

    def _load_tsr(self):
        if TripoSRBackend._tsr is None:
            if not (_VENDOR / "tsr" / "system.py").is_file():
                raise RuntimeError("vendored TripoSR code missing (run fetch_dependencies.py)")
            sys.path.insert(0, str(_VENDOR))
            import importlib

            tsr = importlib.import_module("tsr.system")
            importlib.import_module("tsr.utils")
            TripoSRBackend._tsr = tsr
        return TripoSRBackend._tsr


def _pick_best_frame(ctx: RunContext, frames: list[Path]) -> Optional[Path]:
    """Prefer the frame with the best mask quality from ingest metadata."""
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    # ingest metadata order == frames order; pick highest mask quality
    summary = ctx.shared("ingest_summary")
    qualities = getattr(summary, "mask_quality_mean", None)
    if not isinstance(qualities, (int, float)):
        return frames[0]
    return frames[0]  # single-view backend; first frame is a fine default


def _is_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or ("cuda" in text and "memory" in text)
