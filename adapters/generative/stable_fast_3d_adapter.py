"""Stable Fast 3D generative backend (secondary fallback).

Status: the weights live in the gated HF repo ``stabilityai/stable-fast-3d``
(GatedRepoError for anonymous downloads). This adapter is fully implemented
but *skips itself* with a clear warning until the weights are present at
``models/stable_fast_3d/model.safetensors`` (fetch with ``HF_TOKEN`` set::

    set HF_TOKEN=hf_... && python scripts/setup/fetch_dependencies.py --only stable_fast_3d

The ``sf3d`` pip package and its ``texture_baker`` dependency are imported
lazily so the rest of the pipeline never depends on them.
"""

from __future__ import annotations

from pathlib import Path

from adapters.generative.base import BackendResult, MeshBackend, is_available
from core.config import Config
from core.logging import get_logger
from core.model_manager import free_memory, select_device
from core.pipeline import RunContext

log = get_logger("sf3d")


class StableFast3DBackend(MeshBackend):
    name = "stable_fast_3d"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def available(self) -> bool:
        return is_available(self.cfg, "stable_fast_3d")

    def generate(self, ctx: RunContext, input_frames: list[Path], out_path: Path) -> BackendResult:
        if not self.available():
            return BackendResult(
                success=False,
                error=(
                    "stable_fast_3d weights missing (gated HF repo). "
                    "Set HF_TOKEN and re-run fetch_dependencies.py, or rely on "
                    "the depth-relief fallback."
                ),
            )
        try:
            from sf3d.system import SF3D
        except ImportError as exc:
            return BackendResult(
                success=False,
                error=f"sf3d package not installed ({exc}); pip install stable-fast-3d",
            )

        preset = self.cfg.preset(ctx.preset)
        gen_cfg = self.cfg.generative_cfg
        model_dir = str(self.cfg.resolve_path("paths.models", "./models") / "stable_fast_3d")

        try:
            model = SF3D.from_pretrained(model_dir, config_name="config.yaml",
                                         weight_name="model.safetensors")
            device = select_device(self.cfg)
            model = model.to(device).eval()

            frame = input_frames[0] if input_frames else None
            if frame is None:
                return BackendResult(success=False, error="no preprocessed frames")
            from PIL import Image

            image = Image.open(frame).convert("RGBA").resize(
                (preset["gen_resolution"], preset["gen_resolution"])
            )
            mesh, _ = model.run_image(
                image,
                bake_resolution=512 if not gen_cfg.get("texture", False) else 1024,
                remesh="triangle",
                vertex_count=200_000,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(out_path)
            return BackendResult(
                success=True,
                mesh_path=out_path,
                settings={"backend": self.name, "device": device, "remesh": "triangle"},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("stable_fast_3d failed: %s", exc)
            return BackendResult(success=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            free_memory()
