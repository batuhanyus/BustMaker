"""Hunyuan3D-2.1 generative backend (primary for balanced/high presets).

Runs the official ``hy3dgen`` inference code (vendored under
``vendor/hunyuan3d/``, pinned to Tencent/Hunyuan3D-2 commit
``f8db63096c8282cb27354314d896feba5ba6ff8a``) with the local checkpoints from
``models/hunyuan3d/``::

    models/hunyuan3d/
    ├── dit-v2-1/      (config.yaml + model.fp16.ckpt, 7.4 GB)   # required
    ├── vae-v2-1/      (config.yaml + model.fp16.ckpt, 656 MB)   # optional spare
    └── LICENSE

The dit-v2-1 checkpoint bundles the shape VAE and the DINOv2 conditioner
weights, so ``vae-v2-1/`` is not required at runtime. The official HF layout
(``hunyuan3d-dit-v2-1/``) is also accepted.

VRAM ladder: fp16 cuda -> fp32 cuda -> fp16 + accelerate model CPU offload
-> CPU. The fp32 rung will normally OOM on a 12 GB card for this 3B-param
model and simply advances to the offload rung.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image

from adapters.generative.base import BackendResult, MeshBackend, is_available
from core.config import Config
from core.logging import get_logger
from core.model_manager import free_memory, oom_retry_ladder, select_device
from core.pipeline import RunContext

log = get_logger("hunyuan3d")

_VENDOR = Path(__file__).resolve().parent.parent.parent / "vendor" / "hunyuan3d"

# ---------------------------------------------------------------------------
# RAM-safe model construction
# ---------------------------------------------------------------------------
# hy3dgen's from_single_file instantiates the DiT/VAE/conditioner in fp32
# (~16 GB for the 3B DiT) while the 7.3 GB fp16 checkpoint is still resident,
# which exceeds 32 GB of system RAM and segfaults. Since every weight comes
# from the fp16 ckpt, constructing directly under fp16 default dtype is
# numerically identical (fp16 -> fp32 -> fp16 would round-trip exactly) and
# cuts the peak by ~8 GB. The pipeline is cast afterwards if a fp32 device
# (CPU rung) is requested.
_patched = False


def _patch_fp16_construction() -> None:
    global _patched
    if _patched:
        return
    import hy3dgen.shapegen.pipelines as hy_pipelines  # noqa: PLC0415

    original = hy_pipelines.Hunyuan3DDiTFlowMatchingPipeline.from_single_file.__func__

    def wrapper(cls, ckpt_path, config_path, device="cuda", dtype=torch.float16,
                use_safetensors=None, **kwargs):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        try:
            pipeline = original(
                cls, ckpt_path, config_path,
                device=device, dtype=torch.float16,
                use_safetensors=use_safetensors, **kwargs,
            )
        finally:
            torch.set_default_dtype(prev)
        if dtype != torch.float16:
            pipeline.to(dtype=dtype)
        return pipeline

    hy_pipelines.Hunyuan3DDiTFlowMatchingPipeline.from_single_file = classmethod(wrapper)
    _patched = True


# Official HF repo uses the "hunyuan3d-" prefix; the canonical local layout
# (see fetch_dependencies.py) drops it. Accept both.
_DIT_SUBFOLDERS = ("dit-v2-1", "hunyuan3d-dit-v2-1")
_MV_SUBFOLDERS = ("dit-v2-mv", "hunyuan3d-dit-v2-mv")

def find_dit_subfolder(models_dir: Path) -> Optional[str]:
    """Return the subfolder name that actually contains the DiT weights."""
    for name in _DIT_SUBFOLDERS:
        d = models_dir / "hunyuan3d" / name
        if (d / "model.fp16.ckpt").is_file() and (d / "config.yaml").is_file():
            return name
    return None


def find_mv_subfolder(models_dir: Path) -> Optional[str]:
    """Return the multi-view (Hunyuan3D-2mv) subfolder, or None if absent.

    The mv model is optional: without it the backend falls back to
    single-image conditioning with dit-v2-1.
    """
    for name in _MV_SUBFOLDERS:
        d = models_dir / "hunyuan3d-mv" / name
        if (d / "model.fp16.ckpt").is_file() and (d / "config.yaml").is_file():
            return name
    return None


def _select_multiview_views(ctx: RunContext, frames: list[Path]) -> Optional[dict[str, Path]]:
    """Pick up to 4 viewpoint frames (front/left/back/right) for mv conditioning.

    Returns None when fewer than 2 distinct views are available or the mv
    weights are missing — callers then use single-image conditioning.
    """
    if len(frames) < 2:
        return None
    from scripts.preprocess.select_keyframes import (  # noqa: PLC0415
        fill_yaws_by_temporal_interpolation,
        select_viewpoints,
    )

    raw_yaws: list[Optional[float]] = ctx.shared("frame_yaws", []) or []
    if len(raw_yaws) != len(frames):
        raw_yaws = [None] * len(frames)
    filled = fill_yaws_by_temporal_interpolation(raw_yaws)
    # yaw_guided_only: the mv model is sensitive to wrong view semantics, so
    # never feed it views whose tags are temporal guesses.
    selections = select_viewpoints(filled, max_views=min(4, len(frames)), yaw_guided_only=True)
    if len(selections) < 2:
        return None
    views: dict[str, Path] = {}
    for i, tag in selections:
        views.setdefault(tag, frames[i])  # first frame per tag wins
    if len(views) < 2:
        return None
    log.info("multiview input: %s", {k: v.name for k, v in views.items()})
    return views


class Hunyuan3DBackend(MeshBackend):
    name = "hunyuan3d"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # -- availability ----------------------------------------------------------

    def available(self) -> bool:
        return is_available(self.cfg, "hunyuan3d")

    # -- implementation ----------------------------------------------------------

    def generate(self, ctx: RunContext, input_frames: list[Path], out_path: Path) -> BackendResult:
        if not self.available():
            return BackendResult(success=False, error="hunyuan3d weights missing (see README / fetch_dependencies.py)")

        preset = self.cfg.preset(ctx.preset)
        gen_cfg = self.cfg.generative_cfg
        models_dir = self.cfg.resolve_path("paths.models", "./models")
        subfolder = find_dit_subfolder(models_dir)
        if subfolder is None:
            return BackendResult(success=False, error="hunyuan3d weights missing")
        mv_subfolder = find_mv_subfolder(models_dir)

        # Multi-view conditioning (Hunyuan3D-2mv) when enough distinct views
        # exist; otherwise single best frame through dit-v2-1.
        views = _select_multiview_views(ctx, input_frames) if mv_subfolder else None
        if views is not None:
            cond = views
            cond_desc = {"multiview": True, "views": {k: v.name for k, v in views.items()}}
        else:
            frame = _pick_best_frame(ctx, input_frames)
            if frame is None:
                return BackendResult(success=False, error="no preprocessed frames to reconstruct")
            cond = frame
            cond_desc = {"multiview": False, "views": 1}

        octree = int(preset.get("octree_resolution", 384))
        steps = int(preset.get("gen_steps", 50))
        seed = int(gen_cfg.get("seed", 42))

        for settings in oom_retry_ladder(self.cfg):
            log.info(
                "hunyuan3d attempt: %s (device=%s, fp16=%s, octree=%d, steps=%d, %s)",
                settings["step"], settings["device"], settings["fp16"], octree, steps,
                cond_desc,
            )
            try:
                mesh = self._infer(cond, subfolder, mv_subfolder, models_dir,
                                   octree, steps, seed, settings,
                                   progress=self._progress_cb(ctx))
            except Exception as exc:  # noqa: BLE001 - try next ladder rung
                free_memory()
                log.warning("hunyuan3d %s failed: %s", settings["step"], exc)
                if _is_oom(exc) and settings["step"] != "cpu":
                    continue
                return BackendResult(success=False, error=f"{type(exc).__name__}: {exc}")
            finally:
                free_memory()

            if mesh is None or len(mesh.vertices) == 0:
                return BackendResult(success=False, error="hunyuan3d produced an empty mesh")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(out_path)
            return BackendResult(
                success=True,
                mesh_path=out_path,
                settings={
                    "backend": self.name,
                    "device": settings["device"],
                    "fp16": settings["fp16"],
                    "low_vram": settings["low_vram"],
                    "octree_resolution": octree,
                    "num_inference_steps": steps,
                    "subfolder": subfolder,
                    **cond_desc,
                    "vertices": int(len(mesh.vertices)),
                    "faces": int(len(mesh.faces)),
                },
            )

        return BackendResult(success=False, error="hunyuan3d: all retry rungs failed")

    # -- internals ---------------------------------------------------------------

    @staticmethod
    def _progress_cb(ctx: RunContext) -> Callable[[str, float], None]:
        """Return a (message, fraction) reporter wired to the pipeline's ProgressHub.

        Never raises: progress reporting must not be able to abort inference.
        """
        def cb(message: str, fraction: float) -> None:
            try:
                ctx.progress.emit("generate_mesh", "progress", message, fraction)
            except Exception:  # noqa: BLE001 - progress must never break inference
                log.debug("progress emit failed: %s", message, exc_info=True)
        return cb

    def _infer(self, cond, subfolder: str, mv_subfolder: Optional[str], models_dir: Path,
               octree: int, steps: int, seed: int, settings: dict,
               progress: Optional[Callable[[str, float], None]] = None):
        """Load the pipeline and generate one mesh. Returns a trimesh.Trimesh.

        ``cond`` is either a single frame Path (dit-v2-1) or a
        {view_tag: Path} dict (Hunyuan3D-2mv multi-view).
        ``progress`` is an optional ``(message, fraction)`` reporter called
        with per-denoise-step updates and phase-boundary messages.
        """
        def safe_progress(message: str, fraction: float) -> None:
            if progress is None:
                return
            try:
                progress(message, fraction)
            except Exception:  # noqa: BLE001 - progress must never break inference
                pass
        if not (_VENDOR / "hy3dgen").is_dir():
            raise RuntimeError("vendored hy3dgen code missing (run fetch_dependencies.py)")
        sys.path.insert(0, str(_VENDOR))
        # smart_load_model resolves local weights via HY3DGEN_MODELS + model_path + subfolder
        os.environ.setdefault("HY3DGEN_MODELS", str(models_dir))
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: PLC0415
        _patch_fp16_construction()

        multiview = isinstance(cond, dict)
        model_path = "hunyuan3d-mv" if multiview else "hunyuan3d"
        model_subfolder = mv_subfolder if multiview else subfolder

        dtype = torch.float16 if (settings["fp16"] or settings["low_vram"]) else torch.float32
        device = select_device(self.cfg, prefer_cpu=(settings["device"] == "cpu"))

        safe_progress(f"hunyuan3d: loading weights ({device}, fp16={settings['fp16']})...", 0.0)
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=model_subfolder,
            use_safetensors=False,   # weights are model.fp16.ckpt
            variant="fp16",
            device=device,
            dtype=dtype,
        )
        if settings["low_vram"] and device == "cuda":
            pipeline.enable_model_cpu_offload()

        image = (
            {tag: Image.open(p).convert("RGBA") for tag, p in cond.items()}
            if multiview else Image.open(cond).convert("RGBA")
        )
        # Exactly `steps` callbacks fire (one per denoise iteration, i in 0..steps-1),
        # so counting calls is robust even if the scheduler order > 1 skews step_idx.
        calls = [0]

        def on_step(step_idx: int, t, outputs) -> None:
            calls[0] += 1
            fraction = min(1.0, calls[0] / steps)
            if calls[0] >= steps:
                # Denoise finished; the rest of the call is the VAE volume
                # decode (no callback hook there) — announce it instead.
                safe_progress(
                    f"hunyuan3d: extracting mesh volume (octree={octree})...",
                    fraction,
                )
            else:
                safe_progress(
                    f"hunyuan3d denoise {calls[0]}/{steps}",
                    fraction,
                )

        safe_progress("hunyuan3d: encoding condition...", 0.0)
        meshes = pipeline(
            image=image,
            num_inference_steps=steps,
            octree_resolution=octree,
            generator=torch.manual_seed(seed),
            output_type="trimesh",
            enable_pbar=False,   # progress is reported through `callback` instead
            callback=on_step,
            callback_steps=1,
        )
        mesh = meshes[0] if meshes else None
        if isinstance(mesh, (list, tuple)):
            mesh = mesh[0] if mesh else None
        return mesh


def _pick_best_frame(ctx: RunContext, frames: list[Path]) -> Optional[Path]:
    """Single-image backend: use the first preprocessed frame (ingest already
    quality-filtered and sorted the set)."""
    return frames[0] if frames else None


def _is_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or ("cuda" in text and "memory" in text)
