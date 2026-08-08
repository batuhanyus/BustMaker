"""Background removal adapter (rembg, u2net default).

The interface is deliberately small so RVM/SAM2 adapters can be added later
with the same surface::

    remove(image_path, out_path, cfg, model="u2net") -> MaskResult

Integrity details:

* rembg is told where its weights live via the ``U2NET_HOME`` env var, which
  points at ``models/rembg/`` (pre-fetched by scripts/setup/fetch_dependencies.py).
  The env var must be set *before* rembg is imported, so we import lazily.
* onnxruntime-GPU is used when available; a session-level fallback retries on
  CPU providers automatically (low-VRAM/robustness requirement).
* Every mask gets a quality score (foreground fraction, bounding-box coverage)
  so the ingest stage can flag weak masks instead of failing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from core.config import Config
from core.logging import get_logger

log = get_logger("rembg")

_session_cache: dict[str, object] = {}


def _configure_env(cfg: Config) -> None:
    """Point rembg at our pre-fetched models dir (must run before import)."""
    models_dir = cfg.resolve_path("paths.models", "./models") / "rembg"
    os.environ.setdefault("U2NET_HOME", str(models_dir))


def _get_session(cfg: Config, model: str):
    _configure_env(cfg)
    if model in _session_cache:
        return _session_cache[model]
    from rembg import new_session

    session = None
    try:
        session = new_session(model)  # GPU providers preferred automatically
    except Exception as exc:  # noqa: BLE001 - fall back to CPU providers
        log.warning("rembg GPU session failed (%s); retrying with CPU providers", exc)
        from onnxruntime import get_available_providers

        providers = [p for p in get_available_providers() if p != "CUDAExecutionProvider"]
        session = new_session(model, providers=providers)
    _session_cache[model] = session
    return session


@dataclass
class MaskResult:
    out_path: Path
    foreground_fraction: float
    bbox_coverage: float
    quality: float  # 0..1 heuristic; < config background_removal.min_mask_quality = weak
    used_cpu_fallback: bool = False


def remove_background(
    cfg: Config,
    image_path: Path,
    out_path: Path,
    model: str = "u2net",
    mask_dir: Optional[Path] = None,
) -> MaskResult:
    """Cut the subject out of ``image_path`` and save RGBA PNG to ``out_path``.

    ``mask_dir`` (optional): also write the raw binary mask there for debug.
    """
    # 1. load + normalize orientation
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    # 2. matting (rembg >= 2.0.7x predict() takes a PIL image, may return (1,H,W))
    session = _get_session(cfg, model)
    alpha = np.squeeze(np.asarray(session.predict(img)))
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    # 3. composite RGBA
    rgba = np.dstack([np.asarray(img), alpha])
    out = Image.fromarray(rgba, mode="RGBA")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)

    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(alpha).save(mask_dir / (out_path.stem + "_mask.png"))

    quality, fg_frac, bbox_cov = _mask_quality(alpha)
    result = MaskResult(
        out_path=out_path,
        foreground_fraction=fg_frac,
        bbox_coverage=bbox_cov,
        quality=quality,
    )
    if quality < cfg.get("background_removal.min_mask_quality", 0.5):
        log.warning("weak mask for %s (quality=%.2f)", image_path.name, quality)
    return result


def _mask_quality(alpha: np.ndarray) -> tuple[float, float, float]:
    """Heuristic mask quality in [0,1].

    * foreground fraction: subject should occupy a sane share of the frame,
    * bbox coverage: the subject should span a large part of its bounding box
      (a scattered/dashed mask gets penalized).
    """
    mask = alpha > 127
    total = mask.size
    fg = int(mask.sum())
    fg_frac = fg / total
    if fg == 0:
        return 0.0, 0.0, 0.0
    ys, xs = np.where(mask)
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    bbox_cov = fg / max(bbox_area, 1)
    # Subject should occupy a sane share of the frame; tiny silhouettes are
    # almost always matting errors, so scale the score down with size.
    size_score = min(1.0, fg_frac / 0.15) if fg_frac < 0.15 else 1.0
    if fg_frac > 0.85:  # subject filling the whole frame = mask probably failed
        size_score = 0.5
    quality = float(bbox_cov) * size_score
    return min(quality, 1.0), float(fg_frac), float(bbox_cov)


def clear_cache() -> None:
    _session_cache.clear()
