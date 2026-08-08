"""Depth Anything V2 adapter: monocular depth estimation on a single frame.

Used by the depth-relief fallback. Runs the local transformers checkpoint
(``models/depth_anything_v2``) via the ``transformers`` pipeline API, or a
direct ``DptForDepthEstimation`` forward as fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from core.config import Config
from core.logging import get_logger
from core.model_manager import select_device

log = get_logger("depth")

_pipeline_cache: dict = {}


def estimate_depth(cfg: Config, image: Image.Image, size: str = "small") -> np.ndarray:
    """Return a float32 depth map (H, W) in [0, 1] (near -> far).

    Uses the local checkpoint in models/depth_anything_v2; never downloads.
    """
    key = (size,)
    if key not in _pipeline_cache:
        from transformers import pipeline

        model_dir = cfg.resolve_path("paths.models", "./models") / "depth_anything_v2"
        if not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"Depth Anything V2 weights missing at {model_dir} "
                "(run fetch_dependencies.py)"
            )
        device = select_device(cfg)
        _pipeline_cache[key] = pipeline(
            "depth-estimation", model=str(model_dir), device=device
        )
    pipe = _pipeline_cache[key]

    result = pipe(image.convert("RGB"))
    depth = result["depth"]  # PIL image
    arr = np.asarray(depth, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return arr


def clear_cache() -> None:
    _pipeline_cache.clear()
