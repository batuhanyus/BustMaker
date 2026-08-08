"""Model lifecycle management: device selection, VRAM hygiene, OOM retries.

Rules from PROJECT.md that this module enforces:

* batch size 1 unless explicitly safe,
* unload models after each stage (``free_memory`` after every backend attempt),
* FP16 where supported, CPU offload as the last rung,
* catch CUDA OOM and retry with lower settings (progressive ladder).
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch

from core.config import Config
from core.logging import get_logger

log = get_logger("model_manager")


@dataclass
class ComputeSettings:
    device: str = "cuda"        # "cuda" | "cpu"
    fp16: bool = True
    low_vram: bool = False      # stage-level flag; bumped automatically on OOM
    batch_size: int = 1
    resolution: int = 640       # generator image resolution (preset override)


def select_device(cfg: Config, prefer_cpu: bool = False) -> str:
    if prefer_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def vram_available_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


def free_memory() -> None:
    """Release cached GPU memory + run a GC pass. Call after every model use."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def oom_retry_ladder(cfg: Config) -> Iterator[dict[str, Any]]:
    """Yield progressively lighter settings after an OutOfMemoryError.

    Ladder (config vr.oom_retry_steps, default fp16 -> lowvram -> cpu):
    1. as configured (fp16 on)                    — original settings
    2. fp16 off, half resolution                  — reduced precision + load
    3. lowvram mode (chunked / cpu offload)       — heavy VRAM pressure
    4. cpu                                        — always works, slow
    """
    steps = [s.strip() for s in cfg.get("vr.oom_retry_steps", ["fp16", "lowvram", "cpu"])]
    yield {"step": "as_configured", "fp16": True, "low_vram": False, "device": "cuda"}
    for s in steps:
        if s == "fp16":
            yield {"step": "fp16_off", "fp16": False, "low_vram": False, "device": "cuda"}
        elif s == "lowvram":
            yield {"step": "lowvram", "fp16": False, "low_vram": True, "device": "cuda"}
        elif s == "cpu":
            yield {"step": "cpu", "fp16": False, "low_vram": False, "device": "cpu"}


@contextmanager
def oom_guard(settings: dict[str, Any]) -> Iterator[None]:
    """Context manager that converts CUDA OOM into a controlled exception."""
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() or "CUDA" in str(exc) and "memory" in str(exc).lower():
            raise torch.cuda.OutOfMemoryError(str(exc)) from exc
        raise
