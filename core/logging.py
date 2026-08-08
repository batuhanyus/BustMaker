"""Logging setup for the pipeline.

Every stage logs through the standard ``logging`` module on the
``bustforge`` logger tree. In addition to console/file output, a
:class:`ProgressHub` collects stage events in memory and on disk (JSONL) so
the CLI and the Gradio GUI can stream progress identically.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

LOGGER_NAME = "bustforge"

# ---------------------------------------------------------------------------
# File + console logging
# ---------------------------------------------------------------------------


def setup_logging(level: str = "INFO", log_dir: Optional[Path] = None) -> logging.Logger:
    """Configure the root ``bustforge`` logger once. Safe to call repeatedly."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        file_h.setFormatter(fmt)
        logger.addHandler(file_h)

    # Keep third-party logs (torch, huggingface, onnxruntime) quieter.
    for noisy in ("urllib3", "PIL", "huggingface_hub", "onnxruntime", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(LOGGER_NAME + (f".{name}" if name else ""))


# ---------------------------------------------------------------------------
# Progress hub: stage events for CLI/GUI streaming
# ---------------------------------------------------------------------------


@dataclass
class ProgressHub:
    """Collects stage lifecycle events; each event is a JSON-serializable dict.

    Event shape::

        {"ts": 1730000000.0, "stage": "preprocess", "status": "started",
         "message": "extracting frames...", "progress": 0.0}

    ``status`` is one of ``started | progress | finished | failed | skipped``.
    """

    log_path: Optional[Path] = None
    _events: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._events = []
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        stage: str,
        status: str,
        message: str = "",
        progress: Optional[float] = None,
    ) -> None:
        ev = {
            "ts": time.time(),
            "stage": stage,
            "status": status,
            "message": message,
            "progress": progress,
        }
        self._events.append(ev)
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def as_progress_callback(self) -> Callable[[str, float, str], None]:
        """Return a ``(message, fraction, stage_name)`` callback for stages."""

        def cb(message: str, fraction: float, stage_name: str = "") -> None:
            self.emit(stage_name or "pipeline", "progress", message, fraction)

        return cb


__all__ = ["LOGGER_NAME", "setup_logging", "get_logger", "ProgressHub"]
