"""Keyframe selection: filter frames by blur and exposure quality.

Used after raw frame extraction (video) and on still photos. Each frame gets:

* a sharpness score = variance of the Laplacian (higher = sharper),
* an exposure verdict from mean luma (configurable band),
* a final accept/reject decision.

When more frames survive than allowed, the survivors are evenly sub-sampled
*in temporal order* — photogrammetry (COLMAP) benefits from an ordered,
evenly-spaced view set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from core.logging import get_logger

log = get_logger("keyframes")


@dataclass
class FrameVerdict:
    path: Path
    sharpness: float
    mean_luma: float
    accepted: bool
    reason: str = ""
    index: int = 0


@dataclass
class SelectionResult:
    accepted: list[FrameVerdict] = field(default_factory=list)
    rejected: list[FrameVerdict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "rejected_reasons": _reason_counts(self.rejected),
            "mean_sharpness": round(
                float(np.mean([v.sharpness for v in self.accepted])), 2
            )
            if self.accepted
            else None,
        }

    def save_metadata(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "frames": [
                {
                    "file": v.path.name,
                    "index": v.index,
                    "sharpness": round(float(v.sharpness), 2),
                    "mean_luma": round(float(v.mean_luma), 1),
                    "accepted": v.accepted,
                    "reason": v.reason,
                }
                for v in self.accepted + self.rejected
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


def _reason_counts(rejected: list[FrameVerdict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in rejected:
        counts[v.reason] = counts.get(v.reason, 0) + 1
    return counts


def assess_frame(
    path: Path,
    blur_threshold: float,
    exposure_ok_range: tuple[float, float],
    index: int = 0,
) -> FrameVerdict:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:  # unreadable/corrupt
        return FrameVerdict(path, 0.0, 0.0, False, "unreadable", index)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_luma = float(gray.mean())

    if sharpness < blur_threshold:
        return FrameVerdict(path, sharpness, mean_luma, False, "blurry", index)
    lo, hi = exposure_ok_range
    if not (lo <= mean_luma <= hi):
        return FrameVerdict(path, sharpness, mean_luma, False, "bad_exposure", index)
    return FrameVerdict(path, sharpness, mean_luma, True, "", index)


def select_keyframes(
    frame_paths: list[Path],
    max_frames: int,
    blur_threshold: float = 60.0,
    exposure_ok_range: tuple[float, float] = (25.0, 235.0),
    progress: Optional[Callable[[str, float], None]] = None,
    order_by_quality: bool = False,
) -> SelectionResult:
    """Assess every frame, reject bad ones, then cap to ``max_frames``.

    ``order_by_quality=False`` keeps temporal order (photogrammetry-friendly).
    """
    verdicts = [
        assess_frame(p, blur_threshold, exposure_ok_range, i)
        for i, p in enumerate(frame_paths)
    ]
    accepted = [v for v in verdicts if v.accepted]
    rejected = [v for v in verdicts if not v.accepted]

    if len(accepted) > max_frames:
        if order_by_quality:
            accepted.sort(key=lambda v: v.sharpness, reverse=True)
            accepted = accepted[:max_frames]
            accepted.sort(key=lambda v: v.index)
        else:
            step = len(accepted) / max_frames
            accepted = [accepted[round(i * step)] for i in range(max_frames)]
        log.info("capped keyframes to %d (from %d good)", len(accepted), len(accepted) + len(rejected))

    if progress:
        progress("keyframe selection done", 1.0)
    log.info("keyframes: %d accepted, %d rejected", len(accepted), len(rejected))
    return SelectionResult(accepted=accepted, rejected=rejected)
