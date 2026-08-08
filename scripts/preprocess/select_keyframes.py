"""Keyframe selection: filter frames by blur and exposure quality.

Used after raw frame extraction (video) and on still photos. Each frame gets:

* a sharpness score = variance of the Laplacian (higher = sharper),
* an exposure verdict from mean luma (configurable band),
* a final accept/reject decision.

When more frames survive than allowed, the survivors are evenly sub-sampled
*in temporal order* — photogrammetry (COLMAP) benefits from an ordered,
evenly-spaced view set.

Multi-view support (generative backends): when ``estimate_yaws=True``, every
accepted frame also gets an optional head-yaw estimate (mediapipe FaceMesh,
decomposed from nose-vs-ear 3D landmarks). :func:`fill_yaws_by_temporal_interpolation`
and :func:`select_viewpoints` then turn that into a compact front/left/back/right
view set for multi-view image-to-3D (Hunyuan3D-2mv).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from core.logging import get_logger

log = get_logger("keyframes")

# Target views for multi-view fusion: (tag, target_yaw_deg). yaw convention:
# 0 = facing camera, +90 = subject's left, 180 = back, -90 = subject's right.
TARGET_VIEWS: tuple[tuple[str, float], ...] = (
    ("front", 0.0),
    ("left", 90.0),
    ("back", 180.0),
    ("right", -90.0),
)


@dataclass
class FrameVerdict:
    path: Path
    sharpness: float
    mean_luma: float
    accepted: bool
    reason: str = ""
    index: int = 0
    yaw: Optional[float] = None  # degrees (-180..180); None = no face visible


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
                    "yaw": v.yaw,
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


# ---------------------------------------------------------------------------
# Head-yaw estimation + multi-view viewpoint selection
# ---------------------------------------------------------------------------


def _circular_distance(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def estimate_frame_yaws(paths: list[Path]) -> list[Optional[float]]:
    """Estimate head yaw (degrees) per frame with mediapipe FaceLandmarker.

    Returns a list aligned with ``paths``; ``None`` when no face is visible or
    when mediapipe / the landmarker model is unavailable (callers fall back to
    temporal sampling). The model file is fetched by fetch_dependencies.py to
    ``models/mediapipe/face_landmarker.task`` (Google's official float16
    FaceLandmarker, ~3.6 MB).

    Yaw convention: 0 = facing the camera, +90 = subject's left, 180 = back,
    -90 = subject's right (angle of the nose-tip vector relative to the ear
    midpoint, using the landmarker's metric-ish z-depth).
    """
    if not paths:
        return []
    try:
        import mediapipe as mp  # noqa: PLC0415
        from mediapipe.tasks import python as mp_python  # noqa: PLC0415
        from mediapipe.tasks.python import vision  # noqa: PLC0415
    except ImportError:
        log.info("mediapipe not installed; skipping yaw estimation")
        return [None] * len(paths)

    model = _landmarker_model_path()
    if model is None:
        log.info("face_landmarker.task missing; skipping yaw estimation")
        return [None] * len(paths)

    yaws: list[Optional[float]] = []
    try:
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            num_faces=1,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        for p in paths:
            try:
                mp_image = mp.Image.create_from_file(str(p))
            except Exception:  # noqa: BLE001 - unreadable frame
                yaws.append(None)
                continue
            result = landmarker.detect(mp_image)
            if not result.face_landmarks:
                yaws.append(None)
                continue
            lm = result.face_landmarks[0]
            # nose tip (1); approximate ears (454 right, 234 left in image)
            nose, ear_l, ear_r = lm[1], lm[454], lm[234]
            mid_x = (ear_l.x + ear_r.x) / 2.0
            mid_z = (ear_l.z + ear_r.z) / 2.0
            dx = nose.x - mid_x
            dz = nose.z - mid_z  # negative when facing the camera
            yaw = math.degrees(math.atan2(dx, -dz))
            yaws.append(round(yaw, 1))
    except Exception as exc:  # noqa: BLE001 - never break ingest on a helper
        log.warning("yaw estimation failed (%s); continuing without yaws", exc)
        return [None] * len(paths)
    log.info("yaw estimates: %d/%d frames with face",
             sum(y is not None for y in yaws), len(yaws))
    return yaws


def _landmarker_model_path() -> Optional[Path]:
    """models/mediapipe/face_landmarker.task relative to the project root."""
    root = Path(__file__).resolve().parent.parent.parent
    model = root / "models" / "mediapipe" / "face_landmarker.task"
    return model if model.is_file() else None


def fill_yaws_by_temporal_interpolation(
    yaws: list[Optional[float]],
) -> list[Optional[float]]:
    """Fill unknown yaws between real detections by circular interpolation.

    Walkaround videos are roughly monotonic in viewpoint angle, so a frame
    between two face-detected frames is assumed to sit between their yaws
    along the *shortest* arc (correct across the +/-180 wrap, e.g. 170 deg ->
    -170 deg passes through the back at 180). Nothing is fabricated beyond
    the first/last detection — entries there stay None so downstream logic
    never mistakes an extrapolated guess for a real viewpoint. Returns a
    copy; values are normalized to (-180, 180].
    """
    known = [(i, y) for i, y in enumerate(yaws) if y is not None]
    if len(known) < 2:
        return list(yaws)
    out: list[Optional[float]] = list(yaws)
    for (i1, y1), (i2, y2) in zip(known, known[1:]):
        delta = (y2 - y1 + 180.0) % 360.0 - 180.0  # shortest signed arc
        for i in range(i1 + 1, i2):
            if out[i] is None:
                t = (i - i1) / (i2 - i1)
                out[i] = round(((y1 + delta * t + 180.0) % 360.0) - 180.0, 1)
    return out


def _circular_spread(yaws: list[float]) -> float:
    """Angular spread of a set of yaws (0..360); 0 when <2 values."""
    if len(yaws) < 2:
        return 0.0
    s = sorted(yaws)
    gaps = [
        (s[(i + 1) % len(s)] - s[i]) % 360.0
        for i in range(len(s))
    ]
    return 360.0 - max(gaps)


def select_viewpoints(
    yaws: list[Optional[float]],
    max_views: int = 4,
    min_separation_deg: float = 25.0,
    yaw_guided_only: bool = False,
) -> list[tuple[int, str]]:
    """Pick up to ``max_views`` frames for multi-view fusion.

    Returns ``[(frame_index, view_tag), ...]`` with tags from
    front/left/back/right (see :data:`TARGET_VIEWS`).

    Two regimes:

    * **yaw-guided** — when face-detected yaws span >= 60 deg, each target
      view takes the frame closest to its yaw (with ``min_separation_deg``
      enforced so the views stay distinct, and a 60 deg tag-coverage gate so
      a tag is omitted rather than mislabeled).
    * **temporal walkaround** — when yaws are missing or clustered (e.g. the
      face is only visible during the frontal arc of a 360 deg orbit), the
      front view is anchored at the face-visible frame closest to 0 deg (or
      the first frame when no face was ever seen) and the other views sit at
      quarter-turn positions of the orbit (left/back/right). These tags are
      *guesses* — ``yaw_guided_only=True`` (used by multi-view model
      conditioning, which is sensitive to wrong view semantics) returns []
      in this regime instead.
    """
    n = len(yaws)
    if n == 0:
        return []
    usable = [(i, y) for i, y in enumerate(yaws) if y is not None]
    if usable and _circular_spread([y for _, y in usable]) >= 60.0:
        # --- yaw-guided quadrant selection ------------------------------
        chosen_idx: list[int] = []
        chosen_yaws: list[float] = []
        picks: list[tuple[int, str]] = []
        for tag, target in TARGET_VIEWS:
            if len(chosen_idx) >= max_views:
                break
            candidates = [
                (i, y) for i, y in usable
                if all(_circular_distance(y, cy) >= min_separation_deg for cy in chosen_yaws)
            ]
            if not candidates:
                continue
            best_i, best_y = min(candidates, key=lambda t: _circular_distance(t[1], target))
            # Tag-coverage gate: a tag whose best frame is far from its target
            # yaw is a mislabel (e.g. calling a 55 deg profile "back") — the
            # mv model is sensitive to wrong view semantics, so omit the tag.
            if _circular_distance(best_y, target) > 60.0:
                continue
            chosen_idx.append(best_i)
            chosen_yaws.append(best_y)
            picks.append((best_i, tag))
        picks.sort(key=lambda t: t[0])
        return picks

    # --- temporal walkaround selection ----------------------------------
    if yaw_guided_only:
        return []
    if usable:
        front_i = min(usable, key=lambda t: abs(t[1]))[0]
    else:
        front_i = 0
    picks: list[tuple[int, str]] = [(front_i, "front")]
    if n >= 2:
        step = n / 4.0
        for k, tag in ((1, "left"), (2, "back"), (3, "right")):
            picks.append((round((front_i + k * step) % n), tag))
    seen: set[int] = set()
    deduped: list[tuple[int, str]] = []
    for i, tag in picks:
        if i in seen:
            continue
        seen.add(i)
        deduped.append((i, tag))
    return deduped[:max_views]


def attach_yaws(verdicts: list[FrameVerdict], yaws: list[Optional[float]]) -> None:
    """Mutate verdicts in place with yaw estimates aligned by index."""
    for v, y in zip(verdicts, yaws):
        v.yaw = y


def select_keyframes(
    frame_paths: list[Path],
    max_frames: int,
    blur_threshold: float = 60.0,
    exposure_ok_range: tuple[float, float] = (25.0, 235.0),
    progress: Optional[Callable[[str, float], None]] = None,
    order_by_quality: bool = False,
    estimate_yaws: bool = False,
) -> SelectionResult:
    """Assess every frame, reject bad ones, then cap to ``max_frames``.

    ``order_by_quality=False`` keeps temporal order (photogrammetry-friendly).
    ``estimate_yaws=True`` additionally attaches head-yaw estimates to the
    accepted verdicts (mediapipe; needed for multi-view generative input).
    """
    verdicts = [
        assess_frame(p, blur_threshold, exposure_ok_range, i)
        for i, p in enumerate(frame_paths)
    ]
    accepted = [v for v in verdicts if v.accepted]
    rejected = [v for v in verdicts if not v.accepted]

    if rejected:
        reasons = _reason_counts(rejected)
        sharpness_vals = [v.sharpness for v in rejected]
        log.info("quality rejects: %s | sharpness range [%.1f, %.1f]",
                 reasons, min(sharpness_vals), max(sharpness_vals))

    if len(accepted) > max_frames:
        if order_by_quality:
            accepted.sort(key=lambda v: v.sharpness, reverse=True)
            accepted = accepted[:max_frames]
            accepted.sort(key=lambda v: v.index)
        else:
            step = len(accepted) / max_frames
            accepted = [accepted[round(i * step)] for i in range(max_frames)]
        log.info("capped keyframes to %d (from %d good)", len(accepted), len(accepted) + len(rejected))

    # Best-effort fallback: if every frame fails quality checks (e.g. phone
    # video with motion blur where nothing reaches the threshold), keep the
    # sharpest frames anyway so the run can proceed.
    if not accepted:
        usable = [v for v in verdicts if v.reason != "unreadable"]
        if usable:
            log.warning(
                "all %d frames failed quality checks; falling back to %d sharpest",
                len(verdicts), min(max_frames, len(usable)),
            )
            usable.sort(key=lambda v: v.sharpness, reverse=True)
            accepted = usable[:max_frames]
            for v in accepted:
                v.accepted = True
                v.reason = f"fallback_{v.reason}"
            accepted.sort(key=lambda v: v.index)  # temporal order (COLMAP-friendly)
            rejected = [v for v in verdicts if not v.accepted]

    # Yaw estimation runs last so the fallback path gets estimates too.
    if estimate_yaws and accepted:
        yaws = estimate_frame_yaws([v.path for v in accepted])
        attach_yaws(accepted, yaws)

    if progress:
        progress("keyframe selection done", 1.0)
    log.info("keyframes: %d accepted, %d rejected", len(accepted), len(rejected))
    return SelectionResult(accepted=accepted, rejected=rejected)
