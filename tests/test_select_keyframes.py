"""Keyframe selection tests: blur/exposure verdicts, capping, metadata,
viewpoint (yaw) selection."""

import json

import numpy as np
from PIL import Image

from scripts.preprocess.select_keyframes import (
    TARGET_VIEWS,
    assess_frame,
    fill_yaws_by_temporal_interpolation,
    select_keyframes,
    select_viewpoints,
)


def _make_image(path, sharp=True, luma=120):
    rng = np.random.default_rng(0)
    if sharp:
        if luma == 120:
            arr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        else:
            # full-contrast noise around a shifted mean (JPEG would smooth
            # low-amplitude noise into a flat image)
            arr = np.clip(luma + (rng.random((64, 64, 3)) - 0.5) * 120, 0, 255).astype(np.uint8)
    else:
        arr = np.full((64, 64, 3), luma, dtype=np.uint8)  # flat = blurry
    Image.fromarray(arr).save(path)


def test_assess_blur_and_exposure(tmp_path):
    sharp = tmp_path / "sharp.jpg"
    blur = tmp_path / "blur.jpg"
    dark = tmp_path / "dark.jpg"
    _make_image(sharp, sharp=True)
    _make_image(blur, sharp=False)
    _make_image(dark, sharp=True, luma=5)  # sharp but underexposed

    assert assess_frame(sharp, 60.0, (25, 235)).accepted
    assert assess_frame(blur, 60.0, (25, 235)).reason == "blurry"
    assert assess_frame(dark, 60.0, (25, 235)).reason == "bad_exposure"


def test_capping_keeps_order(tmp_path):
    paths = []
    for i in range(10):
        p = tmp_path / f"f{i:02d}.jpg"
        _make_image(p)
        paths.append(p)
    res = select_keyframes(paths, max_frames=4, blur_threshold=1.0)
    assert len(res.accepted) == 4
    indices = [v.index for v in res.accepted]
    assert indices == sorted(indices)  # temporal order preserved


def test_metadata_json(tmp_path):
    p = tmp_path / "f.jpg"
    _make_image(p)
    res = select_keyframes([p], max_frames=5, blur_threshold=1.0)
    meta = tmp_path / "meta.json"
    res.save_metadata(meta)
    data = json.loads(meta.read_text())
    assert data["summary"]["accepted"] == 1
    assert data["frames"][0]["file"] == "f.jpg"
    assert data["frames"][0]["yaw"] is None


# ---------------------------------------------------------------------------
# Viewpoint (yaw) selection
# ---------------------------------------------------------------------------


def test_select_viewpoints_picks_each_quadrant():
    # one frame per target view: front(0), left(90), back(180), right(-90)
    yaws = [0.0, 90.0, 180.0, -90.0, 45.0]
    picks = select_viewpoints(yaws, max_views=4)
    assert picks == [(0, "front"), (1, "left"), (2, "back"), (3, "right")]


def test_select_viewpoints_no_yaw_falls_back_to_walkaround():
    # no face ever visible: front = first frame, others at quarter turns
    picks = select_viewpoints([None] * 40, max_views=4)
    assert picks[0] == (0, "front")
    assert [tag for _, tag in picks] == ["front", "left", "back", "right"]
    assert len({i for i, _ in picks}) == 4  # distinct frames
    assert select_viewpoints([None], max_views=4) == [(0, "front")]


def test_select_viewpoints_clustered_yaws_use_walkaround():
    # faces only visible in a narrow frontal arc -> temporal walkaround,
    # front anchored at the face-visible frame closest to 0 deg
    yaws = [None, None, 5.0, None, None, None, None, None, 0.0]
    picks = select_viewpoints(yaws, max_views=4)
    assert picks[0] == (8, "front")  # yaw 0.0 anchor wins over yaw 5.0
    assert len(picks) == 4


def test_select_viewpoints_enforces_min_separation():
    # narrow spread (< 60 deg) -> walkaround regime, front at the frame
    # closest to 0 deg, others at quarter turns of the orbit
    picks = select_viewpoints([0.0, -5.0, 10.0, 20.0], max_views=4)
    assert picks[0] == (0, "front")  # yaw 0 is closest to 0
    assert [tag for _, tag in picks] == ["front", "left", "back", "right"]
    assert len({i for i, _ in picks}) == 4


def test_select_viewpoints_spread_yaws_use_quadrants():
    # wide spread -> yaw-guided: front + left picked, rest too close
    picks = select_viewpoints([0.0, 5.0, 100.0], max_views=4)
    assert picks[0] == (0, "front")  # closest to front target
    assert (2, "left") in picks  # 100 deg -> left
    assert len(picks) == 2


def test_fill_yaws_interpolates_without_fabrication():
    filled = fill_yaws_by_temporal_interpolation(
        [None, 0.0, None, None, 90.0, None, None, None]
    )
    assert filled[2] == 30.0 and filled[3] == 60.0
    # nothing beyond the last anchor and before the first anchor
    assert filled[0] is None
    assert filled[5] is None and filled[6] is None and filled[7] is None
    assert fill_yaws_by_temporal_interpolation([None, None]) == [None, None]
    assert fill_yaws_by_temporal_interpolation([None, 5.0, None]) == [None, 5.0, None]


def test_fill_yaws_circular_wrap_via_back():
    # 170 -> -170 is a 20 deg arc THROUGH the back (180), not through front
    filled = fill_yaws_by_temporal_interpolation([170.0, None, None, -170.0])
    assert filled[1] == 176.7  # 20 deg arc through the back
    assert filled[2] == -176.7
    # shortest-arc rule: 170 -> 10 is -160 deg (through 90), not +200
    filled2 = fill_yaws_by_temporal_interpolation([170.0, None, 10.0])
    assert filled2[1] == 90.0


def test_select_viewpoints_yaw_guided_only_rejects_guesses():
    # clustered yaws -> temporal regime -> [] when guesses are not allowed
    assert select_viewpoints([0.0, -5.0, 10.0, 20.0], max_views=4, yaw_guided_only=True) == []
    assert select_viewpoints([None] * 10, max_views=4, yaw_guided_only=True) == []
    # wide spread -> yaw-guided regime still allowed
    picks = select_viewpoints([0.0, 90.0, 180.0], max_views=4, yaw_guided_only=True)
    assert len(picks) == 3


def test_select_viewpoints_tag_coverage_gate():
    # 160 deg frontal arc (-80..+80): front/left/right are honest tags, but
    # no frame is within 60 deg of "back" -> the tag must be omitted
    picks = select_viewpoints([-80.0, -40.0, 0.0, 40.0, 80.0], max_views=4)
    tags = [t for _, t in picks]
    assert "front" in tags and "left" in tags and "right" in tags
    assert "back" not in tags


def test_target_views_cover_the_compass():
    assert [tag for tag, _ in TARGET_VIEWS] == ["front", "left", "back", "right"]
