"""Keyframe selection tests: blur/exposure verdicts, capping, metadata."""

import numpy as np
from PIL import Image

from scripts.preprocess.select_keyframes import (
    assess_frame,
    select_keyframes,
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
    import json

    data = json.loads(meta.read_text())
    assert data["summary"]["accepted"] == 1
    assert data["frames"][0]["file"] == "f.jpg"
