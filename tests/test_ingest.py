"""Ingest tests: input normalization (RGBA/alpha inputs must survive the
JPEG re-encode step without raising OSError)."""

import numpy as np
from PIL import Image

from scripts.preprocess.ingest import _exif_normalize_copy


def _rgba(shape=(64, 64), color=(200, 40, 40, 255)):
    arr = np.zeros((*shape, 4), np.uint8)
    arr[16:48, 16:48] = color
    return arr


def test_exif_normalize_copy_flattens_rgba_to_rgb_jpeg(tmp_path):
    src = tmp_path / "input.png"
    Image.fromarray(_rgba(), "RGBA").save(src)
    dst = tmp_path / "out.jpg"

    _exif_normalize_copy(src, dst)

    out = Image.open(dst)
    assert out.mode == "RGB"  # regression: RGBA->JPEG used to raise OSError
    px = out.convert("RGBA").load()
    assert px[0, 0][:3] == (255, 255, 255)  # transparent -> white, not black
    assert px[16, 16][:3] != (0, 0, 0)  # opaque subject kept


def test_exif_normalize_copy_palette_with_transparency(tmp_path):
    src = tmp_path / "pal.png"
    img = Image.new("P", (64, 64), 0)
    img.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    img.putdata([0] * 1024 + [1] * 3072)
    img.info["transparency"] = 0
    img.save(src)
    dst = tmp_path / "out.jpg"

    _exif_normalize_copy(src, dst)

    assert Image.open(dst).mode == "RGB"


def test_exif_normalize_copy_plain_rgb(tmp_path):
    src = tmp_path / "rgb.png"
    Image.fromarray(np.full((32, 32, 3), 120, np.uint8)).save(src)
    dst = tmp_path / "out.jpg"

    _exif_normalize_copy(src, dst)

    assert Image.open(dst).mode == "RGB"
