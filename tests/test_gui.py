"""GUI functional test: generate() streams events and returns artifacts."""

import os
import pathlib

import pytest

import app


@pytest.fixture()
def sample_image(tmp_path):
    img = tmp_path / "sample.png"
    from PIL import Image

    Image.new("RGB", (128, 128), (150, 120, 90)).save(img)
    return str(img)


def test_generate_streams_and_returns_artifacts(sample_image, tmp_path):
    gen = app.generate([sample_image], "", "generative", "fast")
    lines, final = 0, None
    for item in gen:
        if item[1] is not None:
            final = item
        else:
            lines += 1
    assert lines > 3, "expected several progress lines"
    assert final is not None
    assert final[1] is not None and pathlib.Path(final[1]).is_file()
    assert final[3] is not None and pathlib.Path(final[3]).is_file()
    assert "DONE" in final[0]


def test_generate_rejects_empty_input():
    gen = app.generate([], "", "auto", "fast")
    first = next(gen)
    assert first[0].startswith("ERROR")
    assert first[1] is None
