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
    # 3D review panel: GLB viewer gets preview.glb, STL viewer gets bust.stl
    assert final[5] == final[2] and pathlib.Path(final[5]).is_file()
    assert final[6] == final[1] and pathlib.Path(final[6]).is_file()
    assert pathlib.Path(final[5]).suffix.lower() == ".glb"
    # stats line pulled from report.json -> mesh.stats
    assert final[7] is None or "triangles" in final[7]


def test_generate_rejects_empty_input():
    gen = app.generate([], "", "auto", "fast")
    first = next(gen)
    assert first[0].startswith("ERROR")
    assert first[1:] == (None, None, None, None, None, None, None)


def test_build_app_has_3d_review_panel():
    demo = app.build_app()
    models = [c for c in demo.config["components"] if c["type"] == "model3d"]
    assert len(models) == 2, "expected preview.glb + bust.stl viewers"
    assert all(m["props"]["interactive"] is False for m in models)
    click_deps = [d for d in demo.config["dependencies"] if d.get("targets")]
    assert len(click_deps) == 1 and len(click_deps[0]["outputs"]) == 8
