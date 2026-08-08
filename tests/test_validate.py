"""Validation stage tests: printability checks on known-good/bad meshes."""

import trimesh

from core.config import Config
from scripts.validate.check_mesh import validate_stl


def _make_box(path, size_mm=100.0, watertight=True):
    m = trimesh.creation.box(extents=[size_mm, size_mm, size_mm])
    if not watertight:  # punch a hole: drop the last two faces
        m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces[:-2], process=False)
    m.export(path)


def test_validate_good_mesh(tmp_path):
    stl = tmp_path / "good.stl"
    _make_box(stl, watertight=True)
    v = validate_stl(Config.load(), stl)
    assert v["watertight"] is True
    assert v["printable"] is True
    assert v["within_build_volume"] is True
    assert v["connected_components"] == 1
    assert v["main_component_share"] == 1.0
    assert v["volume_mm3"] > 0


def test_validate_bad_mesh_attempts_repair(tmp_path):
    stl = tmp_path / "bad.stl"
    _make_box(stl, watertight=False)
    v = validate_stl(Config.load(), stl)
    assert v["watertight"] is False
    assert v["repair_attempted"] is True
    assert v["printable"] is False
    assert any("watertight" in f for f in v["failures"])


def test_validate_oversized_mesh(tmp_path):
    stl = tmp_path / "big.stl"
    _make_box(stl, size_mm=500.0, watertight=True)
    v = validate_stl(Config.load(), stl)
    assert v["within_build_volume"] is False
    assert v["printable"] is False
    assert any("build volume" in f for f in v["failures"])
