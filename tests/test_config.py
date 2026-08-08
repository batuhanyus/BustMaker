"""Config loader tests: presets, mode chains, path resolution, validation."""

import pytest

from core.config import Config


def test_loads_project_config(cfg):
    assert cfg.path.name == "config.yaml"
    assert cfg.get("print.target_height_mm") == 120.0


def test_presets_exist_and_typed(cfg):
    for name in ("fast", "balanced", "high"):
        p = cfg.preset(name)
        assert p["gen_resolution"] > 0
        assert 0 < p["decimate_ratio"] <= 1
        assert p["voxel_size_mm"] > 0
    fast, high = cfg.preset("fast"), cfg.preset("high")
    assert fast["gen_resolution"] < high["gen_resolution"]


def test_mode_chains(cfg):
    assert cfg.mode_chain("auto") == ["capture", "generative", "depth_relief"]
    assert cfg.mode_chain("depth_relief") == ["depth_relief"]
    assert cfg.mode_chain("generative")[-1] == "depth_relief"  # always has fallback
    with pytest.raises(KeyError):
        cfg.mode_chain("nope")


def test_missing_preset_raises(cfg):
    with pytest.raises(KeyError):
        cfg.preset("ultra")


def test_resolve_path_relative_to_config(cfg):
    p = cfg.resolve_path("paths.output", "./output")
    assert p.is_absolute()
    assert p.name == "output"


def test_bad_config_file_missing():
    with pytest.raises(FileNotFoundError):
        Config.load("does-not-exist.yaml")


def test_vr_defaults(cfg):
    assert cfg.vr_cfg.get("vram_limit_gb") == 12
    assert cfg.get("vr.oom_retry_steps") == ["fp16", "lowvram", "cpu"]
