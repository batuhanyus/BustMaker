"""Mesh generation tests: fallback chain orchestration + depth relief geometry."""

import numpy as np
import pytest

from core.config import Config
from core.logging import ProgressHub
from core.paths import JobPaths
from core.pipeline import RunContext, StageError
from core.report import JobReport
from scripts.generate import generate_mesh
from scripts.generate.generate_depth_relief import build_relief_mesh


class _FakeBackend:
    """Scriptable backend: fails or succeeds per the configured script."""

    def __init__(self, name, script):
        self.name = name
        self._script = list(script)

    def available(self):
        return True

    def run(self, ctx, frames, out_path):
        step = self._script.pop(0) if self._script else "success"
        if step == "fail":
            from adapters.generative.base import BackendResult
            return BackendResult(success=False, error="boom")
        from adapters.generative.base import BackendResult
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("fake mesh")
        return BackendResult(success=True, mesh_path=out_path)


def _make_ctx(tmp_path, mode="generative"):
    cfg = Config.load()
    job = JobPaths.create(cfg, tmp_path / "in.png", output_dir=tmp_path / "job")
    job.ensure_dirs()
    report = JobReport.new(job_id="t", input_path=job.input_path, input_type="single_image",
                           config_snapshot={"mode": mode})
    ctx = RunContext(cfg=cfg, job=job, report=report, progress=ProgressHub(),
                     mode=mode, preset="fast")
    ctx.set_shared("preprocessed_frames", [tmp_path / "f.png"])
    return ctx


def test_chain_falls_through_to_success(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path, mode="generative")
    monkeypatch.setattr(generate_mesh, "STRATEGY_BACKENDS",
                        {"generative": ["a", "b", "c"]})
    scripts = {"a": ["fail"], "b": ["fail"], "c": ["success"]}
    monkeypatch.setattr(generate_mesh, "_backend_for",
                        lambda name, cfg: _FakeBackend(name, scripts[name]))

    result = generate_mesh.run_generate(ctx)
    assert result.status == "success"
    assert result.fallback_used == "c"
    attempts = result.artifacts["attempts"]
    assert [a["status"] for a in attempts] == ["failed", "failed", "success"]
    assert attempts[0]["error"] == "boom"
    assert ctx.job.raw_obj_path.is_file()


def test_chain_all_fail_raises(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path, mode="generative")
    monkeypatch.setattr(generate_mesh, "STRATEGY_BACKENDS",
                        {"generative": ["a", "b"]})
    monkeypatch.setattr(generate_mesh, "_backend_for",
                        lambda name, cfg: _FakeBackend(name, ["fail"]))

    with pytest.raises(StageError, match="all mesh backends failed"):
        generate_mesh.run_generate(ctx)


def test_unavailable_backend_skipped(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path, mode="generative")
    monkeypatch.setattr(generate_mesh, "STRATEGY_BACKENDS",
                        {"generative": ["a", "b"]})

    class _Unavailable(_FakeBackend):
        def available(self):
            return False

    monkeypatch.setattr(generate_mesh, "_backend_for",
                        lambda name, cfg: _Unavailable(name, []))
    with pytest.raises(StageError):
        generate_mesh.run_generate(ctx)


# ---------------------------------------------------------------------------
# depth relief geometry
# ---------------------------------------------------------------------------


def test_relief_watertight_and_oriented():
    rng = np.random.default_rng(3)
    depth = rng.random((48, 48)).astype(np.float32)
    alpha = np.zeros((48, 48), np.uint8)
    alpha[8:40, 6:42] = 255
    m = build_relief_mesh(depth, alpha=alpha, target_height_mm=120.0,
                          relief_depth_mm=25.0, grid=64)
    assert m.is_watertight
    assert m.volume > 0
    # relief face spans the target height
    assert abs(m.bounds[1][1] - m.bounds[0][1] - 120.0) < 1.0
    assert m.bounds[0][2] < 0 < m.bounds[1][2]  # spans both sides of z=0


def test_relief_background_masked_out():
    depth = np.full((32, 32), 0.9, dtype=np.float32)
    alpha = np.zeros((32, 32), np.uint8)
    alpha[8:24, 8:24] = 255
    m = build_relief_mesh(depth, alpha=alpha, grid=32)
    # masked background -> front face is mostly flat; still watertight
    assert m.is_watertight
