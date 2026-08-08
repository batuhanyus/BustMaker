"""Orchestrator tests: resume skips successful stages, records them as skipped."""

from pathlib import Path

from core.config import Config
from core.logging import ProgressHub
from core.paths import JobPaths
from core.pipeline import STAGES, run_pipeline


def test_resume_skips_successful_stages(tmp_path, monkeypatch):
    cfg = Config.load()
    job = JobPaths.create(cfg, tmp_path / "in.jpg", output_dir=tmp_path / "job")

    # stub stages that succeed and write a marker
    calls = []

    def fake_stage(ctx):
        calls.append(ctx.job.job_id)
        return __import__("core.pipeline", fromlist=["StageResult"]).StageResult(status="success")

    stages = [__import__("core.pipeline", fromlist=["Stage"]).Stage("s", fake_stage)]

    run_pipeline(cfg, job, mode="auto", preset="fast", stages=stages)
    assert len(calls) == 1

    # second run with resume: stage is skipped, not executed
    run_pipeline(cfg, job, mode="auto", preset="fast", stages=stages,
                 cli={"resume": True})
    assert len(calls) == 1  # still only the first run executed it
    import json

    report = json.loads(job.report_path.read_text(encoding="utf-8"))
    assert report["stages"][0]["status"] == "skipped"
    assert report["stages"][0]["warnings"] == ["resumed from previous report"]
