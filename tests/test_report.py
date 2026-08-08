"""Report schema tests: lifecycle, serialization round-trip, JSON validity."""

import json

from core.report import JobReport


def test_stage_lifecycle(tmp_path):
    rep = JobReport.new(job_id="j1", input_path=tmp_path / "in.mp4", input_type="video",
                        config_snapshot={"mode": "auto"})
    rep.mark_stage("ingest", "success", duration_s=1.5, artifacts={"frames": 42})
    rep.mark_stage("generate_mesh", "failed", error="OOM", fallback_used=None)
    rep.add_warning("low mask quality")
    rep.finalize(total_duration_s=2.0, success=False)

    d = rep.to_dict()
    assert d["schema_version"] == 1
    assert d["input"]["type"] == "video"
    assert d["stages"][0]["status"] == "success"
    assert d["stages"][0]["artifacts"]["frames"] == 42
    assert d["stages"][1]["error"] == "OOM"
    assert d["summary"]["success"] is False
    assert d["summary"]["total_duration_s"] == 2.0


def test_save_and_reload(tmp_path):
    rep = JobReport.new(job_id="j2", input_path=tmp_path / "img.png", input_type="single_image",
                        config_snapshot={})
    rep.mark_stage("ingest", "skipped", warnings=["stub"])
    out = rep.save(tmp_path / "report.json")

    assert out.is_file()
    with open(out, encoding="utf-8") as fh:
        raw = json.load(fh)
    assert raw["job_id"] == "j2"
    assert raw["stages"][0]["status"] == "skipped"
    # schema_version is a class constant, not a constructor arg — excluded here
    assert raw["schema_version"] == JobReport.SCHEMA_VERSION


def test_fallback_tracking():
    rep = JobReport.new(job_id="j3", input_path=__import__("pathlib").Path("x"), input_type="images",
                        config_snapshot={})
    rep.mark_stage("generate_mesh", "success", fallback_used="stable_fast_3d")
    assert rep.fallbacks_used == ["stable_fast_3d"]
    # same fallback not duplicated
    rep.mark_stage("generate_mesh", "success", fallback_used="stable_fast_3d")
    assert len(rep.fallbacks_used) == 1
