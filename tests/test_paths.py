"""Path resolution tests: tool discovery order, job layout, input classification."""

from pathlib import Path

from core.config import Config
from core.paths import JobPaths, find_blender, find_ffmpeg, project_root
from core.pipeline import classify_input


def test_project_root():
    assert (project_root() / "config.yaml").is_file()


def test_find_ffmpeg_on_path():
    # ffmpeg is installed on this machine's PATH; resolution must find it.
    found = find_ffmpeg(Config.load())
    assert found is not None and found.exists()


def test_find_blender_vendored():
    # fetch_dependencies installs the portable build into vendor/blender/.
    found = find_blender(Config.load())
    assert found is not None and found.exists()
    assert "vendor" in found.parts


def test_job_paths_layout(tmp_path, cfg):
    job = JobPaths.create(cfg, tmp_path / "subject_01.mp4")
    assert job.job_dir.name != "subject_01"  # timestamped subdir
    job.ensure_dirs()
    for d in (job.preprocessed_dir, job.raw_mesh_dir, job.final_dir, job.debug_dir):
        assert d.is_dir()
    assert job.stl_path.name == "bust.stl"
    assert job.report_path.name == "report.json"
    assert job.raw_obj_path.name == "raw_mesh.obj"


def test_job_paths_explicit_output(tmp_path, cfg):
    out = tmp_path / "custom_out"
    job = JobPaths.create(cfg, tmp_path / "img.png", output_dir=out)
    assert job.job_dir == out.resolve()


def test_classify_input(tmp_path):
    vid = tmp_path / "clip.MOV"
    vid.touch()
    img = tmp_path / "pic.jpg"
    img.touch()
    assert classify_input(vid) == "video"
    assert classify_input(img) == "single_image"
    assert classify_input(tmp_path) == "images"
    assert classify_input(tmp_path / "notes.txt") == "unknown"
