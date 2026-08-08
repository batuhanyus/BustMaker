"""Progress-bar rendering tests: stage-reset + monotonic fill in GUI and CLI."""

import pathlib

import app
import run_pipeline as rp


class FakeProgress:
    """Records (fraction, desc) calls like gr.Progress would."""

    def __init__(self):
        self.calls = []

    def __call__(self, frac, desc=None, **kwargs):
        self.calls.append((frac, desc))


def test_tqdm_hub_resets_per_stage_and_never_goes_backward(tmp_path):
    hub = rp._TqdmProgressHub(log_path=tmp_path / "e.jsonl")
    try:
        hub.emit("ingest", "progress", "frames", 0.5)
        assert hub._bar.n == 50
        hub.emit("ingest", "progress", "frames", 0.8)
        assert hub._bar.n == 80
        hub.emit("generate_mesh", "started", "stage started")       # no fraction
        assert hub._bar.n == 80  # hold
        hub.emit("generate_mesh", "progress", "trying backend", 0.0)  # stage reset
        assert hub._bar.n == 0
        hub.emit("generate_mesh", "progress", "denoise 20/50", 0.4)
        assert hub._bar.n == 40
        hub.emit("generate_mesh", "progress", "retry rung", 0.2)    # dip -> hold
        assert hub._bar.n == 40
        hub.emit("generate_mesh", "progress", "denoise 50/50", 1.0)
        assert hub._bar.n == 100
        hub.emit("print_prep", "started", "stage started")          # hold
        assert hub._bar.n == 100
    finally:
        hub.close()


def test_gui_bar_resets_per_stage_and_holds_dips():
    progress = FakeProgress()

    def fake_run_pipeline(cfg, job, mode, preset, cli, progress):
        progress.emit("ingest", "started", "stage started")
        progress.emit("ingest", "progress", "masking frames", 0.5)
        progress.emit("ingest", "finished", "done", 1.0)
        progress.emit("generate_mesh", "started", "stage started")
        progress.emit("generate_mesh", "progress", "trying backend 'hunyuan3d'", 0.0)
        progress.emit("generate_mesh", "progress", "hunyuan3d denoise 1/50", 0.02)
        progress.emit("generate_mesh", "progress", "hunyuan3d denoise 20/50", 0.4)
        progress.emit("generate_mesh", "progress", "retry rung restarts", 0.0)  # dip
        progress.emit("generate_mesh", "progress",
                      "hunyuan3d: extracting mesh volume (octree=384)...", 1.0)
        progress.emit("generate_mesh", "finished", "backend succeeded", 1.0)
        progress.emit("print_prep", "started", "stage started")     # no fraction -> hold
        from core.report import JobReport
        r = JobReport.new(job_id=job.job_id, input_path=job.input_path,
                          input_type="single_image", config_snapshot={})
        r.finalize(total_duration_s=0.1, success=True)
        r.save(job.report_path)
        return r

    orig = app.run_pipeline
    app.run_pipeline = fake_run_pipeline
    try:
        img = pathlib.Path(app.tempfile.mkdtemp()) / "s.png"
        from PIL import Image
        Image.new("RGB", (64, 64), (120, 90, 60)).save(img)
        list(app.generate([str(img)], "", "generative", "fast", progress=progress))
    finally:
        app.run_pipeline = orig

    fracs = [f for f, _ in progress.calls]
    descs = [d for _, d in progress.calls]
    # init + per-stage monotonic: ingest 0.5 -> 1.0, generate_mesh resets 0.0 -> 1.0,
    # plus the final DONE call at 1.0
    assert fracs == [0.0, 0.0, 0.5, 1.0, 1.0, 0.0, 0.02, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0], fracs
    # the dip ("retry rung restarts" at 0.0) held the bar at 0.4, not reset it
    assert "retry rung restarts" in descs[8]
    # fraction-less events hold: print_prep started keeps 1.0
    assert "print_prep | stage started" in descs
    # final bar state: full with the DONE summary
    assert progress.calls[-1] == (1.0, descs[-1]) and "DONE" in descs[-1]


def test_gui_bar_error_path():
    gen = app.generate([], "", "auto", "fast", progress=FakeProgress())
    first = next(gen)
    assert first[0].startswith("ERROR")
    assert first[1:] == (None, None, None, None, None, None, None)
