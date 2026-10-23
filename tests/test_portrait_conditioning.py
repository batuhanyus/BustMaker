"""Portrait-mode preprocessing and Hunyuan view-selection tests."""

from pathlib import Path

import numpy as np
from PIL import Image

from adapters.generative.hunyuan3d_adapter import _load_condition_image, _select_multiview_views
from core.config import Config
from core.logging import ProgressHub
from core.paths import JobPaths
from core.pipeline import RunContext
from core.report import JobReport
from scripts.preprocess.ingest import _view_tag_from_name
from scripts.preprocess.portrait_conditioning import prepare_portrait_condition


def _ctx(tmp_path: Path) -> RunContext:
    cfg = Config.load()
    inp = tmp_path / "in.png"
    Image.new("RGB", (32, 32), "white").save(inp)
    job = JobPaths.create(cfg, inp, output_dir=tmp_path / "job")
    report = JobReport.new(
        job_id="portrait-test", input_path=inp, input_type="images", config_snapshot={}
    )
    return RunContext(
        cfg=cfg,
        job=job,
        report=report,
        progress=ProgressHub(),
        mode="portrait",
        preset="high",
    )


def test_view_tag_from_filename():
    assert _view_tag_from_name("subject_front") == "front"
    assert _view_tag_from_name("portrait-left-02") == "left"
    assert _view_tag_from_name("rear_view") == "back"
    assert _view_tag_from_name("RIGHT") == "right"
    assert _view_tag_from_name("front_left") is None
    assert _view_tag_from_name("IMG_0123") is None


def test_multiview_prefers_explicit_filename_tags(tmp_path):
    ctx = _ctx(tmp_path)
    frames = []
    for name in ("a.png", "b.png", "c.png"):
        p = tmp_path / name
        Image.new("RGBA", (16, 16), (120, 120, 120, 255)).save(p)
        frames.append(p)
    ctx.set_shared("frame_view_tags", ["front", "left", "back"])
    ctx.set_shared("frame_yaws", [None, None, None])

    views = _select_multiview_views(ctx, frames)

    assert views == {"front": frames[0], "left": frames[1], "back": frames[2]}


def test_load_condition_image_uses_gen_resolution(tmp_path):
    p = tmp_path / "wide.png"
    Image.new("RGBA", (1200, 600), (10, 20, 30, 255)).save(p)

    out = _load_condition_image(p, 640)

    assert out.size == (640, 320)
    assert out.mode == "RGBA"


def test_portrait_condition_alpha_fallback_preserves_subject(tmp_path, monkeypatch):
    # No face detector: the alpha silhouette becomes the conservative fallback.
    from scripts.preprocess import portrait_conditioning as pc

    monkeypatch.setattr(pc, "_mediapipe_face_bbox", lambda img: None)
    monkeypatch.setattr(pc, "_opencv_face_bbox", lambda img: None)

    arr = np.zeros((500, 400, 4), dtype=np.uint8)
    arr[40:480, 60:340, :3] = (180, 120, 90)
    arr[40:480, 60:340, 3] = 255
    src = tmp_path / "subject.png"
    Image.fromarray(arr, "RGBA").save(src)
    dst = tmp_path / "condition.png"

    result = prepare_portrait_condition(Config.load(), src, dst, resolution=512)
    out = Image.open(dst)

    assert result.detector == "alpha_bbox"
    assert out.size == (512, 512)
    assert out.mode == "RGBA"
    alpha = np.asarray(out.getchannel("A"))
    assert alpha.max() == 255
    assert alpha.min() == 0
    ys, xs = np.where(alpha > 127)
    assert xs.max() - xs.min() > 250
    assert ys.max() - ys.min() > 400


def test_hunyuan_infer_forwards_guidance_chunks_and_resolution(tmp_path, monkeypatch):
    import sys
    import types
    from adapters.generative import hunyuan3d_adapter as hy

    captured = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured["load"] = (args, kwargs)
            return cls()

        def __call__(self, **kwargs):
            captured["call"] = kwargs
            cb = kwargs.get("callback")
            for i in range(kwargs["num_inference_steps"]):
                cb(i, None, None)
            return ["mesh"]

        def enable_model_cpu_offload(self):
            captured["offload"] = True

    fake = types.ModuleType("hy3dgen.shapegen")
    fake.Hunyuan3DDiTFlowMatchingPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "hy3dgen.shapegen", fake)
    monkeypatch.setattr(hy, "_patch_fp16_construction", lambda: None)
    vendor = tmp_path / "vendor"
    (vendor / "hy3dgen").mkdir(parents=True)
    monkeypatch.setattr(hy, "_VENDOR", vendor)

    img_path = tmp_path / "input.png"
    Image.new("RGBA", (1200, 600), (20, 30, 40, 255)).save(img_path)
    backend = hy.Hunyuan3DBackend(Config.load())
    mesh = backend._infer(
        img_path,
        "dit-v2-1",
        None,
        tmp_path / "models",
        octree=384,
        steps=3,
        seed=7,
        settings={"device": "cpu", "fp16": False, "low_vram": False},
        guidance_scale=7.5,
        num_chunks=20000,
        conditioning_resolution=640,
    )

    assert mesh == "mesh"
    assert captured["call"]["guidance_scale"] == 7.5
    assert captured["call"]["num_chunks"] == 20000
    assert captured["call"]["octree_resolution"] == 384
    assert captured["call"]["image"].size == (640, 320)
