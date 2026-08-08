"""Bust Forge local GUI.

    python app.py          # serves on http://127.0.0.1:7860 (local only)

Thin wrapper over the same pipeline the CLI runs: upload a video / photos /
single image (or type a local folder path), pick mode + quality preset, click
Generate, watch live progress, download bust.stl / preview.glb / report.json
(+ debug zip when requested).

Security: bound to 127.0.0.1, share disabled, no telemetry.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator, Optional

import gradio as gr

from core.config import Config
from core.logging import ProgressHub
from core.paths import JobPaths
from core.pipeline import MODES, run_pipeline

PRESETS = ("fast", "balanced", "high")


def _resolve_input(files: list, folder_path: str) -> Optional[Path]:
    if folder_path and folder_path.strip():
        p = Path(folder_path.strip())
        if p.is_dir():
            return p
        raise ValueError(f"Folder not found: {p}")
    if files:
        paths = [Path(f) for f in files]
        if len(paths) == 1:
            return paths[0]
        # multiple uploads -> stage them into one temp folder
        tmp = Path(tempfile.mkdtemp(prefix="bustforge_upload_"))
        for i, p in enumerate(paths):
            shutil.copy2(p, tmp / f"upload_{i:03d}{p.suffix.lower()}")
        return tmp
    raise ValueError("Provide a video/image upload or a local folder path.")


def _fmt_event(ev: dict) -> str:
    ts = time.strftime("%H:%M:%S", time.localtime(ev["ts"]))
    prog = f" [{ev['progress'] * 100:.0f}%]" if ev.get("progress") is not None else ""
    return f"{ts} | {ev['stage']:14s} | {ev['status']:8s}{prog} | {ev.get('message', '')}"


def _make_debug_zip(job: JobPaths) -> Optional[str]:
    if not job.debug_dir.is_dir() or not any(job.debug_dir.rglob("*")):
        return None
    zpath = job.job_dir / "debug.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(job.debug_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(job.debug_dir))
    return str(zpath)


def generate(
    files: list,
    folder_path: str,
    mode: str,
    preset: str,
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple]:
    try:
        input_path = _resolve_input(files, folder_path)
    except ValueError as exc:
        yield (f"ERROR: {exc}", None, None, None, None)
        return

    cfg = Config.load()
    job = JobPaths.create(cfg, input_path)
    hub = ProgressHub()
    result: dict[str, Any] = {}

    def _worker() -> None:
        try:
            result["report"] = run_pipeline(
                cfg, job, mode=mode, preset=preset,
                cli={"debug": True}, progress=hub,
            )
        except Exception as exc:  # noqa: BLE001 - surface to the UI
            result["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    seen = 0
    last_event = "starting..."
    while thread.is_alive():
        events = hub.events()
        for ev in events[seen:]:
            line = _fmt_event(ev)
            last_event = line
            yield (line, None, None, None, None)
        seen = len(events)
        time.sleep(0.25)
    thread.join()
    for ev in hub.events()[seen:]:
        yield (_fmt_event(ev), None, None, None, None)

    if "error" in result:
        yield (f"ERROR: {result['error']}", None, None, None, None)
        return

    stl = str(job.stl_path) if job.stl_path.is_file() else None
    glb = str(job.glb_path) if job.glb_path.is_file() else None
    report = str(job.report_path) if job.report_path.is_file() else None
    debug_zip = _make_debug_zip(job)
    ok = bool(result.get("report") and result["report"].summary.get("success"))
    summary = f"{'DONE' if ok else 'FINISHED WITH WARNINGS'} — outputs in {job.job_dir}"
    yield (f"{last_event}\n{summary}", stl, glb, report, debug_zip)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Bust Forge") as demo:
        gr.Markdown(
            "# 🗿 Bust Forge\n"
            "Local-only pipeline: video / photos / single portrait → print-ready **bust.stl** "
            "for your Ender 3 V2. Nothing leaves this machine."
        )
        with gr.Row():
            with gr.Column(scale=2):
                files = gr.File(
                    label="Upload a video or photos (multiple allowed)",
                    file_count="multiple",
                    file_types=["image", "video"],
                )
                folder = gr.Textbox(
                    label="…or a local folder path",
                    placeholder="C:\\Users\\you\\Pictures\\subject_01",
                )
            with gr.Column(scale=1):
                mode = gr.Dropdown(choices=list(MODES), value="auto", label="Mode")
                preset = gr.Dropdown(choices=list(PRESETS), value="balanced", label="Quality preset")
                btn = gr.Button("Generate Bust", variant="primary")
        log = gr.Textbox(label="Progress", lines=14, interactive=False, max_lines=20)
        with gr.Row():
            stl_out = gr.File(label="bust.stl (print-ready)")
            glb_out = gr.File(label="preview.glb")
            report_out = gr.File(label="report.json")
            debug_out = gr.File(label="debug.zip (keyframes/masks)")

        btn.click(
            generate,
            inputs=[files, folder, mode, preset],
            outputs=[log, stl_out, glb_out, report_out, debug_out],
        )
    return demo


def main() -> None:
    cfg = Config.load()
    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",  # local only — never expose to the network
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
