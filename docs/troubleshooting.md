# Troubleshooting

## Setup

**`fetch_dependencies.py` fails on `stable_fast_3d` (401 / GatedRepoError)**

Expected: the repo is gated. The rest of the pipeline works without it (TripoSR
+ depth relief). To enable it: create a free Hugging Face account, visit
<https://huggingface.co/stabilityai/stable-fast-3d>, click **Agree**, then run
`fetch_dependencies.py --only stable_fast_3d` with `HF_TOKEN` set.

**`fetch_dependencies.py --check` reports missing files**

Re-run without `--check`. If a download is interrupted, the fetcher resumes or
re-downloads automatically; `--force` re-downloads everything.

**Blender won't start / `BlenderNotFoundError`**

The pipeline uses only the portable build in `vendor/blender/`. Run
`python scripts\setup\fetch_dependencies.py --only blender`. Never install
Blender system-wide for this project.

**`torch.cuda.is_available()` is False**

The CUDA build of torch was replaced by a CPU build (e.g. after re-installing
requirements). Reinstall from the PyTorch index:

```bat
.venv\Scripts\python -m pip install --force-reinstall --no-deps torch torchvision ^
    --index-url https://download.pytorch.org/whl/cu128
```

## Runs

**"no usable frames after quality filtering" (ingest fails)**

Every frame was blurry or badly exposed. Loosen `video.blur_threshold` /
`video.exposure_ok_range` in `config.yaml`, or use better input (see
`docs/capture_guide.md`). Blurry phone footage is the most common cause.

**"all mesh backends failed"**

Check `report.json` → `stages[].artifacts.attempts` for per-backend errors:

- `triposr weights missing` → run `fetch_dependencies.py`.
- TripoSR OOM → the ladder should handle it; if it still fails, lower the
  preset to `fast` (512 px / smaller marching-cubes grid).
- `stable_fast_3d ... gated` → expected without HF_TOKEN (see above).
- `colmap binary not found` → expected unless you installed COLMAP.

**Output mesh is ugly / melted**

- The **depth-relief fallback** ran (check `report.json` → `fallbacks_used`):
  a bas-relief plaque is the guarantee path, not the pretty path. Improve the
  input (sharpness, framing, lighting) so TripoSR can take over.
- Low preset `fast` decimates aggressively (8 % of triangles). Use `balanced`
  or `high`.

**"mesh failed validation: ..."**

See the exact failure list in `report.json` → `mesh.validation.failures`.
Non-watertight results are auto-repaired into `bust_repaired.stl` when possible.

## GUI

**`app.py` opens but nothing loads**

The GUI binds strictly to `127.0.0.1:7860` — check the browser URL is
`http://127.0.0.1:7860`, not `localhost` with a proxy. If the port is taken,
change `server_port` in `app.py`.

**Progress shows `ERROR: ...`**

The pipeline writes a full `report.json` even on failure — open the *report.json*
download from the failed run for stage-by-stage errors.

## Performance notes (RTX 4070, 12 GB)

- TripoSR: ~0.5 s forward + ~5 s mesh extraction per frame at `fast`.
- Depth Anything V2 (small): ~1 s per frame.
- Blender print-prep: ~2 s.
- Everything unloads after each stage (`core/model_manager.py`); if you see
  unexplained OOMs, close other GPU apps (browsers, games).
