# Bust Forge

**Local-only, hands-off pipeline: smartphone video / photos / single portrait → print-ready 3D bust `.stl` for an Ender 3 V2.**

No cloud APIs. No uploads. After the one-time dependency fetch, everything runs offline on your machine.

- **GPU target:** NVIDIA RTX 4070 (12 GB VRAM) — FP16, low-VRAM ladder, CPU fallback
- **Output:** `bust.stl` (binary, watertight, flat-bottomed, scaled) + `preview.glb` + `report.json`
- **GUI:** local Gradio app on `http://127.0.0.1:7860` (never exposed to the network)
- **Portable deps:** Blender 4.5 LTS is vendored under `vendor/blender/` — no system install required

---

## Quickstart

```bat
:: 1. environment (Python 3.12)
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements.txt

:: 2. one-time dependency fetch (internet needed once; ~7 GB)
.venv\Scripts\python scripts\setup\fetch_dependencies.py
.venv\Scripts\python scripts\setup\fetch_dependencies.py --check

:: 3. run
.venv\Scripts\python run_pipeline.py --input input\subject_01.mp4 --output output\subject_01
.venv\Scripts\python run_pipeline.py --input input\photos_folder --mode auto --preset high
.venv\Scripts\python run_pipeline.py --input input\portrait.jpg --mode generative

:: or launch the GUI
.venv\Scripts\python app.py
```

A successful run produces:

```
output/<subject>/
├── preprocessed/          # masked RGBA frames + metadata.json
├── raw_mesh/              # raw_mesh.obj / raw_mesh.glb (as generated)
├── final/
│   ├── bust.stl           # ← print-ready
│   └── preview.glb
└── report.json            # stages, timings, fallbacks, mesh stats, validation
```

## Modes & presets

| Mode | Strategy chain |
|---|---|
| `auto` (default) | capture → generative → depth relief |
| `generative` | TripoSR → Stable Fast 3D → depth relief |
| `capture` | COLMAP (experimental) → generative → depth relief |
| `depth_relief` | depth-relief plaque only (always works) |

Presets: `fast` / `balanced` / `high` — control generation resolution, keyframe budget, voxel size and decimation (see `config.yaml`).

## Fallback behavior

Every backend failure is recorded in `report.json` (never silent):

1. **TripoSR** (primary, ~1 GB weights) — single-image mesh generation on GPU.
2. **Stable Fast 3D** — opt-in fallback; its HF repo is *gated*, so it is skipped until you fetch it with a token (see below).
3. **Depth relief** — Depth Anything V2 → bas-relief plaque; watertight by construction, always printable.
4. **COLMAP** — experimental photogrammetry for multi-view input; skipped automatically when the binary is absent.

OOM handling: automatic retry ladder (fp16 off → low VRAM → CPU → report).

## Fetching Stable Fast 3D (optional, needs a HF account)

```bat
set HF_TOKEN=hf_your_token
.venv\Scripts\python scripts\setup\fetch_dependencies.py --only stable_fast_3d
```

Visit <https://huggingface.co/stabilityai/stable-fast-3d> first and click **Agree** to the license gate.

## Config

Everything lives in `config.yaml`: paths, quality presets, VRAM limits, print constraints
(default bust height **120 mm**, base 4 mm, min wall 1.5 mm, build volume 220×220×250 mm),
mode chains, and the deferred G-code slicing hook (off by design — PROJECT.md forbids
automatic G-code generation).

## Offline operation

After `fetch_dependencies.py` completes, unplug the network: all models live in `models/`
and Blender in `vendor/blender/`. `U2NET_HOME` (rembg) and local checkpoint dirs keep
everything project-local.

## Notes / known limitations

- TripoSR reconstructs a single view at a time; the sharpest/best-masked frame is used.
- Video input is handled as keyframe extraction + selection (photogrammetry via COLMAP is
  the only true multi-view path and needs the COLMAP binary).
- G-code slicing is intentionally out of scope; the `slicing:` config hook is reserved.

## Tests

```bat
.venv\Scripts\python -m pytest tests/ -q
```

## Layout

```
app.py  run_pipeline.py  config.yaml  requirements.txt  requirements.lock
core/        config, paths, logging, report, pipeline, blender_runner,
             ffmpeg_wrapper, model_manager
adapters/    background/ (rembg) · generative/ (triposr, stable_fast_3d, base)
             capture/ (colmap) · depth/ (depth_anything)
scripts/     setup/ · preprocess/ · generate/ · blender/ · validate/
vendor/      blender/ (portable) · triposr/ (vendored inference code)
models/      downloaded weights (see fetch_dependencies.py)
input/ output/ temp/ logs/  docs/
```
