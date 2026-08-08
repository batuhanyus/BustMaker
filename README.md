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

:: 2. one-time dependency fetch (internet needed once; ~16 GB incl. Hunyuan3D)
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
| `generative` | per-preset backend chain (balanced/high: Hunyuan3D-2.1 → Stable Fast 3D → TripoSR; fast: TripoSR → Stable Fast 3D) |
| `capture` | COLMAP (experimental) → generative → depth relief |
| `depth_relief` | depth-relief plaque only (always works) |

Presets: `fast` / `balanced` / `high` — control the generative backend chain, Hunyuan3D
isosurface resolution (`octree_resolution` 256/384/512) and denoise steps (`gen_steps`),
keyframe budget, and STL detail floors (`min_triangles` / `max_triangles`, see `config.yaml`).

## Fallback behavior

Every backend failure is recorded in `report.json` (never silent):

1. **Hunyuan3D-2.1** (primary for `balanced`/`high`, ~8 GB weights) — Tencent's
   flow-matching DiT image-to-mesh; watertight, high-detail output (~200–500k faces).
   With the optional multi-view model (`hunyuan3d_mv`, ~5 GB) and a video whose
   frames span wide head poses, up to 4 viewpoints (front/left/back/right) are fused.
2. **TripoSR** (primary for `fast`, ~1 GB weights) — fast single-image preview-grade
   mesh generation on GPU.
3. **Stable Fast 3D** — opt-in fallback; its HF repo is *gated*, so it is skipped until you fetch it with a token (see below).
4. **Depth relief** — Depth Anything V2 → bas-relief plaque; watertight by construction, always printable.
5. **COLMAP** — experimental photogrammetry for multi-view input; skipped automatically when the binary is absent.

OOM handling: automatic retry ladder (fp16 off → low VRAM → CPU → report). RAM is
bounded: Hunyuan3D models are constructed in fp16 directly (a 3B-param DiT + its
7.4 GB checkpoint fits in 32 GB of system RAM).

## Hunyuan3D-2.1 weights (manual download, recommended)

`fetch_dependencies.py --only hunyuan3d` downloads from the official repo, but the
weights are large — downloading by hand works too. Place them exactly like this
(the script accepts this layout and skips re-downloading):

```
models/hunyuan3d/dit-v2-1/config.yaml     # + model.fp16.ckpt (7.4 GB)
models/hunyuan3d/vae-v2-1/                # optional spare (656 MB)
```

Source: <https://huggingface.co/tencent/Hunyuan3D-2.1> (folders `hunyuan3d-dit-v2-1`,
`hunyuan3d-vae-v2-1`). Optional multi-view fusion:

```
.venv\Scripts\python scripts\setup\fetch_dependencies.py --only hunyuan3d_mv
```

pulls `Hunyuan3D-2mv` (1.1B, ~5 GB) into `models/hunyuan3d-mv/` — without it the
pipeline still works, using single-image conditioning. Face landmarking for
multi-view keyframe selection is fetched as `models/mediapipe/face_landmarker.task`.

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

- Hunyuan3D-2.1 reconstructs from a single image (or up to 4 fused views when the
  multi-view model is present and the video shows wide head poses); TripoSR always
  uses the single sharpest/best-masked frame.
- Video input is handled as keyframe extraction + viewpoint selection (photogrammetry
  via COLMAP is the experimental multi-view path and needs the COLMAP binary).
- Multi-view fusion only activates when face-detected head yaws span ≥ 60° — guessed
  view labels would corrupt the model's conditioning, so clustered-yaw videos fall
  back to the single-image path.
- Print prep is detail-preserving: watertight raw meshes skip the lossy voxel
  remesh; only genuinely broken meshes are repaired with it.
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
adapters/    background/ (rembg) · generative/ (hunyuan3d, triposr,
             stable_fast_3d, base) · capture/ (colmap) · depth/ (depth_anything)
scripts/     setup/ · preprocess/ · generate/ · blender/ · validate/
vendor/      blender/ (portable) · hunyuan3d/ (vendored hy3dgen) · triposr/
models/      downloaded weights (see fetch_dependencies.py)
input/ output/ temp/ logs/  docs/
```
