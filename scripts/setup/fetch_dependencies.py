"""Fetch and verify portable dependencies for Bust Forge (setup, one-time).

Downloads (internet required for the first run only; afterwards the pipeline
runs fully offline):

    Blender 4.5 LTS (portable)          -> vendor/blender/
    TripoSR inference code (HF space)   -> vendor/triposr/tsr/
    TripoSR weights + config            -> models/triposr/
    DINOv2 config for TripoSR tokenizer -> models/triposr_dino/
    Stable Fast 3D weights + config     -> models/stable_fast_3d/
    Hunyuan3D-2.1 DiT + VAE weights     -> models/hunyuan3d/ (dit-v2-1, vae-v2-1)
    Hunyuan3D-2mv multi-view DiT       -> models/hunyuan3d-mv/ (optional: multi-view input)
    Depth Anything V2 (small) weights   -> models/depth_anything_v2/
    rembg u2net ONNX model              -> models/rembg/
    mediapipe FaceLandmarker model      -> models/mediapipe/face_landmarker.task

Usage::

    python scripts/setup/fetch_dependencies.py            # download missing
    python scripts/setup/fetch_dependencies.py --check    # verify only
    python scripts/setup/fetch_dependencies.py --only blender
    python scripts/setup/fetch_dependencies.py --skip models

Integrity: direct URL downloads are verified against official sidecar hashes
(Blender) or a trust-on-first-use hash file written next to the artifact;
Hugging Face downloads are verified by huggingface_hub's own etag handling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

BLENDER_VERSION = "4.5.12"
BLENDER_ZIP_URL = (
    f"https://download.blender.org/release/Blender{BLENDER_VERSION[:3]}/"
    f"blender-{BLENDER_VERSION}-windows-x64.zip"
)
BLENDER_SHA256_URL = (
    f"https://download.blender.org/release/Blender{BLENDER_VERSION[:3]}/"
    f"blender-{BLENDER_VERSION}.sha256"
)

# rembg downloads u2net.onnx from its GitHub release into U2NET_HOME; the
# adapter points U2NET_HOME at models/rembg/ so the pre-fetched file is used.
REMBG_U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"


@dataclass
class HfEntry:
    name: str
    repo_id: str
    dest_dir: str  # relative to PROJECT_ROOT
    allow_patterns: Optional[list[str]] = None
    repo_type: Optional[str] = None  # "space" for HF spaces
    note: str = ""


@dataclass
class UrlEntry:
    name: str
    url: str
    dest_path: str  # relative to PROJECT_ROOT, exact file path
    note: str = ""


HF_ENTRIES: list[HfEntry] = [
    HfEntry(
        "triposr_code",
        "stabilityai/TripoSR",
        "vendor/triposr",
        allow_patterns=["tsr/*.py", "tsr/**/*.py", "requirements.txt"],
        repo_type="space",
        note="Official TripoSR inference code (MIT)",
    ),
    HfEntry(
        "triposr",
        "stabilityai/TripoSR",
        "models/triposr",
        allow_patterns=["config.yaml", "model.ckpt"],
        note="TripoSR checkpoint (stabilityai mirror of the original weights)",
    ),
    HfEntry(
        "triposr_dino",
        "facebook/dino-vitb16",
        "models/triposr_dino",
        allow_patterns=["config.json"],
        note="DINOv2 config used by the TripoSR image tokenizer (weights live in model.ckpt)",
    ),
    HfEntry(
        "stable_fast_3d",
        "stabilityai/stable-fast-3d",
        "models/stable_fast_3d",
        allow_patterns=["config.yaml", "model.safetensors", "LICENSE.md"],
        note="Stable Fast 3D weights (fallback generative backend)",
    ),
    HfEntry(
        "depth_anything_v2",
        "depth-anything/Depth-Anything-V2-Small-hf",
        "models/depth_anything_v2",
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
        note="Depth Anything V2 small (depth-relief fallback)",
    ),
]

URL_ENTRIES: list[UrlEntry] = [
    UrlEntry("rembg_u2net", REMBG_U2NET_URL, "models/rembg/u2net.onnx", "rembg u2net background-removal model"),
    UrlEntry(
        "mediapipe_face_landmarker",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "models/mediapipe/face_landmarker.task",
        "mediapipe FaceLandmarker (head-yaw estimation for multi-view selection)",
    ),
]

# Files that must exist after a successful fetch (for --check)
_BLENDER_EXE = "blender.exe" if os.name == "nt" else "blender"
CHECK_FILES: dict[str, Path] = {
    "blender": PROJECT_ROOT / "vendor/blender" / _BLENDER_EXE,
    "triposr_code": PROJECT_ROOT / "vendor/triposr/tsr/system.py",
    "triposr": PROJECT_ROOT / "models/triposr/model.ckpt",
    "triposr_dino": PROJECT_ROOT / "models/triposr_dino/config.json",
    "hunyuan3d": PROJECT_ROOT / "models/hunyuan3d/dit-v2-1/model.fp16.ckpt",
    "stable_fast_3d": PROJECT_ROOT / "models/stable_fast_3d/model.safetensors",
    "depth_anything_v2": PROJECT_ROOT / "models/depth_anything_v2/model.safetensors",
    "rembg_u2net": PROJECT_ROOT / "models/rembg/u2net.onnx",
    "mediapipe_face_landmarker": PROJECT_ROOT / "models/mediapipe/face_landmarker.task",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha256: Optional[str] = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        h = hashlib.sha256()
        with open(tmp, "wb") as fh:
            with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as bar:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    h.update(chunk)
                    bar.update(len(chunk))
    if expected_sha256 and h.hexdigest() != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 mismatch for {url}\n  expected {expected_sha256}\n  got      {h.hexdigest()}"
        )
    os.replace(tmp, dest)
    # TOFU hash file for future --check runs
    dest.with_suffix(dest.suffix + ".sha256").write_text(h.hexdigest(), encoding="utf-8")


def verify_url_artifact(url_entry: UrlEntry, force: bool) -> bool:
    dest = PROJECT_ROOT / url_entry.dest_path
    hash_file = dest.with_suffix(dest.suffix + ".sha256")
    if dest.is_file() and not force:
        if hash_file.is_file():
            recorded = hash_file.read_text(encoding="utf-8").strip()
            if sha256_file(dest) == recorded:
                print(f"  [ok] {url_entry.name}: {dest.name} (hash verified)")
                return True
            print(f"  [!!] {url_entry.name}: hash mismatch -> re-downloading")
        else:
            print(f"  [ok] {url_entry.name}: {dest.name} (present, no hash recorded)")
            hash_file.write_text(sha256_file(dest), encoding="utf-8")
            return True
    print(f"  [..] {url_entry.name}: downloading {url_entry.url}")
    download(url_entry.url, dest)
    print(f"  [ok] {url_entry.name}: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


# ---------------------------------------------------------------------------
# Blender (zip + official sha256 sidecar + extraction)
# ---------------------------------------------------------------------------


def fetch_blender(force: bool) -> bool:
    zip_path = PROJECT_ROOT / "vendor/downloads" / f"blender-{BLENDER_VERSION}-windows-x64.zip"
    blender_exe = PROJECT_ROOT / "vendor/blender" / _BLENDER_EXE
    if blender_exe.is_file() and not force:
        print(f"  [ok] blender: {blender_exe} present")
        return True

    # Official sidecar hash from blender.org
    expected = None
    try:
        resp = requests.get(BLENDER_SHA256_URL, timeout=30)
        resp.raise_for_status()
        for line in resp.text.splitlines():
            if "windows-x64.zip" in line:
                expected = line.split()[0].strip()
                break
    except requests.RequestException:
        print("  [!!] blender: could not fetch official sha256 sidecar; verifying TOFU")

    if not zip_path.is_file() or force:
        print(f"  [..] blender: downloading {BLENDER_ZIP_URL}")
        download(BLENDER_ZIP_URL, zip_path, expected_sha256=expected)

    print("  [..] blender: extracting (this takes a moment)...")
    vendor = PROJECT_ROOT / "vendor"
    with tempfile.TemporaryDirectory(dir=vendor) as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        top = next(p for p in Path(tmp).iterdir() if p.is_dir())
        target = vendor / "blender"
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(top), str(target))
    print(f"  [ok] blender: extracted to {target}")
    return True


# ---------------------------------------------------------------------------
# Hugging Face entries
# ---------------------------------------------------------------------------


def fetch_hf(entry: HfEntry, force: bool) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print(f"  [!!] {entry.name}: huggingface_hub not installed ({exc})")
        return False

    dest = PROJECT_ROOT / entry.dest_dir
    marker = dest / ".download_complete"
    if marker.is_file() and not force:
        print(f"  [ok] {entry.name}: {dest} (marked complete)")
        return True

    print(f"  [..] {entry.name}: snapshot_download({entry.repo_id}) -> {dest}")
    try:
        snapshot_download(
            repo_id=entry.repo_id,
            repo_type=entry.repo_type,
            local_dir=dest,
            allow_patterns=entry.allow_patterns,
            token=os.environ.get("HF_TOKEN", None),
        )
    except Exception as exc:  # gated repos / network
        print(
            f"  [!!] {entry.name}: download failed: {exc}\n"
            f"       If the repo is gated, set HF_TOKEN or run: huggingface-cli login"
        )
        return False
    marker.write_text(json.dumps({"repo_id": entry.repo_id, "patterns": entry.allow_patterns}), encoding="utf-8")
    print(f"  [ok] {entry.name}: complete")
    return True


# ---------------------------------------------------------------------------
# Hunyuan3D-2.1 (manual-download friendly)
# ---------------------------------------------------------------------------

# Official repo subfolders -> canonical local layout. Manual downloads placed
# directly at models/hunyuan3d/dit-v2-1 and models/hunyuan3d/vae-v2-1 are
# detected and accepted as-is (no re-download).
HUNYUAN3D_REPO = "tencent/Hunyuan3D-2.1"
HUNYUAN3D_OFFICIAL_TO_CANONICAL = {
    "hunyuan3d-dit-v2-1": "dit-v2-1",
    "hunyuan3d-vae-v2-1": "vae-v2-1",
}


def fetch_hunyuan3d(force: bool) -> bool:
    base = PROJECT_ROOT / "models" / "hunyuan3d"
    marker = base / ".download_complete"
    canonical_ckpt = base / "dit-v2-1" / "model.fp16.ckpt"

    # Manual download already in canonical layout -> just mark complete.
    if not force and canonical_ckpt.is_file():
        marker.write_text(json.dumps({"repo_id": HUNYUAN3D_REPO, "source": "manual"}), encoding="utf-8")
        print(f"  [ok] hunyuan3d: {canonical_ckpt} present (canonical layout)")
        return True
    if marker.is_file() and not force:
        print(f"  [ok] hunyuan3d: {base} (marked complete)")
        return True

    print(f"  [..] hunyuan3d: snapshot_download({HUNYUAN3D_REPO}) -> {base}")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HUNYUAN3D_REPO,
            local_dir=str(base),
            allow_patterns=[
                "hunyuan3d-dit-v2-1/config.yaml",
                "hunyuan3d-dit-v2-1/model.fp16.ckpt",
                "hunyuan3d-vae-v2-1/config.yaml",
                "hunyuan3d-vae-v2-1/model.fp16.ckpt",
                "LICENSE",
            ],
            token=os.environ.get("HF_TOKEN", None),
        )
    except Exception as exc:  # network / gated
        print(f"  [!!] hunyuan3d: download failed: {exc}\n"
              f"       If the repo is gated, set HF_TOKEN or run: huggingface-cli login")
        return False

    # Normalize official subfolder names -> canonical layout. On --force the
    # canonical dir is stale, so replace it with the fresh download.
    for official, canonical in HUNYUAN3D_OFFICIAL_TO_CANONICAL.items():
        src = base / official
        dst = base / canonical
        if not src.is_dir():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))

    if canonical_ckpt.is_file():
        marker.write_text(json.dumps({"repo_id": HUNYUAN3D_REPO, "source": "fetch"}), encoding="utf-8")
        print(f"  [ok] hunyuan3d: {canonical_ckpt} ({canonical_ckpt.stat().st_size / 1e6:.0f} MB)")
        return True
    print(f"  [!!] hunyuan3d: dit-v2-1/model.fp16.ckpt not found after download")
    return False


# ---------------------------------------------------------------------------
# Hunyuan3D-2mv (optional multi-view backend)
# ---------------------------------------------------------------------------

# Only the multi-view DiT is needed: it bundles its own VAE/conditioner, and
# its fp16 checkpoint is 4.93 GB (1.1B params — fits 12 GB VRAM comfortably).
HUNYUAN3D_MV_REPO = "tencent/Hunyuan3D-2mv"
HUNYUAN3D_MV_OFFICIAL_TO_CANONICAL = {"hunyuan3d-dit-v2-mv": "dit-v2-mv"}


def fetch_hunyuan3d_mv(force: bool) -> bool:
    base = PROJECT_ROOT / "models" / "hunyuan3d-mv"
    marker = base / ".download_complete"
    canonical_ckpt = base / "dit-v2-mv" / "model.fp16.ckpt"

    if not force and canonical_ckpt.is_file():
        marker.write_text(json.dumps({"repo_id": HUNYUAN3D_MV_REPO, "source": "manual"}), encoding="utf-8")
        print(f"  [ok] hunyuan3d_mv: {canonical_ckpt} present (canonical layout)")
        return True
    if marker.is_file() and not force:
        print(f"  [ok] hunyuan3d_mv: {base} (marked complete)")
        return True

    print(f"  [..] hunyuan3d_mv: snapshot_download({HUNYUAN3D_MV_REPO}) -> {base}")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HUNYUAN3D_MV_REPO,
            local_dir=str(base),
            allow_patterns=[
                "hunyuan3d-dit-v2-mv/config.yaml",
                "hunyuan3d-dit-v2-mv/model.fp16.ckpt",
            ],
            token=os.environ.get("HF_TOKEN", None),
        )
    except Exception as exc:  # network / gated
        print(f"  [!!] hunyuan3d_mv: download failed: {exc}")
        return False

    for official, canonical in HUNYUAN3D_MV_OFFICIAL_TO_CANONICAL.items():
        src = base / official
        dst = base / canonical
        if not src.is_dir():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))

    if canonical_ckpt.is_file():
        marker.write_text(json.dumps({"repo_id": HUNYUAN3D_MV_REPO, "source": "fetch"}), encoding="utf-8")
        print(f"  [ok] hunyuan3d_mv: {canonical_ckpt} ({canonical_ckpt.stat().st_size / 1e6:.0f} MB)")
        return True
    print(f"  [!!] hunyuan3d_mv: dit-v2-mv/model.fp16.ckpt not found after download")
    return False


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="Only verify presence/integrity of everything.")
    ap.add_argument("--only", default="", help="Comma-separated names to fetch (default: all).")
    ap.add_argument("--skip", default="", help="Comma-separated names to skip.")
    ap.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = ap.parse_args(argv)

    names = {e.name for e in HF_ENTRIES} | {e.name for e in URL_ENTRIES} | {"blender", "hunyuan3d", "hunyuan3d_mv"}
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        unknown = wanted - names
        if unknown:
            print(f"Unknown names: {sorted(unknown)} (available: {sorted(names)})")
            return 2
    else:
        wanted = set(names)
    wanted -= {n.strip() for n in args.skip.split(",") if n.strip()}

    results: dict[str, bool] = {}
    if "blender" in wanted:
        results["blender"] = fetch_blender(args.force)
    if "hunyuan3d" in wanted:
        results["hunyuan3d"] = fetch_hunyuan3d(args.force)
    if "hunyuan3d_mv" in wanted:
        results["hunyuan3d_mv"] = fetch_hunyuan3d_mv(args.force)
    for entry in URL_ENTRIES:
        if entry.name in wanted:
            results[entry.name] = verify_url_artifact(entry, args.force)
    for entry in HF_ENTRIES:
        if entry.name in wanted:
            results[entry.name] = fetch_hf(entry, args.force)

    if args.check:
        print("\n=== Integrity check ===")
        ok = True
        for name, path in CHECK_FILES.items():
            present = path.is_file()
            if not present:
                ok = False
            print(f"  [{'ok' if present else 'MISSING'}] {name:20s} {path}")
        print("\nAll dependencies present." if ok else "\nSome dependencies are missing — run without --check.")
        return 0 if ok else 1

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\nFailed: {failed}")
        return 1
    print("\nDone. Run `python scripts/setup/fetch_dependencies.py --check` to confirm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
