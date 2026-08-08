"""FFmpeg/FFprobe wrapper for video frame extraction.

Resolves ffmpeg via :func:`core.paths.find_ffmpeg` (config -> vendor/ffmpeg ->
PATH) and extracts evenly spaced, de-duplicated frames from a phone video.
Frame de-duplication uses a dependency-free dHash (difference hash): videos
contain long stretches of near-identical frames that only waste VRAM later.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from core.config import Config
from core.logging import get_logger
from core.paths import find_ffmpeg

log = get_logger("ffmpeg")


class FfmpegNotFoundError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    duration_s: float
    fps: float
    width: int
    height: int
    nb_frames: Optional[int] = None

    @property
    def rotation_deg(self) -> int:
        return 0  # ffmpeg CLI auto-rotates by default; kept for future use


def _ffmpeg(cfg: Config) -> Path:
    exe = find_ffmpeg(cfg)
    if exe is None:
        raise FfmpegNotFoundError(
            "ffmpeg not found. Install it or place it in vendor/ffmpeg/ and set "
            "paths.ffmpeg_path in config.yaml."
        )
    return exe


def probe(cfg: Config, video_path: Path) -> VideoInfo:
    """Read duration/fps/resolution via ffprobe (falls back to ffmpeg -i)."""
    if not video_path.is_file():
        raise FileNotFoundError(f"video file not found: {video_path}")
    exe = _ffmpeg(cfg)
    ffprobe = exe.with_name("ffprobe.exe" if exe.suffix == ".exe" else "ffprobe")
    if ffprobe.is_file():
        proc = subprocess.run(
            [
                str(ffprobe), "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(video_path),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            vstream = next(
                (s for s in data.get("streams", []) if s.get("codec_type") == "video"), {}
            )
            duration = float(data.get("format", {}).get("duration")
                             or vstream.get("duration") or 0.0)
            fps = _parse_fps(vstream.get("avg_frame_rate") or vstream.get("r_frame_rate"))
            return VideoInfo(
                duration_s=duration,
                fps=fps,
                width=int(vstream.get("width", 0)),
                height=int(vstream.get("height", 0)),
            )
    # fallback: parse `ffmpeg -i` stderr
    proc = subprocess.run(
        [str(exe), "-hide_banner", "-i", str(video_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    stderr = proc.stderr
    duration = _parse_duration(stderr)
    size = _parse_video_size(stderr)
    fps = _parse_fps_from_stderr(stderr)
    return VideoInfo(duration_s=duration, fps=fps, width=size[0], height=size[1])


def _parse_fps(rate: str) -> float:
    try:
        num, _, den = rate.partition("/")
        return float(num) / float(den) if den else float(num)
    except (ValueError, ZeroDivisionError):
        return 30.0


def _parse_duration(stderr: str) -> float:
    for line in stderr.splitlines():
        if "Duration:" in line:
            try:
                h, m, rest = line.split("Duration:")[1].split(":")[:3]
                return int(h) * 3600 + int(m) * 60 + float(rest.split(",")[0])
            except ValueError:
                return 0.0
    return 0.0


def _parse_video_size(stderr: str) -> tuple[int, int]:
    for line in stderr.splitlines():
        if "Video:" in line:
            for token in line.split(","):
                token = token.strip()
                if "x" in token and token.replace("x", "").replace(" ", "").isdigit():
                    w, h = token.split("x")
                    return int(w), int(h)
    return 0, 0


def _parse_fps_from_stderr(stderr: str) -> float:
    for line in stderr.splitlines():
        if "Video:" in line and "fps" in line:
            try:
                return float(line.split("fps")[0].split(",")[-1].strip())
            except ValueError:
                return 30.0
    return 30.0


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_frames(
    cfg: Config,
    video_path: Path,
    out_dir: Path,
    interval_sec: float = 1.0,
    max_frames: int = 600,
    dedupe: bool = True,
) -> list[Path]:
    """Extract frames at ``interval_sec`` spacing into ``out_dir``.

    A single ffmpeg pass writes ``frame_%05d.jpg``; frames beyond
    ``max_frames`` are never kept. Returns the sorted list of kept paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = _ffmpeg(cfg)
    info = probe(cfg, video_path)
    if info.duration_s <= 0:
        raise ValueError(f"Could not determine duration of {video_path}")

    pattern = str(out_dir / "frame_%05d.jpg")
    cmd = [
        str(exe), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps=1/{interval_sec}",
        "-q:v", "2",
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {proc.stderr[-2000:]}")

    frames = sorted(out_dir.glob("frame_*.jpg"))
    log.info("extracted %d frames from %s", len(frames), video_path.name)

    if dedupe and len(frames) > 1:
        frames = dedupe_frames(frames, max_frames=max_frames)
    elif len(frames) > max_frames:
        frames = _even_sample(frames, max_frames)
    return frames


def dedupe_frames(frames: list[Path], max_frames: int, hamming_threshold: int = 8) -> list[Path]:
    """Drop near-duplicate consecutive frames (dHash hamming distance)."""
    kept: list[Path] = []
    prev_hash: Optional[int] = None
    for frame in frames:
        try:
            h = _dhash(frame)
        except OSError:
            continue
        if prev_hash is not None and _hamming(h, prev_hash) <= hamming_threshold:
            continue
        kept.append(frame)
        prev_hash = h
    if len(kept) > max_frames:
        kept = _even_sample(kept, max_frames)
    log.info("dedupe: %d -> %d frames", len(frames), len(kept))
    return kept


def _even_sample(items: list[Path], n: int) -> list[Path]:
    if n <= 0 or len(items) <= n:
        return items
    idx = {round(i * (len(items) - 1) / (n - 1)) for i in range(n)}
    return [items[i] for i in sorted(idx)]


def _dhash(path: Path, hash_size: int = 8) -> int:
    img = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).ravel()
    return sum(int(b) << i for i, b in enumerate(bits))


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
