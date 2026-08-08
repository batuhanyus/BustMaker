"""ffmpeg_wrapper tests: probe, frame extraction, dHash dedupe (uses real ffmpeg)."""

import subprocess

import pytest

from core.config import Config
from core.ffmpeg_wrapper import dedupe_frames, extract_frames, probe, _dhash, _hamming


def _make_video(path, n_frames=5, pattern="checkers"):
    """Create a tiny test video with animated (frame-distinct) content."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i",
        f"testsrc2=size=160x120:rate=2:duration={n_frames / 2}",
        "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def test_probe(tmp_path, cfg):
    vid = tmp_path / "clip.mp4"
    _make_video(vid, n_frames=6)
    info = probe(cfg, vid)
    assert info.duration_s > 0
    assert info.width == 160 and info.height == 120
    assert info.fps > 0


def test_extract_and_dedupe(tmp_path, cfg):
    vid = tmp_path / "clip.mp4"
    _make_video(vid, n_frames=6)
    out = tmp_path / "frames"
    frames = extract_frames(cfg, vid, out, interval_sec=0.5, max_frames=20, dedupe=False)
    assert len(frames) >= 3
    # all frames readable + hashes sane
    hashes = [_dhash(f) for f in frames]
    assert all(h >= 0 for h in hashes)
    # identical frames dedupe to 1
    dup = tmp_path / "dup"
    dup.mkdir()
    a, b = frames[0], dup / "b.jpg"
    b.write_bytes(a.read_bytes())
    kept = dedupe_frames([a, b], max_frames=10)
    assert len(kept) == 1
    # unrelated hashes differ
    assert _hamming(hashes[0], hashes[-1]) > 0 or len(frames) == 1


def test_probe_missing_file(tmp_path, cfg):
    with pytest.raises(FileNotFoundError):
        probe(cfg, tmp_path / "nope.mp4")
