"""Run portable Blender headlessly from Python.

All Blender invocations go through :func:`run_blender_script` so that:

* the executable is always the vendored portable build (``vendor/blender``)
  resolved via :func:`core.paths.find_blender` — never a system install,
* user config/cache are isolated per-run (env vars below) so the pipeline
  never touches the user's normal Blender profile,
* stdout/stderr are captured, logged, and returned for diagnostics.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from core.config import Config
from core.logging import get_logger
from core.paths import find_blender

log = get_logger("blender")

# Keep Blender from reading/writing the user's real profile.
BLENDER_ISOLATION_ENV = {
    "BLENDER_USER_CONFIG": "vendor/blender/user/config",
    "BLENDER_USER_SCRIPTS": "vendor/blender/user/scripts",
    "BLENDER_USER_EXTENSIONS": "vendor/blender/user/extensions",
    "BLENDER_USER_RESOURCES": "vendor/blender/user/resources",
}


class BlenderNotFoundError(RuntimeError):
    pass


def blender_executable(cfg: Config) -> Path:
    exe = find_blender(cfg)
    if exe is None:
        raise BlenderNotFoundError(
            "Blender not found. Run: python scripts/setup/fetch_dependencies.py"
        )
    return exe


def run_blender_script(
    cfg: Config,
    script_path: Path,
    script_args: Sequence[str] = (),
    timeout: Optional[float] = None,
    extra_env: Optional[dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``blender --background --factory-startup --python <script> -- <args>``.

    Returns the completed process; raises ``BlenderNotFoundError`` when the
    portable build is missing and ``subprocess.CalledProcessError`` on
    non-zero exit (stdout/stderr attached to the exception).
    """
    exe = blender_executable(cfg)
    cmd = [str(exe), "--background", "--factory-startup", "--python", str(script_path)]
    if script_args:
        cmd += ["--", *map(str, script_args)]

    env = os.environ.copy()
    base = cfg.resolve_path("paths.vendor", "./vendor")
    for key, rel in BLENDER_ISOLATION_ENV.items():
        env[key] = str(base / rel)
    env["PYTHONPATH"] = str(cfg.path.parent)  # core/ scripts/ importable inside Blender
    if extra_env:
        env.update(extra_env)

    t0 = time.monotonic()
    log.info("blender: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
        cwd=str(cwd or cfg.path.parent),
        check=False,
    )
    duration = time.monotonic() - t0
    log.info("blender: exited %s in %.1fs", proc.returncode, duration)

    if proc.stdout:
        log.debug("blender stdout:\n%s", proc.stdout[-8000:])
    if proc.stderr:
        log.debug("blender stderr:\n%s", proc.stderr[-8000:])

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
        )
    return proc


def blender_version(cfg: Config) -> str:
    """Return e.g. ``Blender 4.5.12`` by running the portable binary."""
    exe = blender_executable(cfg)
    proc = subprocess.run(
        [str(exe), "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    first = (proc.stdout or proc.stderr).strip().splitlines()
    return first[0] if first else "unknown"
