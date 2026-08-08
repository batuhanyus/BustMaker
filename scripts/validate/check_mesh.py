"""Printability validation of the final STL (runs in the main process).

Checks (all recorded in report.json ``mesh.validation``):

* watertight / non-manifold edge count
* volume (positive, sane for the target height)
* bounds within the printer build volume (config ``print.max_build_volume_mm``)
* floating islands: connected components + main-component volume share
* base flatness: fraction of bottom-plane vertices within 0.2 mm
* triangle budget (sanity vs. FDM slicing)

Auto-repair: when the mesh is not watertight, attempt a manifold3d repair and
re-export as ``bust_repaired.stl``; the verdict reflects the repaired state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.config import Config
from core.logging import get_logger

log = get_logger("validate")


def validate_stl(cfg: Config, stl_path: Path) -> dict[str, Any]:
    import trimesh

    mesh = trimesh.load(stl_path, force="mesh")
    checks: dict[str, Any] = {"file": str(stl_path)}
    max_build = list(cfg.get("print.max_build_volume_mm", [220, 220, 250]))

    # --- geometry ------------------------------------------------------------
    checks["vertices"] = len(mesh.vertices)
    checks["triangles"] = len(mesh.faces)
    checks["watertight"] = bool(mesh.is_watertight)
    checks["volume_mm3"] = round(float(mesh.volume), 1)

    bounds = mesh.bounds
    extents = [float(bounds[1][i] - bounds[0][i]) for i in range(3)]
    checks["bounds_mm"] = [round(e, 2) for e in extents]
    checks["within_build_volume"] = all(e <= b + 1.0 for e, b in zip(extents, max_build))
    checks["max_build_volume_mm"] = max_build

    # --- islands ---------------------------------------------------------------
    try:
        components = mesh.split(only_watertight=False)
        volumes = sorted((float(c.volume) for c in components), reverse=True)
        total = sum(volumes) or 1.0
        checks["connected_components"] = len(components)
        checks["main_component_share"] = round(volumes[0] / total, 4) if volumes else 0.0
        checks["floating_islands"] = len(components) > 1 and volumes[0] / total < 0.99
    except Exception as exc:  # noqa: BLE001 - split can be expensive/finicky
        log.warning("component analysis failed: %s", exc)
        checks["connected_components"] = None
        checks["floating_islands"] = None

    # --- base flatness -----------------------------------------------------------
    z_min = float(mesh.vertices[:, 2].min())
    on_floor = int((np.abs(mesh.vertices[:, 2] - z_min) < 0.2).sum())
    # Scale-independent: a real base rim (>= 4 verts, excludes a sphere's
    # contact point) plus a small ratio — the old fixed 0.5% fails on
    # detailed busts where the rim is a shrinking vertex fraction.
    checks["base_flat"] = on_floor >= 4 and on_floor / max(len(mesh.vertices), 1) > 0.001
    checks["base_min_z_mm"] = round(z_min, 3)

    # --- triangle budget ----------------------------------------------------------
    checks["triangle_budget_ok"] = checks["triangles"] <= 2_000_000

    # --- auto-repair ----------------------------------------------------------------
    checks["repair_attempted"] = False
    checks["repair_succeeded"] = None
    if not checks["watertight"]:
        repaired = _repair_with_manifold3d(mesh, stl_path)
        if repaired is not None:
            checks["repair_attempted"] = True
            checks["repair_succeeded"] = True
            checks["watertight"] = True
            checks["repaired_file"] = str(repaired)
            log.info("mesh auto-repaired -> %s", repaired)
        else:
            checks["repair_attempted"] = True
            checks["repair_succeeded"] = False

    # --- verdict -------------------------------------------------------------------
    failures = []
    if not checks["watertight"]:
        failures.append("not watertight")
    if checks["volume_mm3"] <= 0:
        failures.append("non-positive volume")
    if not checks["within_build_volume"]:
        failures.append("exceeds build volume")
    if checks.get("floating_islands"):
        failures.append("floating islands")
    if not checks["base_flat"]:
        failures.append("base not flat")
    checks["printable"] = not failures
    checks["failures"] = failures
    return checks


def _repair_with_manifold3d(mesh, stl_path: Path) -> Optional[Path]:
    """Attempt watertight repair via manifold3d; returns repaired STL path or None."""
    try:
        from manifold3d import Manifold

        if hasattr(Manifold, "of_trimesh"):
            m = Manifold.of_trimesh(mesh)
            out = m.get_trimesh()
        else:  # older API: Manifold(mesh)
            m = Manifold(mesh)
            out = m.to_mesh()
    except Exception as exc:  # noqa: BLE001
        log.warning("manifold3d repair failed: %s", exc)
        return None

    if out is None or len(out.faces) == 0 or not out.is_watertight:
        return None
    repaired_path = stl_path.with_name("bust_repaired.stl")
    out.export(repaired_path)
    return repaired_path


def load_validation(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
