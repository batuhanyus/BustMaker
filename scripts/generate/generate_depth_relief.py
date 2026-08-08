"""Depth-relief bust fallback: the ultra-reliable last rung.

Turns the best preprocessed frame into a *bas-relief* plaque — a solid,
watertight box whose front face is the subject's depth map, so it is always
printable even when every full-3D backend fails.

Geometry:
    front face  = depth map (Z = f(depth)), masked to the subject,
    back face   = flat plane behind the relief,
    sides       = connecting quads, bottom stays flat for printing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from adapters.generative.base import BackendResult, MeshBackend, is_available
from core.config import Config
from core.logging import get_logger
from core.model_manager import free_memory
from core.pipeline import RunContext

log = get_logger("depth_relief")


class DepthReliefBackend(MeshBackend):
    name = "depth_relief"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def available(self) -> bool:
        return is_available(self.cfg, "depth_relief")

    def generate(self, ctx: RunContext, input_frames: list[Path], out_path: Path) -> BackendResult:
        if not self.available():
            return BackendResult(success=False, error="depth_anything_v2 weights missing")

        frame = input_frames[0] if input_frames else None
        if frame is None:
            return BackendResult(success=False, error="no preprocessed frames")

        try:
            from adapters.depth.depth_anything_adapter import estimate_depth

            img = Image.open(frame).convert("RGBA")
            depth = estimate_depth(self.cfg, img)
            mesh = build_relief_mesh(
                depth,
                alpha=np.asarray(img)[:, :, 3],
                target_height_mm=float(self.cfg.get("print.target_height_mm", 120.0)),
                relief_depth_mm=float(self.cfg.get("depth.relief_depth_mm", 25.0)),
                grid=384,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(out_path)
            return BackendResult(
                success=True,
                mesh_path=out_path,
                settings={"backend": self.name, "grid": 384,
                          "relief_depth_mm": self.cfg.get("depth.relief_depth_mm", 25.0)},
            )
        except Exception as exc:  # noqa: BLE001
            return BackendResult(success=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            free_memory()


def build_relief_mesh(
    depth: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    target_height_mm: float = 120.0,
    relief_depth_mm: float = 25.0,
    grid: int = 384,
) -> "trimesh.Trimesh":
    """Build a watertight relief box from a [0,1] depth map.

    The mesh is scaled so the *relief face* spans target_height_mm; the
    back plate adds ``base_mm`` thickness beyond ``relief_depth_mm``.
    """
    import trimesh

    h, w = depth.shape
    # down/upsample to the grid resolution, keep aspect
    scale = min(grid / w, grid / h)
    nw, nh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
    img = Image.fromarray((depth * 255).astype(np.uint8)).resize((nw, nh))
    z = np.asarray(img, dtype=np.float32) / 255.0  # [0,1]

    if alpha is not None:
        amask = np.asarray(Image.fromarray(alpha).resize((nw, nh)), dtype=np.float32) / 255.0
        z = z * (amask > 0.1)  # zero out background
        if z.max() <= 0:
            z = np.asarray(img, dtype=np.float32) / 255.0  # mask failed; use raw depth

    # world scale: relief face height == target bust height
    aspect = nw / nh
    face_w, face_h = target_height_mm * aspect, target_height_mm
    back_mm = max(2.0, 0.3 * relief_depth_mm)  # backplate behind the relief

    xs = np.linspace(0, face_w, nw)
    ys = np.linspace(0, face_h, nh)
    xx, yy = np.meshgrid(xs, ys)
    zz = z * relief_depth_mm

    # front-face vertices (row-major: y, then x)
    verts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    n = nw

    def quad(i, j):
        # two triangles for grid cell (i, j) — (i row, j col)
        a = i * n + j
        b = i * n + j + 1
        c = (i + 1) * n + j
        d = (i + 1) * n + j + 1
        return a, b, c, d

    faces = []
    for i in range(nh - 1):
        for j in range(nw - 1):
            a, b, c, d = quad(i, j)
            faces += [(a, b, d), (a, d, c)]  # front face, outward +z

    # back plate (flat, offset behind) with reversed winding (faces -z)
    z_back = -back_mm
    base = len(verts)
    verts = np.vstack([verts, np.stack([xx.ravel(), yy.ravel(), np.full_like(zz.ravel(), z_back)], -1)])
    for i in range(nh - 1):
        for j in range(nw - 1):
            a, b, c, d = quad(i, j)
            faces += [(base + a, base + d, base + b), (base + a, base + c, base + d)]

    # side walls: quads between front border and back border
    def wall(a0, a1, b0, b1):
        # front edge (a0->a1), back edge (b0->b1), connect with two triangles
        return [(a0, b0, b1), (a0, b1, a1)]

    for j in range(nw - 1):  # top edge (y = 0)
        faces += wall(j, j + 1, base + j, base + j + 1)
    for j in range(nw - 1):  # bottom edge (y = max)
        f0, f1 = (nh - 1) * n + j, (nh - 1) * n + j + 1
        faces += wall(f0, f1, base + f0, base + f1)
    for i in range(nh - 1):  # left edge (x = 0)
        f0, f1 = i * n, (i + 1) * n
        faces += wall(f0, f1, base + f0, base + f1)
    for i in range(nh - 1):  # right edge (x = max)
        f0, f1 = i * n + nw - 1, (i + 1) * n + nw - 1
        faces += wall(f0, f1, base + f0, base + f1)

    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64))
    if mesh.volume < 0:  # consistent winding outward
        mesh.invert()
    # Orientation is normalized by the Blender print-prep stage (Phase 5):
    # tallest axis up, bottom flattened. Keep raw geometry here.
    return mesh
