"""Mesh statistics (runs INSIDE headless Blender).

Computes printability-relevant stats for an STL/OBJ file and writes JSON::

    blender --background --factory-startup --python scripts/blender/mesh_stats.py -- \
        --input <mesh.stl> --output <stats.json>

Stats: vertices, triangles, bounds_mm [x,y,z] extents, volume_mm3,
non_manifold_edges, watertight, base_flat (min Z is uniform at the lowest plane).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> dict:
    args = {}
    for i, tok in enumerate(argv):
        if tok.startswith("--") and i + 1 < len(argv):
            args[tok[2:]] = argv[i + 1]
    return args


def main() -> int:
    import bpy
    import bmesh
    from mathutils import Vector

    args = _parse_args(sys.argv[sys.argv.index("--") + 1:])
    input_path = Path(args["input"])
    output_path = Path(args["output"])

    suffix = input_path.suffix.lower()
    if suffix == ".stl":
        bpy.ops.wm.stl_import(filepath=str(input_path))
    else:
        bpy.ops.wm.obj_import(filepath=str(input_path))

    obj = bpy.context.active_object
    if obj is None:
        print("ERROR: no mesh imported")
        return 1

    deps = bpy.context.evaluated_depsgraph_get()
    mesh = obj.evaluated_get(deps).to_mesh()

    bm = bmesh.new()
    bm.from_mesh(mesh)

    n_non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    volume = bm.calc_volume(signed=False)
    z_min = min(v.co.z for v in bm.verts)
    eps = 0.1
    on_base = sum(1 for v in bm.verts if abs(v.co.z - z_min) <= eps)
    base_flat = on_base / max(len(bm.verts), 1) > 0.005
    bm.free()

    # world-space bounds in mm (scene unit == mm)
    bb = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    xs = [v.x for v in bb]
    ys = [v.y for v in bb]
    zs = [v.z for v in bb]
    bounds_mm = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]

    stats = {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.polygons),
        "bounds_mm": [round(b, 2) for b in bounds_mm],
        "volume_mm3": round(volume, 1),
        "non_manifold_edges": n_non_manifold,
        "watertight": n_non_manifold == 0,
        "base_flat": base_flat,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("MESH_STATS_OK", json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
