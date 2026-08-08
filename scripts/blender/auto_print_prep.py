"""Automated print-proofing of the raw mesh (runs INSIDE headless Blender).

Usage (via core/blender_runner.py)::

    blender --background --factory-startup --python scripts/blender/auto_print_prep.py -- \
        --input <raw.obj> --output-dir <dir> --target-height 120.0 \
        --base-thickness 4.0 --voxel-size 1.2 --decimate-ratio 0.15

Operations (deterministic, no GUI):
    1. import raw mesh, join into one object
    2. orient: the axis with the largest extent becomes +Z (up)
    3. scale to target bust height (mm), center, drop min Z to 0
    4. repair: if the input is already watertight (Hunyuan3D output), only
       remove disconnected islands — the voxel remesh (which erases facial
       detail) is reserved for broken meshes (``--input-watertight false``)
    5. union a flat printable base cylinder
    6. safety-cut anything below Z=0 (guaranteed flat bottom)
    7. adaptive decimate: planar collapse first (lossless), then a
       triangle-budget collapse (keeps detail, unlike a blind ratio)
    8. export bust.stl (binary) + preview.glb

Exit code 0 = success; the stage runner collects the exported files.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _activate(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="OBJECT")


def _parse_args(argv: list[str]) -> dict:
    args = {}
    for i, tok in enumerate(argv):
        if tok.startswith("--") and i + 1 < len(argv):
            args[tok[2:]] = argv[i + 1]
    return args


def main() -> int:
    args = _parse_args(sys.argv[sys.argv.index("--") + 1:])
    input_path = Path(args["input"])
    out_dir = Path(args["output-dir"])
    target_height = float(args.get("target-height", 120.0))
    base_thickness = float(args.get("base-thickness", 4.0))
    voxel_size = float(args.get("voxel-size", 1.2))
    decimate_ratio = float(args.get("decimate-ratio", 0.15))
    input_watertight = args.get("input-watertight", "true").lower() == "true"
    min_triangles = int(args.get("min-triangles", 25_000))
    max_triangles = int(args.get("max-triangles", 2_000_000))

    out_dir.mkdir(parents=True, exist_ok=True)

    # work in millimeters: 1 Blender unit = 1 mm
    scene = bpy.context.scene
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = "MILLIMETERS"

    # --- 1. import + join ------------------------------------------------------
    bpy.ops.wm.obj_import(filepath=str(input_path))
    objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not objs:
        print("ERROR: no mesh imported from", input_path)
        return 1
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    bust = bpy.context.active_object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # --- 2. orient up axis = largest extent ------------------------------------
    dims = bust.dimensions  # (x, y, z) world
    up_axis = max(range(3), key=lambda i: dims[i])  # 0=x, 1=y, 2=z
    if up_axis != 2:
        angle = math.pi / 2 * (1 if up_axis == 0 else -1)
        if up_axis == 0:
            bust.rotation_euler = (0.0, -angle, 0.0)  # x -> z
        else:
            bust.rotation_euler = (angle, 0.0, 0.0)  # y -> z
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        bpy.context.view_layer.update()

    # --- 3. scale to target height + center + drop to Z=0 ----------------------
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bb = [bust.matrix_world @ Vector(c) for c in bust.bound_box]
    z_min = min(v.z for v in bb)
    z_max = max(v.z for v in bb)
    height = z_max - z_min
    if height <= 0:
        print("ERROR: degenerate mesh (zero height)")
        return 1
    scale = target_height / height
    bust.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bb = [bust.matrix_world @ Vector(c) for c in bust.bound_box]
    cx = (min(v.x for v in bb) + max(v.x for v in bb)) / 2
    cy = (min(v.y for v in bb) + max(v.y for v in bb)) / 2
    z_min = min(v.z for v in bb)
    bust.location = (-cx, -cy, -z_min)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    # --- 4. repair: detail-preserving for watertight meshes ------------------
    # The voxel remesh guarantees a clean solid but erases sub-millimeter
    # facial features. Hunyuan3D (and other marching-cubes/poisson outputs)
    # are already watertight, so only drop disconnected islands; broken
    # meshes still get the remesh as a repair.
    bust.data.validate(verbose=False)
    if input_watertight:
        _remove_small_islands(bust)
    else:
        _activate(bust)
        mod = bust.modifiers.new(name="remesh", type="REMESH")
        mod.voxel_size = voxel_size
        mod.use_remove_disconnected = True
        bpy.ops.object.modifier_apply(modifier="remesh")
        bpy.context.view_layer.update()

    # --- 5+6. union printable base + flat-bottom cut (with repair retry) ------
    bb = [bust.matrix_world @ Vector(c) for c in bust.bound_box]
    extent_x = max(v.x for v in bb) - min(v.x for v in bb)
    extent_y = max(v.y for v in bb) - min(v.y for v in bb)
    base_radius = 0.5 * max(extent_x, extent_y) + 6.0  # mm margin
    try:
        _union_base(bust, base_radius, base_thickness)
        _flat_bottom_cut(bust)
    except Exception as exc:  # noqa: BLE001 - boolean ops are sensitive to
        # residual non-manifoldness on skip-remesh inputs; repair and retry.
        print(f"boolean ops failed ({exc}); repairing with voxel remesh")
        # drop any leftover base/cutter objects AND failed boolean modifiers
        # from the failed attempt so the retry starts from a clean state
        # (stale modifiers would be re-evaluated at export)
        bust.modifiers.clear()
        for o in list(bpy.data.objects):
            if o.type == "MESH" and o is not bust:
                bpy.data.objects.remove(o, do_unlink=True)
        _activate(bust)
        mod = bust.modifiers.new(name="remesh_retry", type="REMESH")
        mod.voxel_size = voxel_size
        mod.use_remove_disconnected = True
        bpy.ops.object.modifier_apply(modifier="remesh_retry")
        bpy.context.view_layer.update()
        _union_base(bust, base_radius, base_thickness)
        _flat_bottom_cut(bust)
    # booleans can leave microscopic slivers and micro-holes (coincident
    # face artifacts at the base junction); purge them so the exported STL
    # is a single clean solid
    _remove_small_islands(bust)
    _fix_non_manifold(bust)

    # --- 7. adaptive decimate --------------------------------------------------
    # Planar collapse first: removes coplanar redundancy with zero visible
    # loss. Then a budget collapse down to the preset's keep-ratio (floored
    # at min_triangles so the STL never gets too coarse for slicing), capped
    # at max_triangles for slicer friendliness.
    tris = len(bust.data.polygons)
    target = max(int(tris * decimate_ratio), min_triangles)
    target = min(target, max_triangles)
    if tris > target:
        _activate(bust)
        # Budget collapse only: planar/dissolve decimation is wrong for
        # organic meshes (smooth generative surfaces are within a few
        # degrees everywhere and would dissolve to a handful of faces).
        dec = bust.modifiers.new(name="decimate_budget", type="DECIMATE")
        dec.decimate_type = "COLLAPSE"
        dec.ratio = max(0.02, min(target / tris, 1.0))
        bpy.ops.object.modifier_apply(modifier="decimate_budget")
        bpy.context.view_layer.update()
        print(f"decimate: {tris} -> final {len(bust.data.polygons)} (target {target})")

    # --- 8. export ----------------------------------------------------------------
    bpy.ops.object.select_all(action="DESELECT")
    bust.select_set(True)
    bpy.context.view_layer.objects.active = bust
    print(f"export: active={bpy.context.active_object.name} "
          f"bust_faces={len(bust.data.polygons)} nm={_count_non_manifold(bust)}")

    stl_path = out_dir / "bust.stl"
    bpy.ops.wm.stl_export(
        filepath=str(stl_path),
        apply_modifiers=True,
        ascii_format=False,
        export_selected_objects=True,
    )
    glb_path = out_dir / "preview.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        use_selection=True,
    )
    print(f"PRINT_PREP_OK stl={stl_path} glb={glb_path}")
    return 0


def _union_base(bust, base_radius: float, base_thickness: float) -> None:
    """Union a flat printable base cylinder into the bust."""
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=96,
        radius=base_radius,
        depth=base_thickness,
        location=(0.0, 0.0, base_thickness / 2),
    )
    base = bpy.context.active_object
    bool_mod = bust.modifiers.new(name="union_base", type="BOOLEAN")
    bool_mod.operation = "UNION"
    bool_mod.object = base
    _activate(bust)  # modifier_apply acts on the ACTIVE object
    bpy.ops.object.modifier_apply(modifier="union_base")
    bpy.data.objects.remove(base, do_unlink=True)
    _activate(bust)


def _flat_bottom_cut(bust) -> None:
    """Safety-cut anything below Z=0 (guaranteed flat bottom).

    Cutter top face at z=-0.01: coincident faces at exactly z=0 make the
    boolean solver eat the base plate, so we cut a hair below the floor.
    """
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, location=(0.0, 0.0, -0.51)
    )
    cutter = bpy.context.active_object
    cutter.scale = (400.0, 400.0, 1.0)  # 800x800x1 mm box, top face at Z=-0.01
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    cut_mod = bust.modifiers.new(name="flat_bottom", type="BOOLEAN")
    cut_mod.operation = "DIFFERENCE"
    cut_mod.object = cutter
    _activate(bust)  # modifier_apply acts on the ACTIVE object
    bpy.ops.object.modifier_apply(modifier="flat_bottom")
    bpy.data.objects.remove(cutter, do_unlink=True)
    _activate(bust)


def _fix_non_manifold(obj, max_attempts: int = 3) -> None:
    """Repair boolean micro-artifacts: delete non-manifold faces, fill holes.

    Boolean unions/cuts on faceted meshes occasionally leave a tiny
    non-manifold face pair and a micro-hole at the junction (typically on
    the base plate). Deleting the offending faces and filling the resulting
    boundary loop closes them; on clean meshes this is a no-op.
    """
    last_nm = None
    prev_faces = len(obj.data.polygons)
    for attempt in range(max_attempts):
        # Voxel remesh output is quads/ngons; STL export triangulates, which
        # can surface microscopic artifacts. Triangulate first so the
        # non-manifold pass works on the exact geometry that will be written.
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.quads_convert_to_tris(
            quad_method="BEAUTY", ngon_method="BEAUTY"
        )
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold(extend=False)
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.fill_holes(sides=8)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        nm = _count_non_manifold(obj)
        faces = len(obj.data.polygons)
        print(f"fix_non_manifold: faces={faces} nm={nm}")
        if nm == 0:
            return
        # erosion guard: stop when an attempt made no progress or ate more
        # than 2% of the mesh (the hole is not a boolean micro-artifact)
        if last_nm is not None and (nm >= last_nm or faces < prev_faces * 0.98):
            break
        last_nm, prev_faces = nm, faces
    print("WARNING: non-manifold geometry remains after repair attempts")


def _count_non_manifold(obj) -> int:
    """Count edges used by != 2 faces (same rule as mesh_stats)."""
    from collections import Counter

    counts: Counter = Counter()
    for p in obj.data.polygons:
        for e in p.edge_keys:
            counts[e] += 1
    return sum(1 for n in counts.values() if n != 2)


def _remove_small_islands(obj) -> None:
    """Keep only the largest connected component (bmesh flood fill).

    Watertight meshes can still contain detached floaters (e.g. stray
    blobs from the generator). This removes every component except the
    largest without touching surface detail.
    """
    import bmesh

    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    comp = [-1] * len(bm.verts)
    sizes: list[int] = []
    next_id = 0
    for start in bm.verts:
        if comp[start.index] != -1:
            continue
        stack = [start]
        comp[start.index] = next_id
        count = 0
        while stack:
            v = stack.pop()
            count += 1
            for e in v.link_edges:
                for nv in (e.verts[0], e.verts[1]):
                    if comp[nv.index] == -1:
                        comp[nv.index] = next_id
                        stack.append(nv)
        sizes.append(count)
        next_id += 1

    if len(sizes) > 1:
        largest = max(range(len(sizes)), key=lambda i: sizes[i])
        doomed = [v for v in bm.verts if comp[v.index] != largest]
        bmesh.ops.delete(bm, geom=doomed, context="VERTS")
        print(f"island cleanup: kept {sizes[largest]} verts, removed "
              f"{sum(sizes) - sizes[largest]} verts in {len(sizes) - 1} floater(s)")

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")


if __name__ == "__main__":
    sys.exit(main())
