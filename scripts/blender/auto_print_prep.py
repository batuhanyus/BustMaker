"""Automated print-proofing of the raw mesh (runs INSIDE headless Blender).

Usage (via core/blender_runner.py)::

    blender --background --factory-startup --python scripts/blender/auto_print_prep.py -- \
        --input <raw.obj> --output-dir <dir> --target-height 120.0 \
        --base-thickness 4.0 --voxel-size 1.2 --decimate-ratio 0.15

Operations (deterministic, no GUI):
    1. import raw mesh, join into one object
    2. orient: the axis with the largest extent becomes +Z (up)
    3. scale to target bust height (mm), center, drop min Z to 0
    4. voxel remesh (unified solid, removes disconnected islands)
    5. union a flat printable base cylinder
    6. safety-cut anything below Z=0 (guaranteed flat bottom)
    7. decimate for FDM slicing
    8. export bust.stl (binary) + preview.glb

Exit code 0 = success; the stage runner collects the exported files.
"""

from __future__ import annotations

import math
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
    from mathutils import Vector

    args = _parse_args(sys.argv[sys.argv.index("--") + 1:])
    input_path = Path(args["input"])
    out_dir = Path(args["output-dir"])
    target_height = float(args.get("target-height", 120.0))
    base_thickness = float(args.get("base-thickness", 4.0))
    voxel_size = float(args.get("voxel-size", 1.2))
    decimate_ratio = float(args.get("decimate-ratio", 0.15))

    out_dir.mkdir(parents=True, exist_ok=True)

    # work in millimeters: 1 Blender unit = 1 mm
    scene = bpy.context.scene
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = "MILLIMETERS"

    def _activate(obj) -> None:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="OBJECT")

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

    # --- 4. voxel remesh (solid, removes disconnected islands) ------------------
    _activate(bust)
    mod = bust.modifiers.new(name="remesh", type="REMESH")
    mod.voxel_size = voxel_size
    mod.use_remove_disconnected = True
    bpy.ops.object.modifier_apply(modifier="remesh")
    bpy.context.view_layer.update()

    # --- 5. union printable base ------------------------------------------------
    bb = [bust.matrix_world @ Vector(c) for c in bust.bound_box]
    extent_x = max(v.x for v in bb) - min(v.x for v in bb)
    extent_y = max(v.y for v in bb) - min(v.y for v in bb)
    base_radius = 0.5 * max(extent_x, extent_y) + 6.0  # mm margin
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

    # --- 6. safety-cut below Z=0 (flat bottom guarantee) ------------------------
    # Cutter top face at z=-0.01: coincident faces at exactly z=0 make the
    # boolean solver eat the base plate, so we cut a hair below the floor.
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

    # --- 7. decimate -------------------------------------------------------------
    dec = bust.modifiers.new(name="decimate", type="DECIMATE")
    dec.decimate_type = "COLLAPSE"
    dec.ratio = max(0.02, min(decimate_ratio, 1.0))
    bpy.ops.object.modifier_apply(modifier="decimate")
    bpy.context.view_layer.update()

    # --- 8. export ----------------------------------------------------------------
    bpy.ops.object.select_all(action="DESELECT")
    bust.select_set(True)
    bpy.context.view_layer.objects.active = bust

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


if __name__ == "__main__":
    sys.exit(main())
