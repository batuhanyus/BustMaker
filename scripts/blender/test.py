"""Smoke test: verify the vendored Blender runs headlessly.

Usage (see core/blender_runner.py)::

    blender --background --factory-startup --python scripts/blender/test.py

Exits 0 and prints ``BUSTFORGE_BLENDER_OK <version>`` on success.
"""

import bpy  # noqa: F401  (importing proves the API is available)
import sys

version = bpy.app.version_string

# Create + delete a mesh to prove the API actually works headlessly.
mesh = bpy.data.meshes.new("smoke")
bpy.data.meshes.remove(mesh)

print(f"BUSTFORGE_BLENDER_OK {version}")
sys.exit(0)
