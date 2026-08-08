"""COLMAP photogrammetry backend (experimental, capture mode).

Pipeline: COLMAP (vendored or PATH) feature extraction -> exhaustive matching
-> sparse reconstruction -> image undistortion -> patch-match stereo ->
stereo fusion -> Open3D Poisson surface reconstruction -> OBJ.

Robustness rules baked in:

* if the COLMAP binary is missing, the backend reports *unavailable* and the
  orchestrator moves on (never a hard failure),
* every sub-step failure surfaces as a BackendResult error with the failing
  command's stderr tail,
* the fused point cloud must contain a minimum number of points, otherwise we
  report failure (a sparse/broken capture is worse than falling back to
  generative reconstruction).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from adapters.generative.base import BackendResult, MeshBackend
from core.config import Config
from core.logging import get_logger
from core.model_manager import free_memory
from core.paths import find_colmap
from core.pipeline import RunContext

log = get_logger("colmap")

MIN_POINTS = 20_000


class ColmapBackend(MeshBackend):
    name = "colmap"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def available(self) -> bool:
        return find_colmap(self.cfg) is not None

    def generate(self, ctx: RunContext, input_frames: list[Path], out_path: Path) -> BackendResult:
        colmap = find_colmap(self.cfg)
        if colmap is None:
            return BackendResult(
                success=False,
                error=(
                    "COLMAP binary not found (experimental capture backend). "
                    "Install it and set paths.colmap_path, or use generative mode."
                )
            )
        if len(input_frames) < int(self.cfg.get("capture.min_views", 12)):
            return BackendResult(
                success=False,
                error=f"capture needs >= {self.cfg.get('capture.min_views')} views, got {len(input_frames)}",
            )

        work = ctx.job.temp_dir / "colmap"
        try:
            self._run_photogrammetry(colmap, input_frames, work)
            mesh = self._poisson_surface(work / "dense" / "fused.ply", out_path)
            return BackendResult(
                success=True,
                mesh_path=out_path,
                settings={"backend": self.name, "views": len(input_frames),
                          "poisson_depth": 9, "min_points": MIN_POINTS},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("colmap failed: %s", exc)
            return BackendResult(success=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            free_memory()

    # -- internals -----------------------------------------------------------------

    def _run(self, colmap: Path, args: list[str], work: Path) -> None:
        cmd = [str(colmap), *args]
        log.info("colmap: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", cwd=str(work))
        if proc.returncode != 0:
            raise RuntimeError(f"colmap {' '.join(args[:3])} failed: {proc.stderr[-1500:]}")

    def _run_photogrammetry(self, colmap: Path, frames: list[Path], work: Path) -> None:
        (work / "images").mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            shutil.copy2(f, work / "images" / f"img_{i:05d}.png")
        db = work / "db.db"

        self._run(colmap, ["feature_extractor", "--database_path", str(db),
                           "--image_path", str(work / "images"),
                           "--ImageReader.single_camera", "1",
                           "--SiftExtraction.use_gpu", str(int(self.cfg.get("capture.gpu", True)))], work)
        self._run(colmap, ["exhaustive_matcher", "--database_path", str(db),
                           "--SiftMatching.use_gpu", str(int(self.cfg.get("capture.gpu", True)))], work)
        self._run(colmap, ["mapper", "--database_path", str(db),
                           "--image_path", str(work / "images"),
                           "--output_path", str(work / "sparse")], work)

        sparse_dir = work / "sparse" / "0"
        if not sparse_dir.is_dir():
            raise RuntimeError("COLMAP mapper produced no reconstruction (insufficient overlap?)")

        self._run(colmap, ["image_undistorter", "--image_path", str(work / "images"),
                           "--input_path", str(sparse_dir),
                           "--output_path", str(work / "dense")], work)
        self._run(colmap, ["patch_match_stereo", "--workspace_path", str(work / "dense"),
                           "--PatchMatchStereo.geom_consistency", "true"], work)
        self._run(colmap, ["stereo_fusion", "--workspace_path", str(work / "dense"),
                           "--output_path", str(work / "dense" / "fused.ply")], work)

    def _poisson_surface(self, ply_path: Path, out_path: Path) -> None:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(ply_path))
        if len(pcd.points) < MIN_POINTS:
            raise RuntimeError(f"fused cloud too sparse ({len(pcd.points)} pts < {MIN_POINTS})")
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(30)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=True)
