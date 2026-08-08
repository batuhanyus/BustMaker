"""Job report accumulation and serialization.

Every pipeline run produces a ``report.json`` at the job root. Schema::

    {
      "schema_version": 1,
      "job_id": "abc123def456",
      "created_at": "2026-08-08T14:05:00.000Z",
      "input": {"path": "...", "type": "video|images|single_image"},
      "config": {"mode": "auto", "preset": "balanced", "quality_preset": {...}},
      "stages": [
        {"name": "ingest", "status": "success|failed|skipped",
         "duration_s": 12.3, "warnings": [...],
         "fallback_used": null|"triposr", "artifacts": {"count": 42}}
      ],
      "mesh": {"raw_mesh": "...", "stl": "...", "glb": "...",
               "stats": {"vertices": ..., "triangles": ..., "volume_mm3": ...,
                         "bounds_mm": [...], "is_watertight": ...}},
      "warnings": [...],
      "fallbacks_used": [...],
      "summary": {"success": true, "total_duration_s": 123.4,
                  "outputs": {"stl": "...", "glb": "...", "report": "..."}}
    }

``stages`` and ``mesh`` are filled in by the orchestrator; backends attach
free-form results under ``stage_result`` when needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class StageRecord:
    name: str
    status: str = "pending"  # pending | running | success | failed | skipped
    duration_s: Optional[float] = None
    warnings: list[str] = field(default_factory=list)
    fallback_used: Optional[str] = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_s": self.duration_s,
            "warnings": self.warnings,
            "fallback_used": self.fallback_used,
            "artifacts": self.artifacts,
            "error": self.error,
        }


@dataclass
class JobReport:
    job_id: str
    created_at: str
    input: dict[str, Any]
    config: dict[str, Any]
    stages: list[StageRecord] = field(default_factory=list)
    mesh: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    fallbacks_used: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    SCHEMA_VERSION = 1

    # -- construction ---------------------------------------------------------

    @classmethod
    def new(
        cls,
        job_id: str,
        input_path: Path,
        input_type: str,
        config_snapshot: dict[str, Any],
    ) -> "JobReport":
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return cls(
            job_id=job_id,
            created_at=now,
            input={"path": str(input_path), "type": input_type},
            config=config_snapshot,
        )

    # -- stage lifecycle --------------------------------------------------------

    def stage(self, name: str) -> StageRecord:
        for s in self.stages:
            if s.name == name:
                return s
        rec = StageRecord(name=name)
        self.stages.append(rec)
        return rec

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def mark_stage(
        self,
        name: str,
        status: str,
        duration_s: Optional[float] = None,
        warnings: Optional[list[str]] = None,
        fallback_used: Optional[str] = None,
        artifacts: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> StageRecord:
        rec = self.stage(name)
        rec.status = status
        if duration_s is not None:
            rec.duration_s = round(duration_s, 3)
        if warnings:
            rec.warnings.extend(warnings)
        if fallback_used:
            rec.fallback_used = fallback_used
            if fallback_used not in self.fallbacks_used:
                self.fallbacks_used.append(fallback_used)
        if artifacts:
            rec.artifacts.update(artifacts)
        if error:
            rec.error = error
        return rec

    # -- finalization -------------------------------------------------------------

    def finalize(self, total_duration_s: float, success: bool) -> None:
        self.summary = {
            "success": success,
            "total_duration_s": round(total_duration_s, 3),
            "outputs": {
                "stl": self.mesh.get("stl"),
                "glb": self.mesh.get("glb"),
                "report": self.mesh.get("report"),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "input": self.input,
            "config": self.config,
            "stages": [s.to_dict() for s in self.stages],
            "mesh": self.mesh,
            "warnings": self.warnings,
            "fallbacks_used": self.fallbacks_used,
            "summary": self.summary,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.summary.setdefault("outputs", {})["report"] = str(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return path


def load_report(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
