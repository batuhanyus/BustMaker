"""Configuration loading and typed access for the Bust Forge pipeline.

The single source of truth is ``config.yaml`` at the project root. This module
parses it into a :class:`Config` object with attribute access and small helper
methods (presets, mode chains, resolved paths).

Values are *validated on load* so that a bad config fails fast at startup
instead of deep inside a stage.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# ---------------------------------------------------------------------------
# Small typed views over the YAML dict. We keep the raw dict as the source of
# truth and expose typed helpers on top, so unknown keys survive round-trips
# and stages can read their own nested settings without schema friction.
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Parsed pipeline configuration.

    ``data`` holds the full raw YAML dict (deep-copied). Typed accessors below
    return validated values with defaults, so stages can rely on them.
    """

    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "Config":
        cfg_path = Path(path) if path else _default_config_path()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}: {cfg_path}")
        cfg = cls(path=cfg_path, data=raw)
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        presets = self.data.get("quality_presets", {})
        for name in ("fast", "balanced", "high"):
            if name not in presets:
                raise ValueError(f"config: missing quality preset '{name}'")
        modes = self.data.get("modes", {})
        for name in ("auto", "generative", "capture", "depth_relief"):
            if name not in modes:
                raise ValueError(f"config: missing mode '{name}'")
        for name, chain in modes.items():
            if not chain:
                raise ValueError(f"config: mode '{name}' has an empty strategy chain")
        for preset in presets.values():
            for key in ("gen_resolution", "video_max_frames", "decimate_ratio", "voxel_size_mm"):
                if key not in preset:
                    raise ValueError(f"config: preset missing key '{key}'")

    # -- generic helpers ------------------------------------------------------

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value by dotted path, e.g. ``print.target_height_mm``."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        return self.data.get(name, {})

    # -- typed accessors -------------------------------------------------------

    def preset(self, name: str) -> dict[str, Any]:
        presets = self.data.get("quality_presets", {})
        if name not in presets:
            raise KeyError(f"Unknown quality preset '{name}' (have {sorted(presets)})")
        return copy.deepcopy(presets[name])

    def mode_chain(self, mode: str) -> list[str]:
        modes = self.data.get("modes", {})
        if mode not in modes:
            raise KeyError(f"Unknown mode '{mode}' (have {sorted(modes)})")
        return list(modes[mode])

    def resolve_path(self, key: str, default: str) -> Path:
        """Resolve a config path (relative to the config file's directory)."""
        raw = self.get(key, default)
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = self.path.parent / p
        return p.resolve()

    # -- sections ---------------------------------------------------------------

    @property
    def paths(self) -> dict[str, Any]:
        return self.data.get("paths", {})

    @property
    def print_cfg(self) -> dict[str, Any]:
        return self.data.get("print", {})

    @property
    def vr_cfg(self) -> dict[str, Any]:
        return self.data.get("vr", {})

    @property
    def generative_cfg(self) -> dict[str, Any]:
        return self.data.get("generative", {})

    @property
    def logging_level(self) -> str:
        return str(self.get("logging.level", "INFO")).upper()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Config({self.path})"


def _default_config_path() -> Path:
    """Project root is two levels above this file (core/config.py -> root)."""
    return Path(__file__).resolve().parent.parent / "config.yaml"
