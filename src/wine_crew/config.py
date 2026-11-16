"""Load config.yaml and resolve repo-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml (or the given file) into a dict."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Return config['paths'][key] as an absolute path (relative paths resolve against the repo root)."""
    return REPO_ROOT / config["paths"][key]
