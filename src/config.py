"""Load and expose config from YAML."""
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_data_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("data", {})


def get_train_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("train", {})


def get_model_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("model", {})


def get_paths_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("paths", {})
