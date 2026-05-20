"""Configuration file helpers for DEM-processing commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback.
    tomllib = None


def load_config_file(path: str | Path | None) -> Dict[str, Any]:
    """Load a JSON or TOML config file."""
    if not path:
        return {}
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML config files require Python 3.11+ tomllib.")
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    else:
        raise ValueError(f"Unsupported config format {suffix!r}. Use .json or .toml.")

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain an object/table: {config_path}")
    return normalize_config_keys(data)


def normalize_config_keys(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize dashed config keys to snake_case dataclass field names."""
    return {str(key).replace("-", "_"): value for key, value in data.items()}


def explicit_cli_flags(argv: Iterable[str] | None = None) -> set[str]:
    """Return command-line flags explicitly present in argv."""
    if argv is None:
        argv = sys.argv[1:]
    flags: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0]
        flags.add(flag)
    return flags


def config_to_cli_args(config: Mapping[str, Any]) -> list[str]:
    """Convert a dict of config values to command-line arguments."""
    args: list[str] = []
    for key, value in config.items():
        flag = f"--{str(key).replace('_', '-')}"
        if value is True:
            args.append(flag)
        elif value is False or value is None:
            continue
        else:
            args.extend([flag, str(value)])
    return args
