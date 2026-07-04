from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "defaults.json"


def namespace_to_dict(namespace: argparse.Namespace) -> dict[str, Any]:
    return {key: encode_value(value) for key, value in vars(namespace).items()}


def encode_value(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return str(value.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(value)
    return value


def read_json(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, [f"config_missing={path}; using fallback defaults"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, [f"config_invalid={path}: {exc}; using fallback defaults"]
    if not isinstance(data, dict):
        return {}, [f"config_invalid={path}: root must be an object; using fallback defaults"]
    return data, []


def merge_config(
    fallback: argparse.Namespace,
    config_path: Path,
) -> tuple[argparse.Namespace, list[str]]:
    fallback_values = vars(fallback)
    raw, messages = read_json(config_path)
    values = dict(fallback_values)
    for key, raw_value in raw.items():
        if key not in fallback_values:
            messages.append(f"config_ignored_unknown_key={key}")
            continue
        try:
            values[key] = coerce_value(raw_value, fallback_values[key])
        except (TypeError, ValueError) as exc:
            messages.append(f"config_ignored_invalid_key={key}: {exc}")
    return argparse.Namespace(**values), messages


def coerce_value(value: Any, fallback: Any) -> Any:
    if isinstance(fallback, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off"):
                return False
        raise TypeError("expected bool")
    if isinstance(fallback, int) and not isinstance(fallback, bool):
        if isinstance(value, bool):
            raise TypeError("expected int")
        return int(value)
    if isinstance(fallback, float):
        if isinstance(value, bool):
            raise TypeError("expected float")
        return float(value)
    if isinstance(fallback, Path):
        if not isinstance(value, str):
            raise TypeError("expected path string")
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    if isinstance(fallback, str):
        return str(value)
    return value


def save_config(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = {key: encode_value(value) for key, value in sorted(values.items())}
    path.write_text(json.dumps(encoded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
