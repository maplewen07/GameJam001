#!/usr/bin/env python3
"""Start the full live camera pipeline from one file."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def app_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


ROOT = app_root()
PIPELINE_PATH = ROOT / "process_multi_live" / "live_multi_red_tracking.py"


def load_pipeline():
    if not PIPELINE_PATH.exists():
        raise SystemExit(f"pipeline file not found: {PIPELINE_PATH}")

    deps_path = ROOT / ".deps"
    if deps_path.exists():
        sys.path.insert(0, str(deps_path))

    spec = importlib.util.spec_from_file_location("live_multi_red_tracking", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load pipeline file: {PIPELINE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_pipeline()
    module.main()


if __name__ == "__main__":
    main()
