from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str
    unique_id: str
    model_id: str = ""


def discover_camera_devices() -> list[CameraDevice]:
    devices = discover_macos_cameras()
    if devices:
        return devices
    return [CameraDevice(index=i, name=f"Camera {i}", unique_id=str(i)) for i in range(4)]


def discover_macos_cameras() -> list[CameraDevice]:
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            check=False,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return parse_system_profiler_cameras(result.stdout)


def parse_system_profiler_cameras(text: str) -> list[CameraDevice]:
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped == "Camera:":
            continue
        if raw_line.startswith("    ") and not raw_line.startswith("      ") and stripped.endswith(":"):
            if current:
                records.append(current)
            current = {"name": stripped[:-1]}
            continue
        if current is None:
            continue
        if stripped.startswith("Model ID:"):
            current["model_id"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Unique ID:"):
            current["unique_id"] = stripped.split(":", 1)[1].strip()
    if current:
        records.append(current)

    devices = []
    for index, record in enumerate(records):
        name = record.get("name", f"Camera {index}")
        unique_id = record.get("unique_id") or name
        devices.append(
            CameraDevice(
                index=index,
                name=name,
                unique_id=unique_id,
                model_id=record.get("model_id", ""),
            )
        )
    return devices


def device_label(device: CameraDevice, all_devices: list[CameraDevice]) -> str:
    duplicate_name = sum(1 for item in all_devices if item.name == device.name) > 1
    return f"{device.name} [{device.unique_id}]" if duplicate_name else device.name


def resolve_camera_index(args: argparse.Namespace) -> tuple[int, str | None]:
    fallback_index = int(getattr(args, "camera", 0))
    requested = str(getattr(args, "camera_device", "") or "").strip()
    if not requested:
        return fallback_index, None

    devices = discover_camera_devices()
    requested_lower = requested.lower()
    for device in devices:
        candidates = {
            str(device.index).lower(),
            device.name.lower(),
            device.unique_id.lower(),
            device_label(device, devices).lower(),
        }
        if requested_lower in candidates:
            return device.index, f"camera_device={device.name} index={device.index}"
    return fallback_index, f"camera_device_not_found={requested}; using camera={fallback_index}"
