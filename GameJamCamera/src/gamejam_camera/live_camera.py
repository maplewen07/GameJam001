from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import camera_devices
from . import space_preprocess as stable

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def import_runtime_packages():
    deps = str(PROJECT_ROOT / ".deps")
    if Path(deps).exists():
        sys.path.insert(0, deps)
    try:
        import cv2 as cv2_module
        import numpy as np_module
        return cv2_module, np_module
    except Exception:
        for name in list(sys.modules):
            if name == "cv2" or name.startswith("cv2.") or name == "numpy" or name.startswith("numpy."):
                del sys.modules[name]
        try:
            sys.path.remove(deps)
        except ValueError:
            pass
        import cv2 as cv2_module
        import numpy as np_module
        return cv2_module, np_module


cv2, np = import_runtime_packages()


def backend_value(name: str) -> int:
    return {
        "any": 0,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }.get(name.lower(), cv2.CAP_DSHOW)


def resize_to(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def hsv(frame: np.ndarray, use_clahe: bool) -> np.ndarray:
    if use_clahe:
        return stable.hsv(frame)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def clamp01(value: object) -> float:
    return max(0.0, min(1.0, float(value)))


def roi_is_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "roi_enabled", False))


def roi_bounds(args: argparse.Namespace) -> tuple[float, float, float, float]:
    x1, x2 = sorted((clamp01(getattr(args, "roi_x_min", 0.0)), clamp01(getattr(args, "roi_x_max", 1.0))))
    y1, y2 = sorted((clamp01(getattr(args, "roi_y_min", 0.0)), clamp01(getattr(args, "roi_y_max", 1.0))))
    return x1, x2, y1, y2


def roi_rect(shape: tuple[int, ...], args: argparse.Namespace) -> tuple[int, int, int, int] | None:
    if not roi_is_enabled(args):
        return None
    height, width = shape[:2]
    if height <= 0 or width <= 0:
        return None
    x_min, x_max, y_min, y_max = roi_bounds(args)
    x1 = int(round(x_min * (width - 1)))
    x2 = int(round(x_max * (width - 1)))
    y1 = int(round(y_min * (height - 1)))
    y2 = int(round(y_max * (height - 1)))
    return x1, y1, max(x1, x2), max(y1, y2)


def calibration_roi_mask(shape: tuple[int, ...], args: argparse.Namespace) -> np.ndarray:
    height, width = shape[:2]
    if not roi_is_enabled(args):
        return np.full((height, width), 255, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    rect = roi_rect(shape, args)
    if rect is None:
        return mask
    x1, y1, x2, y2 = rect
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def draw_roi_rect(frame: np.ndarray, rect: tuple[int, int, int, int] | None, label: str = "ROI") -> None:
    if rect is None:
        return
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 2)
    cv2.putText(frame, label, (x1 + 6, max(18, y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)


def red_blue_masks(
    frame: np.ndarray,
    space: np.ndarray,
    red_h1_max: int,
    red_h2_min: int,
    red_s_min: int,
    red_v_min: int,
    blue_h_min: int,
    blue_h_max: int,
    blue_s_min: int,
    blue_v_min: int,
    use_clahe: bool,
) -> tuple[np.ndarray, np.ndarray]:
    hh = hsv(frame, use_clahe)
    red = cv2.inRange(hh, np.array((0, red_s_min, red_v_min)), np.array((red_h1_max, 255, 255)))
    red |= cv2.inRange(hh, np.array((red_h2_min, red_s_min, red_v_min)), np.array((179, 255, 255)))
    blue = cv2.inRange(hh, np.array((blue_h_min, blue_s_min, blue_v_min)), np.array((blue_h_max, 255, 255)))
    red = stable.clean(cv2.bitwise_and(red, space))
    blue = stable.clean(cv2.bitwise_and(blue, space))
    return red, blue


def mask_view(frame: np.ndarray, args: argparse.Namespace, use_clahe: bool) -> np.ndarray:
    full = calibration_roi_mask(frame.shape, args)
    red, blue = red_blue_masks(
        frame,
        full,
        args.red_h1_max,
        args.red_h2_min,
        args.red_s_min,
        args.red_v_min,
        args.blue_h_min,
        args.blue_h_max,
        args.blue_s_min,
        args.blue_v_min,
        use_clahe,
    )
    view = np.zeros_like(frame)
    view[red > 0] = (0, 0, 255)
    view[blue > 0] = (255, 80, 20)
    return view


def add_inset(frame: np.ndarray, inset: np.ndarray, title: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    inset_w = max(1, w // 4)
    inset_h = max(1, int(inset.shape[0] * inset_w / inset.shape[1]))
    small = cv2.resize(inset, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST)
    x = w - inset_w - 12
    y = 12
    out[y : y + inset_h, x : x + inset_w] = small
    cv2.rectangle(out, (x, y), (x + inset_w, y + inset_h), (255, 255, 255), 1)
    cv2.putText(out, title, (x, y + inset_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def detect_red_cup(frame: np.ndarray, args: argparse.Namespace) -> dict[str, float]:
    hh = hsv(frame, args.clahe)
    red = cv2.inRange(hh, np.array((0, args.red_s_min, args.red_v_min)), np.array((args.red_h1_max, 255, 255)))
    red |= cv2.inRange(hh, np.array((args.red_h2_min, args.red_s_min, args.red_v_min)), np.array((179, 255, 255)))
    red = cv2.bitwise_and(red, calibration_roi_mask(frame.shape, args))
    blobs = stable.merge_close(stable.components(stable.clean(red), args.red_min_area), 28)
    blobs = [
        b
        for b in blobs
        if args.red_cup_min_w <= b["w"] <= args.red_cup_max_w
        and args.red_cup_min_h <= b["h"] <= args.red_cup_max_h
        and b["h"] / max(b["w"], 1) >= args.red_cup_min_aspect
    ]
    if not blobs:
        raise RuntimeError("need red z cup, got 0")
    return max(blobs, key=lambda b: b["area"])


def blue_candidates(frame: np.ndarray, args: argparse.Namespace) -> list[dict[str, float]]:
    hh = hsv(frame, args.clahe)
    mask = cv2.inRange(
        hh,
        np.array((args.blue_h_min, args.blue_s_min, args.blue_v_min)),
        np.array((args.blue_h_max, 255, 255)),
    )
    mask = cv2.bitwise_and(mask, calibration_roi_mask(frame.shape, args))
    mask[: int(frame.shape[0] * args.blue_roi_top), :] = 0
    blobs = stable.merge_close(stable.components(stable.clean(mask), args.blue_min_area), 28)
    return [
        b
        for b in blobs
        if args.blue_cup_min_w <= b["w"] <= args.blue_cup_max_w
        and args.blue_cup_min_h <= b["h"] <= args.blue_cup_max_h
    ]


def detect_blue_cups(frame: np.ndarray, args: argparse.Namespace, z_ref: dict[str, float]) -> dict[str, dict[str, float]]:
    blobs = blue_candidates(frame, args)
    front_choices = [b for b in blobs if b["area"] >= args.front_blue_min_area and b["cy"] > z_ref["cy"]]
    if not front_choices:
        raise RuntimeError("need front-left blue cup below red z cup")

    front_left = min(front_choices, key=lambda b: abs((b["x"] + b["w"] * 0.5) - z_ref["cx"]))
    fl = calibration_point(front_left)
    right_choices = [
        b
        for b in front_choices
        if b is not front_left and (b["x"] + b["w"] * 0.5) > fl["cx"] + args.front_min_dx
    ]
    if not right_choices:
        raise RuntimeError("need front-right blue cup right of front-left")
    front_right = max(right_choices, key=lambda b: (b["y"] + b["h"], b["area"]))
    fr = calibration_point(front_right)

    x_min = min(fl["cx"], fr["cx"]) - args.back_x_margin
    x_max = max(fl["cx"], fr["cx"]) + args.back_x_margin
    front_top_y = min(fl["cy"], fr["cy"]) - args.back_y_gap
    back = [
        b
        for b in blobs
        if b is not front_left
        and b is not front_right
        and b["area"] >= args.back_blue_min_area
        and x_min <= b["x"] + b["w"] * 0.5 <= x_max
        and b["y"] + b["h"] <= front_top_y
    ]
    if len(back) < 2:
        raise RuntimeError(f"need 2 back blue cups between front cups, got {len(back)}")
    back = sorted(back, key=lambda b: b["x"] + b["w"] * 0.5)[:2]
    return {
        "front_left": fl,
        "front_right": fr,
        "back_left": calibration_point(back[0]),
        "back_right": calibration_point(back[1]),
    }


def calibration_point(blob: dict[str, float]) -> dict[str, float]:
    out = dict(blob)
    out["raw_cx"] = blob["cx"]
    out["raw_cy"] = blob["cy"]
    out["cx"] = blob["x"] + blob["w"] * 0.5
    out["cy"] = blob["y"] + blob["h"]
    return out


def calibrate(frame: np.ndarray, args: argparse.Namespace) -> dict[str, object]:
    z_ref = detect_red_cup(frame, args)
    blue = detect_blue_cups(frame, args, z_ref)
    polygon = np.array(
        [
            [blue["front_left"]["cx"], blue["front_left"]["cy"]],
            [blue["front_right"]["cx"], blue["front_right"]["cy"]],
            [blue["back_right"]["cx"], blue["back_right"]["cy"]],
            [blue["back_left"]["cx"], blue["back_left"]["cy"]],
        ],
        dtype=np.int32,
    )
    detection_polygon = stable.space_polygon(blue, z_ref, frame.shape[1], frame.shape[0])
    space = stable.make_space_mask(frame.shape[:2], detection_polygon)
    return {
        "blue": blue,
        "z_ref": z_ref,
        "polygon": polygon,
        "detection_polygon": detection_polygon,
        "space": space,
        "roi_rect": roi_rect(frame.shape, args),
        "xy_h": stable.xy_homography(blue),
        "axis_z": stable.z_axis(blue, z_ref),
    }


def median_blob(blobs: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for blob in blobs for key, value in blob.items() if isinstance(value, (int, float))})
    return {key: float(np.median([float(blob[key]) for blob in blobs if key in blob])) for key in keys}


def lock_calibration(
    samples: list[dict[str, object]],
    shape: tuple[int, int, int],
    args: argparse.Namespace | None = None,
) -> dict[str, object]:
    blue = {
        name: median_blob([sample["blue"][name] for sample in samples])
        for name in ("front_left", "front_right", "back_left", "back_right")
    }
    z_ref = median_blob([sample["z_ref"] for sample in samples])
    xy_polygon = np.array(
        [
            [blue["front_left"]["cx"], blue["front_left"]["cy"]],
            [blue["front_right"]["cx"], blue["front_right"]["cy"]],
            [blue["back_right"]["cx"], blue["back_right"]["cy"]],
            [blue["back_left"]["cx"], blue["back_left"]["cy"]],
        ],
        dtype=np.int32,
    )
    detection_polygon = stable.space_polygon(blue, z_ref, shape[1], shape[0])
    space = stable.make_space_mask(shape[:2], detection_polygon)
    return {
        "blue": blue,
        "z_ref": z_ref,
        "polygon": xy_polygon,
        "detection_polygon": detection_polygon,
        "space": space,
        "roi_rect": None,
        "xy_h": stable.xy_homography(blue),
        "axis_z": stable.z_axis(blue, z_ref),
        "samples": len(samples),
    }


def red_mask(frame: np.ndarray, space: np.ndarray, args: argparse.Namespace, use_clahe: bool) -> np.ndarray:
    hh = hsv(frame, use_clahe)
    red = cv2.inRange(hh, np.array((0, args.red_s_min, args.red_v_min)), np.array((args.red_h1_max, 255, 255)))
    red |= cv2.inRange(hh, np.array((args.red_h2_min, args.red_s_min, args.red_v_min)), np.array((179, 255, 255)))
    return stable.clean(cv2.bitwise_and(red, space))


def draw_space_overlay(frame: np.ndarray, cal: dict[str, object]) -> np.ndarray:
    overlay = frame.copy()
    cv2.polylines(overlay, [cal["detection_polygon"]], True, (0, 180, 255), 1)
    cv2.polylines(overlay, [cal["polygon"]], True, (0, 255, 255), 2)
    draw_roi_rect(overlay, cal.get("roi_rect"))

    blue_cups = cal["blue"]
    origin = (int(blue_cups["front_left"]["cx"]), int(blue_cups["front_left"]["cy"]))
    x_ref = (int(blue_cups["front_right"]["cx"]), int(blue_cups["front_right"]["cy"]))
    y_ref = (int(blue_cups["back_left"]["cx"]), int(blue_cups["back_left"]["cy"]))
    z_ref = (int(cal["z_ref"]["cx"]), int(cal["z_ref"]["cy"]))
    cv2.line(overlay, origin, x_ref, (0, 255, 0), 2)
    cv2.line(overlay, origin, y_ref, (255, 255, 0), 2)
    cv2.line(overlay, origin, z_ref, (255, 0, 255), 2)
    cv2.circle(overlay, z_ref, 5, (255, 0, 255), -1)
    cv2.putText(overlay, "z_ref", (z_ref[0] + 6, z_ref[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    for name, blob in blue_cups.items():
        p = (int(blob["cx"]), int(blob["cy"]))
        cv2.circle(overlay, p, 5, (255, 255, 255), -1)
        cv2.putText(overlay, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return overlay


class Preview:
    def __init__(self) -> None:
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("live_camera_720_clahe")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.key = ""
        self.closed = False
        self.root.bind("<Key>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_key(self, event: object) -> None:
        self.key = getattr(event, "char", "") or getattr(event, "keysym", "")

    def _on_close(self) -> None:
        self.closed = True

    def show(self, frame: np.ndarray) -> str:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        data = f"P6\n{w} {h}\n255\n".encode("ascii") + rgb.tobytes()
        image = self.tk.PhotoImage(data=data, format="PPM")
        self.label.configure(image=image)
        self.label.image = image
        self.root.update_idletasks()
        self.root.update()
        key, self.key = self.key, ""
        return "q" if self.closed else key

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    camera_index, device_message = camera_devices.resolve_camera_index(args)
    if device_message:
        print(device_message)
    tried = []
    cap = None
    for name in dict.fromkeys([args.backend, "any", "msmf", "dshow"]):
        backend = backend_value(name)
        current = cv2.VideoCapture(camera_index, backend) if backend else cv2.VideoCapture(camera_index)
        tried.append(f"{name}:configured")
        if current.isOpened() and configure_and_probe(current, args, configure=True):
            cap = current
            print(f"camera_backend={name}")
            break
        current.release()
        current = cv2.VideoCapture(camera_index, backend) if backend else cv2.VideoCapture(camera_index)
        tried.append(f"{name}:default")
        if current.isOpened() and configure_and_probe(current, args, configure=False):
            cap = current
            print(f"camera_backend={name} camera_config=default")
            break
        current.release()
    if cap is None:
        raise SystemExit(f"cannot read camera {camera_index}; tried backends: {', '.join(tried)}")
    return cap


def configure_and_probe(cap: cv2.VideoCapture, args: argparse.Namespace, configure: bool) -> bool:
    if configure:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc[:4]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(args.camera_warmup_reads):
        ok, _ = cap.read()
        if ok:
            return True
        time.sleep(0.05)
    return False
