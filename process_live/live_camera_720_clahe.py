import argparse
import importlib.util
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))

import cv2
import numpy as np


STABLE_PATH = ROOT / "process_2487" / "stabilize_space_mask_preprocess.py"
spec = importlib.util.spec_from_file_location("stable_preprocess", STABLE_PATH)
stable = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(stable)


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
    full = np.full(frame.shape[:2], 255, dtype=np.uint8)
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
    blobs = stable.merge_close(stable.components(stable.clean(red), args.red_min_area), 28)
    blobs = [
        b
        for b in blobs
        if 20 <= b["w"] <= 260 and 40 <= b["h"] <= 360 and b["h"] / max(b["w"], 1) >= 0.75
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
    mask[: int(frame.shape[0] * args.blue_roi_top), :] = 0
    blobs = stable.merge_close(stable.components(stable.clean(mask), args.blue_min_area), 28)
    return [b for b in blobs if 12 <= b["w"] <= 320 and 12 <= b["h"] <= 300]


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
    # ponytail: red cup anchors front-left; fail instead of using background blue noise.
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
    return {
        "blue": blue,
        "z_ref": z_ref,
        "polygon": polygon,
        "detection_polygon": detection_polygon,
        "space": stable.make_space_mask(frame.shape[:2], detection_polygon),
        "xy_h": stable.xy_homography(blue),
        "axis_z": stable.z_axis(blue, z_ref),
    }


def median_blob(blobs: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for blob in blobs for key, value in blob.items() if isinstance(value, (int, float))})
    return {key: float(np.median([float(blob[key]) for blob in blobs if key in blob])) for key in keys}


def lock_calibration(samples: list[dict[str, object]], shape: tuple[int, int, int]) -> dict[str, object]:
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
    return {
        "blue": blue,
        "z_ref": z_ref,
        "polygon": xy_polygon,
        "detection_polygon": detection_polygon,
        "space": stable.make_space_mask(shape[:2], detection_polygon),
        "xy_h": stable.xy_homography(blue),
        "axis_z": stable.z_axis(blue, z_ref),
        "samples": len(samples),
    }


def red_mask(frame: np.ndarray, space: np.ndarray, args: argparse.Namespace, use_clahe: bool) -> np.ndarray:
    hh = hsv(frame, use_clahe)
    red = cv2.inRange(hh, np.array((0, args.red_s_min, args.red_v_min)), np.array((args.red_h1_max, 255, 255)))
    red |= cv2.inRange(hh, np.array((args.red_h2_min, args.red_s_min, args.red_v_min)), np.array((179, 255, 255)))
    return stable.clean(cv2.bitwise_and(red, space))


def marker_row(blob: dict[str, float], role: str, cal: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    space_x, space_y = stable.map_xy(cal["xy_h"], blob["cx"], blob["cy"])
    space_z = stable.map_z(cal["axis_z"], blob["cx"], blob["cy"]) / 100.0 * args.z_height
    return {**blob, "role": role, "space_x": space_x, "space_y": space_y, "space_z": space_z}


def process_tracking_frame(
    frame: np.ndarray,
    cal: dict[str, object],
    args: argparse.Namespace,
    use_clahe: bool,
) -> tuple[np.ndarray, list[dict[str, object]], str]:
    red = red_mask(frame, cal["space"], args, use_clahe)
    candidates = []
    for blob in stable.components(red, args.blob_min_area):
        if np.hypot(blob["cx"] - cal["z_ref"]["cx"], blob["cy"] - cal["z_ref"]["cy"]) <= args.z_exclude_px:
            continue
        candidates.append(blob)

    rows: list[dict[str, object]] = []
    status = "missing_red"
    if len(candidates) >= 2:
        foot = max(candidates, key=lambda b: b["cy"])
        head = min((b for b in candidates if b is not foot), key=lambda b: b["cy"])
        rows = [marker_row(head, "head", cal, args), marker_row(foot, "foot", cal, args)]
        status = "ok"
    elif candidates:
        rows = [marker_row(candidates[0], "candidate", cal, args)]
        status = "partial"

    return draw_tracking_overlay(frame, red, rows, cal), rows, status


def draw_space_overlay(frame: np.ndarray, cal: dict[str, object]) -> np.ndarray:
    overlay = frame.copy()
    cv2.polylines(overlay, [cal["detection_polygon"]], True, (0, 180, 255), 1)
    cv2.polylines(overlay, [cal["polygon"]], True, (0, 255, 255), 2)

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


def draw_tracking_overlay(
    frame: np.ndarray,
    red: np.ndarray,
    rows: list[dict[str, object]],
    cal: dict[str, object],
) -> np.ndarray:
    overlay = draw_space_overlay(frame, cal)
    overlay[red > 0] = (overlay[red > 0] * 0.35 + np.array((0, 0, 255)) * 0.65).astype(np.uint8)
    for row in rows:
        color = (255, 255, 255)
        if row["role"] == "head":
            color = (0, 0, 255)
        elif row["role"] == "foot":
            color = (0, 255, 255)
        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        label = f"{row['role']} x={row['space_x']:.1f} y={row['space_y']:.1f} z={row['space_z']:.2f}"
        cv2.putText(overlay, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
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
    tried = []
    cap = None
    for name in dict.fromkeys([args.backend, "any", "msmf", "dshow"]):
        backend = backend_value(name)
        current = cv2.VideoCapture(args.camera, backend) if backend else cv2.VideoCapture(args.camera)
        tried.append(f"{name}:configured")
        if current.isOpened() and configure_and_probe(current, args, configure=True):
            cap = current
            print(f"camera_backend={name}")
            break
        current.release()
        current = cv2.VideoCapture(args.camera, backend) if backend else cv2.VideoCapture(args.camera)
        tried.append(f"{name}:default")
        if current.isOpened() and configure_and_probe(current, args, configure=False):
            cap = current
            print(f"camera_backend={name} camera_config=default")
            break
        current.release()
    if cap is None:
        raise SystemExit(f"cannot read camera {args.camera}; tried backends: {', '.join(tried)}")
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


def scan_cameras(args: argparse.Namespace) -> None:
    for camera in range(args.scan_cameras):
        probe = argparse.Namespace(**vars(args))
        probe.camera = camera
        for name in dict.fromkeys([args.backend, "any", "msmf", "dshow"]):
            backend = backend_value(name)
            cap = cv2.VideoCapture(camera, backend) if backend else cv2.VideoCapture(camera)
            ok = cap.isOpened() and configure_and_probe(cap, probe, configure=True)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if ok:
                print(f"camera={camera} backend={name} actual={actual_w}x{actual_h}@{actual_fps:.1f}")
                break
        else:
            print(f"camera={camera} unreadable")


def run_camera(args: argparse.Namespace) -> None:
    cap = open_camera(args)
    preview = Preview()
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"camera={args.camera} actual={actual_w}x{actual_h}@{actual_fps:.1f}")
    print(f"process={args.process_width}x{args.process_height} clahe={args.clahe}")
    print("keys: c=recalibrate h=toggle CLAHE q/esc=quit")

    locked_cal = None
    last_sample = None
    samples: list[dict[str, object]] = []
    calibration_start = time.perf_counter()
    use_clahe = args.clahe
    fps_smooth = 0.0
    last_status = time.perf_counter()
    last = time.perf_counter()
    frame_i = 0
    rows: list[dict[str, object]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break
        frame = resize_to(frame, args.process_width, args.process_height)

        now = time.perf_counter()
        dt = max(1e-6, now - last)
        last = now
        fps_smooth = (0.85 * fps_smooth + 0.15 / dt) if fps_smooth else 1.0 / dt
        args.clahe = use_clahe

        if locked_cal is None:
            elapsed = now - calibration_start
            try:
                last_sample = calibrate(frame, args)
                samples.append(last_sample)
                if elapsed >= args.calibration_seconds and len(samples) >= args.min_calibration_frames:
                    locked_cal = lock_calibration(samples, frame.shape)
                    overlay = draw_space_overlay(frame, locked_cal)
                    status = f"calibration locked samples={len(samples)} fps={fps_smooth:.1f} clahe={'on' if use_clahe else 'off'}"
                    print(status)
                else:
                    overlay = draw_space_overlay(frame, last_sample)
                    status = (
                        f"calibrating valid={len(samples)}/required={args.min_calibration_frames} "
                        f"t={elapsed:.1f}/{args.calibration_seconds:.1f}s fps={fps_smooth:.1f}"
                    )
            except Exception as exc:
                overlay = draw_space_overlay(frame, last_sample) if last_sample else frame.copy()
                status = (
                    f"calibrating valid={len(samples)}/required={args.min_calibration_frames} "
                    f"failed: {exc}"
                )
            if args.show_calibration_mask:
                overlay = add_inset(overlay, mask_view(frame, args, use_clahe), "calibration mask")
        else:
            overlay, rows, tracking_status = process_tracking_frame(frame, locked_cal, args, use_clahe)
            status = (
                f"locked {tracking_status} markers={len(rows)} fps={fps_smooth:.1f} "
                f"clahe={'on' if use_clahe else 'off'}"
            )

        cv2.putText(overlay, status, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        key = preview.show(overlay)

        if now - last_status >= 1.0:
            print(f"frame={frame_i} {status}")
            last_status = now
        frame_i += 1

        if key in ("Escape", "q"):
            break
        if key == "c":
            locked_cal = None
            last_sample = None
            samples = []
            calibration_start = time.perf_counter()
            print("restarting calibration")
        if key == "h":
            use_clahe = not use_clahe
            args.clahe = use_clahe
            print(f"clahe={'on' if use_clahe else 'off'}")

    cap.release()
    preview.close()


def benchmark(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(str(args.benchmark_video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.benchmark_video}")

    frames = []
    while len(frames) < args.benchmark_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(resize_to(frame, args.process_width, args.process_height))
    cap.release()
    if not frames:
        raise SystemExit("cannot read benchmark frames")

    samples = []
    for frame in frames:
        try:
            samples.append(calibrate(frame, args))
        except RuntimeError:
            pass
    if len(samples) < args.min_calibration_frames:
        raise SystemExit(
            f"benchmark calibration failed: valid={len(samples)}/required={args.min_calibration_frames}"
        )
    cal = lock_calibration(samples, frames[0].shape)

    blobs = 0
    t0 = time.perf_counter()
    for frame in frames:
        _, rows, _ = process_tracking_frame(frame, cal, args, args.clahe)
        blobs += len(rows)
    dt = time.perf_counter() - t0
    print("benchmark_note=preloaded_720p_processing_only")
    print(f"benchmark_frames={len(frames)}")
    print(f"benchmark_calibration_samples={len(samples)}")
    print(f"benchmark_seconds={dt:.3f}")
    print(f"benchmark_fps={len(frames) / dt:.1f}")
    print(f"benchmark_blobs={blobs}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--process-width", type=int, default=1280)
    parser.add_argument("--process-height", type=int, default=720)
    parser.add_argument("--backend", default="dshow", choices=["any", "dshow", "msmf"])
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--clahe", dest="clahe", action="store_true", default=True)
    parser.add_argument("--no-clahe", dest="clahe", action="store_false")
    parser.add_argument("--red-h1-max", type=int, default=12)
    parser.add_argument("--red-h2-min", type=int, default=168)
    parser.add_argument("--red-s-min", type=int, default=160)
    parser.add_argument("--red-v-min", type=int, default=100)
    parser.add_argument("--red-min-area", type=int, default=500)
    parser.add_argument("--blue-h-min", type=int, default=108)
    parser.add_argument("--blue-h-max", type=int, default=129)
    parser.add_argument("--blue-s-min", type=int, default=145)
    parser.add_argument("--blue-v-min", type=int, default=41)
    parser.add_argument("--blue-min-area", type=int, default=150)
    parser.add_argument("--blue-roi-top", type=float, default=0.25)
    parser.add_argument("--front-blue-min-area", type=int, default=800)
    parser.add_argument("--back-blue-min-area", type=int, default=250)
    parser.add_argument("--front-min-dx", type=float, default=80.0)
    parser.add_argument("--back-x-margin", type=float, default=20.0)
    parser.add_argument("--back-y-gap", type=float, default=15.0)
    parser.add_argument("--blob-min-area", type=int, default=60)
    parser.add_argument("--calibration-seconds", type=float, default=10.0)
    parser.add_argument("--min-calibration-frames", type=int, default=30)
    parser.add_argument("--z-exclude-px", type=float, default=80.0)
    parser.add_argument("--z-height", type=float, default=1.0)
    parser.add_argument("--show-calibration-mask", dest="show_calibration_mask", action="store_true", default=True)
    parser.add_argument("--hide-calibration-mask", dest="show_calibration_mask", action="store_false")
    parser.add_argument("--benchmark-video", type=Path)
    parser.add_argument("--benchmark-frames", type=int, default=180)
    parser.add_argument("--scan-cameras", type=int, default=0)
    parser.add_argument("--camera-warmup-reads", type=int, default=40)
    args = parser.parse_args()

    if args.scan_cameras:
        scan_cameras(args)
    elif args.benchmark_video:
        benchmark(args)
    else:
        run_camera(args)


if __name__ == "__main__":
    main()
