import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


sys.path.insert(0, str(Path(__file__).with_name(".deps")))

import cv2
import numpy as np

from cups_3d_tracker import Blob, blobs_from_mask, draw_blob, hsv_mask, merge_close


VIDEO_PATH = Path(r"C:\Users\Administrator\Downloads\IMG_2483.MOV")
OUT_VIDEO = Path("wristband_3d_tracker_overlay.mp4")
OUT_CSV = Path("wristband_3d_positions.csv")

# ponytail: relative units; replace with measured room/cup values if needed.
WORLD_W = 100.0
WORLD_D = 100.0
Z_REF_HEIGHT = 25.0
OUTPUT_SCALE = 0.5

# ponytail: this MOV has clear first-frame marker positions; use them to avoid white-shirt false picks.
INITIAL_HAND_BBOX = (2620, 1030, 190, 200)
INITIAL_FOOT_BBOX = (2540, 1530, 320, 210)


@dataclass
class Calibration:
    image_to_world: np.ndarray
    blue: dict[str, Blob]
    z_ref: Blob
    z_base: Blob
    z_unit: np.ndarray
    z_pixels: float
    frame_index: int


def detect_blue(frame: np.ndarray) -> dict[str, Blob] | None:
    h, _ = frame.shape[:2]
    mask = hsv_mask(frame, [((95, 35, 20), (135, 255, 235))])
    mask[: int(h * 0.52), :] = 0
    blobs = merge_close(blobs_from_mask(mask, 45, 20000), 28)
    blobs = [b for b in blobs if 10 <= b.w <= 180 and 10 <= b.h <= 180]
    blobs = sorted(blobs, key=lambda b: b.area, reverse=True)[:4]
    if len(blobs) < 4:
        return None

    front = sorted(sorted(blobs, key=lambda b: b.cy, reverse=True)[:2], key=lambda b: b.cx)
    back = sorted(sorted(blobs, key=lambda b: b.cy, reverse=True)[2:], key=lambda b: b.cx)
    return {"front_left": front[0], "front_right": front[1], "back_left": back[0], "back_right": back[1]}


def detect_red_z(frame: np.ndarray) -> Blob | None:
    h, w = frame.shape[:2]
    mask = hsv_mask(frame, [((0, 70, 60), (12, 255, 255)), ((170, 70, 60), (179, 255, 255))])
    blobs = merge_close(blobs_from_mask(mask, 80, 20000), 24)
    choices = [
        b
        for b in blobs
        if b.cx < w * 0.22 and h * 0.48 < b.cy < h * 0.78 and 20 <= b.w <= 180 and 25 <= b.h <= 180
    ]
    return max(choices, key=lambda b: b.area, default=None)


def calibrate(video: Path) -> tuple[Calibration, np.ndarray, float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first_frame = None
    for frame_index in range(0, max(frame_count, 1), 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        first_frame = frame if first_frame is None else first_frame
        blue = detect_blue(frame)
        z_ref = detect_red_z(frame)
        if not blue or not z_ref:
            continue

        img = np.array(
            [
                [blue["front_left"].cx, blue["front_left"].cy],
                [blue["front_right"].cx, blue["front_right"].cy],
                [blue["back_right"].cx, blue["back_right"].cy],
                [blue["back_left"].cx, blue["back_left"].cy],
            ],
            dtype=np.float32,
        )
        world = np.array([[0, 0], [WORLD_W, 0], [WORLD_W, WORLD_D], [0, WORLD_D]], dtype=np.float32)
        image_to_world = cv2.getPerspectiveTransform(img, world)

        below = [b for b in blue.values() if b.cy > z_ref.cy]
        z_base = min(below or list(blue.values()), key=lambda b: np.hypot(b.cx - z_ref.cx, b.cy - z_ref.cy))
        z_vec = np.array([z_ref.cx - z_base.cx, z_ref.cy - z_base.cy], dtype=float)
        z_pixels = float(np.linalg.norm(z_vec))
        if z_pixels < 1:
            continue

        cap.release()
        return Calibration(image_to_world, blue, z_ref, z_base, z_vec / z_pixels, z_pixels, frame_index), frame, fps, (
            frame.shape[1],
            frame.shape[0],
        )

    cap.release()
    if first_frame is None:
        raise RuntimeError("cannot read video frames")
    raise RuntimeError("could not find a frame with 4 blue cups and the red z cup")


def white_candidates(frame: np.ndarray) -> list[Blob]:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0, 0, 145)), np.array((179, 78, 255)))
    mask[: int(h * 0.18), :] = 0
    blobs = blobs_from_mask(mask, 70, 30000)
    out = []
    for b in blobs:
        ratio = b.h / max(b.w, 1)
        if not (10 <= b.w <= 280 and 10 <= b.h <= 240 and 0.10 <= ratio <= 7.0):
            continue
        if not (w * 0.08 <= b.cx <= w * 0.92 and h * 0.28 <= b.cy <= h * 0.90):
            continue
        out.append(b)
    return out


def blob_from_bbox(bbox: tuple[int, int, int, int], area: int) -> Blob:
    x, y, w, h = bbox
    return Blob(x=x, y=y, w=w, h=h, area=area, cx=x + w * 0.5, cy=y + h * 0.5)


def init_pick(candidates: list[Blob], shape: tuple[int, int, int], kind: str) -> Blob | None:
    h, w = shape[:2]
    if kind == "hand":
        choices = [b for b in candidates if w * 0.20 < b.cx < w * 0.90 and h * 0.35 < b.cy < h * 0.67]
        return max(choices, key=lambda b: b.area - 0.10 * abs(b.cy - h * 0.52), default=None)
    choices = [b for b in candidates if w * 0.20 < b.cx < w * 0.90 and h * 0.62 < b.cy < h * 0.88]
    return max(choices, key=lambda b: b.area - 0.08 * abs(b.cy - h * 0.74), default=None)


def track_pick(
    candidates: list[Blob], prev: Blob | None, shape: tuple[int, int, int], kind: str, other: Blob | None
) -> Blob | None:
    if prev is None:
        return init_pick(candidates, shape, kind)

    choices = []
    for b in candidates:
        dist = np.hypot(b.cx - prev.cx, b.cy - prev.cy)
        if dist > 260:
            continue
        if kind == "hand" and other and b.cy > other.cy - 80:
            continue
        if kind == "foot" and other and b.cy < other.cy + 80:
            continue
        size_penalty = 0.02 * abs(b.area - prev.area)
        score = -dist - size_penalty + 0.01 * b.area
        choices.append((score, b))
    return max(choices, default=(None, None))[1]


def project_to_world(cal: Calibration, point: tuple[float, float]) -> np.ndarray:
    return cv2.perspectiveTransform(np.array([[point]], dtype=np.float32), cal.image_to_world)[0, 0].astype(float)


def marker_position(cal: Calibration, marker: Blob, kind: str, foot: Blob | None) -> np.ndarray:
    center = np.array([marker.cx, marker.cy], dtype=float)
    if kind == "foot":
        ground = np.array([marker.cx, marker.maxy], dtype=float)
    elif foot:
        ground = np.array([marker.cx, foot.maxy], dtype=float)
    else:
        front_y = (cal.blue["front_left"].cy + cal.blue["front_right"].cy) * 0.5
        back_y = (cal.blue["back_left"].cy + cal.blue["back_right"].cy) * 0.5
        ground = np.array([marker.cx, (front_y + back_y) * 0.5], dtype=float)

    xy = project_to_world(cal, (float(ground[0]), float(ground[1])))
    z = max(0.0, float(np.dot(center - ground, cal.z_unit)) / cal.z_pixels * Z_REF_HEIGHT)
    return np.array([xy[0], xy[1], z])


def draw_calibration(frame: np.ndarray, cal: Calibration) -> None:
    for name, blob in cal.blue.items():
        draw_blob(frame, blob, name, (255, 80, 20), None)
    draw_blob(frame, cal.z_ref, "z_ref", (0, 0, 255), None)
    fl = cal.blue["front_left"]
    fr = cal.blue["front_right"]
    bl = cal.blue["back_left"]
    cv2.line(frame, (int(fl.cx), int(fl.cy)), (int(fr.cx), int(fr.cy)), (0, 0, 255), 5)
    cv2.line(frame, (int(fl.cx), int(fl.cy)), (int(bl.cx), int(bl.cy)), (0, 255, 0), 5)
    cv2.line(frame, (int(cal.z_base.cx), int(cal.z_base.cy)), (int(cal.z_ref.cx), int(cal.z_ref.cy)), (255, 0, 0), 5)


def resize_for_output(frame: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return frame
    return cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--out-video", type=Path, default=OUT_VIDEO)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--scale", type=float, default=OUTPUT_SCALE)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    cal, sample_frame, fps, size = calibrate(args.video)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")

    out_size = (int(size[0] * args.scale), int(size[1] * args.scale))
    writer = cv2.VideoWriter(str(args.out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    prev_hand = blob_from_bbox(INITIAL_HAND_BBOX, 500)
    prev_foot = blob_from_bbox(INITIAL_FOOT_BBOX, 3000)
    rows = []
    frame_i = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if args.max_frames and frame_i >= args.max_frames:
            break

        candidates = white_candidates(frame)
        hand = track_pick(candidates, prev_hand, frame.shape, "hand", prev_foot)
        foot = track_pick(candidates, prev_foot, frame.shape, "foot", hand or prev_hand)
        prev_hand = hand or prev_hand
        prev_foot = foot or prev_foot

        hand_pos = marker_position(cal, hand, "hand", foot or prev_foot) if hand else None
        foot_pos = marker_position(cal, foot, "foot", foot) if foot else None

        draw_calibration(frame, cal)
        draw_blob(frame, hand, "hand_band", (0, 0, 255), hand_pos)
        draw_blob(frame, foot, "foot_band", (255, 255, 255), foot_pos)
        cv2.putText(
            frame,
            f"calibration frame={cal.frame_index}",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        writer.write(resize_for_output(frame, args.scale))
        if args.display:
            cv2.imshow("wristband_3d_tracker", resize_for_output(frame, args.scale))
            if cv2.waitKey(1) == 27:
                break

        row = {"frame": frame_i, "time": frame_i / fps}
        for prefix, pos in (("hand", hand_pos), ("foot", foot_pos)):
            for axis, value in zip("xyz", pos if pos is not None else [None, None, None]):
                row[f"{prefix}_{axis}"] = "" if value is None else f"{float(value):.4f}"
        rows.append(row)

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "time", "hand_x", "hand_y", "hand_z", "foot_x", "foot_y", "foot_z"]
        out = csv.DictWriter(f, fieldnames=fieldnames)
        out.writeheader()
        out.writerows(rows)

    print(f"calibration_frame={cal.frame_index}")
    print(f"frames={len(rows)}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_video}")


if __name__ == "__main__":
    main()
