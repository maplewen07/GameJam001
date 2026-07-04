import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


sys.path.insert(0, str(Path(__file__).with_name(".deps")))

import cv2
import numpy as np


VIDEO_PATH = Path(r"C:\Users\Administrator\Downloads\VID_20260703_222816848.mp4")
OUT_VIDEO = Path("cups_3d_tracker_overlay.mp4")
OUT_CSV = Path("cups_3d_positions.csv")

# ponytail: relative units; replace with measured width/depth/height when available.
WORLD_W = 100.0
WORLD_D = 100.0
Z_REF_HEIGHT = 25.0
CUP_HEIGHT = 3.0


@dataclass
class Blob:
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float

    @property
    def minx(self) -> int:
        return self.x

    @property
    def miny(self) -> int:
        return self.y

    @property
    def maxx(self) -> int:
        return self.x + self.w

    @property
    def maxy(self) -> int:
        return self.y + self.h


@dataclass
class Calibration:
    image_to_world: np.ndarray
    blue: dict[str, Blob]
    z_ref: Blob
    z_unit: np.ndarray
    z_pixels: float


def blobs_from_mask(mask: np.ndarray, min_area: int, max_area: int) -> list[Blob]:
    n, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if min_area <= area <= max_area:
            cx, cy = [float(v) for v in centers[i]]
            blobs.append(Blob(x, y, w, h, area, cx, cy))
    return blobs


def merge_close(blobs: list[Blob], pad: int) -> list[Blob]:
    groups: list[list[Blob]] = []
    for blob in blobs:
        hits = [
            g
            for g in groups
            if any(
                blob.minx <= other.maxx + pad
                and other.minx <= blob.maxx + pad
                and blob.miny <= other.maxy + pad
                and other.miny <= blob.maxy + pad
                for other in g
            )
        ]
        if not hits:
            groups.append([blob])
            continue
        hits[0].append(blob)
        for extra in hits[1:]:
            hits[0].extend(extra)
            groups.remove(extra)

    merged = []
    for group in groups:
        area = sum(b.area for b in group)
        sx = sum(b.cx * b.area for b in group)
        sy = sum(b.cy * b.area for b in group)
        minx = min(b.minx for b in group)
        miny = min(b.miny for b in group)
        maxx = max(b.maxx for b in group)
        maxy = max(b.maxy for b in group)
        merged.append(Blob(minx, miny, maxx - minx, maxy - miny, area, sx / area, sy / area))
    return merged


def hsv_mask(frame: np.ndarray, ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]]) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def detect_blue(frame: np.ndarray) -> dict[str, Blob]:
    h, w = frame.shape[:2]
    mask = hsv_mask(frame, [((95, 40, 25), (135, 255, 230))])
    mask[: int(h * 0.52), :] = 0
    blobs = merge_close(blobs_from_mask(mask, 25, 5000), 22)
    blobs = [b for b in blobs if 5 <= b.w <= 80 and 5 <= b.h <= 80]
    blobs = sorted(blobs, key=lambda b: b.area, reverse=True)[:4]
    if len(blobs) < 4:
        raise RuntimeError(f"need 4 blue calibration points, got {len(blobs)}")

    front = sorted(sorted(blobs, key=lambda b: b.cy, reverse=True)[:2], key=lambda b: b.cx)
    back = sorted(sorted(blobs, key=lambda b: b.cy, reverse=True)[2:], key=lambda b: b.cx)
    return {"front_left": front[0], "front_right": front[1], "back_left": back[0], "back_right": back[1]}


def detect_red(frame: np.ndarray) -> list[Blob]:
    h, _ = frame.shape[:2]
    mask = hsv_mask(frame, [((0, 80, 60), (12, 255, 255)), ((170, 80, 60), (179, 255, 255))])
    mask[: int(h * 0.30), :] = 0
    blobs = merge_close(blobs_from_mask(mask, 60, 5000), 28)
    return [b for b in blobs if 8 <= b.w <= 140 and 10 <= b.h <= 140 and 0.35 <= b.h / max(b.w, 1) <= 3.5]


def detect_white(frame: np.ndarray, blue: dict[str, Blob], prev: Blob | None, hand: Blob | None) -> Blob | None:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0, 0, 80)), np.array((179, 90, 255)))
    mask[: int(h * 0.48), :] = 0
    blobs = blobs_from_mask(mask, 30, 350)
    blue_points = np.array([(b.cx, b.cy) for b in blue.values()], dtype=float)
    candidates = []
    for b in blobs:
        if not (4 <= b.w <= 60 and 8 <= b.h <= 80 and b.h / max(b.w, 1) > 0.25):
            continue
        if not (w * 0.22 <= b.cx <= w * 0.76 and h * 0.48 <= b.cy <= h * 0.76):
            continue
        if blue_points.size and np.min(np.linalg.norm(blue_points - np.array([b.cx, b.cy]), axis=1)) < 45:
            continue
        if prev:
            score = -np.hypot(b.cx - prev.cx, b.cy - prev.cy) + 0.10 * b.area
        else:
            score = b.area - 0.20 * abs(b.cx - w * 0.48) - 0.10 * abs(b.cy - h * 0.60)
        if hand:
            if abs(b.cx - hand.cx) > 260:
                continue
            score -= 0.35 * abs(b.cx - hand.cx)
        candidates.append((score, b))
    return max(candidates, default=(None, None))[1]


def choose_z_ref(red: list[Blob], frame_shape: tuple[int, int, int]) -> Blob:
    h, w = frame_shape[:2]
    choices = [b for b in red if b.cx > w * 0.75 and h * 0.52 < b.cy < h * 0.76]
    if not choices:
        raise RuntimeError("no right-side static red z-reference point detected")
    return max(choices, key=lambda b: b.area)


def choose_hand(red: list[Blob], z_ref: Blob, frame_shape: tuple[int, int, int]) -> Blob | None:
    h, w = frame_shape[:2]
    choices = []
    for b in red:
        if np.hypot(b.cx - z_ref.cx, b.cy - z_ref.cy) < 120:
            continue
        if not (w * 0.30 < b.cx < w * 0.78 and h * 0.30 < b.cy < h * 0.78):
            continue
        choices.append(b)
    return max(choices, key=lambda b: b.area, default=None)


def calibrate(frame: np.ndarray) -> Calibration:
    blue = detect_blue(frame)
    red = detect_red(frame)
    z_ref = choose_z_ref(red, frame.shape)

    img = np.array(
        [
            [blue["front_left"].cx, blue["front_left"].cy],
            [blue["front_right"].cx, blue["front_right"].cy],
            [blue["back_right"].cx, blue["back_right"].cy],
            [blue["back_left"].cx, blue["back_left"].cy],
        ],
        dtype=np.float32,
    )
    world = np.array(
        [[0, 0], [WORLD_W, 0], [WORLD_W, WORLD_D], [0, WORLD_D]],
        dtype=np.float32,
    )
    image_to_world = cv2.getPerspectiveTransform(img, world)
    z_base = np.array([blue["front_right"].cx, blue["front_right"].cy], dtype=float)
    z_tip = np.array([z_ref.cx, z_ref.cy], dtype=float)
    z_vec = z_tip - z_base
    z_pixels = float(np.linalg.norm(z_vec))
    if z_pixels < 1:
        raise RuntimeError("z reference is too close to its blue base point")
    return Calibration(image_to_world, blue, z_ref, z_vec / z_pixels, z_pixels)


def project_to_world(cal: Calibration, point: tuple[float, float]) -> np.ndarray:
    p = cv2.perspectiveTransform(np.array([[point]], dtype=np.float32), cal.image_to_world)[0, 0]
    return p.astype(float)


def y_on_line(a: Blob, b: Blob, x: float) -> float:
    if abs(b.cx - a.cx) < 1e-6:
        return (a.cy + b.cy) * 0.5
    t = (x - a.cx) / (b.cx - a.cx)
    return a.cy + t * (b.cy - a.cy)


def blob_position_3d(blob: Blob, cal: Calibration, mode: str) -> np.ndarray:
    if mode == "foot":
        ground = np.array([blob.cx, blob.maxy], dtype=float)
    else:
        front_y = y_on_line(cal.blue["front_left"], cal.blue["front_right"], blob.cx)
        back_y = y_on_line(cal.blue["back_left"], cal.blue["back_right"], blob.cx)
        ground = np.array([blob.cx, (front_y + back_y) * 0.5], dtype=float)

    p = np.array([blob.cx, blob.cy], dtype=float)
    xy = project_to_world(cal, (float(ground[0]), float(ground[1])))
    z = max(0.0, float(np.dot(p - ground, cal.z_unit)) / cal.z_pixels * Z_REF_HEIGHT)
    return np.array([xy[0], xy[1], z])


def draw_blob(frame: np.ndarray, blob: Blob | None, name: str, color: tuple[int, int, int], pos: np.ndarray | None) -> None:
    if blob is None:
        return
    cv2.rectangle(frame, (blob.minx, blob.miny), (blob.maxx, blob.maxy), color, 2)
    cv2.circle(frame, (int(blob.cx), int(blob.cy)), 4, color, -1)
    label = name if pos is None else f"{name} x={pos[0]:.1f} y={pos[1]:.1f} z={pos[2]:.1f}"
    cv2.putText(frame, label, (blob.minx, max(20, blob.miny - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_calibration(frame: np.ndarray, cal: Calibration) -> None:
    for name, blob in cal.blue.items():
        draw_blob(frame, blob, name, (255, 80, 20), None)
    draw_blob(frame, cal.z_ref, "z_ref", (0, 0, 255), None)
    fl = cal.blue["front_left"]
    fr = cal.blue["front_right"]
    bl = cal.blue["back_left"]
    cv2.line(frame, (int(fl.cx), int(fl.cy)), (int(fr.cx), int(fr.cy)), (0, 0, 255), 3)
    cv2.line(frame, (int(fl.cx), int(fl.cy)), (int(bl.cx), int(bl.cy)), (0, 255, 0), 3)
    cv2.line(frame, (int(fr.cx), int(fr.cy)), (int(cal.z_ref.cx), int(cal.z_ref.cy)), (255, 0, 0), 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--out-video", type=Path, default=OUT_VIDEO)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    ok, first = cap.read()
    if not ok:
        raise SystemExit("cannot read first frame")

    cal = calibrate(first)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (first.shape[1], first.shape[0])
    writer = cv2.VideoWriter(str(args.out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    prev_foot = None
    rows = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if args.max_frames and frame_i >= args.max_frames:
            break

        red = detect_red(frame)
        hand = choose_hand(red, cal.z_ref, frame.shape)
        foot = detect_white(frame, cal.blue, prev_foot, hand)
        prev_foot = foot or prev_foot

        hand_pos = blob_position_3d(hand, cal, "hand") if hand else None
        foot_pos = blob_position_3d(foot, cal, "foot") if foot else None

        draw_calibration(frame, cal)
        draw_blob(frame, hand, "hand_red", (0, 0, 255), hand_pos)
        draw_blob(frame, foot, "foot_white", (255, 255, 255), foot_pos)
        writer.write(frame)
        if args.display:
            cv2.imshow("cups_3d_tracker", frame)
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

    print(f"frames={len(rows)}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_video}")


if __name__ == "__main__":
    main()
