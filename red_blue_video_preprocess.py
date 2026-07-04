import argparse
import csv
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).with_name(".deps")))

import cv2
import numpy as np


VIDEO_PATH = Path(r"C:\Users\Administrator\Downloads\IMG_2485.MOV")
OUT_OVERLAY = Path("red_blue_preprocess_overlay.mp4")
OUT_MASK = Path("red_blue_preprocess_mask.mp4")
OUT_CSV = Path("red_blue_preprocess_blobs.csv")


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def hsv_for_detection(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # ponytail: cheap lighting normalization; enough for this fixed-camera indoor video.
    v = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(v)
    return cv2.merge((h, s, v))


def marker_masks(frame: np.ndarray, roi_top: float, roi_right: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    hsv = hsv_for_detection(frame)
    red = cv2.inRange(hsv, np.array((0, 85, 45)), np.array((12, 255, 255)))
    red |= cv2.inRange(hsv, np.array((168, 85, 45)), np.array((179, 255, 255)))
    blue = cv2.inRange(hsv, np.array((92, 35, 30)), np.array((136, 255, 255)))

    red[: int(h * roi_top), :] = 0
    blue[: int(h * roi_top), :] = 0
    red[:, int(w * roi_right) :] = 0
    blue[:, int(w * roi_right) :] = 0
    return clean_mask(red), clean_mask(blue)


def blobs(mask: np.ndarray, color: str, frame_i: int) -> list[dict[str, object]]:
    n, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    rows = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < 45 or w < 4 or h < 4:
            continue
        cx, cy = [float(v) for v in centers[i]]
        rows.append(
            {
                "frame": frame_i,
                "color": color,
                "cx": f"{cx:.2f}",
                "cy": f"{cy:.2f}",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
            }
        )
    return rows


def draw_blobs(frame: np.ndarray, rows: list[dict[str, object]], scale: float) -> None:
    for row in rows:
        color = (0, 0, 255) if row["color"] == "red" else (255, 80, 20)
        x = int(int(row["x"]) * scale)
        y = int(int(row["y"]) * scale)
        w = int(int(row["w"]) * scale)
        h = int(int(row["h"]) * scale)
        area = int(row["area"])
        if area < 120:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + w // 2, y + h // 2), 3, color, -1)


def resize(frame: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return frame
    return cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--out-overlay", type=Path, default=OUT_OVERLAY)
    parser.add_argument("--out-mask", type=Path, default=OUT_MASK)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--roi-top", type=float, default=0.25)
    parser.add_argument("--roi-right", type=float, default=0.88)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (int(w * args.scale), int(h * args.scale))
    overlay_writer = cv2.VideoWriter(str(args.out_overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    mask_writer = cv2.VideoWriter(str(args.out_mask), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)

    all_rows: list[dict[str, object]] = []
    frame_i = -1
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if args.max_frames and frame_i >= args.max_frames:
            break
        processed += 1

        red, blue = marker_masks(frame, args.roi_top, args.roi_right)
        rows = blobs(red, "red", frame_i) + blobs(blue, "blue", frame_i)
        all_rows.extend(rows)

        overlay = resize(frame.copy(), args.scale)
        red_small = resize(red, args.scale)
        blue_small = resize(blue, args.scale)
        overlay[red_small > 0] = (overlay[red_small > 0] * 0.35 + np.array((0, 0, 255)) * 0.65).astype(np.uint8)
        overlay[blue_small > 0] = (overlay[blue_small > 0] * 0.35 + np.array((255, 80, 20)) * 0.65).astype(np.uint8)
        draw_blobs(overlay, rows, args.scale)
        cv2.putText(overlay, f"frame={frame_i}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        overlay_writer.write(overlay)

        mask_view = np.zeros_like(overlay)
        mask_view[red_small > 0] = (0, 0, 255)
        mask_view[blue_small > 0] = (255, 80, 20)
        mask_writer.write(mask_view)

    cap.release()
    overlay_writer.release()
    mask_writer.release()

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "color", "cx", "cy", "x", "y", "w", "h", "area"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"frames={processed}")
    print(f"blobs={len(all_rows)}")
    print(f"wrote {args.out_overlay}")
    print(f"wrote {args.out_mask}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
