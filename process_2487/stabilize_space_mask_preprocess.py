import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))

import cv2
import numpy as np


VIDEO_PATH = Path(r"F:\Github\GameJam\IMG_2487.MOV")
OUT_DIR = Path(__file__).resolve().parent
COORD_MAX = 100.0


def hsv(frame: np.ndarray) -> np.ndarray:
    out = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(out)
    v = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(v)
    return cv2.merge((h, s, v))


def clean(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def components(mask: np.ndarray, min_area: int = 30) -> list[dict[str, float]]:
    n, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    rows = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < min_area or w < 3 or h < 3:
            continue
        cx, cy = [float(v) for v in centers[i]]
        rows.append({"x": x, "y": y, "w": w, "h": h, "area": area, "cx": cx, "cy": cy})
    return rows


def merge_close(blobs: list[dict[str, float]], pad: int = 16) -> list[dict[str, float]]:
    groups: list[list[dict[str, float]]] = []
    for blob in blobs:
        bx1, by1 = blob["x"], blob["y"]
        bx2, by2 = bx1 + blob["w"], by1 + blob["h"]
        hits = []
        for group in groups:
            for other in group:
                ox1, oy1 = other["x"], other["y"]
                ox2, oy2 = ox1 + other["w"], oy1 + other["h"]
                if bx1 <= ox2 + pad and ox1 <= bx2 + pad and by1 <= oy2 + pad and oy1 <= by2 + pad:
                    hits.append(group)
                    break
        if not hits:
            groups.append([blob])
            continue
        hits[0].append(blob)
        for extra in hits[1:]:
            hits[0].extend(extra)
            groups.remove(extra)

    merged = []
    for group in groups:
        area = sum(b["area"] for b in group)
        x1 = min(b["x"] for b in group)
        y1 = min(b["y"] for b in group)
        x2 = max(b["x"] + b["w"] for b in group)
        y2 = max(b["y"] + b["h"] for b in group)
        cx = sum(b["cx"] * b["area"] for b in group) / area
        cy = sum(b["cy"] * b["area"] for b in group) / area
        merged.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "area": area, "cx": cx, "cy": cy})
    return merged


def detect_blue_cups(frame: np.ndarray) -> dict[str, dict[str, float]]:
    h = frame.shape[0]
    mask = cv2.inRange(hsv(frame), np.array((92, 35, 25)), np.array((136, 255, 255)))
    mask[: int(h * 0.50), :] = 0
    blobs = merge_close(components(clean(mask), 60), 18)
    blobs = [b for b in blobs if 8 <= b["w"] <= 120 and 8 <= b["h"] <= 120]
    blobs = sorted(blobs, key=lambda b: b["area"], reverse=True)[:4]
    if len(blobs) < 4:
        raise RuntimeError(f"need 4 blue cups, got {len(blobs)}")
    front = sorted(sorted(blobs, key=lambda b: b["cy"], reverse=True)[:2], key=lambda b: b["cx"])
    back = sorted(sorted(blobs, key=lambda b: b["cy"], reverse=True)[2:], key=lambda b: b["cx"])
    return {"front_left": front[0], "front_right": front[1], "back_left": back[0], "back_right": back[1]}


def red_mask_from_hsv(hh: np.ndarray, red_s_min: int, red_v_min: int) -> np.ndarray:
    red = cv2.inRange(hh, np.array((0, red_s_min, red_v_min)), np.array((12, 255, 255)))
    red |= cv2.inRange(hh, np.array((168, red_s_min, red_v_min)), np.array((179, 255, 255)))
    return clean(red)


def red_mask(frame: np.ndarray, red_s_min: int, red_v_min: int) -> np.ndarray:
    return red_mask_from_hsv(hsv(frame), red_s_min, red_v_min)


def detect_z_cup(frame: np.ndarray, red_s_min: int, red_v_min: int, blue: dict[str, dict[str, float]]) -> dict[str, float]:
    blobs = merge_close(components(red_mask(frame, red_s_min, red_v_min), 60), 18)
    blobs = [b for b in blobs if 8 <= b["w"] <= 120 and 8 <= b["h"] <= 120]
    fl = blue["front_left"]
    candidates = [b for b in blobs if b["cx"] <= fl["cx"] + 120 and b["cy"] < fl["cy"]]
    candidates = candidates or blobs
    if not candidates:
        raise RuntimeError("need fixed red z cup, got 0")
    return max(candidates, key=lambda b: b["area"])


def blue_points(blue: dict[str, dict[str, float]]) -> np.ndarray:
    return np.float32(
        [
            [blue["front_left"]["cx"], blue["front_left"]["cy"]],
            [blue["front_right"]["cx"], blue["front_right"]["cy"]],
            [blue["back_right"]["cx"], blue["back_right"]["cy"]],
            [blue["back_left"]["cx"], blue["back_left"]["cy"]],
        ]
    )


def xy_homography(blue: dict[str, dict[str, float]]) -> np.ndarray:
    dst = np.float32([[0, 0], [COORD_MAX, 0], [COORD_MAX, COORD_MAX], [0, COORD_MAX]])
    return cv2.getPerspectiveTransform(blue_points(blue), dst)


def topdown_homography(blue: dict[str, dict[str, float]], size: int) -> np.ndarray:
    dst = np.float32([[0, size - 1], [size - 1, size - 1], [size - 1, 0], [0, 0]])
    return cv2.getPerspectiveTransform(blue_points(blue), dst)


def map_xy(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = cv2.perspectiveTransform(np.float32([[[x, y]]]), h)[0, 0]
    return float(p[0]), float(p[1])


def z_axis(blue: dict[str, dict[str, float]], z_cup: dict[str, float]) -> dict[str, list[float] | float]:
    origin = np.float32([blue["front_left"]["cx"], blue["front_left"]["cy"]])
    ref = np.float32([z_cup["cx"], z_cup["cy"]])
    vec = ref - origin
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        raise RuntimeError("z cup overlaps xy origin")
    unit = vec / length
    return {"origin": origin.tolist(), "ref": ref.tolist(), "unit": unit.tolist(), "length": length}


def map_z(axis: dict[str, list[float] | float], x: float, y: float) -> float:
    origin = np.float32(axis["origin"])
    unit = np.float32(axis["unit"])
    length = float(axis["length"])
    p = np.float32([x, y])
    return float(np.dot(p - origin, unit) / length * COORD_MAX)


def x_on_axis_at_y(origin: dict[str, float], dx: float, dy: float, y: float) -> float:
    if abs(dy) < 1e-6:
        return origin["cx"]
    t = (y - origin["cy"]) / dy
    return origin["cx"] + dx * t


def space_polygon(blue: dict[str, dict[str, float]], z_cup: dict[str, float], width: int, height: int) -> np.ndarray:
    fl, fr = blue["front_left"], blue["front_right"]
    top_y = 0.0
    dx = z_cup["cx"] - fl["cx"]
    dy = z_cup["cy"] - fl["cy"]
    top_left = (max(0.0, min(float(width - 1), x_on_axis_at_y(fl, dx, dy, top_y))), top_y)
    top_right = (max(0.0, min(float(width - 1), x_on_axis_at_y(fr, dx, dy, top_y))), top_y)
    bottom_left = (fl["cx"], fl["cy"])
    bottom_right = (fr["cx"], fr["cy"])
    return np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.int32)


def make_space_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def red_blue_masks(
    frame: np.ndarray,
    space: np.ndarray,
    red_s_min: int,
    red_v_min: int,
    blue_s_min: int,
    blue_v_min: int,
) -> tuple[np.ndarray, np.ndarray]:
    hh = hsv(frame)
    red = red_mask_from_hsv(hh, red_s_min, red_v_min)
    blue = cv2.inRange(hh, np.array((92, blue_s_min, blue_v_min)), np.array((136, 255, 255)))
    red = clean(cv2.bitwise_and(red, space))
    blue = clean(cv2.bitwise_and(blue, space))
    return red, blue


def resize(frame: np.ndarray, scale: float) -> np.ndarray:
    return cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def stabilization_reference(frame: np.ndarray, scale: float) -> tuple[np.ndarray, list[cv2.KeyPoint], np.ndarray | None]:
    small = resize(frame, scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1600)
    kp, des = orb.detectAndCompute(gray, None)
    return gray, kp, des


def affine_to_reference(frame: np.ndarray, ref_kp: list[cv2.KeyPoint], ref_des: np.ndarray | None, scale: float) -> np.ndarray:
    if ref_des is None:
        return np.eye(2, 3, dtype=np.float32)
    small = resize(frame, scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1600)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < 12:
        return np.eye(2, 3, dtype=np.float32)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des, ref_des)
    matches = sorted(matches, key=lambda m: m.distance)[:220]
    if len(matches) < 12:
        return np.eye(2, 3, dtype=np.float32)
    src = np.float32([kp[m.queryIdx].pt for m in matches])
    dst = np.float32([ref_kp[m.trainIdx].pt for m in matches])
    m, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if m is None or inliers is None or int(inliers.sum()) < 10:
        return np.eye(2, 3, dtype=np.float32)
    return m.astype(np.float32)


def scale_affine(m_small: np.ndarray, output_scale: float, stabilize_scale: float) -> np.ndarray:
    full = m_small.copy()
    full[:, 2] /= stabilize_scale
    out = full.copy()
    out[:, 2] *= output_scale
    return out


def draw_overlay(
    frame: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    rows: list[dict[str, object]],
    polygon: np.ndarray,
    blue_cups: dict[str, dict[str, float]],
    z_cup: dict[str, float],
    frame_i: int,
) -> np.ndarray:
    overlay = frame.copy()
    overlay[red > 0] = (overlay[red > 0] * 0.35 + np.array((0, 0, 255)) * 0.65).astype(np.uint8)
    overlay[blue > 0] = (overlay[blue > 0] * 0.35 + np.array((255, 80, 20)) * 0.65).astype(np.uint8)
    cv2.polylines(overlay, [polygon], True, (0, 255, 255), 3)
    origin = (int(blue_cups["front_left"]["cx"]), int(blue_cups["front_left"]["cy"]))
    x_ref = (int(blue_cups["front_right"]["cx"]), int(blue_cups["front_right"]["cy"]))
    y_ref = (int(blue_cups["back_left"]["cx"]), int(blue_cups["back_left"]["cy"]))
    z_ref = (int(z_cup["cx"]), int(z_cup["cy"]))
    cv2.line(overlay, origin, x_ref, (0, 255, 0), 3)
    cv2.line(overlay, origin, y_ref, (255, 255, 0), 3)
    cv2.line(overlay, origin, z_ref, (255, 0, 255), 3)
    cv2.circle(overlay, z_ref, 8, (255, 0, 255), -1)
    cv2.putText(overlay, "X", x_ref, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(overlay, "Y", y_ref, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(overlay, "Z", z_ref, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    for row in rows:
        if int(row["area"]) < 120:
            continue
        color = (0, 0, 255) if row["color"] == "red" else (255, 80, 20)
        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
    cv2.putText(overlay, f"stable cropped mask frame={frame_i}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--stabilize-scale", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--red-s-min", type=int, default=85)
    parser.add_argument("--red-v-min", type=int, default=45)
    parser.add_argument("--blue-s-min", type=int, default=35)
    parser.add_argument("--blue-v-min", type=int, default=25)
    parser.add_argument("--blob-min-area", type=int, default=40)
    parser.add_argument("--tag", default="")
    parser.add_argument("--only-frame", type=int, default=-1)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--topdown-size", type=int, default=600)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")

    ok, first = cap.read()
    if not ok:
        raise SystemExit("cannot read first frame")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_size = (int(first.shape[1] * args.scale), int(first.shape[0] * args.scale))
    first_out = resize(first, args.scale)
    _, ref_kp, ref_des = stabilization_reference(first, args.stabilize_scale)
    blue = detect_blue_cups(first_out)
    z_cup = detect_z_cup(first_out, args.red_s_min, args.red_v_min, blue)
    polygon = space_polygon(blue, z_cup, *out_size)
    space = make_space_mask((out_size[1], out_size[0]), polygon)
    xy_h = xy_homography(blue)
    topdown_h = topdown_homography(blue, args.topdown_size)
    axis_z = z_axis(blue, z_cup)

    prefix = f"{Path(args.tag).name}_" if args.tag else ""
    stable_path = OUT_DIR / f"{prefix}stabilized_preview.mp4"
    overlay_path = OUT_DIR / f"{prefix}space_mask_overlay.mp4"
    mask_path = OUT_DIR / f"{prefix}space_red_blue_mask.mp4"
    topdown_path = OUT_DIR / f"{prefix}topdown_red_blue_mask.mp4"
    csv_path = OUT_DIR / f"{prefix}space_red_blue_blobs.csv"
    calibration_path = OUT_DIR / f"{prefix}space_calibration.json"

    stable_writer = cv2.VideoWriter(str(stable_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    overlay_writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    mask_writer = cv2.VideoWriter(str(mask_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    topdown_writer = cv2.VideoWriter(
        str(topdown_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (args.topdown_size, args.topdown_size),
    )

    rows: list[dict[str, object]] = []
    start_frame = args.only_frame if args.only_frame >= 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    last_m = np.eye(2, 3, dtype=np.float32)
    frame_i = start_frame - 1
    processed = 0
    while True:
        if args.only_frame >= 0 and processed:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if args.max_frames and processed >= args.max_frames:
            break

        m_small = affine_to_reference(frame, ref_kp, ref_des, args.stabilize_scale)
        if np.allclose(m_small, np.eye(2, 3), atol=1e-6) and frame_i:
            m_small = last_m
        last_m = m_small

        frame_out = resize(frame, args.scale)
        m_out = scale_affine(m_small, args.scale, args.stabilize_scale)
        stable = cv2.warpAffine(frame_out, m_out, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        red, blue_mask = red_blue_masks(stable, space, args.red_s_min, args.red_v_min, args.blue_s_min, args.blue_v_min)

        frame_rows = []
        for color, mask in (("red", red), ("blue", blue_mask)):
            for b in components(mask, args.blob_min_area):
                space_x, space_y = map_xy(xy_h, b["cx"], b["cy"])
                space_z = map_z(axis_z, b["cx"], b["cy"])
                row = {
                    "frame": frame_i,
                    "color": color,
                    "cx": f"{b['cx']:.2f}",
                    "cy": f"{b['cy']:.2f}",
                    "space_x": f"{space_x:.2f}",
                    "space_y": f"{space_y:.2f}",
                    "space_z": f"{space_z:.2f}",
                    "x": int(b["x"]),
                    "y": int(b["y"]),
                    "w": int(b["w"]),
                    "h": int(b["h"]),
                    "area": int(b["area"]),
                }
                frame_rows.append(row)
                rows.append(row)

        mask_view = np.zeros_like(stable)
        mask_view[red > 0] = (0, 0, 255)
        mask_view[blue_mask > 0] = (255, 80, 20)
        stable_writer.write(stable)
        mask_writer.write(mask_view)
        topdown_writer.write(cv2.warpPerspective(mask_view, topdown_h, (args.topdown_size, args.topdown_size)))
        overlay = draw_overlay(stable, red, blue_mask, frame_rows, polygon, blue, z_cup, frame_i)
        overlay_writer.write(overlay)
        if args.save_frames:
            cv2.imwrite(str(OUT_DIR / f"{prefix}stable_{frame_i:03d}.jpg"), stable)
            cv2.imwrite(str(OUT_DIR / f"{prefix}mask_{frame_i:03d}.jpg"), mask_view)
            cv2.imwrite(str(OUT_DIR / f"{prefix}topdown_{frame_i:03d}.jpg"), cv2.warpPerspective(mask_view, topdown_h, (args.topdown_size, args.topdown_size)))
            cv2.imwrite(str(OUT_DIR / f"{prefix}overlay_{frame_i:03d}.jpg"), overlay)
        processed += 1

    cap.release()
    stable_writer.release()
    overlay_writer.release()
    mask_writer.release()
    topdown_writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "color", "cx", "cy", "space_x", "space_y", "space_z", "x", "y", "w", "h", "area"])
        writer.writeheader()
        writer.writerows(rows)

    calibration = {
        "source_video": str(args.video),
        "scale": args.scale,
        "stabilize_scale": args.stabilize_scale,
        "output_size": list(out_size),
        "blue_cups": blue,
        "z_cup": z_cup,
        "xy_coord_max": COORD_MAX,
        "xy_homography": xy_h.tolist(),
        "z_axis": axis_z,
        "topdown_size": args.topdown_size,
        "space_polygon": polygon.astype(float).tolist(),
        "space_z_note": "relative image-axis coordinate; z_cup is 100, not metric height",
        "frames": processed,
    }
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(f"frames={processed}")
    print(f"blobs={len(rows)}")
    print(f"wrote {stable_path}")
    print(f"wrote {overlay_path}")
    print(f"wrote {mask_path}")
    print(f"wrote {topdown_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {calibration_path}")


if __name__ == "__main__":
    main()
