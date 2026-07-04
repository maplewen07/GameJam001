from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COORD_MAX = 100.0


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
