import argparse
import csv
import json
import math
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
SPACE_FIELDS = ["foot_space_x", "foot_space_y", "hand_space_x", "hand_space_y", "hand_space_z"]


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def dist(a: dict[str, str], b: dict[str, float]) -> float:
    return math.hypot(f(a, "cx") - float(b["cx"]), f(a, "cy") - float(b["cy"]))


def choose_foot(candidates: list[dict[str, str]], y_band: float) -> dict[str, str]:
    # ponytail: y/area heuristic; replace with frame-to-frame tracking if IDs drift.
    max_y = max(f(r, "cy") for r in candidates)
    lower = [r for r in candidates if f(r, "cy") >= max_y - y_band]
    return max(lower, key=lambda r: int(r["area"]))


def fill_empty_from_previous(rows: list[dict[str, str | int]], fields: list[str]) -> int:
    last: dict[str, str | int] = {}
    filled = 0
    for row in rows:
        for key in fields:
            if row.get(key) == "" and key in last:
                row[key] = last[key]
                filled += 1
            elif row.get(key) != "":
                last[key] = row[key]
    return filled


def smooth_space(rows: list[dict[str, str | int]], radius: int) -> None:
    if radius <= 0:
        return
    for key in SPACE_FIELDS:
        values = [float(row[key]) if row.get(key) != "" else None for row in rows]
        smoothed: list[float | None] = []
        for i in range(len(values)):
            total = 0.0
            weight_sum = 0.0
            for j in range(max(0, i - radius), min(len(values), i + radius + 1)):
                value = values[j]
                if value is None:
                    continue
                weight = radius + 1 - abs(i - j)
                total += value * weight
                weight_sum += weight
            smoothed.append(total / weight_sum if weight_sum else None)
        for row, value in zip(rows, smoothed):
            if value is None:
                continue
            row[key] = f"{value:.4f}" if key == "hand_space_z" else f"{value:.2f}"


def recompute_distances(rows: list[dict[str, str | int]]) -> None:
    for row in rows:
        if any(row.get(key) == "" for key in SPACE_FIELDS):
            continue
        dx = float(row["hand_space_x"]) - float(row["foot_space_x"])
        dy = float(row["hand_space_y"]) - float(row["foot_space_y"])
        dz = float(row["hand_space_z"])
        row["hand_to_foot_dx"] = f"{dx:.2f}"
        row["hand_to_foot_dy"] = f"{dy:.2f}"
        row["hand_to_foot_dz"] = f"{dz:.4f}"
        row["hand_to_foot_distance_2d"] = f"{math.hypot(dx, dy):.2f}"
        row["hand_to_foot_distance_3d"] = f"{math.sqrt(dx * dx + dy * dy + dz * dz):.2f}"


def write_positions(
    blob_csv: Path,
    calibration_json: Path,
    out_csv: Path,
    z_height: float,
    z_exclude_px: float,
    foot_y_band: float,
    smooth_radius: int,
) -> tuple[int, int, int, int]:
    calibration = json.loads(calibration_json.read_text(encoding="utf-8"))
    z_cup = calibration["z_cup"]

    by_frame: dict[int, list[dict[str, str]]] = {}
    with blob_csv.open(newline="", encoding="utf-8") as inp:
        for row in csv.DictReader(inp):
            if row["color"] != "red":
                continue
            if dist(row, z_cup) <= z_exclude_px:
                continue
            by_frame.setdefault(int(row["frame"]), []).append(row)

    fields = [
        "frame",
        "status",
        "foot_space_x",
        "foot_space_y",
        "foot_cx",
        "foot_cy",
        "foot_area",
        "hand_space_x",
        "hand_space_y",
        "hand_space_z",
        "hand_cx",
        "hand_cy",
        "hand_area",
        "hand_to_foot_dx",
        "hand_to_foot_dy",
        "hand_to_foot_dz",
        "hand_to_foot_distance_2d",
        "hand_to_foot_distance_3d",
        "red_candidates",
    ]

    rows = []
    complete = 0
    filled = 0
    last_usable: dict[str, str | int] | None = None
    max_frame = int(calibration.get("frames", max(by_frame) + 1 if by_frame else 0)) - 1
    for frame in range(max_frame + 1):
        candidates = by_frame.get(frame, [])
        out = {k: "" for k in fields}
        out["frame"] = frame
        out["red_candidates"] = len(candidates)
        if not candidates:
            if last_usable:
                out = dict(last_usable)
                out["frame"] = frame
                out["status"] = "filled_from_previous"
                out["red_candidates"] = 0
                filled += 1
            else:
                out["status"] = "missing_red"
            rows.append(out)
            if out.get("foot_space_x"):
                last_usable = out
            continue

        foot = choose_foot(candidates, foot_y_band)
        out.update(
            {
                "status": "foot_only",
                "foot_space_x": f"{f(foot, 'space_x'):.2f}",
                "foot_space_y": f"{f(foot, 'space_y'):.2f}",
                "foot_cx": f"{f(foot, 'cx'):.2f}",
                "foot_cy": f"{f(foot, 'cy'):.2f}",
                "foot_area": foot["area"],
            }
        )

        hand_candidates = [r for r in candidates if r is not foot]
        if hand_candidates:
            hand = min(hand_candidates, key=lambda r: f(r, "cy"))
            dx = f(hand, "space_x") - f(foot, "space_x")
            dy = f(hand, "space_y") - f(foot, "space_y")
            dz = f(hand, "space_z") / 100.0 * z_height
            out.update(
                {
                    "status": "ok",
                    "hand_space_x": f"{f(hand, 'space_x'):.2f}",
                    "hand_space_y": f"{f(hand, 'space_y'):.2f}",
                    "hand_space_z": f"{dz:.4f}",
                    "hand_cx": f"{f(hand, 'cx'):.2f}",
                    "hand_cy": f"{f(hand, 'cy'):.2f}",
                    "hand_area": hand["area"],
                    "hand_to_foot_dx": f"{dx:.2f}",
                    "hand_to_foot_dy": f"{dy:.2f}",
                    "hand_to_foot_dz": f"{dz:.4f}",
                    "hand_to_foot_distance_2d": f"{math.hypot(dx, dy):.2f}",
                    "hand_to_foot_distance_3d": f"{math.sqrt(dx * dx + dy * dy + dz * dz):.2f}",
                }
            )
            complete += 1
        rows.append(out)
        last_usable = out

    filled_fields = fill_empty_from_previous(rows, SPACE_FIELDS)
    smooth_space(rows, smooth_radius)
    recompute_distances(rows)

    with out_csv.open("w", newline="", encoding="utf-8") as outp:
        writer = csv.DictWriter(outp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), complete, filled, filled_fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blobs", type=Path, default=OUT_DIR / "tune1_3d_space_red_blue_blobs.csv")
    parser.add_argument("--calibration", type=Path, default=OUT_DIR / "tune1_3d_space_calibration.json")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "tune1_3d_hand_foot_positions.csv")
    parser.add_argument("--z-height", type=float, default=1.0)
    parser.add_argument("--z-exclude-px", type=float, default=80.0)
    parser.add_argument("--foot-y-band", type=float, default=80.0)
    parser.add_argument("--smooth-radius", type=int, default=30)
    args = parser.parse_args()

    frames, complete, filled, filled_fields = write_positions(
        args.blobs,
        args.calibration,
        args.out,
        args.z_height,
        args.z_exclude_px,
        args.foot_y_band,
        args.smooth_radius,
    )
    print(f"frames={frames}")
    print(f"complete_hand_foot={complete}")
    print(f"filled_from_previous={filled}")
    print(f"filled_empty_space_fields={filled_fields}")
    print(f"smooth_radius={args.smooth_radius}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
