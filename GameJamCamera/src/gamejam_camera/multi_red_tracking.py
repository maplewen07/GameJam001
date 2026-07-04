from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import config as app_config
from . import live_camera as live


cv2 = live.cv2
np = live.np
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class HeadlessPreview:
    def show(self, frame: np.ndarray) -> str:
        return ""

    def close(self) -> None:
        pass


class PersonFrameStore:
    def __init__(self, history_limit: int) -> None:
        self.lock = threading.Lock()
        self.latest: dict[str, object] | None = None
        self.history: deque[dict[str, object]] = deque(maxlen=max(1, int(history_limit)))

    def update(self, payload: dict[str, object]) -> None:
        with self.lock:
            self.latest = payload
            self.history.append(payload)

    def latest_payload(self) -> dict[str, object] | None:
        with self.lock:
            return self.latest

    def history_payloads(self, limit: int | None = None) -> list[dict[str, object]]:
        with self.lock:
            rows = list(self.history)
        if limit is None:
            return rows
        return rows[-max(0, limit) :]


class PersonFrameHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: PersonFrameStore) -> None:
        super().__init__(address, PersonFrameRequestHandler)
        self.store = store


class PersonFrameRequestHandler(BaseHTTPRequestHandler):
    server: PersonFrameHttpServer

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/health"):
            latest = self.server.store.latest_payload()
            self.send_json(
                200,
                {
                    "ok": True,
                    "latest_frame": latest.get("frame") if latest else None,
                    "people": len(latest.get("people", [])) if latest else 0,
                    "endpoints": [
                        "/person-frame/latest",
                        "/person-frame/history?limit=60",
                    ],
                },
            )
            return

        if parsed.path in ("/latest", "/person-frame", "/person-frame/latest"):
            latest = self.server.store.latest_payload()
            if latest is None:
                self.send_json(404, {"ok": False, "error": "no person frame available yet"})
                return
            self.send_json(200, latest)
            return

        if parsed.path in ("/history", "/person-frame/history"):
            query = parse_qs(parsed.query)
            limit = parse_positive_int(query.get("limit", ["60"])[0], 60)
            self.send_json(200, {"frames": self.server.store.history_payloads(limit)})
            return

        self.send_json(404, {"ok": False, "error": "not found"})

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class PersonFrameSocketService:
    def __init__(self, host: str, port: int, history_limit: int) -> None:
        self.host = host
        self.port = int(port)
        self.store = PersonFrameStore(history_limit)
        self.server: PersonFrameHttpServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.server = PersonFrameHttpServer((self.host, self.port), self.store)
        self.host, self.port = self.server.server_address[:2]
        self.thread = threading.Thread(target=self.server.serve_forever, name="person-frame-socket", daemon=True)
        self.thread.start()
        print(f"person_socket=http://{self.host}:{self.port}/person-frame/latest")

    def update(self, payload: dict[str, object]) -> None:
        self.store.update(payload)

    def close(self) -> None:
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.server = None
        self.thread = None


def parse_positive_int(value: object, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def red_candidates_and_mask(
    frame: np.ndarray,
    cal: dict[str, object],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], np.ndarray]:
    red = live.red_mask(frame, cal["space"], args, args.clahe)
    rows = []
    blobs = live.stable.merge_close(live.stable.components(red, args.blob_min_area), int(args.marker_merge_px))
    for blob in blobs:
        if math.hypot(blob["cx"] - cal["z_ref"]["cx"], blob["cy"] - cal["z_ref"]["cy"]) <= args.z_exclude_px:
            continue
        space_x, space_y = live.stable.map_xy(cal["xy_h"], blob["cx"], blob["cy"])
        space_z = live.stable.map_z(cal["axis_z"], blob["cx"], blob["cy"]) / 100.0 * args.z_height
        rows.append({**blob, "space_x": space_x, "space_y": space_y, "space_z": space_z})
    return rows, red


def foot_split_y(cal: dict[str, object], args: argparse.Namespace) -> float:
    blue = cal["blue"]
    back_y = (float(blue["back_left"]["cy"]) + float(blue["back_right"]["cy"])) * 0.5
    return back_y + float(args.foot_split_offset_px)


def split_markers(markers: list[dict[str, object]], split_y: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    heads = [m for m in markers if float(m["cy"]) < split_y]
    feet = [m for m in markers if float(m["cy"]) >= split_y]
    return heads, feet


def init_person_detections(markers: list[dict[str, object]], args: argparse.Namespace, split_y: float) -> list[dict[str, object]]:
    heads, feet = split_markers(markers, split_y)
    feet = merge_init_feet(feet, args)
    detections = []
    used_heads: set[int] = set()
    for foot in sorted(feet, key=lambda m: float(m["cx"])):
        choices = []
        for i, head in enumerate(heads):
            if i in used_heads or float(head["cy"]) >= float(foot["cy"]):
                continue
            dx = abs(float(head["cx"]) - float(foot["cx"]))
            dist = math.hypot(dx, float(head["cy"]) - float(foot["cy"]))
            if dx <= args.person_init_x_margin and dist <= args.pair_max_px:
                choices.append((dx, dist, i, head))
        head = None
        if choices:
            _, _, i, head = min(choices, key=lambda item: (item[0], item[1]))
            used_heads.add(i)
        detections.append({"head": head, "foot": foot, "center": blob_center(foot)})
    return detections


def merge_init_feet(feet: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    clusters: list[list[dict[str, object]]] = []
    for foot in sorted(feet, key=lambda m: float(m["cx"])):
        if clusters and abs(float(foot["cx"]) - float(clusters[-1][-1]["cx"])) <= args.person_init_foot_merge_px:
            clusters[-1].append(foot)
        else:
            clusters.append([foot])
    return [max(cluster, key=lambda m: (float(m["cy"]), float(m["area"]))) for cluster in clusters]


def blob_center(blob: dict[str, object]) -> tuple[float, float]:
    return float(blob["cx"]), float(blob["cy"])


def detection_center(det: dict[str, object]) -> tuple[float, float]:
    return blob_center(det["foot"])


def update_init_tracks(
    tracks: dict[int, dict[str, object]],
    detections: list[dict[str, object]],
    next_id: int,
    args: argparse.Namespace,
) -> int:
    unused = set(range(len(detections)))
    for track_id, track in sorted(tracks.items()):
        choices = []
        tx, ty = track["center"]
        for i in unused:
            dx, dy = detection_center(detections[i])
            dist = math.hypot(float(dx) - float(tx), float(dy) - float(ty))
            choices.append((dist, i))
        if not choices:
            track["missing"] = int(track["missing"]) + 1
            continue
        dist, i = min(choices, key=lambda item: item[0])
        if dist > args.track_max_jump:
            track["missing"] = int(track["missing"]) + 1
            continue
        det = detections[i]
        track["center"] = detection_center(det)
        track["missing"] = 0
        track["last"] = det
        track["samples"].append(det)
        unused.remove(i)

    for i in sorted(unused):
        det = detections[i]
        tracks[next_id] = {"center": detection_center(det), "missing": 0, "last": det, "samples": [det]}
        next_id += 1

    for track_id in [tid for tid, track in tracks.items() if int(track["missing"]) > args.track_max_missing]:
        del tracks[track_id]
    return next_id


def stable_init_count(tracks: dict[int, dict[str, object]], args: argparse.Namespace) -> int:
    return sum(1 for track in tracks.values() if len(track["samples"]) >= args.min_person_init_frames)


def median_blob(samples: list[dict[str, object]], role: str) -> dict[str, object]:
    blobs = [sample[role] for sample in samples if sample.get(role) is not None]
    if not blobs:
        return {}
    keys = [key for key, value in blobs[0].items() if isinstance(value, (int, float, np.number))]
    return {key: float(np.median([float(blob[key]) for blob in blobs])) for key in keys}


def body_box(
    head_center: tuple[float, float],
    foot_center: tuple[float, float],
    shape: tuple[int, int, int],
    margin: float,
) -> tuple[float, float, float, float]:
    height, width = shape[:2]
    hx, hy = head_center
    fx, fy = foot_center
    x1 = max(0.0, min(hx, fx) - margin)
    y1 = max(0.0, min(hy, fy) - margin)
    x2 = min(float(width - 1), max(hx, fx) + margin)
    y2 = min(float(height - 1), max(hy, fy) + margin)
    return x1, y1, x2, y2


def make_person(person_id: int, track: dict[str, object], shape: tuple[int, int, int], args: argparse.Namespace) -> dict[str, object]:
    samples = track["samples"]
    head = median_blob(samples, "head")
    foot = median_blob(samples, "foot")
    foot_center = blob_center(foot)
    head_center = blob_center(head) if head else (foot_center[0], max(0.0, foot_center[1] - args.person_default_height))
    return {
        "id": person_id,
        "head": None,
        "foot": None,
        "head_center": head_center,
        "foot_center": foot_center,
        "box": body_box(head_center, foot_center, shape, args.person_box_margin),
        "split_y": (head_center[1] + foot_center[1]) * 0.5,
        "missing": 0,
        "status": "init",
    }


def lock_people(
    tracks: dict[int, dict[str, object]],
    shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    stable = [track for track in tracks.values() if len(track["samples"]) >= args.min_person_init_frames]
    if not stable:
        raise ValueError("no stable person tracks")
    stable.sort(key=lambda track: float(np.median([sample["foot"]["cx"] for sample in track["samples"]])))
    return [make_person(i + 1, track, shape, args) for i, track in enumerate(stable)]


def point_in_box(blob: dict[str, object], box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= float(blob["cx"]) <= x2 and y1 <= float(blob["cy"]) <= y2


def nearest_blob(blobs: list[dict[str, object]], target: tuple[float, float]) -> dict[str, object] | None:
    if not blobs:
        return None
    tx, ty = target
    return min(blobs, key=lambda blob: math.hypot(float(blob["cx"]) - tx, float(blob["cy"]) - ty))


def ema_point(old: tuple[float, float], new: tuple[float, float], alpha: float) -> tuple[float, float]:
    return old[0] * (1.0 - alpha) + new[0] * alpha, old[1] * (1.0 - alpha) + new[1] * alpha


def ema_box(
    old: tuple[float, float, float, float],
    new: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    return tuple(old[i] * (1.0 - alpha) + new[i] * alpha for i in range(4))


def assign_markers(markers: list[dict[str, object]], people: list[dict[str, object]], split_y: float) -> dict[int, list[dict[str, object]]]:
    assigned = {int(person["id"]): [] for person in people}
    for marker in markers:
        choices = []
        for person in people:
            box = person["box"]
            if not point_in_box(marker, box):
                continue
            target = person["head_center"] if float(marker["cy"]) < split_y else person["foot_center"]
            choices.append(
                (
                    math.hypot(float(marker["cx"]) - float(target[0]), float(marker["cy"]) - float(target[1])),
                    int(person["id"]),
                )
            )
        if choices:
            _, person_id = min(choices, key=lambda item: item[0])
            assigned[person_id].append(marker)
    return assigned


def update_person(
    person: dict[str, object],
    markers: list[dict[str, object]],
    shape: tuple[int, int, int],
    args: argparse.Namespace,
    split_y: float,
) -> None:
    head = nearest_blob([m for m in markers if float(m["cy"]) <= split_y], person["head_center"])
    foot = nearest_blob([m for m in markers if float(m["cy"]) > split_y], person["foot_center"])
    alpha = float(args.person_smooth_alpha)

    if head is not None:
        person["head_center"] = ema_point(person["head_center"], blob_center(head), alpha)
    if foot is not None:
        person["foot_center"] = ema_point(person["foot_center"], blob_center(foot), alpha)

    person["head"] = head
    person["foot"] = foot
    if head is not None and foot is not None:
        person["missing"] = 0
        person["status"] = "ok"
    elif head is not None or foot is not None:
        person["missing"] = 0
        person["status"] = "partial"
    else:
        person["missing"] = int(person["missing"]) + 1
        person["status"] = "lost" if int(person["missing"]) > args.track_max_missing else "missing"

    new_box = body_box(person["head_center"], person["foot_center"], shape, args.person_box_margin)
    person["box"] = ema_box(person["box"], new_box, alpha)
    person["split_y"] = split_y


def process_person_init_frame(
    frame: np.ndarray,
    cal: dict[str, object],
    init_tracks: dict[int, dict[str, object]],
    next_init_id: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, int, int, int]:
    markers, red = red_candidates_and_mask(frame, cal, args)
    split_y = foot_split_y(cal, args)
    detections = init_person_detections(markers, args, split_y)
    next_init_id = update_init_tracks(init_tracks, detections, next_init_id, args)
    overlay = draw_overlay(frame, red, markers, detections, cal, split_y)
    overlay = live.add_inset(overlay, red_mask_inset(red), "red mask")
    return overlay, next_init_id, len(markers), len(detections), stable_init_count(init_tracks, args)


def red_mask_inset(red: np.ndarray) -> np.ndarray:
    inset = np.zeros((red.shape[0], red.shape[1], 3), dtype=np.uint8)
    inset[red > 0] = (0, 0, 255)
    return inset


def process_multi_frame(
    frame: np.ndarray,
    cal: dict[str, object],
    people: list[dict[str, object]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    markers, red = red_candidates_and_mask(frame, cal, args)
    split_y = foot_split_y(cal, args)
    assigned = assign_markers(markers, people, split_y)
    for person in people:
        update_person(person, assigned[int(person["id"])], frame.shape, args, split_y)
    active = sum(1 for person in people if person["status"] != "lost")
    return draw_overlay(frame, red, markers, people, cal, split_y), f"people={active}/{len(people)} markers={len(markers)}"


def rounded_pair(point: tuple[float, float], digits: int = 2) -> list[float]:
    return [round(float(point[0]), digits), round(float(point[1]), digits)]


def person_center(person: dict[str, object], role: str) -> tuple[float, float] | None:
    center = person.get(f"{role}_center")
    if center is not None:
        return float(center[0]), float(center[1])
    blob = person.get(role)
    return blob_center(blob) if blob is not None else None


def reference_2d(head_center: tuple[float, float], foot_center: tuple[float, float]) -> tuple[dict[str, object], float]:
    hx, hy = head_center
    fx, fy = foot_center
    up_x = hx - fx
    up_y = hy - fy
    distance = math.hypot(up_x, up_y)
    if distance > 1e-6:
        up_x /= distance
        up_y /= distance
    else:
        up_x, up_y = 0.0, -1.0
    right_x, right_y = -up_y, up_x
    return (
        {
            "origin": "foot",
            "origin_px": rounded_pair(foot_center),
            "x_axis_px_unit": rounded_pair((right_x, right_y), 4),
            "y_axis_px_unit": rounded_pair((up_x, up_y), 4),
        },
        distance,
    )


def person_frame_payload(frame_i: int, people: list[dict[str, object]], cal: dict[str, object]) -> dict[str, object]:
    rows = []
    for index, person in enumerate(people):
        head_center = person_center(person, "head")
        foot_center = person_center(person, "foot")
        if head_center is None or foot_center is None:
            continue
        plane_x, plane_y = live.stable.map_xy(cal["xy_h"], foot_center[0], foot_center[1])
        ref, distance = reference_2d(head_center, foot_center)
        rows.append(
            {
                "id": int(person.get("id", index + 1)),
                "status": str(person.get("status", "")),
                "plane": {
                    "source": "foot",
                    "x": round(float(plane_x), 2),
                    "y": round(float(plane_y), 2),
                },
                "reference_2d": ref,
                "head_px": rounded_pair(head_center),
                "foot_px": rounded_pair(foot_center),
                "head_to_foot_vector_px": rounded_pair((foot_center[0] - head_center[0], foot_center[1] - head_center[1])),
                "head_to_foot_distance_px": round(distance, 2),
                "detected": {
                    "head": person.get("head") is not None,
                    "foot": person.get("foot") is not None,
                },
            }
        )
    return {"frame": frame_i, "people": rows}


def emit_person_frame_output(frame_i: int, payload: dict[str, object], args: argparse.Namespace) -> None:
    if not args.person_frame_output:
        return
    interval = max(1, int(args.person_frame_output_interval))
    if frame_i % interval:
        return
    print("person_frame=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def publish_person_frame(
    frame_i: int,
    people: list[dict[str, object]],
    cal: dict[str, object],
    args: argparse.Namespace,
    socket_service: PersonFrameSocketService | None,
) -> None:
    payload = person_frame_payload(frame_i, people, cal)
    if socket_service is not None:
        socket_service.update(payload)
    emit_person_frame_output(frame_i, payload, args)


def draw_overlay(
    frame: np.ndarray,
    red: np.ndarray,
    markers: list[dict[str, object]],
    people: list[dict[str, object]],
    cal: dict[str, object],
    split_y: float | None = None,
) -> np.ndarray:
    overlay = live.draw_space_overlay(frame, cal)
    overlay[red > 0] = (overlay[red > 0] * 0.35 + np.array((0, 0, 255)) * 0.65).astype(np.uint8)
    if split_y is not None:
        y = int(split_y)
        cv2.line(overlay, (0, y), (overlay.shape[1] - 1, y), (255, 255, 255), 1)
        cv2.putText(overlay, "head / foot split", (18, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for marker in markers:
        x, y, w, h = int(marker["x"]), int(marker["y"]), int(marker["w"]), int(marker["h"])
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (170, 170, 170), 1)
    for index, person in enumerate(people):
        pid = int(person.get("id", index + 1))
        color = id_color(pid)
        head = person.get("head")
        foot = person.get("foot")
        hc = blob_center(head) if head is not None else person.get("head_center")
        fc = blob_center(foot) if foot is not None else person.get("foot_center")
        if "box" in person:
            x1, y1, x2, y2 = [int(v) for v in person["box"]]
            split_y = int(person["split_y"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.line(overlay, (x1, split_y), (x2, split_y), color, 1)
            cv2.putText(
                overlay,
                f"P{pid} {person['status']}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        if hc is not None and fc is not None:
            cv2.line(overlay, (int(hc[0]), int(hc[1])), (int(fc[0]), int(fc[1])), color, 2)
            _, distance = reference_2d((float(hc[0]), float(hc[1])), (float(fc[0]), float(fc[1])))
            mid = (int((float(hc[0]) + float(fc[0])) * 0.5) + 6, int((float(hc[1]) + float(fc[1])) * 0.5))
            cv2.putText(overlay, f"{distance:.0f}px", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        for role, blob in (("head", head), ("foot", foot)):
            if blob is None:
                continue
            x, y, w, h = int(blob["x"]), int(blob["y"]), int(blob["w"]), int(blob["h"])
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            label = f"P{pid} {role} x={blob['space_x']:.1f} y={blob['space_y']:.1f} z={blob['space_z']:.2f}"
            cv2.putText(overlay, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1)
    cv2.putText(overlay, f"people={len(people)}", (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return overlay


def id_color(track_id: int) -> tuple[int, int, int]:
    colors = [(0, 255, 255), (0, 180, 255), (255, 120, 0), (255, 0, 255), (0, 255, 120), (255, 255, 0)]
    return colors[(track_id - 1) % len(colors)]


def run_camera(args: argparse.Namespace) -> None:
    cap = live.open_camera(args)
    preview = live.Preview() if args.preview else HeadlessPreview()
    print(f"process={args.process_width}x{args.process_height} clahe={args.clahe}")
    print("keys: c=restart flow h=toggle CLAHE q/esc=quit")

    state = "calibrating"
    locked_cal = None
    last_sample = None
    samples: list[dict[str, object]] = []
    calibration_start = time.perf_counter()
    person_init_start = 0.0
    person_init_valid = 0
    init_tracks: dict[int, dict[str, object]] = {}
    next_init_id = 1
    people: list[dict[str, object]] = []
    fps_smooth = 0.0
    last = time.perf_counter()
    last_status = last
    frame_i = 0
    writer = None
    person_socket = None
    if args.record_video:
        args.record_dir.mkdir(parents=True, exist_ok=True)
        video_path = args.record_dir / time.strftime("multi_live_%Y%m%d_%H%M%S.mp4")
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.fps),
            (args.process_width, args.process_height),
        )
        print(f"recording {video_path}")
    if args.person_socket_output:
        person_socket = PersonFrameSocketService(args.person_socket_host, args.person_socket_port, args.person_socket_history)
        person_socket.start()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break
        frame = live.resize_to(frame, args.process_width, args.process_height)
        now = time.perf_counter()
        dt = max(1e-6, now - last)
        last = now
        fps_smooth = (0.85 * fps_smooth + 0.15 / dt) if fps_smooth else 1.0 / dt

        if state == "calibrating":
            elapsed = now - calibration_start
            try:
                last_sample = live.calibrate(frame, args)
                samples.append(last_sample)
                if elapsed >= args.calibration_seconds and len(samples) >= args.min_calibration_frames:
                    locked_cal = live.lock_calibration(samples, frame.shape, args)
                    state = "initializing_people"
                    person_init_start = now
                    person_init_valid = 0
                    init_tracks = {}
                    next_init_id = 1
                    people = []
                    overlay = live.draw_space_overlay(frame, locked_cal)
                    status = f"calibration locked samples={len(samples)}; starting person init fps={fps_smooth:.1f}"
                    print(status)
                else:
                    overlay = live.draw_space_overlay(frame, last_sample)
                    status = f"calibrating valid={len(samples)}/required={args.min_calibration_frames} t={elapsed:.1f}/{args.calibration_seconds:.1f}s"
            except Exception as exc:
                    overlay = live.draw_space_overlay(frame, last_sample) if last_sample else frame.copy()
                    status = f"calibrating valid={len(samples)}/required={args.min_calibration_frames} failed: {exc}"
            if args.show_calibration_mask:
                overlay = live.add_inset(overlay, live.mask_view(frame, args, args.clahe), "calibration mask")
        elif state == "initializing_people":
            overlay, next_init_id, marker_count, pair_count, stable_count = process_person_init_frame(
                frame, locked_cal, init_tracks, next_init_id, args
            )
            if pair_count:
                person_init_valid += 1
            elapsed = now - person_init_start
            status = (
                f"person init valid={person_init_valid}/required={args.min_person_init_frames} "
                f"stable={stable_count} pairs={pair_count} markers={marker_count} "
                f"t={elapsed:.1f}/{args.person_init_seconds:.1f}s fps={fps_smooth:.1f}"
            )
            if elapsed >= args.person_init_seconds and person_init_valid >= args.min_person_init_frames:
                try:
                    people = lock_people(init_tracks, frame.shape, args)
                    state = "tracking"
                    status = f"person init locked people={len(people)} fps={fps_smooth:.1f}"
                    print(status)
                except Exception as exc:
                    status = f"person init failed stable={stable_count}: {exc}"
        else:
            overlay, tracking = process_multi_frame(frame, locked_cal, people, args)
            status = f"tracking {tracking} fps={fps_smooth:.1f}"
            publish_person_frame(frame_i, people, locked_cal, args, person_socket)

        cv2.putText(overlay, status, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if writer is not None:
            writer.write(overlay)
        key = preview.show(overlay)
        if now - last_status >= 1.0:
            print(f"frame={frame_i} {status}")
            last_status = now
        frame_i += 1

        if args.max_frames and frame_i >= args.max_frames:
            break
        if key in ("Escape", "q"):
            break
        if key == "c":
            state = "calibrating"
            locked_cal = None
            last_sample = None
            samples = []
            init_tracks = {}
            next_init_id = 1
            person_init_valid = 0
            people = []
            calibration_start = time.perf_counter()
            print("restarting calibration and person init")
        if key == "h":
            args.clahe = not args.clahe
            print(f"clahe={'on' if args.clahe else 'off'}")

    cap.release()
    if writer is not None:
        writer.release()
    if person_socket is not None:
        person_socket.close()
    preview.close()


def default_args() -> argparse.Namespace:
    return argparse.Namespace(
        camera=0,
        camera_device="",
        width=1920,
        height=1080,
        fps=30,
        process_width=1280,
        process_height=720,
        backend="dshow",
        fourcc="MJPG",
        camera_warmup_reads=40,
        clahe=True,
        red_h1_max=12,
        red_h2_min=168,
        red_s_min=160,
        red_v_min=100,
        red_min_area=500,
        red_cup_min_w=20,
        red_cup_max_w=260,
        red_cup_min_h=40,
        red_cup_max_h=360,
        red_cup_min_aspect=0.75,
        blue_h_min=108,
        blue_h_max=129,
        blue_s_min=145,
        blue_v_min=41,
        blue_min_area=150,
        blue_cup_min_w=12,
        blue_cup_max_w=320,
        blue_cup_min_h=12,
        blue_cup_max_h=300,
        blue_roi_top=0.25,
        roi_enabled=False,
        roi_x_min=0.0,
        roi_x_max=1.0,
        roi_y_min=0.0,
        roi_y_max=1.0,
        front_blue_min_area=200,
        back_blue_min_area=50,
        front_min_dx=80.0,
        back_x_margin=20.0,
        back_y_gap=15.0,
        blob_min_area=60,
        calibration_seconds=10.0,
        min_calibration_frames=30,
        z_exclude_px=80.0,
        z_height=1.0,
        show_calibration_mask=True,
        marker_merge_px=35,
        foot_split_offset_px=100.0,
        pair_max_px=450.0,
        track_max_jump=35.0,
        track_max_missing=10,
        person_init_seconds=20.0,
        min_person_init_frames=30,
        person_box_margin=100.0,
        person_smooth_alpha=0.5,
        person_init_x_margin=180.0,
        person_init_foot_merge_px=90.0,
        person_default_height=360.0,
        preview=True,
        max_frames=0,
        person_frame_output=False,
        person_frame_output_interval=1,
        person_socket_output=True,
        person_socket_host="127.0.0.1",
        person_socket_port=8765,
        person_socket_history=120,
        record_video=True,
        record_dir=PROJECT_ROOT / "recordings",
    )


def resolve_config_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else app_config.PROJECT_ROOT / path


def parser_with_defaults(defaults: argparse.Namespace | None = None, suppress_defaults: bool = False) -> argparse.ArgumentParser:
    defaults = defaults or default_args()
    default_value = argparse.SUPPRESS if suppress_defaults else app_config.DEFAULT_CONFIG_PATH
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS if suppress_defaults else None)
    parser.add_argument("--config", type=Path, default=default_value)
    for name, value in vars(defaults).items():
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(option, dest=name, action="store_true", default=argparse.SUPPRESS if suppress_defaults else value)
            parser.add_argument(
                "--no-" + name.replace("_", "-"),
                dest=name,
                action="store_false",
                default=argparse.SUPPRESS if suppress_defaults else value,
            )
        elif isinstance(value, int):
            parser.add_argument(option, type=int, default=argparse.SUPPRESS if suppress_defaults else value)
        elif isinstance(value, float):
            parser.add_argument(option, type=float, default=argparse.SUPPRESS if suppress_defaults else value)
        elif isinstance(value, Path):
            parser.add_argument(option, type=Path, default=argparse.SUPPRESS if suppress_defaults else value)
        else:
            choices = ["any", "dshow", "msmf"] if name == "backend" else None
            parser.add_argument(option, default=argparse.SUPPRESS if suppress_defaults else value, choices=choices)
    return parser


def load_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    fallback = default_args()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=app_config.DEFAULT_CONFIG_PATH)
    config_probe, _ = config_parser.parse_known_args(argv)
    config_path = resolve_config_path(config_probe.config)

    configured, messages = app_config.merge_config(fallback, config_path)
    parser = parser_with_defaults(configured, suppress_defaults=True)
    parsed = parser.parse_args(argv)

    merged = vars(configured).copy()
    cli_values = vars(parsed)
    if "config" in cli_values:
        config_path = resolve_config_path(cli_values.pop("config"))
    merged.update(cli_values)
    merged["config"] = config_path
    return argparse.Namespace(**merged), messages


def main() -> None:
    args, messages = load_args()
    for message in messages:
        print(message)
    run_camera(args)


if __name__ == "__main__":
    main()
