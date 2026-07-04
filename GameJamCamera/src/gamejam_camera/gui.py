from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

from . import camera_devices
from . import config as app_config
from . import live_camera as live
from . import multi_red_tracking


MASK_PARAMS = (
    "clahe",
    "red_h1_max",
    "red_h2_min",
    "red_s_min",
    "red_v_min",
    "red_min_area",
    "blue_h_min",
    "blue_h_max",
    "blue_s_min",
    "blue_v_min",
    "blue_min_area",
    "blob_min_area",
    "marker_merge_px",
)

SECTIONS = {
    "Mask": MASK_PARAMS,
    "Space": (
        "roi_enabled",
        "roi_x_min",
        "roi_x_max",
        "roi_y_min",
        "roi_y_max",
    ),
    "Camera": (
        "camera_device",
        "camera",
        "width",
        "height",
        "fps",
        "process_width",
        "process_height",
        "backend",
        "fourcc",
        "camera_warmup_reads",
    ),
    "Calibration": (
        "calibration_seconds",
        "min_calibration_frames",
        "z_exclude_px",
        "z_height",
        "show_calibration_mask",
        "red_cup_min_w",
        "red_cup_max_w",
        "red_cup_min_h",
        "red_cup_max_h",
        "red_cup_min_aspect",
        "blue_cup_min_w",
        "blue_cup_max_w",
        "blue_cup_min_h",
        "blue_cup_max_h",
        "blue_roi_top",
        "front_blue_min_area",
        "back_blue_min_area",
        "front_min_dx",
        "back_x_margin",
        "back_y_gap",
    ),
    "Person Init": (
        "foot_split_offset_px",
        "pair_max_px",
        "person_init_seconds",
        "min_person_init_frames",
        "person_init_x_margin",
        "person_init_foot_merge_px",
        "person_default_height",
    ),
    "Tracking": (
        "track_max_jump",
        "track_max_missing",
        "person_box_margin",
        "person_smooth_alpha",
    ),
    "Output": (
        "preview",
        "max_frames",
        "person_frame_output",
        "person_frame_output_interval",
        "person_socket_output",
        "person_socket_host",
        "person_socket_port",
        "person_socket_history",
        "record_video",
        "record_dir",
    ),
}

SLIDERS = {
    "red_h1_max": (0, 30, 1),
    "red_h2_min": (140, 179, 1),
    "red_s_min": (0, 255, 1),
    "red_v_min": (0, 255, 1),
    "red_min_area": (0, 3000, 10),
    "blue_h_min": (0, 179, 1),
    "blue_h_max": (0, 179, 1),
    "blue_s_min": (0, 255, 1),
    "blue_v_min": (0, 255, 1),
    "blue_min_area": (0, 3000, 10),
    "blob_min_area": (0, 1000, 5),
    "marker_merge_px": (0, 150, 1),
    "red_cup_min_w": (0, 200, 1),
    "red_cup_max_w": (0, 500, 1),
    "red_cup_min_h": (0, 200, 1),
    "red_cup_max_h": (0, 600, 1),
    "red_cup_min_aspect": (0.0, 3.0, 0.05),
    "blue_cup_min_w": (0, 200, 1),
    "blue_cup_max_w": (0, 600, 1),
    "blue_cup_min_h": (0, 200, 1),
    "blue_cup_max_h": (0, 600, 1),
    "roi_x_min": (0.0, 1.0, 0.01),
    "roi_x_max": (0.0, 1.0, 0.01),
    "roi_y_min": (0.0, 1.0, 0.01),
    "roi_y_max": (0.0, 1.0, 0.01),
}

CHOICES = {
    "backend": ("any", "dshow", "msmf"),
}


class ConfigGui:
    def __init__(self, config_path: Path) -> None:
        self.config_path = multi_red_tracking.resolve_config_path(config_path)
        self.fallback = multi_red_tracking.default_args()
        self.defaults = vars(self.fallback)
        self.values: dict[str, Any] = {}
        self.vars: dict[str, tk.Variable] = {}
        self.camera_combo: ttk.Combobox | None = None
        self.camera_devices = camera_devices.discover_camera_devices()
        self.cap = None
        self.after_id: str | None = None
        self.last_preview_error = 0.0
        self.config_messages: list[str] = []

        self.root = tk.Tk()
        self.root.title("GameJamCamera Config")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status = tk.StringVar()
        self.preview_label = ttk.Label()
        self.preview_status = tk.StringVar()

        self.load_config_values()
        self.build_actions()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        self.build_tabs()
        self.set_status("; ".join(self.config_messages) if self.config_messages else f"Loaded {self.config_path}")
        self.restart_camera()

    def load_config_values(self) -> None:
        configured, messages = app_config.merge_config(self.fallback, self.config_path)
        self.values = vars(configured).copy()
        self.config_messages = messages

    def build_tabs(self) -> None:
        self.build_mask_tab()
        for section in ("Space", "Camera", "Calibration", "Person Init", "Tracking", "Output"):
            self.build_form_tab(section)

    def build_mask_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="Mask")
        self.preview_label = ttk.Label(tab)
        self.preview_label.pack(fill="both", expand=True)
        ttk.Label(tab, textvariable=self.preview_status, anchor="w").pack(fill="x", pady=(4, 8))

        controls = ttk.Frame(tab)
        controls.pack(fill="x")
        self.create_control(controls, "clahe", 0, 0)
        for index, key in enumerate(key for key in MASK_PARAMS if key != "clahe"):
            self.create_control(controls, key, 1 + index // 4, index % 4)
        for col in range(4):
            controls.columnconfigure(col, weight=1)

    def build_form_tab(self, section: str) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text=section)
        for index, key in enumerate(SECTIONS[section]):
            self.create_control(tab, key, index, 0)
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)

    def build_actions(self) -> None:
        actions = ttk.Frame(self.root, padding=8)
        actions.pack(fill="x")
        ttk.Button(actions, text="Save", command=self.save).pack(side="left")
        ttk.Button(actions, text="Reload", command=self.reload).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Reset", command=self.reset).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Restart Camera", command=self.restart_camera).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Refresh Devices", command=self.refresh_devices).pack(side="left", padx=(8, 0))
        ttk.Label(actions, textvariable=self.status, anchor="w").pack(side="left", fill="x", expand=True, padx=(12, 0))

    def create_control(self, parent: ttk.Frame, key: str, row: int, col: int) -> None:
        default = self.defaults[key]
        value = self.values[key]
        if isinstance(default, bool):
            var = tk.BooleanVar(value=bool(value))
            self.vars[key] = var
            ttk.Checkbutton(parent, text=key, variable=var).grid(row=row, column=col, sticky="w", padx=4, pady=3)
            return

        if key == "camera_device":
            ttk.Label(parent, text=key).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            var = tk.StringVar(value=self.camera_label_for_value(str(value), int(self.values.get("camera", 0))))
            self.vars[key] = var
            self.camera_combo = ttk.Combobox(
                parent,
                textvariable=var,
                values=self.camera_labels(),
                state="readonly",
            )
            self.camera_combo.grid(row=row, column=1, sticky="ew", pady=4)
            self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_device_selected)
            return

        if key in SLIDERS:
            lo, hi, resolution = SLIDERS[key]
            var = tk.DoubleVar(value=float(value)) if isinstance(default, float) else tk.IntVar(value=int(value))
            self.vars[key] = var
            scale = tk.Scale(
                parent,
                from_=lo,
                to=hi,
                resolution=resolution,
                orient="horizontal",
                label=key,
                variable=var,
                showvalue=True,
            )
            scale.grid(row=row, column=col, sticky="ew", padx=4, pady=3)
            return

        ttk.Label(parent, text=key).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        if key in CHOICES:
            var = tk.StringVar(value=str(value))
            self.vars[key] = var
            widget = ttk.Combobox(parent, textvariable=var, values=CHOICES[key], state="readonly")
        else:
            var = tk.StringVar(value=str(app_config.encode_value(value)))
            self.vars[key] = var
            widget = ttk.Entry(parent, textvariable=var)
        widget.grid(row=row, column=1, sticky="ew", pady=4)

    def collect_values(self) -> dict[str, Any] | None:
        values: dict[str, Any] = {}
        try:
            for key, default in self.defaults.items():
                raw = self.vars[key].get()
                if key == "camera_device":
                    values[key] = self.camera_device_from_label(str(raw))
                else:
                    values[key] = app_config.coerce_value(raw, default)
            if values.get("camera_device"):
                index, _ = camera_devices.resolve_camera_index(argparse.Namespace(**values))
                values["camera"] = index
        except (tk.TclError, TypeError, ValueError) as exc:
            self.set_status(f"Invalid value: {exc}")
            return None
        return values

    def save(self) -> None:
        values = self.collect_values()
        if values is None:
            return
        app_config.save_config(self.config_path, values)
        self.values = values
        self.set_status(f"Saved {self.config_path}")

    def reload(self) -> None:
        self.load_config_values()
        self.apply_values_to_widgets()
        self.set_status("; ".join(self.config_messages) if self.config_messages else f"Reloaded {self.config_path}")

    def reset(self) -> None:
        self.values = vars(multi_red_tracking.default_args()).copy()
        self.apply_values_to_widgets()
        self.set_status("Reset to fallback defaults")

    def apply_values_to_widgets(self) -> None:
        for key, value in self.values.items():
            if key not in self.vars:
                continue
            if key == "camera_device":
                self.vars[key].set(self.camera_label_for_value(str(value), int(self.values.get("camera", 0))))
            else:
                self.vars[key].set(app_config.encode_value(value))

    def refresh_devices(self) -> None:
        self.camera_devices = camera_devices.discover_camera_devices()
        if self.camera_combo is not None:
            self.camera_combo.configure(values=self.camera_labels())
            current = str(self.vars["camera_device"].get())
            if current not in self.camera_labels():
                camera = int(self.vars["camera"].get())
                self.vars["camera_device"].set(self.camera_label_for_value("", camera))
        self.set_status(f"Found {len(self.camera_devices)} camera device(s)")

    def camera_labels(self) -> list[str]:
        return [camera_devices.device_label(device, self.camera_devices) for device in self.camera_devices]

    def camera_label_for_value(self, value: str, fallback_index: int) -> str:
        value = value.strip()
        for device in self.camera_devices:
            if value and value in (device.unique_id, device.name, camera_devices.device_label(device, self.camera_devices)):
                return camera_devices.device_label(device, self.camera_devices)
        for device in self.camera_devices:
            if device.index == fallback_index:
                return camera_devices.device_label(device, self.camera_devices)
        return value

    def camera_device_from_label(self, label: str) -> str:
        label = label.strip()
        for device in self.camera_devices:
            if label == camera_devices.device_label(device, self.camera_devices):
                return device.unique_id
        return label

    def on_camera_device_selected(self, _event: object | None = None) -> None:
        values = self.collect_values()
        if values is None:
            return
        device = str(values.get("camera_device", ""))
        index, message = camera_devices.resolve_camera_index(argparse.Namespace(**values))
        if "camera" in self.vars:
            self.vars["camera"].set(index)
        self.set_status(message or f"camera_device={device} index={index}")
        self.restart_camera()

    def restart_camera(self) -> None:
        self.close_camera()
        values = self.collect_values()
        if values is None:
            return
        args = argparse.Namespace(**values)
        try:
            self.cap = live.open_camera(args)
            suffix = f"; {'; '.join(self.config_messages)}" if self.config_messages else ""
            self.set_status(f"Camera running{suffix}")
            self.schedule_preview()
        except Exception as exc:
            self.cap = None
            suffix = f"; {'; '.join(self.config_messages)}" if self.config_messages else ""
            self.set_status(f"Camera failed: {exc}{suffix}")

    def close_camera(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def schedule_preview(self) -> None:
        self.after_id = self.root.after(30, self.update_preview)

    def update_preview(self) -> None:
        self.after_id = None
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        values = self.collect_values()
        if ok and values is not None:
            try:
                frame = live.resize_to(frame, int(values["process_width"]), int(values["process_height"]))
                view, status = self.preview_frame(frame, values)
                self.preview_label.configure(image=self.ppm_image(view))
                self.preview_status.set(status)
            except Exception as exc:
                now = time.perf_counter()
                if now - self.last_preview_error > 1.0:
                    self.preview_status.set(f"preview failed: {exc}")
                    self.last_preview_error = now
        self.schedule_preview()

    def preview_frame(self, frame: Any, values: dict[str, Any]) -> tuple[Any, str]:
        args = argparse.Namespace(**values)
        space = live.np.full(frame.shape[:2], 255, dtype=live.np.uint8)
        red, blue = live.red_blue_masks(
            frame,
            space,
            args.red_h1_max,
            args.red_h2_min,
            args.red_s_min,
            args.red_v_min,
            args.blue_h_min,
            args.blue_h_max,
            args.blue_s_min,
            args.blue_v_min,
            args.clahe,
        )
        red_blobs = live.stable.merge_close(live.stable.components(red, args.blob_min_area), int(args.marker_merge_px))
        blue_blobs = live.stable.components(blue, args.blue_min_area)

        original = self.draw_roi_preview(frame, args)
        mask = live.np.zeros_like(frame)
        mask[red > 0] = (0, 0, 255)
        mask[blue > 0] = (255, 80, 20)
        live.draw_roi_rect(mask, live.roi_rect(frame.shape, args))
        overlay = self.draw_roi_preview(frame, args)
        overlay[red > 0] = (overlay[red > 0] * 0.35 + live.np.array((0, 0, 255)) * 0.65).astype(live.np.uint8)
        overlay[blue > 0] = (overlay[blue > 0] * 0.35 + live.np.array((255, 80, 20)) * 0.65).astype(live.np.uint8)
        for blob in red_blobs:
            self.draw_blob(overlay, blob, (0, 255, 255))
        for blob in blue_blobs:
            self.draw_blob(overlay, blob, (255, 255, 255))

        view = live.np.hstack((original, mask, overlay))
        view = self.scale_for_window(view)
        roi = "space-only" if args.roi_enabled else "off"
        status = (
            f"red blobs={len(red_blobs)} blue blobs={len(blue_blobs)} "
            f"clahe={'on' if args.clahe else 'off'} roi={roi}"
        )
        return view, status

    def draw_roi_preview(self, frame: Any, args: argparse.Namespace) -> Any:
        preview = frame.copy()
        live.draw_roi_rect(preview, live.roi_rect(frame.shape, args))
        return preview

    def draw_blob(self, frame: Any, blob: dict[str, float], color: tuple[int, int, int]) -> None:
        x, y, w, h = int(blob["x"]), int(blob["y"]), int(blob["w"]), int(blob["h"])
        live.cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    def scale_for_window(self, frame: Any) -> Any:
        max_width = 1320
        max_height = 720
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        if scale >= 1.0:
            return frame
        return live.cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=live.cv2.INTER_AREA)

    def ppm_image(self, frame: Any) -> tk.PhotoImage:
        rgb = live.cv2.cvtColor(frame, live.cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        data = f"P6\n{w} {h}\n255\n".encode("ascii") + rgb.tobytes()
        image = tk.PhotoImage(data=data, format="PPM")
        self.preview_label.image = image
        return image

    def set_status(self, message: str) -> None:
        self.status.set(message)

    def close(self) -> None:
        self.close_camera()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=app_config.DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    ConfigGui(args.config).run()


if __name__ == "__main__":
    main()
