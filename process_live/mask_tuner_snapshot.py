import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIVE_PATH = Path(__file__).with_name("live_camera_720_clahe.py")
SNAPSHOT = Path(__file__).with_name("mask_tuner_snapshot.png")

spec = importlib.util.spec_from_file_location("live", LIVE_PATH)
live = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(live)

cv2 = live.cv2
np = live.np


def capture(args: argparse.Namespace) -> Path:
    cap = live.open_camera(args)
    frame = None
    for _ in range(args.capture_reads):
        ok, current = cap.read()
        if ok:
            frame = current
    cap.release()
    if frame is None:
        raise SystemExit("cannot capture snapshot")
    frame = live.resize_to(frame, args.process_width, args.process_height)
    cv2.imwrite(str(args.image), frame)
    print(f"wrote {args.image}")
    return args.image


def hsv(frame: np.ndarray, clahe: bool) -> np.ndarray:
    return live.hsv(frame, clahe)


def masks(frame: np.ndarray, values: dict[str, int | bool]) -> tuple[np.ndarray, np.ndarray]:
    hh = hsv(frame, bool(values["clahe"]))
    red = cv2.inRange(
        hh,
        np.array((0, values["red_s_min"], values["red_v_min"])),
        np.array((values["red_h1_max"], 255, 255)),
    )
    red |= cv2.inRange(
        hh,
        np.array((values["red_h2_min"], values["red_s_min"], values["red_v_min"])),
        np.array((179, 255, 255)),
    )
    blue = cv2.inRange(
        hh,
        np.array((values["blue_h_min"], values["blue_s_min"], values["blue_v_min"])),
        np.array((values["blue_h_max"], 255, 255)),
    )
    if values["clean"]:
        red = live.stable.clean(red)
        blue = live.stable.clean(blue)
    return red, blue


def view(frame: np.ndarray, values: dict[str, int | bool]) -> np.ndarray:
    red, blue = masks(frame, values)
    mask = np.zeros_like(frame)
    mask[red > 0] = (0, 0, 255)
    mask[blue > 0] = (255, 80, 20)
    overlay = frame.copy()
    overlay[red > 0] = (overlay[red > 0] * 0.35 + np.array((0, 0, 255)) * 0.65).astype(np.uint8)
    overlay[blue > 0] = (overlay[blue > 0] * 0.35 + np.array((255, 80, 20)) * 0.65).astype(np.uint8)
    return np.hstack((frame, mask, overlay))


def ppm(frame: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return f"P6\n{w} {h}\n255\n".encode("ascii") + rgb.tobytes()


def run_ui(image: Path, args: argparse.Namespace) -> None:
    import tkinter as tk

    frame = cv2.imread(str(image))
    if frame is None:
        raise SystemExit(f"cannot read image: {image}")

    values: dict[str, int | bool] = {
        "clahe": args.clahe,
        "clean": True,
        "red_h1_max": args.red_h1_max,
        "red_h2_min": args.red_h2_min,
        "red_s_min": args.red_s_min,
        "red_v_min": args.red_v_min,
        "blue_h_min": args.blue_h_min,
        "blue_h_max": args.blue_h_max,
        "blue_s_min": args.blue_s_min,
        "blue_v_min": args.blue_v_min,
    }

    root = tk.Tk()
    root.title("mask tuner snapshot")
    label = tk.Label(root)
    label.pack()
    status = tk.StringVar()
    tk.Label(root, textvariable=status, anchor="w").pack(fill="x")
    controls = tk.Frame(root)
    controls.pack(fill="x")

    def refresh(*_: object) -> None:
        for key, widget in sliders.items():
            values[key] = widget.get()
        values["clahe"] = bool(clahe_var.get())
        values["clean"] = bool(clean_var.get())
        image_obj = tk.PhotoImage(data=ppm(view(frame, values)), format="PPM")
        label.configure(image=image_obj)
        label.image = image_obj
        status.set(params_text(values))

    sliders: dict[str, tk.Scale] = {}
    slider_defs = [
        ("red_h1_max", 0, 30),
        ("red_h2_min", 140, 179),
        ("red_s_min", 0, 255),
        ("red_v_min", 0, 255),
        ("blue_h_min", 0, 179),
        ("blue_h_max", 0, 179),
        ("blue_s_min", 0, 255),
        ("blue_v_min", 0, 255),
    ]
    for i, (key, lo, hi) in enumerate(slider_defs):
        scale = tk.Scale(controls, from_=lo, to=hi, orient="horizontal", label=key, command=refresh)
        scale.set(int(values[key]))
        scale.grid(row=i // 4, column=i % 4, sticky="ew")
        sliders[key] = scale
    for col in range(4):
        controls.columnconfigure(col, weight=1)

    checks = tk.Frame(root)
    checks.pack(fill="x")
    clahe_var = tk.IntVar(value=1 if values["clahe"] else 0)
    clean_var = tk.IntVar(value=1 if values["clean"] else 0)
    tk.Checkbutton(checks, text="CLAHE", variable=clahe_var, command=refresh).pack(side="left")
    tk.Checkbutton(checks, text="morph clean", variable=clean_var, command=refresh).pack(side="left")

    def save() -> None:
        args.out_params.write_text(params_text(values) + "\n", encoding="utf-8")
        print(params_text(values))
        print(f"wrote {args.out_params}")

    tk.Button(checks, text="Save params", command=save).pack(side="left")
    refresh()
    root.mainloop()


def params_text(values: dict[str, int | bool]) -> str:
    return (
        f"--red-h1-max {values['red_h1_max']} --red-h2-min {values['red_h2_min']} "
        f"--red-s-min {values['red_s_min']} --red-v-min {values['red_v_min']} "
        f"--blue-h-min {values['blue_h_min']} --blue-h-max {values['blue_h_max']} "
        f"--blue-s-min {values['blue_s_min']} --blue-v-min {values['blue_v_min']} "
        f"{'--clahe' if values['clahe'] else '--no-clahe'}"
    )


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
    parser.add_argument("--camera-warmup-reads", type=int, default=40)
    parser.add_argument("--capture-reads", type=int, default=5)
    parser.add_argument("--image", type=Path, default=SNAPSHOT)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--use-existing", action="store_true")
    parser.add_argument("--out-params", type=Path, default=Path(__file__).with_name("mask_tuner_params.txt"))
    parser.add_argument("--clahe", dest="clahe", action="store_true", default=True)
    parser.add_argument("--no-clahe", dest="clahe", action="store_false")
    parser.add_argument("--red-h1-max", type=int, default=12)
    parser.add_argument("--red-h2-min", type=int, default=168)
    parser.add_argument("--red-s-min", type=int, default=160)
    parser.add_argument("--red-v-min", type=int, default=100)
    parser.add_argument("--blue-h-min", type=int, default=92)
    parser.add_argument("--blue-h-max", type=int, default=136)
    parser.add_argument("--blue-s-min", type=int, default=65)
    parser.add_argument("--blue-v-min", type=int, default=15)
    args = parser.parse_args()

    image = args.image if args.use_existing and args.image.exists() else capture(args)
    if not args.capture_only:
        run_ui(image, args)


if __name__ == "__main__":
    main()
