# GameJamCamera

Live camera calibration and multi-person red marker tracking for the GameJam setup.

## Project Layout

```text
GameJamCamera/
  pyproject.toml
  requirements.txt
  scripts/
    run_multi_tk.sh
    run_config_gui.sh
  config/
    defaults.json
  src/
    gamejam_camera/
      config.py
      gui.py
      __main__.py
      live_camera.py
      multi_red_tracking.py
      space_preprocess.py
  recordings/
```

## Run

On this Mac, use the cached Python 3.12 + Tk 8.6 environment:

```bash
cd /Users/maplewen/Project/GameJam001/GameJamCamera
./scripts/run_multi_tk.sh
```

Open the config and mask tuning GUI:

```bash
cd /Users/maplewen/Project/GameJam001/GameJamCamera
./scripts/run_config_gui.sh
```

Equivalent command:

```bash
PYTHONPATH=src /Users/maplewen/.cache/gamejam001/tk-python/bin/python -u -m gamejam_camera --config config/defaults.json
```

In the Tk window, press `q` or `Esc` to quit.

## Useful Options

```bash
./scripts/run_multi_tk.sh --no-record-video
./scripts/run_multi_tk.sh --record-video
./scripts/run_multi_tk.sh --max-frames 240
./scripts/run_multi_tk.sh --camera 1
./scripts/run_multi_tk.sh --no-preview
./scripts/run_multi_tk.sh --person-socket-port 8766
./scripts/run_multi_tk.sh --person-socket-host 0.0.0.0
./scripts/run_multi_tk.sh --no-person-socket-output
./scripts/run_multi_tk.sh --person-frame-output
./scripts/run_multi_tk.sh --no-person-frame-output
./scripts/run_multi_tk.sh --person-frame-output-interval 5
./scripts/run_config_gui.sh --config config/defaults.json
```

Settings are loaded from `config/defaults.json`. Command-line options still override the JSON values.

## Space ROI

Open the config GUI and use the `Space` tab to enable a fixed recognition area. `roi_x_min`, `roi_x_max`, `roi_y_min`, and `roi_y_max` are normalized frame coordinates from `0.0` to `1.0`, so the saved area follows the camera image size.

When `roi_enabled` is true, the ROI is applied only while finding calibration reference cups: the red z reference cup mask and blue reference cup mask are limited to this rectangle. The locked space mask used by person init and tracking is generated normally from the selected cups. Normal mask tuning preview still shows the full red/blue color result and only draws the green ROI box as a guide.

The `Calibration` tab also exposes red and blue reference-cup geometry filters such as `red_cup_min_w`, `red_cup_min_h`, `blue_cup_min_w`, and `blue_cup_min_h`. These only filter calibration cups while building the space coordinate system.

## Camera Selection

The config GUI shows camera devices by name in the `Camera` tab. Saving the config writes `camera_device` to `config/defaults.json` using the device Unique ID, while `camera` remains a numeric fallback if the named device is not available.

You can also override it from the command line:

```bash
./scripts/run_multi_tk.sh --camera-device OsmoAction5pro
```

## Notes

The Apple Command Line Tools Python 3.9 on this machine uses Tk/Tcl 8.5 and crashes when creating a Tk window. The cached Python environment above uses Tk/Tcl 8.6 and has been verified with the preview window.

Calibration needs the red z reference cup and the blue cups visible in the camera frame before tracking can lock.

During tracking, a local socket HTTP service is started by default:

```text
GET http://127.0.0.1:8765/person-frame/latest
GET http://127.0.0.1:8765/person-frame/history?limit=60
GET http://127.0.0.1:8765/health
```

Each JSON payload includes every locked person's foot-based plane coordinate, a 2D local reference frame, head/foot pixel centers, and direct head-to-foot pixel distance. Use `--person-frame-output` if you also want the old `person_frame=...` console stream.
