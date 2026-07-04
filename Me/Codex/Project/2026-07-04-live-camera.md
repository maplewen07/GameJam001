# 2026-07-04 live camera

- Task: add a realtime external-camera test entry for the current red/blue marker pipeline.
- Script: `F:\Github\GameJam\process_live\live_camera_720_clahe.py`.
- Input request: `1920x1080@30`, OpenCV default backend, `MJPG`.
- Processing size: `1280x720`.
- CLAHE: enabled by default; press `h` in the preview window to toggle it.
- Calibration: first valid processed frame detects blue cups, fixed red z-cup, space polygon, XY homography, and relative Z axis. Press `c` to recalibrate.
- Realtime loop: camera frame -> resize to 720p -> HSV/CLAHE red-blue masks -> space crop -> connected components -> overlay preview with FPS/blob count.
- No default CSV or video output; this keeps the realtime test focused on latency and detection stability.
- Run:

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\process_live\live_camera_720_clahe.py
```

- Benchmark without camera:

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\process_live\live_camera_720_clahe.py --benchmark-video .\IMG_2487.MOV
```

- Verification: `--benchmark-video .\IMG_2487.MOV --benchmark-frames 180` preloaded frames to 720p, then measured processing only. Result: `180` frames, `3.584s`, `50.2fps`, `1070` blobs.
- Update: OpenCV in `.deps` is headless, so `cv2.imshow` / `cv2.destroyAllWindows` are unavailable. Live preview now uses standard-library `tkinter`.
- Update: camera open now requires a successful first frame read before accepting a backend. Use `--scan-cameras 5` to list readable camera indexes/backends.
- Update: local scan found the readable camera at `camera=0 backend=any actual=1920x1080@30.0`; default backend changed from `dshow` to `any`.
- Update: camera probing now waits up to `--camera-warmup-reads 40` reads before failing, and falls back from forced `MJPG/1080p30` to default camera config if configured reads fail.
- Update: live calibration uses the red cup as `z_ref` and anchors `front_left` to the large blue cup directly below it. `front_right` is selected from large low blue cups to the right; `back_left/back_right` must be valid blue candidates between the front cups and behind them. The old blue-only fallback was removed because it could pick background blue noise.
- Current screenshot check: red `z_ref` is detected at about `(430,150)`, `front_left` at about `(416,441)`, and the left background blue noise is rejected. Only one back blue cup is detected in the accepted range, so calibration now fails instead of drawing a wrong quadrilateral.
- Update: realtime flow is now two-stage. Stage 1 collects successful calibration samples for `--calibration-seconds 10.0` and requires at least `--min-calibration-frames 30`; it then freezes median blue cup points, red `z_ref`, `XY` homography, `Z` axis, and a detection polygon extended along the red-cup direction. Stage 2 no longer detects blue cups; it masks red points inside the frozen detection polygon, excludes the static red cup by `--z-exclude-px 80`, and maps the highest remaining red blob as `head` and the lowest as `foot`.
- Verification: `py_compile` passed. Current screenshot still fails safely with `need 2 back blue cups between front cups, got 1`. A synthetic in-memory calibration/tracking frame locks successfully and returns `status ok` with `head` and `foot` 3D coordinates.
- Update: calibration preview now shows a realtime red/blue mask inset during the 10-second calibration window. It is enabled by default with `--show-calibration-mask`; use `--hide-calibration-mask` to turn it off.
- Update: added `F:\Github\GameJam\process_live\mask_tuner_snapshot.py` for static mask tuning. It captures `F:\Github\GameJam\process_live\mask_tuner_snapshot.png`, then opens a Tkinter preview with HSV sliders for red and blue masks. Verification on the captured snapshot showed default `blue_pixels=44474`, confirming the current blue threshold is too broad.
- Update: applied saved tuner params to `live_camera_720_clahe.py`: `--red-h1-max 12 --red-h2-min 168 --red-s-min 160 --red-v-min 100 --blue-h-min 108 --blue-h-max 129 --blue-s-min 145 --blue-v-min 41 --clahe`. On the saved snapshot, blue mask pixels dropped from `44474` to `3840` while retaining the four blue cup candidates.
- Update: after tuner use, camera probing showed `camera=0 backend=dshow` is the reliable path; `live_camera_720_clahe.py` and `mask_tuner_snapshot.py` now default to `--backend dshow`.
- Update: after tightening the blue HSV mask, blue blob areas became much smaller. The 2026-07-04 failure screenshot had front blue areas about `1086/1184` and back blue areas about `421/383`, so the old `--front-blue-min-area 3000` rejected every front cup. Defaults changed to `--front-blue-min-area 800` and `--back-blue-min-area 250`; the screenshot then calibrates with `front_left/front_right/back_left/back_right` detected.
- Update: added `F:\Github\GameJam\process_multi_live\live_multi_red_tracking.py` for marker-only multi-person tracking. It reuses the single-person camera/calibration/mapping helpers, pairs same-color red head/foot blobs into people, assigns nearest-neighbor track IDs, and records overlay video by default under `F:\Github\GameJam\process_multi_live\recordings`. Use `--no-record-video` to disable recording.
- Update: reviewed `F:\Github\GameJam\process_multi_live\recordings\multi_live_20260704_150238.mp4`. Calibration locked and red markers were detected (`markers=3/4/5`), but `people=0` because the first pairing logic used projected `space_x/space_y` distance; elevated head markers project far from foot markers on the ground plane. Multi-person pairing now splits markers by image vertical order and pairs head/foot by pixel distance (`--pair-max-px 450`), while track IDs use the foot image position as the stable center.
- Update: reviewed `F:\Github\GameJam\process_multi_live\recordings\multi_live_20260704_152129.mp4`. Red head/foot blobs are visible, but global head-foot pairing still crosses people when two testers are close or extra red blobs appear.
- Update: multi-person live flow is now three-stage: cup calibration -> 10-second person-region initialization -> region tracking. After cup calibration locks, the script collects temporary red marker pairs for `--person-init-seconds 10.0`, clusters them into stable person regions, assigns `P1/P2/...` left-to-right, then tracks each person only inside its own slowly-following box. Inside a box, the upper half is fixed as `head` and the lower half is fixed as `foot`, which avoids global cross-person head/foot matching. New defaults: `--person-init-seconds 10.0`, `--min-person-init-frames 30`, `--person-box-margin 80`, `--person-smooth-alpha 0.25`.
- Update: reviewed `F:\Github\GameJam\process_multi_live\recordings\multi_live_20260704_154853.mp4`. The contact sheet shows red blobs are detected, but person initialization often has only one complete head/foot pair while another person only has a foot marker. The old init logic required global temporary pairs, so a bad early pair could create oversized/cross-person boxes.
- Update: person initialization now anchors people by stable foot clusters first, then attaches the nearest upper marker only when it is horizontally close (`--person-init-x-margin 180`). If a head marker is missing during init, the script seeds a temporary head point above the foot (`--person-default-height 360`) so the person box still exists. Defaults changed to `--person-box-margin 100` and `--person-smooth-alpha 0.5` so boxes are less incomplete and follow movement faster.
- Update: reviewed the front 35 seconds of `F:\Github\GameJam\process_multi_live\recordings\multi_live_20260704_160605.mp4` for the three-person case. During person init, duplicate foot blobs and extra lower red blobs could create four temporary people (`stable=4 pairs=4 markers=7`) before settling to three.
- Update: multi-person tracking now merges nearby red marker fragments before candidate extraction (`--marker-merge-px 35`), merges horizontally close init foot anchors (`--person-init-foot-merge-px 90`), and assigns markers in overlapping person boxes by distance to the predicted head/foot point instead of distance to the box center.
- Update: head/foot role classification now uses one fixed screen-space split line from the frozen back blue cups: `split_y = average(back_left.cy, back_right.cy) - --foot-split-offset-px`, default `100`. Red markers below the line are always `foot`; markers above the line are always `head`. The overlay draws this line as `head / foot split`.
