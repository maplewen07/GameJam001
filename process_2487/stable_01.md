# Stable 01 - IMG_2487.MOV 处理记录

## 结论

`IMG_2487.MOV` 已按 `stable_01` 流程处理完成。红色手/脚标识可用，最终 CSV 已输出脚的二维位置、手相对脚的距离，并做了上一帧补值和 30 帧三角权重平滑。

源视频：

```text
F:\Github\GameJam\IMG_2487.MOV
```

处理目录：

```text
F:\Github\GameJam\process_2487
```

## 处理命令

预处理：

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\process_2487\stabilize_space_mask_preprocess.py --tag tune1_3d --red-s-min 160 --red-v-min 100 --blue-s-min 65 --blue-v-min 15 --blob-min-area 60
```

手脚后处理：

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\process_2487\track_hand_foot_stable01.py
```

## 坐标与裁剪

- 四个蓝杯建立 `XY` 地面 homography。
- 前排两个蓝杯作为裁剪底边。
- 固定红杯作为相对 `Z` 轴方向。
- 固定红杯到地面距离默认 `1.0`。
- `space_z` 是相对投影，不是物理高度。

当前裁剪多边形：

```text
[(224,977), (1675,992), (1649,0), (198,0)]
```

## 输出文件

```text
F:\Github\GameJam\process_2487\tune1_3d_stabilized_preview.mp4
F:\Github\GameJam\process_2487\tune1_3d_space_mask_overlay.mp4
F:\Github\GameJam\process_2487\tune1_3d_space_red_blue_mask.mp4
F:\Github\GameJam\process_2487\tune1_3d_topdown_red_blue_mask.mp4
F:\Github\GameJam\process_2487\tune1_3d_space_red_blue_blobs.csv
F:\Github\GameJam\process_2487\tune1_3d_space_calibration.json
F:\Github\GameJam\process_2487\tune1_3d_hand_foot_positions.csv
```

QA 抽帧：

```text
F:\Github\GameJam\process_2487\qa
```

## 结果统计

- 视频帧数：`639`
- 总 blob：`4439`
- 红色 blob：`1939`
- 蓝色 blob：`2500`
- 同帧识别到手和脚：`631`
- 只识别到脚：`8`
- 无可用红点并沿用上一帧：`0`
- 空 2D/3D 距离字段：`0`
- 补齐空的手/脚空间字段：`24`
- 平滑窗口：前后 `30` 帧，越靠近当前帧权重越高

## 关键输出字段

- `foot_space_x/foot_space_y`：脚在二维空间的位置。
- `hand_to_foot_dx/hand_to_foot_dy`：手相对脚的二维偏移。
- `hand_to_foot_dz`：手相对地面的相对高度。
- `hand_to_foot_distance_2d`：手到脚的二维距离。
- `hand_to_foot_distance_3d`：手到脚的三维相对距离。
- `status`：`ok`、`foot_only`、`filled_from_previous`。

## 已知边界

- 当前手/脚分类仍是规则版：排除固定红杯后，低处红点判脚，高处红点判手。
- `topdown` 只用于地面平面预览，不用于手脚高度判断。
