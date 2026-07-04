# Stable 01 - IMG_2485.MOV 预处理流程

## 结论

当前预处理流程可作为稳定版 `stable_01` 使用。手和脚的红色标识判定基本可用，后续可以在此基础上做轨迹筛选和动作分析。

源视频：

```text
C:\Users\Administrator\Downloads\IMG_2485.MOV
```

主脚本：

```text
F:\Github\GameJam\process_2485\stabilize_space_mask_preprocess.py
```

## Stable 01 流程

1. 视频去抖稳定。
2. 对稳定后图像做光照归一化。
3. 检测四个蓝色杯子。
4. 检测固定红色杯子作为相对 `Z` 轴方向。
5. 用四个蓝杯建立 `XY` 地面 homography。
6. 用前排两个蓝杯作为底边，沿固定红杯方向生成裁剪空间。
7. 裁剪空间外不参与红蓝 mask。
8. 对有效空间内做红蓝 HSV mask。
9. 输出每个 blob 的像素坐标和相对空间坐标：`space_x/space_y/space_z`。
10. 排除固定红杯后，用剩余红色标识输出脚的二维位置和手相对脚的距离。

## 坐标约定

- `front_left` 蓝杯近似为 `XY` 原点。
- `front_right` 蓝杯定义 `+X` 方向。
- 后排两个蓝杯参与 `XY` 透视映射。
- 固定红杯定义图像中的相对 `+Z` 方向。
- `space_x/space_y` 是基于蓝杯地面平面的相对坐标，目标范围为 `0-100`。
- `space_z` 是沿固定红杯方向的相对投影，不是物理高度。

当前裁剪多边形：

```text
[(223,895), (1670,903), (1605,0), (159,0)]
```

## 稳定参数

```powershell
--red-s-min 160 --red-v-min 100 --blue-s-min 65 --blue-v-min 15 --blob-min-area 60
```

完整处理命令：

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\process_2485\stabilize_space_mask_preprocess.py --tag tune1_3d --red-s-min 160 --red-v-min 100 --blue-s-min 65 --blue-v-min 15 --blob-min-area 60
```

单帧检查命令：

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\process_2485\stabilize_space_mask_preprocess.py --tag check01 --only-frame 219 --save-frames --red-s-min 160 --red-v-min 100 --blue-s-min 65 --blue-v-min 15 --blob-min-area 60
```

## Stable 01 输出

```text
F:\Github\GameJam\process_2485\tune1_3d_stabilized_preview.mp4
F:\Github\GameJam\process_2485\tune1_3d_space_mask_overlay.mp4
F:\Github\GameJam\process_2485\tune1_3d_space_red_blue_mask.mp4
F:\Github\GameJam\process_2485\tune1_3d_topdown_red_blue_mask.mp4
F:\Github\GameJam\process_2485\tune1_3d_space_red_blue_blobs.csv
F:\Github\GameJam\process_2485\tune1_3d_space_calibration.json
```

手脚结果输出：

```text
F:\Github\GameJam\process_2485\tune1_3d_hand_foot_positions.csv
```

手脚后处理命令：

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\process_2485\track_hand_foot_stable01.py
```

关键字段：

- `foot_space_x/foot_space_y`：脚在二维空间里的位置。
- `hand_to_foot_dx/hand_to_foot_dy`：手相对脚的二维偏移。
- `hand_to_foot_dz`：手相对地面的相对高度，固定红杯到地面距离默认为 `1`。
- `hand_to_foot_distance_2d`：手到脚的二维距离。
- `hand_to_foot_distance_3d`：手到脚的三维相对距离。
- `status`：`ok` 表示同帧有手和脚，`foot_only` 表示只识别到脚，`filled_from_previous` 表示本帧无可用红点并沿用上一帧结果。
- 空的手/脚空间字段会沿用上一帧已有值。
- `foot_space_*`、`hand_space_*` 会在前后 `30` 帧范围内做三角权重平滑，距离当前帧越近权重越高；距离字段由平滑后的坐标重新计算。

统计：

- 视频帧数：`438`
- 总 blob：`2741`
- 红色 blob：`1199`
- 蓝色 blob：`1542`
- 同帧识别到手和脚：`210`
- 只识别到脚：`205`
- 无可用红点并沿用上一帧：`23`
- 空距离字段：`0`

## 已知边界

- `topdown` 是地面平面的透视预览，不适合作为手脚高度判断依据。
- 红色 mask 仍可能包含少量环境红色噪声。
- 后续手脚识别应基于 blob 尺寸、人体区域和帧间轨迹筛选，不再只靠颜色。
- 当前手脚归类是规则版：排除固定红杯后，画面较低且面积较大的红点判为脚，剩余较高红点判为手。
- 如果某帧没有可用红点，当前结果默认复制上一帧的脚/手输出。
- 如果某帧只有脚没有手，手和距离字段默认沿用上一帧，并参与 30 帧平滑。
