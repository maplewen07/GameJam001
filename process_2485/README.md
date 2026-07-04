# IMG_2485.MOV 预处理记录

## 目标

对 `C:\Users\Administrator\Downloads\IMG_2485.MOV` 做预处理，为后续基于蓝色杯子和红色标识的三维坐标计算做准备。

当前阶段只做预处理和相对坐标输出，不做最终手脚轨迹筛选。

## 坐标与空间约定

- 四个蓝色杯子用于建立 `XY` 地面坐标。
- 前排两个蓝杯作为裁剪空间底边。
- 固定红色杯子用于确定图像里的相对 `Z` 方向。
- 裁剪空间由前排两个蓝杯和固定红杯方向生成，不再由后排蓝杯向画面顶部延伸。
- 红色和蓝色 mask 只保留在这个有效空间多边形内。
- `space_z` 是沿固定红杯方向的相对坐标，不是物理高度。

## 脚本

脚本路径：

```text
F:\Github\GameJam\process_2485\stabilize_space_mask_preprocess.py
```

主要处理步骤：

1. 读取视频首帧。
2. 用 ORB 特征对齐做视频去抖，参考帧为首帧。
3. 在首帧中识别四个蓝色杯子和固定红色杯子。
4. 根据四个蓝杯建立 `XY` homography。
5. 用前排两个蓝杯和固定红杯方向建立有效空间多边形。
6. 用固定红杯建立相对 `Z` 方向。
7. 对稳定后的视频帧做 HSV 红/蓝阈值分割。
8. 将红/蓝 mask 限制在有效空间内。
9. 将每个 blob 中心点映射为 `space_x/space_y/space_z`。
10. 输出稳定视频、mask 视频、overlay 视频、topdown 预览、blob CSV 和标定 JSON。

## 最终采用参数

用户确认可用的 mask 参数：

```powershell
--red-s-min 160 --red-v-min 100 --blue-s-min 65 --blue-v-min 15 --blob-min-area 60
```

完整视频处理命令：

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\process_2485\stabilize_space_mask_preprocess.py --tag tune1_3d --red-s-min 160 --red-v-min 100 --blue-s-min 65 --blue-v-min 15 --blob-min-area 60
```

单帧调参命令示例：

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\process_2485\stabilize_space_mask_preprocess.py --tag tune1 --only-frame 1 --save-frames --red-s-min 160 --red-v-min 100 --blue-s-min 65 --blue-v-min 15 --blob-min-area 60
```

## 输出结果

完整 `tune1_3d` 版本处理结果：

- 视频帧数：`438`
- 总 blob 数：`2741`
- 红色 blob：`1199`
- 蓝色 blob：`1542`

输出文件：

```text
F:\Github\GameJam\process_2485\tune1_3d_stabilized_preview.mp4
F:\Github\GameJam\process_2485\tune1_3d_space_mask_overlay.mp4
F:\Github\GameJam\process_2485\tune1_3d_space_red_blue_mask.mp4
F:\Github\GameJam\process_2485\tune1_3d_topdown_red_blue_mask.mp4
F:\Github\GameJam\process_2485\tune1_3d_space_red_blue_blobs.csv
F:\Github\GameJam\process_2485\tune1_3d_space_calibration.json
```

默认参数版本也保留在同目录：

```text
F:\Github\GameJam\process_2485\stabilized_preview.mp4
F:\Github\GameJam\process_2485\space_mask_overlay.mp4
F:\Github\GameJam\process_2485\space_red_blue_mask.mp4
F:\Github\GameJam\process_2485\space_red_blue_blobs.csv
F:\Github\GameJam\process_2485\space_calibration.json
```

## 标定信息

`tune1_3d_space_calibration.json` 中记录：

- 原视频路径
- 输出缩放比例
- 四个蓝杯中心点
- 固定红杯中心点
- `XY` homography
- 相对 `Z` 轴方向
- 有效空间多边形
- topdown 预览尺寸
- 处理帧数

`tune1_3d_space_red_blue_blobs.csv` 中每个 blob 包含：

- `cx/cy`：稳定后视频里的像素中心点
- `space_x/space_y`：由四个蓝杯映射出的地面相对坐标，范围目标为 `0-100`
- `space_z`：沿固定红杯方向的相对坐标，固定红杯约为 `100`

## 当前限制

红色 mask 仍可能包含皮肤、衣服图案、墙面/地面的红色噪声。后续计算手脚位置时，不建议继续只靠颜色阈值硬筛。

topdown 视频只是地面平面的透视矫正预览。手脚标识离地时，按地面 homography 映射可能落到 `0-100` 范围外，这是正常现象。

下一步更适合加入：

- 人体区域约束
- blob 尺寸过滤
- 手/脚大致高度区域
- 帧间连续轨迹约束
- 与蓝杯地面坐标的投影换算
