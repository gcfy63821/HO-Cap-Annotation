# Contact Optimizer (07-4) 开发日志

## 概述

`tools/07-4_contact_optimizer.py` 是手-物体接触优化器，用于在手部和物体6D位姿标注完成后，通过优化手腕平移和旋转来：
1. 最小化指尖与物体表面的距离（尤其是抓取帧）
2. 消除手与物体的穿模（interpenetration）
3. 保持时序平滑性

优化结果用于下游RL框架（仅需指尖-物体距离信息）。

## 文件位置

- 主代码：`tools/07-4_contact_optimizer.py`
- 可视化：`/home/ruoqu/crq_ws/robotool/DataCollection/visualize/viser_viewer.py`（支持 `--source` 切换数据源）
- 输出目录：`<annotated_folder>/processed/contact_optimizer/`

## 输出文件

| 文件 | 内容 |
|------|------|
| `result_hand_optimized.pkl` | 优化后的手部数据（translation + pose/base_rot 已更新） |
| `poses_o.npy` | 物体位姿（直接复制，未修改） |
| `fingertip_distances.npy` | `(num_frames, num_hands, 5)` 指尖到物体的距离（米） |
| `fingertip_meta.json` | 元数据（手的列表、手指名称、阈值等） |

## 使用方法

### 基本运行

```bash
conda activate hocap-annotation
cd /home/ruoqu/crq_ws/robotool/HO-Cap-Annotation

python tools/07-4_contact_optimizer.py \
  --sequence_folder data/videos_XXXX/task_name/sequence_name \
  --object_idx 1
```

### 完整参数

```bash
python tools/07-4_contact_optimizer.py \
  --sequence_folder <path>       # 序列数据文件夹（必需）
  --object_idx 1                 # 物体索引，从1开始（默认1）
  --sdf_resolution 128           # SDF网格分辨率（默认128）
  --sdf_padding 0.15             # 粗SDF的padding，米（默认0.15）
  --grasp_thresh 0.05            # 抓取检测阈值，米（默认0.05）
  --min_grasp_fingers 2          # 至少N个指尖在阈值内才算抓取（默认2）
  --global_steps 100             # Stage 2 全局优化步数（默认100）
  --perframe_steps 200           # Stage 3 逐帧优化步数（默认200）
  --global_lr 0.002              # Stage 2 学习率
  --perframe_lr 0.001            # Stage 3 学习率
  --w_pen 1000                   # 穿模惩罚权重
  --w_tip 200                    # 指尖吸引权重
  --w_prox 50                    # 近表面顶点吸引权重
  --w_reg 5                      # 正则化权重（防止偏离初始值太远）
  --w_smooth 1                   # 时序平滑权重
  --smooth_window 3              # 时序平滑窗口大小
  --output_suffix contact_optimizer  # 输出子文件夹名
```

### 推荐的高质量参数

```bash
python tools/07-4_contact_optimizer.py \
  --sequence_folder <path> \
  --global_steps 500 \
  --perframe_steps 1000 \
  --grasp_thresh 0.08 \
  --w_pen 2000 \
  --w_tip 400 \
  --w_prox 100 \
  --sdf_resolution 192
```

### 可视化结果

```bash
cd /home/ruoqu/crq_ws/robotool/DataCollection/visualize

python viser_viewer.py \
  --data_path <sequence_folder> \
  --source contact_optimizer     # 查看优化结果

python viser_viewer.py \
  --data_path <sequence_folder> \
  --source contact_optimizer \
  --compare original             # 对比优化前后
```

`viser_viewer.py` 支持的 source：original, optimized, contact_optimizer, joint_pose_solver, object_pose_solver（自动扫描 annotated 文件夹）

## 优化流程（4个阶段）

### Stage 0: 诊断

- 用 MANO 前向计算预计算每帧手部顶点/关节在 wrist-local 坐标系下的位置
- 用 trimesh 精确距离计算每帧指尖到物体的距离
- 基于 `grasp_thresh` 和 `min_grasp_fingers` 判定哪些帧是抓取帧

### Stage 1: 解析全局偏移

- 对所有抓取帧，计算指尖到物体最近表面点的中位数向量
- 作为 Stage 2 的初始值，加速收敛

### Stage 2: 全局梯度优化（粗SDF）

- 优化单个 `(delta_rot, delta_trans)` 应用于所有抓取帧
- 使用粗SDF（padding=0.15m），覆盖范围大，适合纠正全局系统性偏移
- Loss = 穿模惩罚 + 指尖吸引 + 近表面顶点吸引

### Stage 3: 逐帧优化（细SDF）

- 每帧独立的 `(delta_rot[f], delta_trans[f])`
- 使用细SDF（padding=0.03m），分辨率更高，穿模检测更精确
- 抓取帧从 Stage 2 结果初始化；非抓取帧仅做穿模修复
- 使用 CosineAnnealingLR 学习率调度
- Loss = 穿模 + 指尖吸引 + 近表面吸引 + 正则化 + 时序平滑

### Stage 4: 后处理

1. 时序平滑（uniform_filter1d）
2. 穿模修复 pass：对平滑引入的穿模做梯度下降修复（100步）

## 关键技术细节

### SDF计算

- 使用 `igl.signed_distance()` 计算有符号距离场（负值=内部/穿模）
- 通过 `F.grid_sample` 实现可微查询
- 双分辨率策略：粗网格覆盖全局偏移，细网格精确检测穿模

### MANO手部模型

- 使用右手 MANO layer（WiLoR 惯例：左手 = 右手 MANO + x翻转 + base_rot）
- 778个顶点，21个关节
- 指尖关节索引：[20, 4, 8, 16, 12]（拇指、食指、中指、无名指、小指）
- 手部pose冻结，仅优化手腕平移和旋转

### 旋转结果保存

- 右手：`delta_rot` 被 bake 进 `pose[:3]`（全局朝向 axis-angle）
- 左手：`delta_rot` 被 bake 进 `base_rot` 矩阵

### 穿模Loss

- 使用 `sum` 而非 `mean`，保证更强的梯度信号

## 已解决的问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `AttributeError: object_cleaned_files` | MySequenceLoader 属性名不同 | 改用 `object_cleaned_mesh_files` |
| SDF计算 OOM | trimesh.proximity 内存消耗大 | 改用 `igl.signed_distance()` |
| igl unpack错误 | 返回4个值 | `signed, _, _, _ = igl.signed_distance(...)` |
| igl faces dtype | 需要 int64 | `np.array(mesh.faces, dtype=np.int64)` |
| 穿模越优化越多 | loss用mean，梯度太弱 | 改为 `sum`，加大 w_pen |
| 远距离SDF不准 | 超出SDF网格范围 | 增大padding + 用trimesh做精确距离 |
| fix_penetration 不考虑原始平移 | 少加了 original_trans | 补上 `orig_t + dt_param` |
| 时序平滑引入穿模 | 平滑后没检查穿模 | 增加 post-smoothing pen fix pass |

## 当前结果（测试序列）

测试序列：`videos_0121/mallet_crush_nuts/20260121_mallet_crush_peanuts_nuts_18`

参数：`--global_steps 200 --perframe_steps 500`

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 右手抓取帧平均指尖距离 | 30.8 mm | 17.9 mm |
| 右手整体平均指尖距离 | 97.6 mm | 87.1 mm |
| 穿模顶点数 | ~3583 | 159 |
| 全局偏移 | — | ~[-14, 30, -1] mm 平移, ~[-1.7, -6.6, 4.7]° 旋转 |

左手因 grasp_thresh=50mm 未检测到抓取帧（最近指尖~74mm），需提高阈值到 80mm。

## 待改进方向

1. **多轮 Stage 2-3 迭代**：每轮结果作为下一轮初始值，逐步收紧 proximity 阈值
2. **增强 fix_penetration_pass**：同时优化 rotation，增加步数
3. **自适应 proximity 阈值**：per-frame 阶段从 0.05 逐步收紧到 0.02
4. **Per-finger 权重**：对距离较大的手指给更高权重
5. **更高 SDF 分辨率**（192或256）：代价是计算时间和显存
6. **左手阈值调整**：根据数据自动选择 grasp_thresh

## 在 pipeline 中的位置

整体流水线顺序：
```
00_convert_videos_to_h5
  → 01 camera calibration
  → 02 hand detection (MediaPipe)
  → 03 3D joints generation
  → 04 object pose tracking (FoundationPose + Kalman)
  → 05 MANO pose solver
  → 06 object pose solver (refinement)
  → 07-2/07-3 joint pose solver (hand-object joint optimization)
  → 07-4 contact optimizer (本工具，最后一步精细化)
```

运行前需要确保：
- `<annotated_folder>/result_hand_optimized.pkl`（或 `result.pkl`）已存在
- `<annotated_folder>/processed/object_pose_solver/poses_o.npy`（或 `fd_pose_solver/fd_poses_merged_fixed.npy`）已存在
- 物体 cleaned mesh 在 `data/models/` 下可访问
