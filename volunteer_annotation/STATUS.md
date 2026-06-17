# 志愿者 SAM2 标注 Pipeline — 实现状态

> 截至 2026-06-17。本文档记录**已实现并验证**的系统现状（与最初的 `ARCHITECTURE.md` 设计稿配套；实际实现已在多处演进，以本文档为准）。

---

## 1. 一句话现状

招募志愿者，在浏览器里**一次性给一个 exp 的 8 个相机视角**用点 prompt 标注三类角色物体（可在多个关键帧中选帧打点），服务端用 CPU 实时预览 SAM2 掩码；提交的点经 GPU 侧全视频双向传播，产出与现有 RoboTool 流水线**字节级兼容**的 `masks.h5`。已在真实数据 `videos_0102`（10 exp × 8 相机）上端到端跑通。

整条链路（precompute → 浏览器标注 → prompts→masks → 下游兼容）**全部实现并验证**。剩余仅为部署期增强（overlay 懒加载、质控汇总、ONNX 解码器）。

---

## 2. 三机拓扑（回顾）

```
[内网 GPU 服务器]            [本地 1T / 枢纽]              [公网云服务器]
离线·GPU·全量数据            权威 embedding 存储           在线·无GPU·≤50G
─ precompute_embeddings      ─ embedding bundle 全量        ─ FastAPI + CPU 解码器
─ prompts_to_masks (传播)    ─ prompt 回流备份              ─ 8 视角标注 UI
                                  ↕ overlay (待接)             ↕ 志愿者
```
- 开发期：三角色全部跑在本地 4090 这台机器（`internal/` 用 GPU，`cloud/` 强制 CPU 解码模拟云端）。
- 核心洞察：SAM2 编码器（重，GPU，与点无关）离线预计算 → embedding 下沉云端；解码器（轻，CPU ~20ms）做交互预览。**云端永不需要 GPU / 不接触原始视频**。
- 关键验证：fp16 embedding + CPU 解码复现原生 SAM2，mean IoU **0.9925**。

---

## 3. 文件清单

```
HO-Cap-Annotation/volunteer_annotation/
├── ARCHITECTURE.md            # 最初设计稿
├── STATUS.md                  # 本文档（实现现状）
├── internal/                  # 内网 GPU 侧（开发期本地 4090）
│   ├── precompute_embeddings.py   # frame → 关键帧 embedding/jpg + 缩略图 + manifest
│   ├── validate_seam.py           # 验证 CPU 解码 == 原生 SAM2（接缝测试）
│   ├── prompts_to_masks.py        # prompt JSON → 全视频双向传播 → masks.h5 + objects.yaml + roles.yaml
│   ├── sbatch_precompute_array.sh # ★ SLURM array 批量生成（对齐 scripts/sbatch_run_hand_array.sh）
│   ├── generate_bundle.sh         # 普通多卡单机批量生成（非 SLURM 备选）
│   ├── merge_manifests.py         # 合并各分片 manifest_k.json → manifest.json
│   └── export_ref_frames.py       # (已被缩略图取代) 单张中段参考帧补全
└── cloud/                     # 公网云侧（无 GPU）
    ├── app.py                     # FastAPI：任务/图像/预览/提交/标坏/进度/列表
    ├── tasks_db.py                # SQLite 任务库（按 exp 一行）+ 角色花名册派生
    ├── decoder.py                 # Sam2CpuDecoder：注入 embedding + 解码（带并发锁）
    ├── seed_tasks.py              # 从 manifest 建任务库
    └── frontend/index.html        # 8 视角 Canvas 标注界面（纯前端）
```

---

## 4. 数据格式

### 4.1 embedding bundle（precompute 产，schema 2）
```
<bundle>/
├── manifest.json                     # 顶层汇总（由各 exp 的 _manifest.json 合并而来）
└── <task>/<exp>/
    ├── _manifest.json                # ★ 该 exp 的 manifest 片段 + “已完成”标记（resume 依据）
    ├── cam{c}_rgb.kf{idx}.embed.npz   # fp16: image_embed[256,64,64] + high_res_feat_0[32,256,256]
    │                                  #       + high_res_feat_1[64,128,128] + orig_hw[2]  (~6MB)
    ├── cam{c}_rgb.kf{idx}.jpg         # 关键帧全分辨率图（标注用）
    ├── cam{c}_rgb.kf{idx}.refmask.npz # 原生 SAM2 multimask 参考（接缝验证；可 --no_refmask 跳过）
    └── cam{c}_rgb.th{idx}.jpg         # 中段浏览缩略图（仅 jpg，~13KB，无 embedding）
```
`manifest.json`：
```json
{ "schema_version": 2, "sam2_model": "sam2.1_hiera_large.pt", "embed_dtype": "float16",
  "cameras": [ { "task","exp","camera","cam_index","width","height","n_frames",
                 "primary_name","keyframes":[0,67,133],"thumbs":[266,400],"image_sha1" } ] }
```
- 默认关键帧分数 `0,0.1,0.2`（frame 0 + 两个靠前帧），缩略图分数 `0.4,0.6`。可用 `--keyframe_fracs/--thumb_fracs` 调。
- `primary_name` 由 `mesh_name_mapping.json` 最长子串匹配 exp 名得到（如 …greenchopstick… → `green_plastic_chopstick`）。
- videos_0102 实测 bundle = **1.5G**（3 关键帧 × 80 相机）。

### 4.2 prompt JSON（志愿者产出，每相机一份）
`<PROMPTS_DIR>/<task>/<exp>/tool_masks/prompts/<camera>.json`：
```json
{ "schema_version":1, "task","exp","camera","cam_index","frame_index":0,"width","height",
  "image_sha1","sam2_model","annotator_id","submitted_at","review_status":"submitted",
  "objects":[
    {"role":"primary_tool","name":"green_plastic_chopstick","frame_index":67,
     "points":[[x,y],...],"labels":[1,0,...],"box":null,"preview_iou":0.9},
    {"role":"manipulated_object","name":"manipulated_object","frame_index":0,"points":[...],"labels":[...]}
  ] }
```
- 原始像素坐标；`labels` 1=正/0=负；**每物体可有自己的 `frame_index`**（在哪个关键帧上标的）。

### 4.3 坏视频标记
`<PROMPTS_DIR>/<task>/<exp>/tool_masks/BAD.json`：`{task,exp,status:"bad",reason,annotator_id,flagged_at}`。

### 4.4 最终产物（prompts_to_masks 产，下游兼容）
`<exp>/tool_masks/`：
- `masks.h5` — dataset `masks`，`(N_frames, N_cams, H, W)` uint8，chunk `(1,1,H,W)` gzip。**全角色 label**：0=背景，1=主工具，2/3=辅助/对象。
- `objects.yaml` — **只列被追踪的主工具**（`generate_meta.py` 优先按它定 `object_ids`，与 mask 标签数无关 → FoundationPose 只追主工具）。
- `roles.yaml` — 记录全角色 `role→object_id→name→tracked`，供 label 2/3 的接触/遮挡推理。

---

## 5. 角色模型（多物体）

三个固定语义角色，顺序决定 object_id：

| object_id | role | 中文 | 名称来源 | 追踪 |
|-----------|------|------|---------|------|
| 1 | `primary_tool` | 主操作工具（手交互） | `mesh_name_mapping.json` 映射 | ✅ FoundationPose + 手物联合优化 |
| 2 | `auxiliary_tool` | 辅助工具（常缺省） | 通用占位名 | ❌ 仅 mask |
| 3 | `manipulated_object` | 操作对象 | 通用占位名 | ❌ 仅 mask |

- 志愿者**从零判定**每角色是否存在（勾「存在」），各角色可在**不同关键帧**上标注。
- `object_id` 由 `prompts_to_masks` 在 **exp 级**按所有相机的角色并集、固定顺序**连续分配**（辅助缺省时对象压缩为 2，无空洞），保证全相机一致、主工具恒为 1。

---

## 6. 云端 API

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/` | 标注 UI |
| GET | `/api/task/next?annotator_id=` | 锁定并返回下一个待标 exp（8 相机 + 角色 + 关键帧/缩略图 url） |
| GET | `/api/task/{id}?annotator_id=` | 加载指定 exp（切换视频；空闲则锁定，已标/坏也可调出复查） |
| GET | `/api/tasks?annotator_id=` | 全部 exp 列表 + 状态 + 是否本人标注 |
| GET | `/api/progress?annotator_id=` | `{total, submitted, bad, mine, remaining}` |
| GET | `/api/image/{id}/{camera}/{frame}` | 关键帧 jpg |
| GET | `/api/thumb/{id}/{camera}/{frame}` | 浏览缩略图 jpg |
| POST | `/api/preview` | `{task_id,camera,frame_index,role,points,labels,box}` → 按角色配色的 mask 叠加 + IoU |
| POST | `/api/submit` | `{task_id,annotator_id,cameras:{cam:[objects(含frame_index)]}}` → 写每相机 prompt JSON，标 submitted |
| POST | `/api/flag_bad` | `{task_id,annotator_id,reason}` → 状态置 bad（不再派发）+ 写 BAD.json |
| GET | `/api/stats` | 按状态计数 |

任务状态机：`unassigned → locked(30min超时回收) → submitted | bad`。

---

## 7. 前端功能（`frontend/index.html`，纯前端）

- **8 视角网格**（4×2），每相机一个格子；相机名（左上，点击聚焦）+ 关键帧按钮（底部 `f0/f67/f133` 选标注帧）叠在图上。
- **三色角色**：🔴主工具 / 🟢辅助 / 🔵对象；点/掩码同色；卡片含「存在」勾选 + 已标视角数 + 平均 IoU。
- **左键正点 / 右键负点**；「预览当前角色(全视角)」并发对所有打点视角解码。
- **浏览栏**：聚焦相机的关键帧（蓝框）+ 中段缩略图滤片，拖看识别工具。
- **顶栏**：进度（总/已标/我标/坏/剩）+ 视频下拉切换（带状态图标）+ 志愿者名。
- 按钮：提交整组 / 跳过下一组 / **⚠标记坏视频** / 撤销点 / 清空当前角色。

---

## 8. 运行方式

```bash
# ① 内网 GPU：预计算（mp4 或 h5 均可直读；--all 批量）
conda activate hocap-annotation
python internal/precompute_embeddings.py --all /data/robotool/videos_0102 --bundle /tmp/va_real
python internal/validate_seam.py --bundle /tmp/va_real --device cpu      # 可选：验证接缝

# ② 云端：建任务库 + 起服务
python cloud/seed_tasks.py --bundle /tmp/va_real --db /tmp/va_real_tasks.db
BUNDLE_DIR=/tmp/va_real PROMPTS_DIR=/tmp/va_real_prompts DB_PATH=/tmp/va_real_tasks.db \
  uvicorn app:app --host 127.0.0.1 --port 8077      # 在 cloud/ 目录下运行

# ③ 志愿者标注后，GPU 侧定稿（对每个 exp）
python internal/prompts_to_masks.py --exp /data/robotool/videos_0102/<exp>   # 全长双向传播
#   产物 masks.h5 + objects.yaml 直接进入现有 run_task_folder.sh / generate_meta.py
```
### 服务器批量生成 + 复制到本地

生成是 **per-exp 独立**任务，按服务器环境二选一（两个脚本都会写 `manifest_*.json` 分片再合并成 `manifest.json`）：

**A. viscam SLURM 集群（推荐，对齐现有 `scripts/sbatch_run_hand_array.sh`）**
```bash
# frontend：扫描 exp → 写 manifest → sbatch --array(每 exp 一块 gpu:1) → 依赖合并作业
bash volunteer_annotation/internal/sbatch_precompute_array.sh \
    --data_root /viscam/projects/robotool/data \
    --bundle    /viscam/projects/robotool/_va_bundle \
    --max_concurrent 16 [--refmask] [--keyframe_fracs 0,0.1,0.2]
#   - 自动跳过 *_annotated；child 每 element 跑 precompute --exp 写 manifest_<id>.json
#   - 结束后依赖作业自动 merge_manifests.py → manifest.json
#   - SBATCH header 已对齐 sbatch_run_hand_array.sh(viscam 账号、chenrq slurm_outs、
#     mem64G/cpus4/time12h/同 exclude)；只需确保 HOCAP_ROOT 指向集群上的本仓库、
#     slurm_outs 目录存在(mkdir)。env 默认 HOCAP_ROOT/CONDA_SH 也是 chenrq 集群路径。
#   - --dry_run 只打印将提交的 sbatch 命令

# B. 普通多 GPU 单机（非 SLURM）：
cd volunteer_annotation/internal
bash generate_bundle.sh /data/robotool/videos_0102 /data/robotool/_bundle 0,1,2,3
#   - 按 GPU 列表分片(exp 取模)，每卡一进程，结束自动 merge；日志 <bundle>/shard_k.log；幂等可重跑
```
两者均默认 `--no_refmask`（接缝已验证）；多个数据根目录就多跑几次（同一 `--bundle`，manifest 累积合并）。

**Resume（分文件夹保存）**：每个 exp 完成后在自己文件夹写 `_manifest.json`（最后一步写，故中断的 exp 没有它、会被重做）。两个脚本都带 `--skip_existing`：重跑时跳过已有 `_manifest.json` 的 exp；SLURM frontend 还会在扫描阶段直接把已完成的 exp 排除出数组（`16 found, 2 done, 14 pending → --array=0-13`）。顶层 `manifest.json` 每次由 `merge_manifests.py` 扫描**所有** exp 文件夹片段重建，所以跨多次/多进程运行能正确累积，无需担心覆盖。

# ② 复制到本地(在本地执行，pull；rsync 增量+断点续传；embedding 已压缩故不用 -z)
rsync -a --partial --info=progress2 user@SERVER:/data/robotool/_bundle/ /data/robotool/_bundle/
#   边生成边传：生成期间在本地循环 rsync 即可(增量)，生成结束后再 rsync 一次确保 manifest.json 最新

# ③ 本地/云：seed + 起服务(指向同一 bundle)
python cloud/seed_tasks.py --bundle /data/robotool/_bundle --db tasks.db
```
体量与时间(参考)：每 exp ≈ 150MB(3 关键帧×8 相机)、~1.2s/相机编码。
50–300 exp → bundle ~7.5–45GB；2–4 卡并行约 2–24 分钟。减到 2 关键帧约省 1/3。

环境/路径：SAM2 在 `mesh_reconstruction/sam2`；image 解码 cfg = hydra 名 `configs/sam2.1/sam2.1_hiera_l.yaml`；video 传播 cfg = 文件路径 `HO-Cap-Annotation/config/sam2_config/sam2.1_hiera_l.yaml`；ckpt `mesh_reconstruction/sam2/checkpoints/sam2.1_hiera_large.pt`。fastapi 0.129 + uvicorn 已在 `hocap-annotation` 环境。

---

## 9. 已验证

| 项 | 结果 |
|----|------|
| 接缝（CPU 解码 vs 原生 SAM2） | 240 关键帧 × 3 点，mean IoU **0.9925**，[SEAM OK] |
| precompute 真实数据 | 80 相机 × (3 关键帧+2 缩略图)，196s，主工具名映射正确 |
| 全流程（含云 API 提交→双向传播→masks.h5） | 8 相机 × 667 帧跑通；`generate_meta.load_object_names_from_yaml` 实测读出 `[green_plastic_chopstick]` |
| 后续帧标注 + 双向传播 | 筷子标在 frame 222 → mask 在 f0 为空（未拿起，正确）、使用时出现 |
| 并发预览竞态 | 加锁后串行 vs 并发 8 相机 mask 面积逐一一致，0 错配 |
| 坏视频标记 | 标记后状态=bad、写 BAD.json、派发跳过 |
| 进度/切换 | per-annotator `mine` 计数正确；按 id 切换加载正常 |

---

## 10. 已知限制与设计决策

- **预览保真度**：个别低置信度模糊**单点**在 bf16 编码+fp16 存储 vs CPU-fp32 下会发散（min IoU 0.63）。**仅影响实时预览**——最终 mask 由 GPU 全精度从存下的点重生成。真实多点标注稳定。若要更清晰预览：fp32 编码+存储（存储翻倍 ~12MB/帧）。
- **细物体**：筷子等细长物体单点易蹭到手，需多点 + 负点排除。
- **帧索引约定**：志愿者标 mp4 绝对帧（含 0），masks.h5 全长；下游 `generate_meta` 按 meta.yaml `start_frame` 切片（志愿者侧不处理 start_frame）。
- **辅助/对象**：仅产 mask，不进 FoundationPose（用户决定），故 objects.yaml 只列主工具。
- 多物体在某物体 cond 帧上偶有另一物体瞬时掉帧（SAM2 多物体交互瑕疵）。

---

## 11. 剩余工作（部署期）

1. **overlay 懒加载**：云 ↔ 本地 1T 的 Tailscale/WireGuard 通道 + `embedding_cache.py`（LRU，按活跃任务池从本地拉 embedding）——万帧规模 + 50G 云盘的关键件。当前云端直接读本地 bundle 目录。
2. **质控**：gold 资格任务、自动启发式（IoU/面积过滤）、审核队列；BAD.json 在批处理中 skip。
3. **ONNX 解码器**：把 torch 从云端剥离，部署最轻。
4. 招募运营：token 登录、计酬报表（按 `mine`/approved 统计）。
