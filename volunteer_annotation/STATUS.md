# 志愿者 SAM2 标注 Pipeline — 实现状态

> 截至 2026-06-21。本文档记录**已实现并验证**的系统现状与部署配置。

---

## 1. 一句话现状

招募志愿者，在浏览器里**一次性给一个 exp 的 8 个相机视角**用点 prompt 标注三类角色物体（可在多个关键帧中选帧打点），服务端用 CPU 实时预览 SAM2 掩码；提交的点经 GPU 侧全视频双向传播，产出与现有 RoboTool 流水线**字节级兼容**的 `masks.h5`。

**当前状态（2026-06-21）：系统已全面部署上线。**

- 云服务器 `http://49.233.81.226:8077` 可公网访问，志愿者可直接打开浏览器标注
- 本地 push agent 实时向云端推送 embedding，按需供给（无需预先推全量）
- 任务库已 seed 4068 个 exp，均来自 `videos_0101` 的真实 bundle

---

## 2. 三机拓扑

```
[viscam 集群 (内网 GPU)]        [本地 4090 (枢纽·权威存储)]     [云服务器 ubuntu@49.233.81.226]
离线·GPU·全量数据                /data/robotool/_va_bundle_v2     公网 http://49.233.81.226:8077
─ precompute_embeddings          ─ 权威 embedding bundle (911G)    ─ FastAPI + CPU 解码器
─ sbatch_precompute_array.sh     ─ manifest.json (4068 exp)        ─ 8 视角标注 UI
─ orchestrate.py (批量管理)      ─ push_agent.py (按需推送)        ─ tasks.db (SQLite)
─ prompts_to_masks (传播)        ─ tmux va_push (常驻进程)         ─ tmux va_server (常驻进程)
                                                                    ─ embedding 缓存 (按需，≤42G)
                                      ↕ rsync over SSH
                                      按需推: locked exp 优先
```

核心洞察：SAM2 编码器（重，GPU，与点无关）离线预计算 → embedding 下沉云端；解码器（轻，CPU ~20ms）做交互预览。**云端永不需要 GPU / 不接触原始视频**。

---

## 3. 已部署环境

### 3.1 云服务器 `ubuntu@49.233.81.226`

```
SSH 别名: va-cloud  (写在 ~/.ssh/config)
公网端口: 8077 (腾讯云安全组已开放 TCP 8077)
磁盘: 50G 总计, ~42G 可用
```

**已安装软件（pip，Python 3.10.12）**:
| 包 | 版本 |
|----|------|
| fastapi | 0.138.0 |
| uvicorn | 0.49.0 |
| torch (CPU only) | 2.12.1+cpu |
| torchvision | 0.27.1+cpu |
| sam2 | 从本地源码安装至 `/tmp/sam2_src/` |
| numpy | 2.2.6 |
| opencv-python-headless | 4.13.0.92 |

**已部署文件**:
```
~/volunteer_annotation/
├── cloud/                        # FastAPI 服务端代码
│   ├── app.py
│   ├── tasks_db.py
│   ├── decoder.py
│   ├── seed_tasks.py
│   └── frontend/index.html
├── data/
│   ├── manifest.json             # 4068 exp, 32544 cameras
│   ├── tasks.db                  # SQLite, 4068 tasks (all unassigned initially)
│   └── <task>/<exp>/             # embedding 缓存 (push_agent 按需推入)
├── prompts/                      # 志愿者提交的 prompt JSON
└── sam2.1_hiera_large.pt         # SAM2 checkpoint (~898MB)
```

**启动命令（tmux `va_server`）**:
```bash
tmux attach -t va_server         # 查看服务状态
# 重启命令:
BUNDLE_DIR=$HOME/volunteer_annotation/data \
PROMPTS_DIR=$HOME/volunteer_annotation/prompts \
DB_PATH=$HOME/volunteer_annotation/data/tasks.db \
SAM2_CHECKPOINT=$HOME/volunteer_annotation/sam2.1_hiera_large.pt \
SAM2_CFG=$HOME/volunteer_annotation/sam2_src/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
PYTHONPATH=$HOME/volunteer_annotation/cloud \
  ~/.local/bin/uvicorn app:app --host 0.0.0.0 --port 8077 --workers 1 \
  --app-dir $HOME/volunteer_annotation/cloud 2>&1 | tee /tmp/va_server.log
```

> ⚠️ 注意：sam2 安装在 `/tmp/sam2_src/`，服务器重启后 `/tmp` 会清空。需重新安装（见下方故障排查）。

### 3.2 本地 4090（枢纽机）

**Push agent（tmux `va_push`，常驻）**:
```bash
# 当前运行命令:
no_proxy='*' NO_PROXY='*' \
python volunteer_annotation/internal/push_agent.py \
  --bundle /data/robotool/_va_bundle_v2 \
  --cloud va-cloud \
  --cloud_dir '~/volunteer_annotation/data' \
  --api http://49.233.81.226:8077 \
  --poll 30 --prefetch 8

# no_proxy 必须设置，否则 requests 走本机代理(7897)超时
```

**本地 bundle 状态**:
- 路径: `/data/robotool/_va_bundle_v2`
- 内容: `videos_0101`（Jan 数据，18 个子文件夹）+ `manifest.json`
- 大小: ~911G
- manifest: 4068 exp, 32544 cameras

### 3.3 SSH 配置（`~/.ssh/config`）

```
Host va-cloud
  HostName 49.233.81.226
  User ubuntu
```

---

## 4. 文件清单

```
HO-Cap-Annotation/volunteer_annotation/
├── ARCHITECTURE.md                    # 最初设计稿
├── STATUS.md                          # 本文档（实现现状）
├── internal/                          # GPU 侧（viscam 集群 / 本地 4090）
│   ├── precompute_embeddings.py       # frame → 关键帧 embedding/jpg + 缩略图 + manifest
│   ├── validate_seam.py               # 验证 CPU 解码 == 原生 SAM2（接缝测试）
│   ├── prompts_to_masks.py            # prompt JSON → 全视频双向传播 → masks.h5
│   ├── orchestrate.py                 # 批量扫描 exp + sbatch 提交(含 --add_frames)
│   ├── push_agent.py                  # ★ 本地常驻：按需 rsync embedding 到云端
│   ├── sbatch_precompute_array.sh     # SLURM array 批量生成
│   ├── sbatch_orchestrate_addframes.sh # SLURM 补关键帧 (50%/100%)
│   ├── merge_manifests.py             # 合并各分片 manifest_k.json → manifest.json
│   ├── generate_bundle.sh             # 普通多卡单机批量生成（非 SLURM 备选）
│   └── export_ref_frames.py           # (已被缩略图取代) 补充参考帧
└── cloud/                             # 云侧（ubuntu@49.233.81.226）
    ├── app.py                         # FastAPI：12 个端点（任务/图像/预览/提交/推送管理）
    ├── tasks_db.py                    # SQLite 任务库（按 exp 一行）+ 角色花名册派生
    ├── decoder.py                     # Sam2CpuDecoder：注入 embedding + 解码（带并发锁）
    ├── seed_tasks.py                  # 从 manifest 建任务库
    └── frontend/index.html            # 8 视角 Canvas 标注界面（纯前端）
```

---

## 5. 数据格式

### 5.1 embedding bundle（schema 2）
```
<bundle>/
├── manifest.json                       # 顶层汇总（由各 exp 的 _manifest.json 合并）
└── <task>/<exp>/
    ├── _manifest.json                  # ★ 该 exp 的 manifest 片段（resume 依据）
    ├── cam{c}_rgb.kf{idx}.embed.npz    # fp16: image_embed[256,64,64] + high_res_feat_0/1 (~6MB)
    ├── cam{c}_rgb.kf{idx}.jpg          # 关键帧全分辨率图（标注用）
    └── cam{c}_rgb.th{idx}.jpg          # 中段浏览缩略图（~13KB，无 embedding）
```

`manifest.json` 结构:
```json
{ "schema_version": 2, "sam2_model": "sam2.1_hiera_large.pt", "embed_dtype": "float16",
  "cameras": [ { "task","exp","camera","cam_index","width","height","n_frames",
                 "primary_name","keyframes":[0,67,133],"thumbs":[266,400],"image_sha1" } ] }
```

### 5.2 prompt JSON（志愿者产出）
`<PROMPTS_DIR>/<task>/<exp>/tool_masks/prompts/<camera>.json`：
```json
{ "schema_version":1, "task","exp","camera","cam_index","frame_index":0,
  "image_sha1","sam2_model","annotator_id","submitted_at","review_status":"submitted",
  "objects":[
    {"role":"primary_tool","name":"green_plastic_chopstick","frame_index":67,
     "points":[[x,y],...],"labels":[1,0,...],"box":null,"preview_iou":0.9}
  ] }
```

### 5.3 最终产物（`prompts_to_masks` 产）
`<exp>/tool_masks/`:
- `masks.h5` — `(N_frames, N_cams, H, W)` uint8；0=背景，1=主工具，2/3=辅助/对象
- `objects.yaml` — 只列被追踪的主工具（`generate_meta.py` 据此定 `object_ids`）
- `roles.yaml` — 全角色 `role→object_id→name→tracked`

---

## 6. 云端 API

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/` | 标注 UI |
| GET | `/api/task/next?annotator_id=` | 锁定并返回下一个待标 exp |
| GET | `/api/task/{id}?annotator_id=` | 加载指定 exp（切换视频） |
| GET | `/api/tasks?annotator_id=` | 全部 exp 列表 + 状态 |
| GET | `/api/progress?annotator_id=` | `{total,submitted,bad,mine,remaining}` |
| GET | `/api/image/{id}/{camera}/{frame}` | 关键帧 jpg |
| GET | `/api/thumb/{id}/{camera}/{frame}` | 浏览缩略图 jpg |
| POST | `/api/preview` | 点 prompt → 实时 SAM2 mask 叠加 + IoU |
| POST | `/api/submit` | 提交全部相机 prompt JSON |
| POST | `/api/flag_bad` | 标记坏视频（状态=bad，写 BAD.json） |
| GET | `/api/stats` | 按状态计数 |
| GET | `/api/needs_embeddings?limit=` | push_agent 轮询：返回缺 embedding 的 exp 列表 |
| GET | `/api/embedding_ready/{task_id}` | 检查单个 exp embedding 是否就绪 |
| DELETE | `/api/bundle/{task}/{exp}` | 删除云端已完成 exp 的 embedding（释放磁盘） |

任务状态机：`unassigned → locked(30min超时回收) → submitted | bad`

---

## 7. 运行方式（完整流程）

### 7.1 预计算 embedding（viscam 集群）

```bash
conda activate hocap-annotation
cd HO-Cap-Annotation

# SLURM array 批量（推荐）
bash volunteer_annotation/internal/sbatch_precompute_array.sh \
    --data_root /viscam/projects/robotool/data \
    --bundle    /viscam/projects/robotool/_va_bundle_v2 \
    --max_concurrent 16

# 补充额外关键帧（50%/100% frame，用于形变物体）
sbatch volunteer_annotation/internal/sbatch_orchestrate_addframes.sh

# 合并 manifest 分片（sbatch 结束后自动触发，也可手动）
python volunteer_annotation/internal/merge_manifests.py \
    --bundle /viscam/projects/robotool/_va_bundle_v2
```

### 7.2 同步到本地（本地 4090）

```bash
# 从集群增量同步（断点续传）
rsync -a --partial --info=progress2 \
    chenrq@viscam.stanford.edu:/viscam/projects/robotool/_va_bundle_v2/ \
    /data/robotool/_va_bundle_v2/

# 合并或更新本地 manifest
python HO-Cap-Annotation/volunteer_annotation/internal/merge_manifests.py \
    --bundle /data/robotool/_va_bundle_v2
```

### 7.3 更新云端任务库（云端）

```bash
ssh va-cloud
python3 ~/volunteer_annotation/cloud/seed_tasks.py \
    --bundle ~/volunteer_annotation/data \
    --db     ~/volunteer_annotation/data/tasks.db
# 幂等，可重复执行，只添加新 exp
```

### 7.4 启动/重启云端服务

```bash
ssh va-cloud
tmux attach -t va_server   # 查看当前状态

# 如需重启（或 /tmp 被清导致 sam2 丢失）：
# 先重装 sam2（仅在 /tmp 清空后需要）
mkdir -p /tmp/sam2_src
tar xzf ~/volunteer_annotation/sam2_src.tar.gz -C /tmp/sam2_src/
cd /tmp/sam2_src && pip install --quiet -e . --no-build-isolation

# 再启动服务
tmux kill-session -t va_server 2>/dev/null
tmux new-session -d -s va_server
tmux send-keys -t va_server "
BUNDLE_DIR=\$HOME/volunteer_annotation/data \\
PROMPTS_DIR=\$HOME/volunteer_annotation/prompts \\
DB_PATH=\$HOME/volunteer_annotation/data/tasks.db \\
SAM2_CHECKPOINT=\$HOME/volunteer_annotation/sam2.1_hiera_large.pt \\
SAM2_CFG=$HOME/volunteer_annotation/sam2_src/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \\
PYTHONPATH=\$HOME/volunteer_annotation/cloud \\
  ~/.local/bin/uvicorn app:app --host 0.0.0.0 --port 8077 --workers 1 \\
  --app-dir \$HOME/volunteer_annotation/cloud 2>&1 | tee /tmp/va_server.log
" Enter
```

### 7.5 启动本地 push agent

```bash
# 本地 4090，在 HO-Cap-Annotation 目录
tmux attach -t va_push   # 查看状态
# 如需重启：
tmux send-keys -t va_push "C-c" ""
tmux send-keys -t va_push "
no_proxy='*' NO_PROXY='*' \\
python volunteer_annotation/internal/push_agent.py \\
  --bundle /data/robotool/_va_bundle_v2 \\
  --cloud va-cloud \\
  --cloud_dir '~/volunteer_annotation/data' \\
  --api http://49.233.81.226:8077 \\
  --poll 30 --prefetch 8
" Enter
```

### 7.6 志愿者开始标注

直接访问 **`http://49.233.81.226:8077`**，无需任何配置。

### 7.7 GPU 侧最终定稿

```bash
# 标注提交后，对每个 exp 跑全视频传播
conda activate hocap-annotation
python volunteer_annotation/internal/prompts_to_masks.py \
    --exp /data/robotool/videos_0101/<task>/<exp>
# 产物: masks.h5 + objects.yaml，直接进入 run_task_folder.sh
```

prompts 若已 rsync 回本地的**独立目录树**（`_va_bundle_v2_prompts/<videos_X>/<task>/<exp>/tool_masks/prompts/`，
与原始视频不同路径），用 `--prompts_dir` / `--out_dir` 指定，不必把文件拷回 exp 目录：

```bash
python volunteer_annotation/internal/prompts_to_masks.py \
    --exp         /data/robotool/videos_0101/<task>/<exp> \
    --prompts_dir /data/robotool/_va_bundle_v2_prompts/videos_0101/<task>/<exp>/tool_masks/prompts \
    --out_dir     /data/robotool/videos_0101_annotated/<task>/<exp>/tool_masks \
    --from_video --resume
```

`--from_video` 强制从 mp4 读帧（志愿者标的是 mp4 绝对帧号；exp 目录里残留的
`data00000000.h5` 可能只覆盖子区间，会把 mask 对错帧）。

### 7.8 接 FoundationPose 多视角标注

见 §14。

---

## 8. Push Agent 工作原理

`push_agent.py` 运行在本地 4090，每 30s 一次循环：

1. **磁盘检查**：云端剩余 < 25G 时，通过 SSH 查询已提交任务，调 `DELETE /api/bundle/...` 释放空间
2. **需求查询**：`GET /api/needs_embeddings?limit=13`，返回缺 embedding 的 exp（locked 优先于 unassigned）
3. **按需推送**：`rsync -a <bundle>/<task>/<exp>/  va-cloud:~/volunteer_annotation/data/<task>/<exp>/`
4. **会话去重**：本次 session 已推过的 exp 不重复推

> `no_proxy='*'` 是必须的，否则本地代理(7897)拦截请求导致超时。

---

## 9. 故障排查

### 服务启动失败：`ModuleNotFoundError: No module named 'sam2'`
`/tmp` 在服务器重启后被清空。需重新安装 sam2:
```bash
ssh va-cloud
mkdir -p /tmp/sam2_src
tar xzf ~/volunteer_annotation/sam2_src.tar.gz -C /tmp/sam2_src/
cd /tmp/sam2_src && pip install --quiet -e . --no-build-isolation
```

### 服务启动失败：checkpoint 找不到
检查环境变量 `SAM2_CHECKPOINT` 是否指向 `~/volunteer_annotation/sam2.1_hiera_large.pt`。

### Push agent 超时 `HTTPConnectionPool(host='127.0.0.1', port=7897)`
本地代理拦截了请求，启动时必须加 `no_proxy='*' NO_PROXY='*'`。

### 云端 8077 端口不通
检查腾讯云控制台安全组，确认入站规则有 `TCP 8077 0.0.0.0/0`。

### 本地 bundle 找不到某个 exp
Push agent 打印 `WARNING: local exp not found`，说明该 exp 未同步到本地。检查 `/data/robotool/_va_bundle_v2/<task>/<exp>/` 是否存在。

---

## 10. 已验证

| 项 | 结果 |
|----|------|
| 接缝（CPU 解码 vs 原生 SAM2） | 240 关键帧 × 3 点，mean IoU **0.9925**，[SEAM OK] |
| fp16 embedding + CPU 解码 | mean IoU **0.9999**（参见 validate_seam.py） |
| precompute 真实数据 | 80 相机 × (3 关键帧+2 缩略图)，196s，主工具名映射正确 |
| 全流程（含云 API 提交→双向传播→masks.h5） | 8 相机 × 667 帧跑通 |
| 后续帧标注 + 双向传播 | 筷子标在 frame 222 → mask 在 f0 正确为空 |
| 并发预览竞态 | 加锁后 8 相机并发无错配 |
| 坏视频标记 | 状态=bad、写 BAD.json、派发跳过 |
| 云端部署 | `http://49.233.81.226:8077` 可访问，4068 tasks seeded |
| Push agent 端到端 | embedding rsync 到云端，`/api/needs_embeddings` 返回正确列表 |

---

## 11. 已知限制与注意事项

- **sam2 安装位置**：`~/volunteer_annotation/sam2_src/`（永久，重启不丢失）。
- **预览保真度**：低置信度单点在 bf16/fp16 下偶有发散（min IoU 0.63），**仅影响实时预览**，最终 mask 由 GPU 全精度重生成，不受影响。
- **帧索引约定**：志愿者标 mp4 绝对帧（含 0），masks.h5 全长；下游 `generate_meta` 按 meta.yaml `start_frame` 切片。
- **辅助/对象角色**：仅产 mask，不进 FoundationPose；objects.yaml 只列主工具。
- **磁盘管理**：云端 42G，每个 exp 约 150MB（3 关键帧×8 相机）。push agent 在空间 < 25G 时自动清理已提交 exp。

---

## 12. 剩余工作

1. **质控**：gold 资格任务、自动启发式（IoU/面积过滤）、审核队列。
3. **prompts 回流**：云端 `prompts/` 定期 rsync 回本地，再跑 `prompts_to_masks.py` 批量定稿。
4. **ONNX 解码器**（可选）：把 torch 从云端剥离，部署更轻。
5. **招募运营**：token 登录、计酬报表（按 `mine`/approved 统计）。

---

## 14. 志愿者 mask → FoundationPose 多视角标注

志愿者产出的 `masks.h5` 与 DINO 自动分割的产物格式完全一致，因此**不需要另写一条流水线**：
`scripts/run_auto_annotator.sh` 的 Stage 2（DINO）在 `<annotated>/tool_masks/masks.h5`
已存在时会自动跳过，后面 generate_meta → `04-1-4_fd_pose_solver_kalman.py`（分块追踪）
→ `04-2-2_fd_pose_merger_cluster.py`（多视角融合）原样复用。

新增三个脚本把两边接上：

| 脚本 | 作用 |
|------|------|
| `scripts/volunteer_exp_index.py` | 扫 prompts 树，为每个 exp 解析出 `(sequence_folder, tool_name, tool_mesh, prompts_dir)`，输出 TSV/JSON worklist |
| `scripts/run_volunteer_annotator.sh` | 单 exp：prompts → masks.h5（Stage 0），然后原样交给 `run_auto_annotator.sh --skip_masks` |
| `scripts/sbatch_run_volunteer_array.sh` | SLURM array：每个 array task 处理 worklist 的一段 |

**tool_name 不再靠文件夹名猜**：志愿者在 UI 里选的 `primary_tool.name` 直接就是工具名
（35 个名字全部落在 `scripts/mesh_name_mapping.json` 的 42 个词表里），mesh 路径由
`mesh_name_mapping.json`（key→tool_name）与 `mapping.json`（同 key→mesh 路径）在 key 上
join 得到，1:1 无歧义。只有极少数早期 prompt 的 name 默认成了 exp 名，才回退到旧的
文件夹名子串匹配。

### 用法

```bash
# 1) 生成 worklist（登录节点即可，纯 stat）
python scripts/volunteer_exp_index.py \
    --prompts_root  /viscam/projects/robotool/_va_bundle_v2_prompts \
    --data_root     /viscam/projects/robotool/data \
    --models_folder /viscam/u/chenrq/models \
    --require_sequence --require_mesh \
    --out /viscam/u/chenrq/crq_ws/volunteer_worklist.tsv

# 2) 提交 array（每个 task 4 个 exp，最多 16 个并发）
sbatch --array=0-99%16 scripts/sbatch_run_volunteer_array.sh \
    --worklist /viscam/u/chenrq/crq_ws/volunteer_worklist.tsv \
    --exps_per_task 4 --fake_optimize

# 单 exp 本地跑（先只出 mask，验证）
bash scripts/run_volunteer_annotator.sh \
    --sequence_folder  /data/robotool/videos_0101/<task>/<exp> \
    --prompts_dir      /data/robotool/_va_bundle_v2_prompts/videos_0101/<task>/<exp>/tool_masks/prompts \
    --calibration_yaml <cal>.yaml --tool_name rubber_mallet \
    --masks_only --mask_max_frames 12
```

`BAD.json` 的 exp 在 index 与 runner 两处都会跳过。所有阶段（含 mask）都按产物存在
与否 resume，重复提交同一个 array 是安全的；`--force` 才会全部重算。

### 成本

mask 传播约 5 帧/秒/相机（4090，fp32 全精度）。2000 帧 × 8 相机 ≈ **1 GPU·小时/exp**，
3758 个 exp ≈ 3700 GPU·小时。这是目前最大的一块开销，排 array 规模时按这个量级估。
