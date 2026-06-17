# 志愿者 SAM2 点 Prompt 标注 Pipeline — 架构设计

> 状态：架构设计稿（未开始编码）
> 适用：招募远程志愿者，为 SAM2 提供点 prompt，产出与现有 RoboTool 标注流水线兼容的 `masks.h5`。

---

## 1. 目标与范围

招募远程志愿者，在多视角 RGB-D 序列的**每个相机视角首帧**上点击正/负点，为 SAM2 提供 prompt；服务端据此复现 SAM2 并传播到整段视频，产出下游所需的 `masks.h5`。

**本 pipeline 替代现有 `mesh_reconstruction/sam2/notebooks/batch_task_annotator_multi.py` 这一步**，下游 `generate_meta.py` / `my_cluster_loader.py` 等**零改动**。

### 已锁定的设计决策

| 维度 | 决策 |
|------|------|
| 志愿者接入方式 | 浏览器 Web 应用（无需安装环境） |
| 交互模型 | 打完点 → 点「预览」跑一次 SAM2 → 不满意补点再预览 → 提交 |
| 标注范围 | 每相机视角首帧（frame 0）；后续帧由 SAM2 自动传播 |
| 数据规模 | 大（万帧以上）→ embedding bundle 上百 GB |
| 云盘容量 | 小（≤ 50 GB）→ 云端只能滚动缓存活跃任务池 |
| 并发量 | 低（个位数）→ 云端解码器同步处理、零队列 |

---

## 2. 核心洞察：拆分 SAM2 的「重编码」与「轻解码」

SAM2 图像预测分两段，算力极不对称：

| 部分 | 算什么 | 成本 | 依赖 |
|------|--------|------|------|
| **Image Encoder**（Hiera backbone） | 图像 → embedding | 重，需 GPU | 只依赖图像，**与点击无关** |
| **Mask Decoder** | embedding + 点 → mask | 极轻，CPU 毫秒级 | 依赖 embedding + 点 |

志愿者反复加点/预览时，真正反复跑的只有**解码器**（轻）；编码器与点无关，对每张首帧只算一次 → **提前在内网 GPU 上批量预计算 embedding**，下沉到云端。

> 这是 Meta 官方 SAM 网页 demo 的标准做法：服务端预计算 image embedding，轻量端只跑解码器。

**推论**：公网云服务器**永不需要 GPU，也永不接触原始视频**——只拿「首帧 JPEG + 预计算 embedding」，跑解码器做交互预览。

---

## 3. 硬件拓扑与机器角色

三台机器构成「内网批处理 ↔ 本地枢纽 ↔ 公网交互」三段，**内网与云永不直连**。

```
   [内网 GPU 服务器]                [本地 1T 电脑]                  [公网云服务器]
   离线·有GPU·全量原始数据            枢纽·权威存储·常在线            在线·无GPU·≤50G盘
   ─────────────────              ─────────────────             ─────────────────
   ① 预计算全部首帧embedding ──批量push(user发起, 内网仅打包时在线)──▶  全量embedding(上百GB)
                                          │  ▲                         滚动缓存(LRU,≤50G)
                                          │  │  ② 云按活跃池懒加载        ◀── overlay ──┐
                                          │  │  ◀────拉取──────────────────────────────┤
                                          │  │                                        │
                                          │  │  ③ prompt 回流(overlay自动)             │
   ④ 批量拉prompt ◀──user发起批量────────┤  └───────────────────────────────────────  志愿者
      全视频GPU传播 → masks.h5              prompt持久备份                   ↕ 解码器预览(CPU同步)
      → 现有下游(零改动)
```

| 机器 | 角色 | 跑什么 | 存什么 |
|------|------|--------|--------|
| **内网 GPU 服务器** | 批处理后端（离线） | ① 预计算首帧 embedding；④ 最终全视频 SAM2 传播 → `masks.h5` | 全量原始数据（H5 视频） |
| **本地 1T 电脑** | 枢纽 / 权威存储 / 中继 | overlay 端点；prompt 持久化 | 全量 embedding bundle；prompt 备份 |
| **公网云服务器** | 交互前端（在线） | FastAPI + 前端 + CPU 解码器 | 首帧 JPEG + embedding LRU 缓存 + prompt + 任务库 |

### 连接方式

- **内网 ↔ 本地**：用户发起的**批量传输**（scp/rsync）。内网只需在「打包 embedding」和「拉取 prompt 定稿」两个时刻在线，平时离线不影响志愿者标注。
- **本地 ↔ 云**：常驻**按需通道**。本地大概率无公网 IP，但能出站连云 → 用**本地出站建链的 overlay**（推荐 Tailscale / WireGuard，备选 `frp` 反向隧道）。本地不暴露任何公网端口；云经 overlay 直接 `rsync`/HTTP 拉取本地 embedding，并回写 prompt。
  - 备选（不引入 overlay）：本地跑 agent **轮询**云端预取请求队列并主动上传——纯出站、零开放端口，但需多写一个 agent。

---

## 4. 关键机制

### 4.1 embedding 分层存储 + 云端滚动缓存

- **本地 1T = 权威全量存储**（上百 GB，1T 容得下），带 manifest 索引。
- **云端 = LRU 滚动缓存**，只持有「当前活跃 + 预取窗口内」的 embedding。
- **工作集估算**：embedding ≈ 8 MB/帧（fp16）。50 GB ÷ 8 MB ≈ **6000+ 帧可同时缓存**；并发个位数 → 实际活跃工作集仅几十帧。预取窗口可开得很大，**驱逐几乎不触发**。万帧总量在云端完全不构成问题，因为云端从不需要全量持有。

### 4.2 任务调度：滑动预取窗口（Wave 式发放）

万帧任务不能一次性全部激活。调度器维护有序任务队列 + 滑动窗口：

- 只把**窗口内**任务的 embedding 预取到云端 LRU；
- 发放任务从窗口头部取，窗口随完成度向前滑动；
- 某帧的所有 (视角 × 物体) 任务全部完成 → 触发该帧 embedding 驱逐。

云端缓存始终只覆盖「正在标 + 即将标」的一小段，与总规模解耦。

### 4.3 低并发简化

并发个位数 → 云端解码器**单请求同步处理**：无需请求队列、无需多实例、无需 GPU 批处理。一个 FastAPI worker + 内存里缓存当前帧 embedding 即可。

---

## 5. 数据流（4 条 + 跨网体量）

```
① 预计算 (内网GPU, 离线)
   imgs[0, cam]  ──Hiera encoder──▶  embedding(.npz, fp16) + frame0.jpg + manifest
                                       │
② 分发 (单向, 派生数据)                  ▼
   内网 ──批量push──▶ 本地1T(权威全量) ──overlay懒加载──▶ 云端LRU
                                       │
③ 交互 (云, 在线, 无GPU)                 ▼
   志愿者 ◀──mask预览(CPU解码)──▶ 云  ──产出──▶ prompt JSON ──overlay──▶ 本地备份
                                       │
④ 回流 & 定稿 (内网GPU, 离线)            ▼
   本地prompt ──user批量拉──▶ 内网 ──全视频传播(GPU)──▶ masks.h5 ──▶ 现有下游
```

| 内容 | 单位体量 | 方向 | 说明 |
|------|---------|------|------|
| frame0 JPEG | ~200 KB/帧 | 内网→本地→云 | 1280×720 |
| embedding (fp16) | **~8 MB/帧** | 内网→本地→云 | image_embed + 2× high_res_feats |
| prompt JSON | ~几 KB/帧 | 云→本地→内网 | 志愿者产出 |
| **原始全量视频** | **GB 级** | **永不跨网** | 留在内网，隐私 + 带宽双赢 |

---

## 6. 数据格式与存储 Schema

### 6.1 embedding bundle（内网产，本地存）

每个 (exp, camera) 的首帧一个条目：

```
<bundle_root>/
├── manifest.json                       # 全量任务索引（见下）
└── <task>/<exp>/
    ├── cam0_rgb.embed.npz              # fp16: image_embed, high_res_feat_0, high_res_feat_1
    ├── cam0_rgb.jpg                    # 首帧显示图
    ├── cam1_rgb.embed.npz
    └── ...
```

`*.embed.npz` 内容（对应 SAM2 `ImagePredictor._features`）：

| key | 形状（hiera_large, 1024 输入） | dtype |
|-----|------|------|
| `image_embed` | [256, 64, 64] | fp16 |
| `high_res_feat_0` | [32, 256, 256] | fp16 |
| `high_res_feat_1` | [64, 128, 128] | fp16 |
| `orig_hw` | [2] (H, W 原生分辨率) | int32 |

> 解码器需用 `orig_hw` 复现 SAM2 的坐标变换（原生像素点 → 1024 空间）。

`manifest.json`：

```json
{
  "schema_version": 1,
  "sam2_model": "sam2_hiera_large.pt",
  "embed_dtype": "float16",
  "frames": [
    {
      "task": "stir_dough", "exp": "exp_0315_01",
      "camera": "cam0_rgb", "cam_index": 0, "frame_index": 0,
      "width": 1280, "height": 720,
      "image_sha1": "ab12…",
      "objects": ["fork", "dough"]
    }
  ]
}
```

### 6.2 prompt JSON（志愿者产出，新的一等公民产物）

存放于每个 exp 的 `tool_masks/prompts/<camera>.json`：

```json
{
  "schema_version": 1,
  "task": "stir_dough", "exp": "exp_0315_01",
  "camera": "cam0_rgb", "cam_index": 0, "frame_index": 0,
  "width": 1280, "height": 720,
  "image_sha1": "ab12…",
  "sam2_model": "sam2_hiera_large.pt",
  "annotator_id": "vol_017",
  "submitted_at": "2026-06-17T10:22:00Z",
  "review_status": "approved",
  "objects": [
    {
      "object_id": 1, "name": "fork",
      "points": [[412, 530], [455, 560], [600, 410]],
      "labels": [1, 1, 0],
      "box": null,
      "preview_iou": 0.94
    },
    {
      "object_id": 2, "name": "dough",
      "points": [[700, 600]], "labels": [1], "box": null, "preview_iou": 0.88
    }
  ]
}
```

字段约定：
- `points`：**原生分辨率像素坐标**（前端缩放显示时换算回原生）。
- `labels`：1=正点，0=负点。
- `image_sha1`：与 manifest 中一致，检测底层数据漂移；不一致则该 prompt 失效需重标。
- `review_status`：`submitted` → `approved` / `rejected`。

**为什么存原始点击而非仅 mask**：
- 可复现/可重跑：换更强 SAM2 权重可批量重生成 mask，无需重新招募。
- 可审计/可纠错：审核员能看到志愿者意图。
- 轻量：KB 级，便于冗余对比与版本管理。

### 6.3 最终产物（与现有下游字节级兼容）

由内网 `prompts_to_masks.py` 生成于 `tool_masks/`：

| 文件 | 格式 | 说明 |
|------|------|------|
| `masks.h5` | dataset `"masks"`，shape `(N_frames, N_cams, H, W)`，uint8，gzip，chunk `(1,1,H,W)` | label mask：0=背景，k=第 k 个物体 |
| `objects.yaml` | `objects: [fork, dough, ...]` | 物体名列表 |

下游通过 `mask == (object_idx + 1)` 提取单物体二值 mask（沿用现状）。

---

## 7. 后端 API（公网云）

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/auth/login` | Token 登录，返回匿名 `annotator_id`（不收 PII） |
| GET | `/task/next` | 从滑动窗口取一个 `unassigned` 任务并加锁（带超时） |
| GET | `/image/{exp}/{camera}` | 返回首帧 JPEG（从 LRU 缓存） |
| POST | `/preview` | 入参 `{exp, camera, object_id, points, labels, box}` → CPU 解码器出 mask（PNG/RLE）+ `preview_iou` |
| POST | `/submit` | 持久化 prompt JSON，跑自动质控，标记任务状态 |
| GET | `/review/queue` | 管理端：待审核任务（传播后多帧叠加图） |
| POST | `/review/{id}` | 管理端：通过 / 打回 |

---

## 8. 质量控制（三层）

1. **资格关**：正式标注前做 1~2 个有标准答案的 gold 任务，提交 mask 与标准 IoU 低于阈值则不放行。
2. **自动启发式**（`/submit` 时，云端）：`preview_iou` 过低 / mask 面积过小或过大（贴边、全图） / 点数为 0 → 自动 `rejected` 重派。
3. **冗余 + 人工抽审**：可选同一任务双人标注，比对两份 mask IoU；管理端审核队列展示传播后若干帧叠加图，一键通过/打回。

招募侧轻量化：Token 登录、匿名 ID、说明页（正/负点图示 + 好坏对比）、计酬按 `approved` 任务数从任务库出报表。

---

## 9. 组件清单（按机器）

| 机器 | 组件 | 职责 |
|------|------|------|
| **内网 GPU** | `precompute_embeddings.py` | 全量首帧 → embedding(fp16) + JPEG + manifest |
| | `prompts_to_masks.py` | 批量拉 prompt → 全视频 `SAM2VideoPredictor` 传播 → `masks.h5` + `objects.yaml` |
| **本地 1T** | embedding store | 权威全量 embedding + manifest 索引 |
| | overlay 端点 + prompt 仓库 | 对云提供按需拉取；接收并持久化 prompt |
| **公网云** | `app.py`（FastAPI） | 鉴权 / 任务发放 / `/preview` / `/submit` / 审核 |
| | `scheduler.py` | 滑动窗口任务调度 + 预取触发（SQLite 任务库） |
| | `decoder.py`（onnxruntime / torch-cpu） | embedding + 点 → mask，CPU 同步 |
| | `embedding_cache.py` | LRU 缓存 + 经 overlay 从本地拉取 |
| | `frontend/index.html` | Canvas 打点 + 预览 + 提交 |
| | `seed_tasks.py` | 扫描 manifest 生成任务库（exp × 相机 × 物体） |

> 解码器建议导出为 **ONNX**：云端只需 `onnxruntime + numpy`，无 torch/CUDA 依赖，部署最轻。

### 建议代码落位

```
HO-Cap-Annotation/volunteer_annotation/
├── ARCHITECTURE.md            # 本文档
├── internal/                  # 内网 GPU 跑
│   ├── precompute_embeddings.py
│   └── prompts_to_masks.py
├── cloud/                     # 公网云跑
│   ├── app.py
│   ├── scheduler.py
│   ├── decoder.py
│   ├── embedding_cache.py
│   ├── seed_tasks.py
│   └── frontend/index.html
└── local/                     # 本地 1T 跑
    └── overlay/               # Tailscale/WireGuard 配置或轮询 agent
```

---

## 10. 落地阶段（风险从高到低排序）

1. **接缝验证（最高风险）**：`precompute_embeddings.py`（内网产 embedding）+ 一段离线脚本：加载 embedding + 手填点 → ONNX 解码器出 mask，**确认离线 embedding + 解码器复现的 mask 与原生 SAM2 一致**。整个架构最不确定的技术点，先打通。
2. **定稿接缝**：`prompts_to_masks.py`，确认 prompt JSON → `masks.h5` 与现有 `generate_meta.py` / loader 字节级兼容。
3. **云后端骨架**：FastAPI + SQLite 任务库 + 同步解码器 + LRU 缓存（先用本地文件模拟 overlay 拉取）。
4. **连接层**：接入 Tailscale/WireGuard，打通云懒加载 + prompt 回流。
5. **前端 Canvas** + 调度器滑动窗口 + 质控。

---

## 11. 与现有流水线的衔接

```
志愿者标注 (本 pipeline)
   └─▶ prompts/*.json ──▶ prompts_to_masks.py ──▶ masks.h5 + objects.yaml
                                                       │
                                                       ▼
   现有主流水线: run_task_folder.sh / generate_meta.py / 01..08 (零改动)
```

本 pipeline 完整替代 `batch_task_annotator_multi.py` 这一步，产物格式与现状一致。
