# Real-Time Object Pose Tracking Pipeline - Implementation Plan

## Goal

Build a real-time 6D object pose tracking system that:
1. Captures live RGB-D from a RealSense camera
2. Lets the user give an initial mask (via SAM2 click) on the first frame
3. Runs FoundationPose++ (FoundationPose + Cutie 2D tracker + Kalman filter) for continuous tracking
4. Displays live pose overlay in real-time

---

## Current Setup Inventory

| Component | Conda Env | Python | PyTorch | CUDA | Key Deps |
|-----------|-----------|--------|---------|------|----------|
| RealSense camera | `camera` | 3.x | 2.5.1 | cu121 | `pyrealsense2` |
| SAM2 mask annotation | `sam2` | 3.x | 2.10.0 | ? | `sam-2` |
| FoundationPose++ tracking | `hocap-annotation` | 3.10 | 2.5.1 | cu118 | `nvdiffrast`, `pytorch3d`, `mycpp`, Cutie, KalmanFilter6D |
| Fast-FoundationStereo | `ffs` | 3.12 | 2.6.0 | cu124 | stereo depth estimation |

**GPU:** RTX 4090 (24 GB VRAM)

### What already works

- **FoundationPose++ is already integrated** into the HO-Cap pipeline via `tools/04-1-4_fd_pose_solver_kalman.py`
  - Uses the wrapper at `hocap_annotation/wrappers/foundationpose.py` (register + track_one)
  - Imports Cutie 2D tracker and KalmanFilter6D from `FoundationPose-plus-plus/src/`
  - Currently processes **pre-recorded sequences** from H5 files (batch mode, not live)
- **FoundationPose++ claims 30+ FPS** tracking on RTX 3090 (tracking = single refinement pass per frame, no scoring). RTX 4090 should be faster.
- **Camera recording** works in `camera` env via `DataCollection/scripts/record_videos.py` (RealSense D435, 640x480@30fps, aligned depth)

### What's missing for real-time

The gap is: **there is no live camera → FoundationPose++ loop**. The current pipeline reads from disk (pre-saved image sequences). We need to wire the RealSense live stream directly into the FoundationPose++ tracking loop.

---

## Key Architecture Decisions

### Q1: What role does Fast-FoundationStereo play?

Fast-FoundationStereo is a **stereo depth estimator** (left+right image → disparity → metric depth). It is **not** a pose tracker. It could optionally replace RealSense's built-in depth with higher-quality stereo depth, but:
- It adds ~30-50ms latency per frame (PyTorch) or ~15-23ms (TensorRT)
- RealSense D435 already provides aligned depth at 30 FPS
- **For v1: use RealSense depth directly. Consider FFS only if depth quality proves insufficient.**

### Q2: Do I need a unified conda env?

**Recommended: extend `hocap-annotation` with `pyrealsense2`.** This is the simplest path.

- `pyrealsense2` is a pure C++ library with Python bindings, no PyTorch dependency, installs cleanly anywhere
- SAM2 is only needed at initialization (one frame) — use a **two-phase approach**:
  1. Capture first frame + generate mask in `sam2` env (or interactively in a separate script)
  2. Run real-time tracking in `hocap-annotation` env

This avoids the torch 2.5.1 vs 2.10.0 conflict entirely.

### Q3: What's the real-time tracking architecture?

The key insight from FoundationPose++ README:

> Tracking only runs the Refinement network once per frame (no Scoring/Ranking). The 2D tracker (Cutie/OSTrack) provides xy guidance, Kalman filter smooths roll/pitch/yaw. This achieves 30+ FPS on 3090.

So the real-time loop is:
1. **Frame 0**: `register()` — full grid search + ranking (~500ms, one-time)
2. **Frame 1+**: For each frame:
   - Cutie 2D tracker → bbox center (xy guidance)
   - Kalman filter predict → smooth pose prior
   - `adjust_pose_to_image_point()` → update xy in `est.pose_last`
   - `est.track_one(rgb, depth, K, iteration=N)` → single refinement pass
   - Total: ~20-30ms per frame

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Phase 1: Init (one-shot, can be separate script/env)        │
│                                                              │
│  RealSense → capture first frame (RGB + Depth + K)           │
│       ↓                                                      │
│  SAM2 point prompt (user click) → binary object mask         │
│       ↓                                                      │
│  Save to disk: rgb.png, depth.npy, mask.npy, K.npy           │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│  Phase 2: Real-time tracking loop (hocap-annotation env)     │
│                                                              │
│  Load mesh → Init FoundationPose (scorer + refiner + glctx)  │
│  Load init frame → register(rgb, depth, mask, K) → pose_0    │
│  Init Cutie tracker with mask                                │
│  Init KalmanFilter6D with pose_0                             │
│       ↓                                                      │
│  while True:                                                 │
│    rgb, depth = RealSense.get_aligned_frame()                │
│    bbox_2d = cutie.track(rgb)                                │
│    kf.update(pose_last) + kf.update_from_xy(bbox_center)     │
│    est.pose_last = kalman_predicted_pose                     │
│    pose = est.track_one(rgb, depth, K, iteration=5)          │
│    kf.predict()                                              │
│    overlay = draw_posed_3d_box(K, rgb, pose, bbox)           │
│    cv2.imshow(overlay)                                       │
│    if 'r' pressed: re-register                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Step 0: Environment Setup

```bash
conda activate hocap-annotation
pip install pyrealsense2

# Verify
python -c "import pyrealsense2; print(pyrealsense2.__version__)"
```

No other env changes needed. SAM2 runs in its own env for init only.

### Step 1: Create `tools/realtime_init_mask.py` (runs in `sam2` env)

Small helper script that:
1. Opens RealSense camera, shows live preview
2. User presses 's' to freeze a frame
3. User clicks on the object → SAM2 `SamPredictor` generates mask from point prompt
4. Saves `rgb.png`, `depth.npy`, `mask.npy`, `K.npy` to a specified directory
5. Exits

This only needs `pyrealsense2` + `sam2` — both available in the `sam2` env (install `pyrealsense2` there too if not present).

### Step 2: Create `tools/realtime_tracker.py` (runs in `hocap-annotation` env)

This is the main script. It reuses existing components:

| Component | Source | Import Path |
|-----------|--------|-------------|
| FoundationPose estimator | `hocap_annotation/wrappers/foundationpose.py` | `FoundationPose`, `ScorePredictor`, `PoseRefinePredictor`, `dr` |
| Cutie 2D tracker | `FoundationPose-plus-plus/src/VOT.py` | `Cutie`, `Tracker_2D` |
| Kalman filter | `FoundationPose-plus-plus/src/utils/kalman_filter_6d.py` | `KalmanFilter6D` |
| Pose utilities | `tools/04-1-4_fd_pose_solver_kalman.py` | `adjust_pose_to_image_point`, `get_6d_pose_arr_from_mat`, etc. |
| Visualization | `FoundationPose/estimater.py` | `draw_posed_3d_box`, `draw_xyz_axis` |
| Camera capture | New, ~30 lines | `pyrealsense2` pipeline + align |

**Script structure:**

```python
# tools/realtime_tracker.py
"""
Real-time 6D object pose tracking with RealSense + FoundationPose++.

Usage:
  # First, generate init mask (in sam2 env):
  conda activate sam2
  python tools/realtime_init_mask.py --save_dir /tmp/init

  # Then run real-time tracking:
  conda activate hocap-annotation
  python tools/realtime_tracker.py \
    --mesh_path /path/to/object.obj \
    --init_dir /tmp/init \
    --track_refine_iter 5 \
    --activate_2d_tracker \
    --activate_kalman_filter

Controls:
  'r' = re-register (use saved init mask)
  'q' = quit
"""
```

**Key implementation details:**

#### 2a. Camera Module (~30 lines)
```python
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# Extract K from profile
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
```

#### 2b. Init from saved mask
```python
# Load init data from realtime_init_mask.py output
init_rgb = cv2.imread(init_dir / "rgb.png")[..., :3]
init_depth = np.load(init_dir / "depth.npy")
init_mask = np.load(init_dir / "mask.npy")
```

#### 2c. FoundationPose++ Init (directly from 04-1-4 pattern)
```python
mesh = trimesh.load(mesh_path)
mesh.apply_scale(apply_scale)

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(
    model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
    mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx,
)

# Register on init frame
pose = est.register(K=K, rgb=init_rgb, depth=init_depth,
                    ob_mask=init_mask, iteration=est_refine_iter)

# Init Cutie + Kalman (same as 04-1-4)
tracker_2D = Cutie()
tracker_2D.initialize(init_rgb, init_info={"mask": init_mask.astype(bool)})
kf = KalmanFilter6D(kf_noise_scale)
kf_mean, kf_cov = kf.initiate(get_6d_pose_arr_from_mat(pose))
```

#### 2d. Real-time Loop (core ~40 lines)
```python
# Modeled directly on the frame loop in 04-1-4_fd_pose_solver_kalman.py
# and obj_pose_track.py, but reading from live camera instead of disk

while True:
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color = np.asanyarray(frames.get_color_frame().get_data())  # BGR
    depth = np.asanyarray(frames.get_depth_frame().get_data()) / 1000.0  # → meters
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    # 2D tracker guidance (same logic as 04-1-4 lines 434-470)
    bbox_2d = tracker_2D.track(rgb)
    if bbox_2d[0] != -1:
        bbox_cx = bbox_2d[0] + bbox_2d[2] / 2
        bbox_cy = bbox_2d[1] + bbox_2d[3] / 2
        kf_mean, kf_cov = kf.update(kf_mean, kf_cov, get_6d_pose_arr_from_mat(est.pose_last))
        measurement_xy = np.array(get_pose_xy_from_image_point(est.pose_last, K_tensor, bbox_cx, bbox_cy))
        kf_mean, kf_cov = kf.update_from_xy(kf_mean, kf_cov, measurement_xy)
        est.pose_last = torch.from_numpy(get_mat_from_6d_pose_arr(kf_mean[:6])).unsqueeze(0).float().cuda()

    # Track
    pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=track_refine_iter)
    kf_mean, kf_cov = kf.predict(kf_mean, kf_cov)

    # Visualize (same as 04-1-4 / obj_pose_track.py)
    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3)

    cv2.imshow("Real-time Tracking", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Re-register using saved init mask
        pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=init_mask, iteration=est_refine_iter)
        kf_mean, kf_cov = kf.initiate(get_6d_pose_arr_from_mat(pose))
```

### Step 3: Performance Tuning

| Parameter | Offline (04-1-4) | Real-time target | Notes |
|-----------|-------------------|------------------|-------|
| `est_refine_iter` | 15 | 15 | Only at init, not latency-critical |
| `track_refine_iter` | 20 | 2-5 | Main speed lever. FP++ claims 30+ FPS @ RTX 3090 |
| Resolution | 640x480 | 640x480 | Matching camera native res |
| Cutie 2D tracker | on | on | ~20ms overhead, important for fast motion |
| Kalman filter | on | on | Negligible overhead (<1ms) |
| Depth masking | on | off | No mask available in real-time (unless Cutie provides one) |
| Image masking | on | off | Same reason |

**Expected latency budget (RTX 4090, 640x480):**

| Component | Time |
|-----------|------|
| RealSense frame capture | ~1ms (non-blocking with align) |
| Cutie 2D track | ~10-20ms |
| Kalman filter update | <1ms |
| FoundationPose track_one (iter=5) | ~15-25ms |
| Visualization + display | ~2ms |
| **Total** | **~30-50ms → 20-30 FPS** |

If Cutie is too slow, switch to `Tracker_2D()` (no-op passthrough) or OSTrack (~10ms).

### Step 4: Optional Enhancements

1. **Threaded camera capture**: Decouple capture from tracking to avoid waiting on camera vsync
2. **Cutie mask as depth mask**: Cutie produces a segmentation mask — use it to mask depth for cleaner tracking
3. **Multi-object**: Multiple FoundationPose instances, one per object (each ~100MB VRAM)
4. **Pose recording**: Save trajectory to `.npy` for offline analysis
5. **Fast-FoundationStereo depth backend**: If RealSense depth is noisy/missing, add FFS as optional depth source
   - Would need stereo camera pair (or use RealSense IR left+right)
   - Adds ~15-23ms (TensorRT) or ~30-50ms (PyTorch) per frame
   - Only worthwhile if depth quality is the bottleneck
6. **In-loop SAM2 re-init**: When tracking is lost, pause and let user click to re-segment

---

## File Plan

```
HO-Cap-Annotation/
├── tools/
│   ├── realtime_tracker.py          # Main real-time tracking (hocap-annotation env)
│   └── realtime_init_mask.py        # Capture first frame + SAM2 mask (sam2 env)
```

All other components already exist — no new modules needed:
- `hocap_annotation/wrappers/foundationpose.py` — FoundationPose wrapper (register, track_one)
- `FoundationPose-plus-plus/src/VOT.py` — Cutie 2D tracker
- `FoundationPose-plus-plus/src/utils/kalman_filter_6d.py` — Kalman filter
- `FoundationPose-plus-plus/FoundationPose/estimater.py` — draw_posed_3d_box, draw_xyz_axis

---

## Required Inputs

1. **Object mesh** (`.obj` or `.ply`) with known scale
2. **RealSense D435/D455** connected via USB3
3. **Initial mask** generated via `realtime_init_mask.py`

---

## Quick Start (After Implementation)

```bash
# Terminal 1: Generate init mask
conda activate sam2
pip install pyrealsense2  # one-time
python tools/realtime_init_mask.py \
  --save_dir /tmp/tracking_init
# → Camera preview opens
# → Press 's' to freeze frame
# → Click on object → mask generated
# → Files saved, script exits

# Terminal 2: Run real-time tracking
conda activate hocap-annotation
python tools/realtime_tracker.py \
  --mesh_path /path/to/object.obj \
  --init_dir /tmp/tracking_init \
  --apply_scale 0.01 \
  --track_refine_iter 5 \
  --activate_2d_tracker \
  --activate_kalman_filter
# → Real-time tracking window opens at ~20-30 FPS
# → Press 'r' to re-register, 'q' to quit
```

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `pyrealsense2` install conflict | Low | Pure C++ lib, no PyTorch deps |
| FoundationPose too slow for real-time | Low | FP++ proven 30+ FPS on 3090; 4090 is faster |
| Cutie tracker drift on fast motion | Medium | Kalman filter smoothing + manual re-register ('r') |
| Init mask from separate env is awkward UX | Medium | Later: install SAM2 in hocap-annotation for single-script flow |
| Depth quality from RealSense insufficient | Low-Medium | RealSense D435 is reasonable; FFS available as fallback |
| VRAM overflow | Low | FP ~2GB + Cutie ~1GB = ~3GB, well within 24GB |

---

## Summary

- **FoundationPose++ is already integrated** in your pipeline (04-1-4). The real-time script reuses the exact same tracking logic — just swaps disk reads for live camera frames.
- **Fast-FoundationStereo is orthogonal** — it's a depth source, not a tracker. Use it later if RealSense depth proves insufficient.
- **Two new scripts** needed: `realtime_init_mask.py` (~80 lines) + `realtime_tracker.py` (~200 lines)
- **No new modules or wrappers** — everything is already in the codebase
- **Environment**: just `pip install pyrealsense2` in `hocap-annotation` (and optionally in `sam2`)
- **Expected performance**: 20-30 FPS real-time tracking on RTX 4090
