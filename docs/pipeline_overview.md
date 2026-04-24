# HO-Cap-Annotation Pipeline

This document summarizes the end-to-end annotation pipeline dispatched by
`scripts/run_task_folder.sh → scripts/run_mydata.sh`, which takes a folder of
synchronized multi-camera RGB-D videos of hand–object interactions and
produces per-frame 6-DoF object poses and articulated hand poses.

## 1. Inputs and Outputs

**Inputs**
- Synchronized RGB-D video streams from *N* calibrated cameras (default *N*=8 RealSense rigs), one `cam{i}_rgb.mp4` + `cam{i}_depth.mkv` per view per sequence
- Global multi-camera extrinsic calibration (`realsense_calibration_*.yaml`)
- Per-object mesh priors ($\mathcal{M}_o$) and optional per-view instance masks produced offline via SAM2 (`tool_masks/` with a label-image convention, `objects.yaml` enumerating object identities)

**Outputs** (under `<videos_root>_annotated/<task>/<sequence>/`)
- `meta.yaml` — sequence metadata (cameras, frame window, objects)
- Per-object 6-DoF trajectories $\{T_{o,t}\}_{t=1}^{F}$
- MANO hand parameters $\{(\theta^{l/r}_t, \beta^{l/r}, \mathbf{t}^{l/r}_t, R^{l/r}_t)\}_{t=1}^{F}$ for both hands
- (Optional) Joint-refined hand–object state under `processed/`

## 2. Stage Overview

The pipeline is a cascade of seven stages. Each stage consumes the previous
stage's outputs from disk; no step retains state in memory across scripts.

```text
Stage 0. Ingest           00_convert_videos_to_h5.py
Stage 1. Calibration      preprocess/generate_meta.py
Stage 2. Per-view tracking 04-1-4_fd_pose_solver_kalman.py   (per object)
Stage 3. Multi-view merge 04-2-2_fd_pose_merger_cluster.py
Stage 4. Hand recovery    HandReconstruction/cluster_reconstruct.py
Stage 5. Hand refinement  HandReconstruction/cluster_optimize_hand.py
Stage 6. Object refine.   06-2_object_pose_solver_cluster.py
Stage 7. Joint optim.     07-2_joint_pose_solver_cluster.py
```

Stages 0–3 yield the per-object tracked trajectories; stages 4–5 yield
independent hand annotations; stages 6–7 (gated by `--optimize`) refine both
jointly using re-projection and contact cues.

## 3. Stage Details

### 3.1 Ingest (Stage 0)

`tools/00_convert_videos_to_h5.py` repacks the per-view video files into a
single HDF5 container with datasets
- `imgs ∈ uint8^{F × N × H × W × 3}`
- `depths ∈ uint16^{F × N × H × W}` (millimetres)

Depth is extracted via `ffmpeg` with `gray16le` to preserve 16-bit precision;
RGB is decoded by OpenCV. Storage uses LZF compression for throughput. The
container is the canonical I/O substrate for all downstream stages.

### 3.2 Calibration & metadata (Stage 1)

`preprocess/generate_meta.py` ingests the HDF5, the calibration YAML, and (if
present) the SAM2 masks, and emits `meta.yaml` with frame count, camera
serials, per-view image dimensions, and the detected object identifiers
(`object_ids`). Object count is inferred from the maximum mask label; object
names are resolved in the order
`tool_masks/objects.yaml > tool_name argument > generic fallback`.
The mask array is also dumped in HDF5 form (`masks.h5`) for random-access
reads downstream.

### 3.3 Single-view 6-DoF tracking (Stage 2)

For every object $o \in \{1,\dots,K\}$, `tools/04-1-4_fd_pose_solver_kalman.py`
produces a per-view pose trajectory $T^{(i)}_{o,t} \in \mathrm{SE}(3)$ for each
camera $i$. The backbone is **FoundationPose** [Wen et al., 2024]
($\mathcal{M}_o + \text{RGB-D} + \text{mask} \mapsto T_{o,t}$) augmented with:

1. **Constant-velocity Kalman filter** on the pose twist
   $\xi \in \mathfrak{se}(3)$, providing temporal smoothing and predictive
   priors for the next frame.
2. **2D visual tracker** (CoTracker / Cutie-style) that propagates object
   mask regions and rejects frames whose IoU with the predicted mask falls
   below a threshold.
3. **Reset logic** keyed on per-frame rotation/translation residuals
   (`--rot_thresh 15°`, `--trans_thresh 0.03 m`) to recover from
   tracking drift by re-initialising FoundationPose from the Kalman prior.

The stage runs independently per camera; no cross-view consistency is
enforced yet.

### 3.4 Multi-view pose merging (Stage 3)

`tools/04-2-2_fd_pose_merger_cluster.py` fuses the *N* per-view trajectories
into a single world-frame trajectory $T_{o,t}$ per object. For each
frame $t$, candidate poses $\{c^{(i)}_t = T^\text{extr}_i \cdot T^{(i)}_{o,t}\}$
are expressed in the global frame, clustered, and the dominant cluster's
centroid is selected. The merger also identifies and fills short occlusions
by interpolating across neighbouring valid frames. Outputs are stored at
`processed/fd_pose_solver/fd_poses_merged_fixed.npy` with shape
$K \times F \times 7$ (quaternion + translation).

### 3.5 Monocular hand reconstruction (Stage 4)

`HandReconstruction/cluster_reconstruct.py` iterates frames; per frame it runs:
1. **WiLoR** [Chen et al., 2024] on every view, producing, for up to two
   detected hands, MANO-aligned `(global_orient, hand_pose θ, betas β,
   joints_2d)` in camera coordinates.
2. **Ray-based triangulation**: the detected 2D wrist keypoint of each view
   defines a 3D ray in the world frame; the hand root $\mathbf{t}_t$ is
   optimised to minimise
   $\sum_{i} d\bigl(\mathbf{t}_t,\,r^{(i)}_t\bigr)$ over rays $r^{(i)}_t$
   (Adam, 2000 iters).
3. **Parameter fusion**: per-frame MANO pose and shape parameters are
   averaged in their respective manifolds (Fréchet mean on $\mathrm{SO}(3)$
   for rotations; Euclidean mean otherwise).

Outputs are packed into a `HandObjectPoseDataset` pickle with keys
`hand_pose.{left,right}_hand_{pose,translation,base_rot,beta}` plus
`hand_joints.{…}_2d` per view, and camera intrinsics/extrinsics.

### 3.6 Hand parameter optimisation (Stage 5)

`HandReconstruction/cluster_optimize_hand.py` refines the Stage-4 estimate by
AdamW over 5 000 iterations, minimising for each hand $h \in \{l,r\}$:

$$
\mathcal{L}^h = \sum_{t=1}^{F} \sum_{i=1}^{N}
    \bigl\| \Pi_i\!\bigl(J^h_t(\theta^h_t,\beta^h,\mathbf{t}^h_t,R^h_t)\bigr)
          - \hat{J}^h_{i,t} \bigr\|_2^2
    + \lambda_{\text{smooth}}\! \sum_{t} \bigl\| V^h_t - V^h_{t-1} \bigr\|_2^2,
$$

where $\Pi_i$ is the projection through camera $i$, $J^h_t$ are the 3D joints
produced by the MANO forward pass for frame $t$, $\hat{J}^h_{i,t}$ are the
detected 2D joints in view $i$, and the smoothness term penalises vertex
drift between successive frames. Output: `result_hand_optimized.pkl` and a
compact `poses_m.npy` of shape $2 \times F \times 51$ (pose + translation per
hand).

### 3.7 Object-pose refinement (Stage 6, optional)

`tools/06-2_object_pose_solver_cluster.py` refines $T_{o,t}$ in a
render-and-compare loop using differentiable SDF/mesh-to-depth losses:

- **Keypoint re-projection** across views;
- **Point-cloud-to-mesh SDF** loss on the masked depth;
- **Temporal smoothness** on the twist $\xi$;
- **Pose-alignment prior** pulling toward the Stage-3 trajectory.

Only object index 1 is optimised (matching the single tool-hand interaction
assumption).

### 3.8 Joint hand–object optimisation (Stage 7, optional)

`tools/07-2_joint_pose_solver_cluster.py` couples the object trajectory with
the MANO hands. In addition to the per-modality terms above, it imposes:

- **Contact consistency**: a soft penalty on signed distance between hand
  vertices and the object surface at predicted contact frames;
- **Mutual non-penetration**: hinge loss on hand-vertex SDF against the
  object mesh;
- **Symmetric grasp prior** (optional, via `07-4_contact_optimizer.py`)
  that stabilises closures when both hands grip the same tool.

The output is written to
`processed/joint_pose_solver/poses_o.npy` (object) and
`processed/joint_pose_solver/result_hand_optimized.pkl` (hands).

## 4. Multi-Object Handling

All stages 2–3 are looped over the object axis; `meta.yaml::object_ids`
drives the iteration. The refinement stages 6–7 are intentionally restricted
to the primary object (first entry), keeping the hand–object interaction
optimisation well-posed. Non-primary objects retain their merged trajectories
from Stage 3 without contact refinement.

## 5. File / Directory Conventions

```text
<data_root>/<videos_XXXX>/<task>/<sequence>/
    cam{i}_rgb.mp4           raw RGB streams
    cam{i}_depth.mkv         raw 16-bit depth streams
    data00000000.h5          Stage 0 container
    meta.yaml                Stage 1 metadata

<data_root>/<videos_XXXX>_annotated/<task>/<sequence>/
    tool_masks/              SAM2 per-view instance labels (optional)
    result.pkl               Stage 4 output
    result_hand_optimized.pkl   Stage 5 output
    poses_m.npy              compact per-frame hand params
    processed/
      fd_pose_solver/        Stage 3 per-object trajectories
      object_pose_solver/    Stage 6 refined object
      joint_pose_solver/     Stage 7 joint refinement
```

## 6. Entry Points

| Script | Scope |
|---|---|
| `scripts/run_task_folder.sh <task>` | iterate every experiment in a task folder, running stages 0–5 (+ 6–7 when `--optimize` is set upstream) |
| `scripts/run_mydata.sh --sequence_name …` | single-sequence dispatcher; exposes `--hand`, `--optimize`, `--start_frame`, `--end_frame`, and tracker thresholds |
| `scripts/batch_task_folder_hand_chunked.sh` | chunked, resume-safe hand-only variant (stages 0 + 1 + 4 + 5) with merged per-experiment output |
| `scripts/run_full_auto.sh` | end-to-end dispatcher that adds Stage 1 calibration auto-alignment before handing off to the chunked batch |

## 7. External Dependencies (non-exhaustive)

- **FoundationPose++** (vendored in `FoundationPose-plus-plus/`) — Stage 2
- **Cutie / SAM-HQ** — video segmentation for `tool_masks/`
- **WiLoR** (vendored in `HandReconstruction/external/WiLoR/`) — Stage 4
- **manopth** — MANO forward model and Jacobians, Stages 4–5 and 7
- **Open3D, trimesh, pyrender** — rendering, geometric queries, visualisation
