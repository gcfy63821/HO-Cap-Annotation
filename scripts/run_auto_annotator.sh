#!/bin/bash
# Full annotation pipeline with DINOv2-based automatic tool segmentation.
# Replaces the interactive SAM2 mask step with tools/dino_tool_segment.py.
#
# Stages (all single-sequence, single-tool):
#   1. videos -> h5                       (tools/00_convert_videos_to_h5.py)
#   2. DINOv2+SAM2 auto tool mask         (tools/dino_tool_segment.py)
#      + export to pipeline format       (--pipeline_tool_masks_dir)
#   3. generate_meta                      (preprocess/generate_meta.py)
#   4. per-object 6-DoF tracking          (tools/04-1-4_fd_pose_solver_kalman.py)
#   5. multi-view pose merge              (tools/04-2-2_fd_pose_merger_cluster.py)
#   6. (optional) hand reconstruction     (HandReconstruction/cluster_reconstruct.py
#                                          + cluster_optimize_hand.py)
#   7. (optional) object + joint optim    (tools/06-2_*, 07-2_*)
#
# Expected sequence folder layout (before running):
#   <sequence_folder>/
#     cam{i}_rgb.mp4
#     cam{i}_depth.mkv
#
# Results go to:
#   <sequence_folder>/data00000000.h5
#   <sequence_folder>/meta.yaml
#   <videos_root>_annotated/<task>/<sequence>/tool_masks/...
#   <videos_root>_annotated/<task>/<sequence>/{result.pkl,result_hand_optimized.pkl,...}
#   <videos_root>_annotated/<task>/<sequence>/processed/...
#
# Resume / force semantics:
#   By default, every stage auto-skips itself if its primary output already
#   exists on disk (per-stage detection table below). This makes re-running
#   the same command after a partial failure cheap and idempotent.
#
#     [Stage 1]  data00000000.h5
#     [Stage 2]  tool_masks/cam*_rgb/0000.npz  OR  tool_masks/masks.h5
#     [Stage 3]  meta.yaml + tool_masks/masks.h5  (with valid calibration_yaml_path)
#     [Stage 4]  per chunk: every abs frame id has ob_in_world/{abs:06d}.txt
#     [Stage 4-5 outer + Stage 5]  fd_poses_merged_fixed.npy
#     [Stage 6]  result_hand_optimized.pkl
#     [Stage 7]  processed/joint_pose_solver/{poses_o,poses_m}.npy
#
#   To force everything to re-run, pass --force (alias --no_resume). To force
#   a single stage, just delete that stage's primary output file before
#   running.
#
# Usage:
#   bash scripts/run_auto_annotator.sh \
#     --sequence_folder /abs/path/to/data/videos_XXXX/task/exp \
#     --calibration_yaml /abs/path/to/realsense_calibration_*.yaml \
#     --tool_name mallet                                # mesh auto-resolved
#     [--tool_mesh /abs/path/to/mesh.obj]               # override if needed
#     [--models_folder /abs/path/to/models]             # default $MODELS_FOLDER
#     [--hand 1] [--optimize 1]
#     [--start_frame 0] [--end_frame 999]
#     [--force | --no_resume]                           # redo every stage
#     [--skip_h5] [--skip_masks] [--skip_tracking]
#     [--rot_thresh 15] [--trans_thresh 0.03] [--track_refine_iter 10]
#     [--hand_chunk_size 500]      # 0 = single-pass hand; otherwise chunk hand
#                                  # into K-frame windows (per-chunk h5+meta,
#                                  # reconstruct + optimize, then merge).
#     [--hand_keep_chunk_files]    # keep per-chunk *_{S}_{E}.pkl after merge
#     [--fake_optimize]            # skip Stage 6+7 optimizers, assemble fake
#                                  # joint_pose_solver outputs from fd merge +
#                                  # result_hand_optimized.pkl. Mutually
#                                  # exclusive with --optimize 1. Useful when
#                                  # the joint optimizer OOMs on long videos.
#     [--frame0_only]       # DINO-register only on frame 0, no re-seeding,
#                           # skips Phase 3-7. Fastest on slow/cluster GPUs.
#
# Mesh auto-resolution: when --tool_mesh is omitted, the script looks in
#   $MODELS_FOLDER/$TOOL_NAME/ for one of (in order):
#     textured_mesh.obj, cleaned_mesh_10000.obj, mesh.obj
# MODELS_FOLDER falls back to $HOCAP_ROOT/data/models if the env var isn't set.

set -eu

# ---------- defaults ----------
SEQUENCE_FOLDER=""
CALIBRATION_YAML=""
TOOL_NAME=""
TOOL_MESH=""
HAND=""
OPTIMIZE=""
# Skip Stage 6 (object_pose_solver) + Stage 7 (joint_pose_solver) optimizations
# and instead assemble the canonical joint_pose_solver outputs from
# fd_poses_merged_fixed.npy + result_hand_optimized.pkl. Mutually exclusive
# with --optimize 1.
FAKE_OPTIMIZE=0
START_FRAME=0
END_FRAME=""
ROT_THRESH=15
TRANS_THRESH=0.03
TRACK_REFINE_ITER=10
SKIP_H5=0
SKIP_MASKS=0
# When the sequence has more than LONG_VIDEO_THRESHOLD frames, automatically
# skip the joint optimization stage so the rest can finish in this job. Hand
# is now handled inline via the chunked path below (HAND_CHUNK_SIZE) and is no
# longer auto-disabled. Set LONG_VIDEO_THRESHOLD=0 to disable optim auto-skip.
LONG_VIDEO_THRESHOLD=1000
# Chunk hand reconstruction + optimization into windows of HAND_CHUNK_SIZE
# frames each (mirrors scripts/batch_task_folder_hand_chunked.sh inline).
# Per chunk: rebuild a chunk-only h5 + meta, run cluster_reconstruct +
# cluster_optimize_hand, rename outputs to *_{S}_{E}.pkl/.npy. After all
# chunks finish, merge_hand_chunks.py stitches them back into the canonical
# result.pkl / result_hand_optimized.pkl / poses_m.npy and the full h5+meta
# are restored so Stage 7 sees the original sequence. Set 0 to disable
# (single-pass hand, like before).
HAND_CHUNK_SIZE=500
# Keep per-chunk result_*_{S}_{E}.pkl files after merging (debug). Default 0.
HAND_KEEP_CHUNK_FILES=0
# Chunk fd_pose_solver into windows of OBJECT_CHUNK_SIZE frames each. We
# rebuild a chunk-only h5 + meta per chunk so the loader's RAM footprint
# stays bounded (MyClusterLoader currently materialises the entire h5 +
# masks.h5 into memory; on long 720p×8-cam sequences that's ~100+ GB and
# will swap to death). The chunk loop re-registers from each chunk's frame 0
# and writes per-frame txts at absolute frame ids via --frame_id_offset, so
# Stage 5's merger reads a unified per-object output dir afterwards. Set to
# 0 to disable (single full-h5 pass; only safe on short sequences).
OBJECT_CHUNK_SIZE=200
# Legacy: previously controlled overlap between frame-range-only chunks
# (same h5 reused). The new chunked-h5 path doesn't overlap (each chunk has
# its own h5 + frame 0). Kept for CLI back-compat; ignored.
OBJECT_CHUNK_OVERLAP=0
SKIP_TRACKING=0
# Resume mode: per-stage auto-skip when the stage's primary output already
# exists. ON by default — re-running the same command after a partial failure
# (or already-finished sequence) is the common case, and forcing re-runs by
# accident is expensive. Use --force / --no_resume to opt out.
RESUME=1
# Visualizations: PNG snapshots (and optional MP4) generated by
# dino_tool_segment.py and the per-stage debug outputs add I/O / time but
# are rarely needed in batch jobs. Default OFF; pass --with_viz to opt in.
WITH_VIZ=0
# Pipeline mask export format from Stage 2 DINO. Choices: 'h5' (default;
# single masks.h5 — saves ~15k inodes per exp on /viscam), 'npz' (legacy
# per-frame compressed), 'npy' (legacy uncompressed). All Stage-3+ paths
# now read masks.h5 directly via generate_meta.py --masks_h5_source.
PIPELINE_MASK_FORMAT="${PIPELINE_MASK_FORMAT:-h5}"
# If set, build h5 in this dir and symlink it into the sequence folder so
# downstream tools (cluster_reconstruct, generate_meta, …) still find it at
# $SEQUENCE_FOLDER/data00000000.h5. Typical: /dev/shm/$USER/$SLURM_JOB_ID .
H5_SCRATCH_DIR=""

# DINO seeding tunables (pass-through to tools/dino_tool_segment.py).
# Empty string = use dino_tool_segment's defaults.
DINO_MESH_SCAN_EVERY=""     # stride for anchor-camera seed scan (default 10)
DINO_DENSE_SCAN_EVERY=""    # stride for non-anchor seed scan    (default 5)
DINO_SEED_MIN_AREA=""       # partial views -> lower this (default 100)
DINO_SEED_MAX_AREA=""       # large tools -> raise this    (default 15000)
DINO_SEED_MIN_SIM=""        # partial views -> lower this (default 0.20)
DINO_SEED_FAST=0            # 1 = first-hit mode (try frame 0, then stride, stop on first OK)
DINO_FRAME0_ONLY=0          # 1 = frame-0-only mode (skip Phase 3-7, no re-seeding; fast path for cluster)
DINO_USE_SIMPLE=0           # 1 = use tools/dino_simple_segment.py (minimal 2-stage version)
                            #     instead of the full tools/dino_tool_segment.py.
                            #     Behaviour ≈ frame0_only, but in a clean ~200-line script.
DINO_SEED_FRAMES=""         # space-separated frame indices for frame0_only's per-cam best-of search,
                            #   e.g. "0 60 200 500". Empty = just frame 0.
DINO_SEED_MIN_MESH_SIM=""   # hard mesh_sim floor for frame0_only seeds (default = seed_min_sim).
                            #   Tighten (e.g. 0.40) to reject wrong-object seeds outright.
DINO_SAM2_CHUNK_SIZE=""     # SAM2 video propagation chunk size (default 100). Larger = fewer
                            #   propagate_in_video calls (faster) but more CPU RAM per chunk.
                            #   Try 200-300 if your node has plenty of RAM.

HOCAP_ROOT="${HOCAP_ROOT:-/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation}"
HAND_ROOT="${HAND_ROOT:-/home/ruoqu/crq_ws/robotool/HandReconstruction}"
ROBOTOOL_ROOT="$(dirname "$HOCAP_ROOT")"

# SAM2 checkpoint resolution order:
#   1. $SAM2_CKPT env var (explicit)
#   2. $SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt (one-shot root override)
#   3. candidate paths below (first existing file wins):
#        local layout:   <robotool>/mesh_reconstruction/sam2/checkpoints/...
#        cluster layout: <robotool>/sam2/checkpoints/...
if [[ -z "${SAM2_CKPT:-}" ]]; then
    if [[ -n "${SAM2_ROOT:-}" ]]; then
        SAM2_CKPT="${SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt"
    else
        _SAM2_CKPT_CANDIDATES=(
            "${ROBOTOOL_ROOT}/mesh_reconstruction/sam2/checkpoints/sam2.1_hiera_large.pt"
            "${ROBOTOOL_ROOT}/sam2/checkpoints/sam2.1_hiera_large.pt"
        )
        for _p in "${_SAM2_CKPT_CANDIDATES[@]}"; do
            if [[ -f "$_p" ]]; then SAM2_CKPT="$_p"; break; fi
        done
        SAM2_CKPT="${SAM2_CKPT:-${ROBOTOOL_ROOT}/mesh_reconstruction/sam2/checkpoints/sam2.1_hiera_large.pt}"
    fi
fi
SAM2_VIDEO_CFG="${SAM2_VIDEO_CFG:-${HOCAP_ROOT}/config/sam2_config/sam2.1_hiera_l.yaml}"
SAM2_IMAGE_CFG="${SAM2_IMAGE_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"

# ---------- arg parsing ----------
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --sequence_folder)     SEQUENCE_FOLDER="$2"; shift 2;;
        --calibration_yaml)    CALIBRATION_YAML="$2"; shift 2;;
        --tool_name)           TOOL_NAME="$2"; shift 2;;
        --tool_mesh)           TOOL_MESH="$2"; shift 2;;
        --hand)                HAND="$2"; shift 2;;
        --optimize)            OPTIMIZE="$2"; shift 2;;
        --start_frame)         START_FRAME="$2"; shift 2;;
        --end_frame)           END_FRAME="$2"; shift 2;;
        --rot_thresh)          ROT_THRESH="$2"; shift 2;;
        --trans_thresh)        TRANS_THRESH="$2"; shift 2;;
        --track_refine_iter)   TRACK_REFINE_ITER="$2"; shift 2;;
        --skip_h5)             SKIP_H5=1; shift;;
        --skip_masks)          SKIP_MASKS=1; shift;;
        --skip_tracking)       SKIP_TRACKING=1; shift;;
        --resume)              RESUME=1; shift;;
        --no_resume|--force)   RESUME=0; shift;;
        --with_viz)            WITH_VIZ=1; shift;;
        --pipeline_mask_format) PIPELINE_MASK_FORMAT="$2"; shift 2;;
        --sam2_ckpt)           SAM2_CKPT="$2"; shift 2;;
        --sam2_video_cfg)      SAM2_VIDEO_CFG="$2"; shift 2;;
        --sam2_image_cfg)      SAM2_IMAGE_CFG="$2"; shift 2;;
        --h5_scratch_dir)      H5_SCRATCH_DIR="$2"; shift 2;;
        --models_folder)       MODELS_FOLDER="$2"; shift 2;;
        --mesh_scan_every)     DINO_MESH_SCAN_EVERY="$2"; shift 2;;
        --dense_scan_every)    DINO_DENSE_SCAN_EVERY="$2"; shift 2;;
        --seed_min_area)       DINO_SEED_MIN_AREA="$2"; shift 2;;
        --seed_max_area)       DINO_SEED_MAX_AREA="$2"; shift 2;;
        --seed_min_sim)        DINO_SEED_MIN_SIM="$2"; shift 2;;
        --seed_fast)           DINO_SEED_FAST=1; shift;;
        --seed_frames)         DINO_SEED_FRAMES="$2"; shift 2;;
        --seed_min_mesh_sim)   DINO_SEED_MIN_MESH_SIM="$2"; shift 2;;
        --sam2_chunk_size)     DINO_SAM2_CHUNK_SIZE="$2"; shift 2;;
        --frame0_only)         DINO_FRAME0_ONLY=1; shift;;
        --simple_dino)         DINO_USE_SIMPLE=1; shift;;
        --long_video_threshold) LONG_VIDEO_THRESHOLD="$2"; shift 2;;
        --object_chunk_size)    OBJECT_CHUNK_SIZE="$2"; shift 2;;
        --object_chunk_overlap) OBJECT_CHUNK_OVERLAP="$2"; shift 2;;
        --hand_chunk_size)      HAND_CHUNK_SIZE="$2"; shift 2;;
        --hand_keep_chunk_files) HAND_KEEP_CHUNK_FILES=1; shift;;
        --fake_optimize)        FAKE_OPTIMIZE=1; shift;;
        -h|--help)             sed -n '2,45p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

for required in SEQUENCE_FOLDER CALIBRATION_YAML TOOL_NAME; do
    if [[ -z "${!required}" ]]; then
        echo "Error: --${required,,} is required"; exit 1
    fi
done

SEQUENCE_FOLDER="$(cd "$SEQUENCE_FOLDER" && pwd)"
CALIBRATION_YAML="$(readlink -f "$CALIBRATION_YAML")"

# --- resolve tool mesh from tool_name if --tool_mesh not given ---
MODELS_FOLDER="${MODELS_FOLDER:-${HOCAP_ROOT}/data/models}"
if [[ -z "$TOOL_MESH" ]]; then
    if [[ ! -d "$MODELS_FOLDER/$TOOL_NAME" ]]; then
        echo "Error: no model folder $MODELS_FOLDER/$TOOL_NAME."
        echo "       Pass --tool_mesh explicitly or set MODELS_FOLDER / --models_folder."
        exit 1
    fi
    for cand in "textured_mesh.obj" "cleaned_mesh_10000.obj" "mesh.obj"; do
        if [[ -f "$MODELS_FOLDER/$TOOL_NAME/$cand" ]]; then
            TOOL_MESH="$MODELS_FOLDER/$TOOL_NAME/$cand"
            break
        fi
    done
    if [[ -z "$TOOL_MESH" ]]; then
        echo "Error: no mesh file found in $MODELS_FOLDER/$TOOL_NAME/"
        echo "       (looked for: textured_mesh.obj, cleaned_mesh_10000.obj, mesh.obj)"
        exit 1
    fi
    echo "[mesh] auto-resolved from tool_name: $TOOL_MESH"
fi
TOOL_MESH="$(readlink -f "$TOOL_MESH")"

# ---------- derive paths ----------
# Expect .../<base>/<videos_XXXX>/<task>/<exp>
EXP_NAME="$(basename "$SEQUENCE_FOLDER")"
TASK_FOLDER="$(dirname "$SEQUENCE_FOLDER")"
TASK_NAME="$(basename "$TASK_FOLDER")"
VIDEOS_ROOT="$(dirname "$TASK_FOLDER")"
VIDEOS_FOLDER_NAME="$(basename "$VIDEOS_ROOT")"
BASE="$(dirname "$VIDEOS_ROOT")"

H5_PATH="${SEQUENCE_FOLDER}/data00000000.h5"   # what downstream scripts expect
# If a fast scratch dir is given, real h5 lives there and H5_PATH is a symlink.
if [[ -n "$H5_SCRATCH_DIR" ]]; then
    mkdir -p "$H5_SCRATCH_DIR"
    # Name unique to this sequence so concurrent jobs don't collide.
    H5_REAL_PATH="${H5_SCRATCH_DIR}/$(echo "${VIDEOS_FOLDER_NAME}_${TASK_NAME}_${EXP_NAME}" | tr / _)_data00000000.h5"

    # /dev/shm cleanup: remove this exp's scratch h5 (+ symlink + Stage 4/6
    # backup files) on script exit. Prevents tmpfs accumulation when sbatch
    # runs many exps in one job — without this, /dev/shm fills up after a
    # few exps and the next one OOM-kills (rc=137) during h5 build or SAM2
    # load. Trap fires on success AND error/signal so failed exps don't
    # leave 25 GB lying in /dev/shm either.
    _cleanup_scratch_h5() {
        local rc=$?
        if [[ -n "${H5_REAL_PATH:-}" && "${H5_REAL_PATH}" != "${H5_PATH:-}" ]]; then
            rm -f "$H5_REAL_PATH" \
                  "$H5_REAL_PATH".full_backup_obj \
                  "$H5_REAL_PATH".full_backup_hand 2>/dev/null || true
        fi
        # Drop the dangling symlink in the sequence folder so RESUME on a
        # later sbatch re-detects "no h5" and rebuilds (the /dev/shm target
        # is gone the moment the slurm job ends anyway).
        if [[ -L "${H5_PATH:-}" ]]; then
            rm -f "$H5_PATH" 2>/dev/null || true
        fi
        return $rc
    }
    trap _cleanup_scratch_h5 EXIT INT TERM HUP
else
    H5_REAL_PATH="$H5_PATH"
fi
ANNOTATED_PATH="${BASE}/${VIDEOS_FOLDER_NAME}_annotated/${TASK_NAME}/${EXP_NAME}"
TOOL_MASKS_DIR="${ANNOTATED_PATH}/tool_masks"
# MODELS_FOLDER already resolved above (used for mesh auto-resolution)

mkdir -p "$ANNOTATED_PATH"

# conda activation (tries common locations)
_CONDA_SH_CANDIDATES=(
    "${CONDA_SH:-}"
    "/home/ruoqu/miniconda3/etc/profile.d/conda.sh"
    "/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
)
for _p in "${_CONDA_SH_CANDIDATES[@]}"; do
    if [[ -n "$_p" && -f "$_p" ]]; then source "$_p"; break; fi
done

echo "=========================================="
echo "sequence         : $SEQUENCE_FOLDER"
echo "tool             : $TOOL_NAME  (mesh: $TOOL_MESH)"
echo "calibration      : $CALIBRATION_YAML"
echo "annotated out    : $ANNOTATED_PATH"
echo "tool_masks_dir   : $TOOL_MASKS_DIR"
echo "hand=${HAND:-0}  optimize=${OPTIMIZE:-0}  fake_optim=${FAKE_OPTIMIZE}"
if [[ "$RESUME" == "1" ]]; then
    echo "resume mode      : ON  (skip stages whose outputs exist; pass --force to redo)"
else
    echo "resume mode      : OFF (--force / --no_resume given; redo all stages)"
fi
echo "=========================================="

conda activate hocap-annotation

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg not on PATH"; exit 1
fi

# ---------- Auto-recover from interrupted chunked Stage 4/6 ----------
# A backup file means a prior run died mid-chunk and didn't restore the full
# h5/meta/masks.h5. The on-disk artifacts are then chunk-only (e.g. 400
# frames out of 1864), which silently corrupts downstream stages: lazy
# merger sees num_frames=400, reads 400 ob_in_world txts, mostly missing,
# writes an all-(-1) fd_poses_merged_fixed.npy, and the viewer shows
# nothing. Restoring before the run pre-empts that whole class of bugs.
for _bk_kind in obj hand; do
    _bk_h5="${H5_REAL_PATH}.full_backup_${_bk_kind}"
    _bk_meta="${SEQUENCE_FOLDER}/meta.yaml.full_backup_${_bk_kind}"
    _bk_masks="${TOOL_MASKS_DIR}/masks.h5.full_backup_${_bk_kind}"
    if [[ -f "$_bk_h5" ]]; then
        echo "[recover] found ${_bk_kind} h5 backup ($_bk_h5); restoring (prior run was interrupted)"
        mv -f "$_bk_h5" "$H5_REAL_PATH"
    fi
    if [[ -f "$_bk_meta" ]]; then
        echo "[recover] found ${_bk_kind} meta backup; restoring"
        mv -f "$_bk_meta" "${SEQUENCE_FOLDER}/meta.yaml"
    fi
    if [[ -f "$_bk_masks" ]]; then
        echo "[recover] found ${_bk_kind} masks.h5 backup; restoring"
        mv -f "$_bk_masks" "${TOOL_MASKS_DIR}/masks.h5"
    fi
done

# ---------- Stage 1: videos -> h5 ----------
if { [[ "$SKIP_H5" == "1" ]] || [[ "$RESUME" == "1" ]]; } && [[ -f "$H5_PATH" ]]; then
    echo "[1/7] [skip] h5 exists: $H5_PATH"
else
    echo "[1/7] building h5 ..."
    # Clean any prior state on both the logical path (may be stale symlink) and the real path.
    rm -f "$H5_PATH" "$H5_REAL_PATH"
    H5_ARGS=(--input_dir "$SEQUENCE_FOLDER" --output_file "$H5_REAL_PATH" --start_frame "$START_FRAME")
    [[ -n "$END_FRAME" ]] && H5_ARGS+=(--end_frame "$END_FRAME")
    python "$HOCAP_ROOT/tools/00_convert_videos_to_h5.py" "${H5_ARGS[@]}"
    # Expose the real file at the logical path so downstream reads transparently.
    if [[ "$H5_REAL_PATH" != "$H5_PATH" ]]; then
        if ! ln -sf "$H5_REAL_PATH" "$H5_PATH" 2>&1; then
            echo "Error: failed to create symlink $H5_PATH -> $H5_REAL_PATH"
            echo "       This is almost always a /viscam quota issue (the symlink"
            echo "       file lives on /viscam, even though the target is in /dev/shm)."
            echo "       Run 'quota -s -u \$USER' on a login node and free space"
            echo "       (try: deleting old slurm_outs/, conda clean -ay,"
            echo "       removing processed/fd_pose_solver/<obj>/ob_in_cam after"
            echo "       fd_poses_merged_fixed.npy is generated)."
            exit 1
        fi
    fi
fi

# ---------- Long-video gate: skip joint optim when N > LONG_VIDEO_THRESHOLD ----------
# Stage 7 (joint optim) is global over the sequence and tends to OOM/timeout
# on long videos. Hand reconstruction is no longer auto-disabled — Stage 6
# below chunks it inline whenever HAND_CHUNK_SIZE > 0 and N_FRAMES exceeds it.
N_FRAMES=$(python3 -c "import h5py; print(h5py.File('$H5_PATH','r')['imgs'].shape[0])" 2>/dev/null || echo 0)
if [[ "$LONG_VIDEO_THRESHOLD" -gt 0 && "$N_FRAMES" -gt "$LONG_VIDEO_THRESHOLD" ]]; then
    if [[ -n "$OPTIMIZE" && "$OPTIMIZE" != "0" ]]; then
        echo "=========================================="
        echo "[long-video] $N_FRAMES frames > threshold $LONG_VIDEO_THRESHOLD"
        echo "[long-video] forcing --optimize 0 for this exp (joint optim is global)."
        echo "[long-video] consider --fake_optimize to assemble joint outputs from fd merge + hand pkl."
        echo "=========================================="
        OPTIMIZE="0"
    fi
fi

# ---------- Stage 2: dino auto tool mask + pipeline-format export ----------
# Discover camera serials from calibration to keep folder naming consistent
# with generate_meta.py (cam{serial}_rgb/).
CAM_SERIALS=$(CAL_YAML="$CALIBRATION_YAML" python - <<'PY'
import os, yaml
with open(os.environ["CAL_YAML"]) as f:
    d = yaml.safe_load(f)
if isinstance(d, list):
    cams = d
else:
    cams = [v for k, v in d.get("extrinsics", {}).items() if not k.startswith("tag_")]
cams = sorted(cams, key=lambda c: c.get("camera_id", 0))
print(" ".join(str(c["camera_id"]).zfill(2) for c in cams))
PY
)
echo "  camera serials: $CAM_SERIALS"

if { [[ "$SKIP_MASKS" == "1" ]] || [[ "$RESUME" == "1" ]]; } && \
   ( ls "$TOOL_MASKS_DIR"/cam*_rgb/0000.npz >/dev/null 2>&1 \
     || ls "$TOOL_MASKS_DIR"/cam*_rgb/0000.npy >/dev/null 2>&1 \
     || [[ -f "$TOOL_MASKS_DIR/masks.h5" ]] ); then
    # Skip DINO if either per-frame npz checkpoints exist OR a full masks.h5
    # is already on disk (e.g. synced from another machine; npz deleted to
    # save space). Stage 4's chunked path can slice the full masks.h5 via
    # generate_meta.py --masks_h5_source instead of re-streaming from npz.
    echo "[2/7] [skip] tool_masks already populated"
else
    echo "[2/7] running DINOv2 + SAM2 auto segmentation ..."
    export PYOPENGL_PLATFORM=egl
    # DINO output dir (contains masks.h5, viz/, seed_info.json for debug)
    DINO_OUT="${ANNOTATED_PATH}/dino_auto"
    mkdir -p "$DINO_OUT"
    DINO_ARGS=(
        --data_h5      "$H5_PATH"
        --calib_yaml   "$CALIBRATION_YAML"
        --tool_mesh    "$TOOL_MESH"
        --output_dir   "$DINO_OUT"
        --sam2_ckpt        "$SAM2_CKPT"
        --sam2_video_cfg   "$SAM2_VIDEO_CFG"
        --sam2_image_cfg   "$SAM2_IMAGE_CFG"
        --pipeline_tool_masks_dir "$TOOL_MASKS_DIR"
        --pipeline_mask_format    "${PIPELINE_MASK_FORMAT:-h5}"
        --tool_name "$TOOL_NAME"
        --no_video
    )
    [[ -n "$CAM_SERIALS" ]]            && DINO_ARGS+=(--cam_serials $CAM_SERIALS)
    [[ -n "$DINO_MESH_SCAN_EVERY" ]]   && DINO_ARGS+=(--mesh_scan_every  "$DINO_MESH_SCAN_EVERY")
    [[ -n "$DINO_DENSE_SCAN_EVERY" ]]  && DINO_ARGS+=(--dense_scan_every "$DINO_DENSE_SCAN_EVERY")
    [[ -n "$DINO_SEED_MIN_AREA" ]]     && DINO_ARGS+=(--seed_min_area    "$DINO_SEED_MIN_AREA")
    [[ -n "$DINO_SEED_MAX_AREA" ]]     && DINO_ARGS+=(--seed_max_area    "$DINO_SEED_MAX_AREA")
    [[ -n "$DINO_SEED_MIN_SIM" ]]      && DINO_ARGS+=(--seed_min_sim     "$DINO_SEED_MIN_SIM")
    [[ "$DINO_SEED_FAST" == "1" ]]     && DINO_ARGS+=(--seed_fast)
    [[ "$DINO_FRAME0_ONLY" == "1" ]]   && DINO_ARGS+=(--frame0_only)
    [[ -n "$DINO_SEED_FRAMES" ]]       && DINO_ARGS+=(--seed_frames $DINO_SEED_FRAMES)
    [[ -n "$DINO_SEED_MIN_MESH_SIM" ]] && DINO_ARGS+=(--seed_min_mesh_sim "$DINO_SEED_MIN_MESH_SIM")
    [[ -n "$DINO_SAM2_CHUNK_SIZE" ]]   && DINO_ARGS+=(--chunk_size "$DINO_SAM2_CHUNK_SIZE")
    [[ "$WITH_VIZ" != "1" ]]           && DINO_ARGS+=(--no_viz)
    if [[ "$DINO_USE_SIMPLE" == "1" ]]; then
        # The simple tool ignores the full tool's drift / phase / scan_every
        # flags, so strip those before invoking. Everything in DINO_ARGS that
        # the simple tool accepts (data_h5, calib_yaml, tool_mesh, output_dir,
        # sam2_*, pipeline_tool_masks_dir, tool_name, cam_serials, seed_min_area,
        # seed_max_area, seed_min_sim, seed_frames, seed_min_mesh_sim,
        # chunk_size, no_viz) is preserved as-is. Drop --no_video / --frame0_only
        # / --seed_fast / --mesh_scan_every / --dense_scan_every which only
        # exist on the full tool.
        SIMPLE_ARGS=()
        skip_next=0
        for tok in "${DINO_ARGS[@]}"; do
            if [[ "$skip_next" == "1" ]]; then skip_next=0; continue; fi
            case "$tok" in
                --no_video|--frame0_only|--seed_fast)
                    continue;;
                --mesh_scan_every|--dense_scan_every)
                    skip_next=1; continue;;
            esac
            SIMPLE_ARGS+=("$tok")
        done
        echo "  [dino] using SIMPLE tool: tools/dino_simple_segment.py"
        python "$HOCAP_ROOT/tools/dino_simple_segment.py" "${SIMPLE_ARGS[@]}"
    else
        python "$HOCAP_ROOT/tools/dino_tool_segment.py" "${DINO_ARGS[@]}"
    fi
fi

# ---------- Stage 3: generate_meta ----------
META_YAML_PATH="${SEQUENCE_FOLDER}/meta.yaml"
META_MASKS_H5="${TOOL_MASKS_DIR}/masks.h5"
cd "$HOCAP_ROOT"
# Treat the cached meta.yaml as stale if its baked-in calibration_yaml_path
# no longer points at an existing file (e.g. the meta was generated on the
# cluster under /viscam/... and we're now running locally). Regenerating is
# cheap (streams masks); reusing a stale path makes Stage 4 explode with a
# FileNotFoundError that's hard to read.
META_STALE=0
if [[ -f "$META_YAML_PATH" ]]; then
    META_CAL=$(python3 -c "import yaml; m=yaml.safe_load(open('$META_YAML_PATH')); print(m.get('calibration_yaml_path',''))" 2>/dev/null || true)
    if [[ -n "$META_CAL" && ! -f "$META_CAL" ]]; then
        echo "[3/7] cached meta.yaml references missing calibration: $META_CAL"
        echo "      regenerating with current --calibration_yaml=$CALIBRATION_YAML"
        META_STALE=1
    fi
fi
if [[ "$RESUME" == "1" && "$META_STALE" == "0" && -f "$META_YAML_PATH" && -f "$META_MASKS_H5" ]]; then
    echo "[3/7] [skip] meta.yaml + masks.h5 exist"
else
    echo "[3/7] generate_meta ..."
    GEN_META_ARGS=(
        --h5_path "$H5_PATH"
        --calibration_yaml_path "$CALIBRATION_YAML"
        --models_folder "$MODELS_FOLDER"
        --tool_name "$TOOL_NAME"
        --start_frame "$START_FRAME"
        --x_min -0.7 --x_max 0.6 --y_min -0.7 --y_max 0.6 --z_min -0.5 --z_max 0.4
    )
    # If masks.h5 already exists (DINO ran with --pipeline_mask_format h5,
    # or a prior run, or it was synced from elsewhere), tell generate_meta
    # to slice/copy from that h5 instead of streaming non-existent npz.
    # Source = a temp copy so we can write to the dest in place; if user
    # has only masks.h5 and no per-frame files, this is the only viable path.
    if [[ -f "$META_MASKS_H5" ]]; then
        TMP_MASKS_SRC="${META_MASKS_H5}.gen_meta_src"
        cp -f "$META_MASKS_H5" "$TMP_MASKS_SRC"
        GEN_META_ARGS+=(--masks_h5_source "$TMP_MASKS_SRC")
    fi
    python preprocess/generate_meta.py "${GEN_META_ARGS[@]}"
    [[ -n "${TMP_MASKS_SRC:-}" && -f "${TMP_MASKS_SRC:-}" ]] && rm -f "$TMP_MASKS_SRC"
fi

# Detect object count (auto-picks 1 for single-object DINO output)
NUM_OBJECTS=$(python3 -c "import yaml; m=yaml.safe_load(open('$SEQUENCE_FOLDER/meta.yaml')); print(len(m['object_ids']))")
echo "  detected $NUM_OBJECTS object(s)"

# ---------- Stage 4: per-object 6-DoF tracking (chunked h5) ----------
# fd_pose_solver writes per-frame poses to processed/fd_pose_solver/<obj>/ob_in_world/{abs_frame_id:06d}.txt
# When chunking is enabled we rebuild a chunk-only h5 + meta per chunk so the
# loader's RAM footprint stays bounded (single full h5 = ~30+ GB at 720p×8 cams),
# then call fd_pose_solver with --frame_id_offset $s so per-frame txts land at
# absolute frame ids in the shared output folder. After all chunks finish the
# original h5 + meta are restored so Stage 5's merger sees the full sequence.
_run_fd_pose_solver_unchunked() {
    local obj_idx="$1"
    python -u tools/04-1-4_fd_pose_solver_kalman.py \
        --no_masked_depth \
        --sequence_folder "$SEQUENCE_FOLDER" \
        --activate_2d_tracker --activate_kalman_filter \
        --object_idx "$obj_idx" \
        --track_refine_iter "$TRACK_REFINE_ITER" \
        --rot_thresh "$ROT_THRESH" --trans_thresh "$TRANS_THRESH" \
        --tool_mesh "$TOOL_MESH" 2>&1
}

FD_POSE_FOLDER="${ANNOTATED_PATH}/processed/fd_pose_solver"
MERGED_FIXED="${FD_POSE_FOLDER}/fd_poses_merged_fixed.npy"

# Per-chunk resume helper: returns 0 (skip) iff every absolute frame id in
# [s, e) already has an ob_in_world/{frame_id:06d}.txt for the given object.
_chunk_already_done() {
    local obj_id="$1" s="$2" e="$3"
    local ob_world_dir="${FD_POSE_FOLDER}/${obj_id}/ob_in_world"
    [[ -d "$ob_world_dir" ]] || return 1
    local fr
    for ((fr=s; fr<e; fr++)); do
        printf -v fname "%06d.txt" "$fr"
        [[ -f "${ob_world_dir}/${fname}" ]] || return 1
    done
    return 0
}

if [[ "$SKIP_TRACKING" != "1" ]]; then
    if [[ "$RESUME" == "1" && -f "$MERGED_FIXED" ]]; then
        echo "[4-5/7] [skip] $MERGED_FIXED already exists"
    else
        # Map object index 1..N to object_id from meta.yaml (used for the
        # per-chunk resume check; matches fd_pose_solver's output layout).
        OBJECT_IDS=( $(python3 -c "import yaml; m=yaml.safe_load(open('$SEQUENCE_FOLDER/meta.yaml')); print(' '.join(m['object_ids']))") )

        # Pre-scan: if every abs ob_in_world txt is on disk for every object,
        # there's nothing for Stage 4 to do — the previous tracking finished
        # but the merger crashed / was interrupted. Skip the whole chunked
        # backup-restore dance and jump straight to Stage 5. Saves the
        # 10-30s of meta/masks regeneration that otherwise runs even when
        # every chunk's _chunk_already_done immediately returns true.
        all_tracking_done=1
        if [[ "$RESUME" == "1" ]]; then
            for OBJ_IDX in $(seq 1 $NUM_OBJECTS); do
                OBJ_ID="${OBJECT_IDS[$((OBJ_IDX-1))]}"
                if ! _chunk_already_done "$OBJ_ID" 0 "$N_FRAMES"; then
                    all_tracking_done=0
                    break
                fi
            done
        else
            all_tracking_done=0
        fi
        if [[ "$all_tracking_done" == "1" ]]; then
            echo "[4/7] [skip] all ob_in_world/{abs:06d}.txt present for $NUM_OBJECTS object(s); going straight to merger"
        fi

        use_chunked=0
        if [[ "$all_tracking_done" == "0" && "$OBJECT_CHUNK_SIZE" -gt 0 && "$N_FRAMES" -gt "$OBJECT_CHUNK_SIZE" ]]; then
            use_chunked=1
        fi

        if [[ "$all_tracking_done" == "1" ]]; then
            : # nothing to do for Stage 4 — fall through to Stage 5
        elif [[ $use_chunked -eq 1 ]]; then
            # Back up the full h5, meta, and (if it exists) the full masks.h5
            # so per-chunk regeneration doesn't destroy them. mv for h5
            # (multi-GB; avoid copy); cp for the small files.
            #
            # IMPORTANT: regenerate the full meta against the *current* h5 +
            # local --models_folder BEFORE backing it up. Sequences synced
            # from a cluster often carry a stale meta.yaml whose
            # `models_folder` points at a path that doesn't exist locally; if
            # we just `cp` that as the backup, the post-Stage-4 restore
            # silently brings back the cluster path and downstream stages
            # (and the viewer) can't find object meshes. A fresh meta is
            # authoritative for the rest of the pipeline.
            H5_FULL_BACKUP_OBJ="${H5_REAL_PATH}.full_backup_obj"
            META_FULL_BACKUP_OBJ="${META_YAML_PATH}.full_backup_obj"
            MASKS_FULL_BACKUP_OBJ="${TOOL_MASKS_DIR}/masks.h5.full_backup_obj"

            # Decide whether to source the full masks.h5 from disk for this
            # whole Stage 4 run. If it's already at $TOOL_MASKS_DIR/masks.h5
            # we copy it aside as a stable source (per-chunk generate_meta
            # would otherwise overwrite it with the chunk slice). When npz
            # are also missing, this is the ONLY way generate_meta can build
            # a chunk masks.h5 — npz streaming would silently produce zeros.
            FULL_MASKS_SOURCE_ARG=()
            if [[ -f "${TOOL_MASKS_DIR}/masks.h5" ]]; then
                cp -f "${TOOL_MASKS_DIR}/masks.h5" "$MASKS_FULL_BACKUP_OBJ"
                FULL_MASKS_SOURCE_ARG=(--masks_h5_source "$MASKS_FULL_BACKUP_OBJ")
                echo "[4/7] using existing full masks.h5 as slice source: $MASKS_FULL_BACKUP_OBJ"
            else
                echo "[4/7] no full masks.h5 found; chunks will stream from per-frame npz"
            fi

            # Pre-Stage-4 full-meta refresh (with the local models_folder).
            python preprocess/generate_meta.py \
                --h5_path "$H5_PATH" \
                --calibration_yaml_path "$CALIBRATION_YAML" \
                --models_folder "$MODELS_FOLDER" \
                --tool_name "$TOOL_NAME" \
                --start_frame "$START_FRAME" \
                --x_min -0.6 --x_max 0.6 --y_min -0.5 --y_max 0.6 --z_min -0.5 --z_max 0.4 \
                "${FULL_MASKS_SOURCE_ARG[@]}"

            # If we didn't have a backup yet (npz-only case), the
            # generate_meta call above just *built* masks.h5 from npz; back
            # it up now so per-chunk slicing has a stable source.
            if [[ ! -f "$MASKS_FULL_BACKUP_OBJ" && -f "${TOOL_MASKS_DIR}/masks.h5" ]]; then
                cp -f "${TOOL_MASKS_DIR}/masks.h5" "$MASKS_FULL_BACKUP_OBJ"
                FULL_MASKS_SOURCE_ARG=(--masks_h5_source "$MASKS_FULL_BACKUP_OBJ")
            fi

            mv -f "$H5_REAL_PATH" "$H5_FULL_BACKUP_OBJ"
            cp -f "$META_YAML_PATH" "$META_FULL_BACKUP_OBJ"
            mkdir -p "${ANNOTATED_PATH}/masks"

            n_chunks=$(( (N_FRAMES + OBJECT_CHUNK_SIZE - 1) / OBJECT_CHUNK_SIZE ))
            for OBJ_IDX in $(seq 1 $NUM_OBJECTS); do
                OBJ_ID="${OBJECT_IDS[$((OBJ_IDX-1))]}"
                echo "[4/7] fd_pose_solver object $OBJ_IDX / $NUM_OBJECTS  (chunked h5: $n_chunks x $OBJECT_CHUNK_SIZE frames)"
                chunk_idx=0
                s=0
                while [[ $s -lt $N_FRAMES ]]; do
                    chunk_idx=$((chunk_idx + 1))
                    e=$((s + OBJECT_CHUNK_SIZE))
                    [[ $e -gt $N_FRAMES ]] && e=$N_FRAMES
                    chunk_len=$((e - s))
                    if [[ "$RESUME" == "1" ]] && _chunk_already_done "$OBJ_ID" "$s" "$e"; then
                        echo "  --- chunk $chunk_idx/$n_chunks: abs frames [$s, $e)  [skip: all per-frame txts exist]"
                        s=$e
                        continue
                    fi
                    echo "  --- chunk $chunk_idx/$n_chunks: abs frames [$s, $e) (chunk_len=$chunk_len) ---"
                    # 1) chunk-only h5 (overwrites $H5_REAL_PATH; symlink at $H5_PATH stays valid)
                    rm -f "$H5_REAL_PATH"
                    python tools/00_convert_videos_to_h5.py \
                        --input_dir "$SEQUENCE_FOLDER" \
                        --output_file "$H5_REAL_PATH" \
                        --start_frame "$s" --end_frame "$e"
                    # 2) chunk-only meta.yaml + chunk masks.h5 (slice from
                    #    full backup if available, else stream from npz).
                    python preprocess/generate_meta.py \
                        --h5_path "$H5_PATH" \
                        --calibration_yaml_path "$CALIBRATION_YAML" \
                        --models_folder "$MODELS_FOLDER" \
                        --tool_name "$TOOL_NAME" \
                        --start_frame "$s" \
                        --x_min -0.6 --x_max 0.6 --y_min -0.5 --y_max 0.6 --z_min -0.5 --z_max 0.4 \
                        "${FULL_MASKS_SOURCE_ARG[@]}"
                    # 3) fd_pose_solver: in-h5 idx [0, chunk_len), txts named by abs id (offset=$s)
                    python -u tools/04-1-4_fd_pose_solver_kalman.py \
                        --no_masked_depth \
                        --sequence_folder "$SEQUENCE_FOLDER" \
                        --activate_2d_tracker --activate_kalman_filter \
                        --object_idx "$OBJ_IDX" \
                        --start_frame 0 --end_frame "$chunk_len" \
                        --frame_id_offset "$s" \
                        --track_refine_iter "$TRACK_REFINE_ITER" \
                        --rot_thresh "$ROT_THRESH" --trans_thresh "$TRANS_THRESH" \
                        --tool_mesh "$TOOL_MESH" 2>&1
                    s=$e
                done
            done

            # Restore full h5 + meta + masks.h5 so Stage 5 merger (and any
            # later stages) see the original sequence.
            mv -f "$H5_FULL_BACKUP_OBJ" "$H5_REAL_PATH"
            mv -f "$META_FULL_BACKUP_OBJ" "$META_YAML_PATH"
            if [[ -f "$MASKS_FULL_BACKUP_OBJ" ]]; then
                mv -f "$MASKS_FULL_BACKUP_OBJ" "${TOOL_MASKS_DIR}/masks.h5"
            fi
        else
            for OBJ_IDX in $(seq 1 $NUM_OBJECTS); do
                OBJ_ID="${OBJECT_IDS[$((OBJ_IDX-1))]}"
                if [[ "$RESUME" == "1" ]] && _chunk_already_done "$OBJ_ID" 0 "$N_FRAMES"; then
                    echo "[4/7] fd_pose_solver object $OBJ_IDX / $NUM_OBJECTS  [skip: all per-frame txts exist]"
                else
                    echo "[4/7] fd_pose_solver object $OBJ_IDX / $NUM_OBJECTS  (single pass, $N_FRAMES frames)"
                    _run_fd_pose_solver_unchunked "$OBJ_IDX"
                fi
            done
        fi

        # ---------- Stage 5: multi-view pose merge ----------
        if [[ "$RESUME" == "1" && -f "$MERGED_FIXED" ]]; then
            echo "[5/7] [skip] fd_poses_merged_fixed.npy exists"
        else
            echo "[5/7] fd_pose_merger ..."
            python -u tools/04-2-2_fd_pose_merger_cluster.py --sequence_folder "$SEQUENCE_FOLDER" 2>&1
        fi
    fi
else
    echo "[4-5/7] [skip] tracking skipped"
fi

# ---------- Stage 6: hand reconstruction (optional, chunked for long videos) ----------
# Idempotent: skip if the final optimized hand result already exists.
# If only the intermediate result.pkl exists (reconstruct done but optimize
# failed/interrupted), re-run just the optimize step.
# Long videos: chunk into HAND_CHUNK_SIZE-frame windows (mirrors
# scripts/batch_task_folder_hand_chunked.sh inline). Per chunk: rebuild a
# chunk-only h5 + meta, run reconstruct + optimize, rename outputs to
# *_{S}_{E}.pkl/.npy. Then merge_hand_chunks.py stitches them and the full
# h5+meta are restored so Stage 7 still sees the original sequence.
HAND_FINAL="${ANNOTATED_PATH}/result_hand_optimized.pkl"
HAND_INTERMEDIATE="${ANNOTATED_PATH}/result.pkl"
if [[ -n "$HAND" && "$HAND" != "0" ]]; then
    if [[ -f "$HAND_FINAL" ]]; then
        echo "[6/7] [skip] hand already annotated: $HAND_FINAL"
    elif [[ "$HAND_CHUNK_SIZE" -gt 0 && "$N_FRAMES" -gt "$HAND_CHUNK_SIZE" ]]; then
        n_hand_chunks=$(( (N_FRAMES + HAND_CHUNK_SIZE - 1) / HAND_CHUNK_SIZE ))
        echo "[6/7] hand reconstruction (chunked: $n_hand_chunks x $HAND_CHUNK_SIZE frames)"

        # Back up the full h5 and meta so per-chunk regeneration doesn't
        # destroy them. Use mv for h5 (avoids copying multi-GB files); meta
        # is small so cp is fine.
        H5_FULL_BACKUP="${H5_REAL_PATH}.full_backup"
        META_FULL_BACKUP="${META_YAML_PATH}.full_backup"
        mv -f "$H5_REAL_PATH" "$H5_FULL_BACKUP"
        cp -f "$META_YAML_PATH" "$META_FULL_BACKUP"
        # generate_meta still writes masks/masks.h5 — pre-create dir to avoid failures.
        mkdir -p "${ANNOTATED_PATH}/masks"

        chunk_idx=0
        s=0
        any_chunk_ran=0
        while [[ $s -lt $N_FRAMES ]]; do
            chunk_idx=$((chunk_idx + 1))
            e=$((s + HAND_CHUNK_SIZE - 1))
            [[ $e -ge $N_FRAMES ]] && e=$((N_FRAMES - 1))
            chunk_tag="${s}_${e}"
            chunk_final="${ANNOTATED_PATH}/result_hand_optimized_${chunk_tag}.pkl"
            chunk_recon="${ANNOTATED_PATH}/result_${chunk_tag}.pkl"
            chunk_poses="${ANNOTATED_PATH}/poses_m_${chunk_tag}.npy"
            echo "  --- hand chunk $chunk_idx/$n_hand_chunks: frames [$s, $e] ---"
            if [[ "$RESUME" == "1" && -f "$chunk_final" ]]; then
                echo "    [skip] $chunk_final exists"
                s=$((e + 1)); continue
            fi
            # Per-chunk h5 (overwrites $H5_REAL_PATH; symlink at $H5_PATH stays valid)
            conda activate hocap-annotation
            cd "$HOCAP_ROOT"
            rm -f "$H5_REAL_PATH"
            python tools/00_convert_videos_to_h5.py \
                --input_dir "$SEQUENCE_FOLDER" \
                --output_file "$H5_REAL_PATH" \
                --start_frame "$s" --end_frame "$e"
            # Per-chunk meta with start_frame=s for absolute mask alignment
            python preprocess/generate_meta.py \
                --h5_path "$H5_PATH" \
                --calibration_yaml_path "$CALIBRATION_YAML" \
                --models_folder "$MODELS_FOLDER" \
                --tool_name "$TOOL_NAME" \
                --start_frame "$s" \
                --x_min -0.6 --x_max 0.6 --y_min -0.5 --y_max 0.6 --z_min -0.5 --z_max 0.4
            # reconstruct + optimize (hand env)
            cd "$HAND_ROOT"
            conda activate reconstruct-hand
            rm -f "${ANNOTATED_PATH}/result.pkl" \
                  "${ANNOTATED_PATH}/result_hand_optimized.pkl" \
                  "${ANNOTATED_PATH}/poses_m.npy"
            python cluster_reconstruct.py --sequence_folder "$SEQUENCE_FOLDER" || \
                echo "    [WARN] reconstruct failed for chunk $chunk_tag"
            opt_ok=0
            if [[ -f "${ANNOTATED_PATH}/result.pkl" ]]; then
                if python cluster_optimize_hand.py --file_name "${ANNOTATED_PATH}/result.pkl"; then
                    opt_ok=1
                else
                    echo "    [WARN] optimize failed for chunk $chunk_tag"
                fi
            fi
            mv -f "${ANNOTATED_PATH}/result.pkl" "$chunk_recon" 2>/dev/null || true
            if [[ $opt_ok -eq 1 ]]; then
                mv -f "${ANNOTATED_PATH}/result_hand_optimized.pkl" "$chunk_final" 2>/dev/null || true
                mv -f "${ANNOTATED_PATH}/poses_m.npy" "$chunk_poses" 2>/dev/null || true
            fi
            cd "$HOCAP_ROOT"
            conda activate hocap-annotation
            any_chunk_ran=1
            s=$((e + 1))
        done

        # Merge per-chunk results into the canonical filenames
        if ls "${ANNOTATED_PATH}"/result_hand_optimized_[0-9]*_[0-9]*.pkl >/dev/null 2>&1; then
            echo "  [merge] merging hand chunks ..."
            python "$HOCAP_ROOT/scripts/merge_hand_chunks.py" --annotated_path "$ANNOTATED_PATH" \
                || echo "  [WARN] merge_hand_chunks failed"
            if [[ "$HAND_KEEP_CHUNK_FILES" != "1" ]]; then
                rm -f "${ANNOTATED_PATH}"/result_hand_optimized_[0-9]*_[0-9]*.pkl \
                      "${ANNOTATED_PATH}"/result_[0-9]*_[0-9]*.pkl \
                      "${ANNOTATED_PATH}"/poses_m_[0-9]*_[0-9]*.npy
            fi
        else
            echo "  [WARN] no hand chunk outputs produced — skipping merge"
        fi

        # Restore full h5 + meta so Stage 7 (and re-runs with --resume) see
        # the original sequence.
        mv -f "$H5_FULL_BACKUP" "$H5_REAL_PATH"
        mv -f "$META_FULL_BACKUP" "$META_YAML_PATH"
    else
        echo "[6/7] hand reconstruction (single pass, $N_FRAMES frames) ..."
        cd "$HAND_ROOT"
        conda activate reconstruct-hand
        if [[ -f "$HAND_INTERMEDIATE" ]]; then
            echo "  [skip] $HAND_INTERMEDIATE exists — resuming at optimize step"
        else
            python cluster_reconstruct.py --sequence_folder "$SEQUENCE_FOLDER"
        fi
        if [[ -f "$HAND_INTERMEDIATE" ]]; then
            python cluster_optimize_hand.py --file_name "$HAND_INTERMEDIATE"
        fi
        cd "$HOCAP_ROOT"
        conda activate hocap-annotation
    fi
fi

# ---------- Stage 7: object + joint optimization (optional) ----------
if [[ -n "$OPTIMIZE" && "$OPTIMIZE" != "0" && "$FAKE_OPTIMIZE" == "1" ]]; then
    echo "[7/7] [warn] both --optimize and --fake_optimize given; running real optimizers"
    FAKE_OPTIMIZE=0
fi
JOINT_DIR="${ANNOTATED_PATH}/processed/joint_pose_solver"
JOINT_POSES_O="${JOINT_DIR}/poses_o.npy"
JOINT_POSES_M="${JOINT_DIR}/poses_m.npy"
if [[ "$RESUME" == "1" && -f "$JOINT_POSES_O" && -f "$JOINT_POSES_M" ]]; then
    echo "[7/7] [skip] joint outputs already exist: $JOINT_DIR"
elif [[ -n "$OPTIMIZE" && "$OPTIMIZE" != "0" ]]; then
    echo "[7/7] object + joint optimization (first object only) ..."
    python tools/06-2_object_pose_solver_cluster.py --sequence_folder "$SEQUENCE_FOLDER" --debug
    python tools/07-2_joint_pose_solver_cluster.py  --sequence_folder "$SEQUENCE_FOLDER" --debug
elif [[ "$FAKE_OPTIMIZE" == "1" ]]; then
    echo "[7/7] fake joint result (skip Stage 6+7 optimizers) ..."
    # When --force, propagate --overwrite to rebuild even if outputs exist;
    # otherwise let build_fake_joint_result.py's own no-clobber check kick in
    # (it'll print "outputs already exist" and return cleanly).
    FAKE_ARGS=(--sequence_folder "$SEQUENCE_FOLDER")
    [[ "$RESUME" == "0" ]] && FAKE_ARGS+=(--overwrite)
    python tools/build_fake_joint_result.py "${FAKE_ARGS[@]}"
fi

echo ""
echo "=========================================="
echo "Done. Annotated output at:"
echo "  $ANNOTATED_PATH"
echo "=========================================="
