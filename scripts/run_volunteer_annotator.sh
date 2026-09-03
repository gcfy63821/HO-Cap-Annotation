#!/bin/bash
# Run the object-tracking pipeline on ONE experiment using the masks that came
# out of the volunteer SAM2 annotation (point prompts), instead of the DINOv2
# auto-segmentation.
#
# This is deliberately a *thin* wrapper: the only thing that changes versus the
# auto path is where the masks come from. Everything after the mask stage is
# scripts/run_auto_annotator.sh verbatim — same h5 build, same generate_meta,
# same chunked tools/04-1-4_fd_pose_solver_kalman.py, same multi-view merge in
# tools/04-2-2_fd_pose_merger_cluster.py, same resume semantics.
#
# How the hand-off works:
#   run_auto_annotator.sh's Stage 2 (DINO) already auto-skips when
#   <annotated>/tool_masks/masks.h5 exists. So we simply produce that file first
#   (via volunteer_annotation/internal/prompts_to_masks.py) and then hand over.
#   --skip_masks is also passed so the skip is explicit rather than incidental.
#
# Stages:
#   0. volunteer prompts -> masks.h5 + objects.yaml + roles.yaml   [this script]
#   1-7. run_auto_annotator.sh (h5, generate_meta, per-object FoundationPose
#        tracking, multi-view merge, optional hand + joint optimization)
#
# Frame indexing: volunteers annotate ABSOLUTE mp4 frame indices, so masks.h5 is
# built over the FULL clip (--from_video) regardless of --start_frame/--end_frame.
# generate_meta.py then slices it down to the h5's range via --masks_h5_source.
#
# Object ids in masks.h5 (set by prompts_to_masks.py):
#   1 = primary_tool (the ONLY object listed in objects.yaml, i.e. the only one
#       FoundationPose tracks), 2 = auxiliary_tool, 3 = manipulated_object.
#   The extra roles stay in the mask file for downstream use but are not tracked.
#
# Usage:
#   bash scripts/run_volunteer_annotator.sh \
#     --sequence_folder  /viscam/projects/robotool/data/videos_0101/mallet_crush_dough/20260104_..._34 \
#     --prompts_dir      /viscam/projects/robotool/_va_bundle_v2_prompts/videos_0101/mallet_crush_dough/20260104_..._34/tool_masks/prompts \
#     --calibration_yaml /viscam/projects/robotool/data/videos_0101/realsense_calibrate_0101/realsense_calibration_0101_global_aligned.yaml \
#     --tool_name        rubber_mallet \
#     [--tool_mesh /abs/mesh.obj]   # skips $MODELS_FOLDER/<tool>/ lookup
#     [--hand 0]                    # default 0 — object multi-view only
#     [--fake_optimize]             # assemble joint outputs without the optimizer
#     [--masks_only]                # stop after masks.h5 (no tracking)
#     [--force]                     # redo masks AND every downstream stage
#     [--object_chunk_size 600] [--start_frame 0] [--end_frame N] ...
#
# Any flag this script doesn't recognise is forwarded to run_auto_annotator.sh
# unchanged, so its full CLI stays available.
#
# The (sequence_folder, prompts_dir, tool_name, tool_mesh) tuple for every
# annotated exp is produced by scripts/volunteer_exp_index.py.

set -eu

HOCAP_ROOT="${HOCAP_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_SCRIPT="$HOCAP_ROOT/scripts/run_auto_annotator.sh"
PROMPTS_TO_MASKS="$HOCAP_ROOT/volunteer_annotation/internal/prompts_to_masks.py"

SEQUENCE_FOLDER=""
PROMPTS_DIR=""
TOOL_NAME=""
TOOL_MESH=""
MASKS_ONLY=0
FORCE=0
HAND_SET=0
MAX_FRAMES=""
PASSTHRU=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sequence_folder)  SEQUENCE_FOLDER="$2"; shift 2 ;;
        --prompts_dir)      PROMPTS_DIR="$2";     shift 2 ;;
        --tool_name)        TOOL_NAME="$2";       shift 2 ;;
        --tool_mesh)        TOOL_MESH="$2";       shift 2 ;;
        --masks_only)       MASKS_ONLY=1;         shift ;;
        --mask_max_frames)  MAX_FRAMES="$2";      shift 2 ;;   # quick smoke test
        --force|--no_resume)
            # Forwarded too: run_auto_annotator interprets it for stages 1-7,
            # we interpret it for the mask stage.
            FORCE=1; PASSTHRU+=("--force"); shift ;;
        --hand)             HAND_SET=1; PASSTHRU+=("$1" "$2"); shift 2 ;;
        *)                  PASSTHRU+=("$1");     shift ;;
    esac
done

[[ -n "$SEQUENCE_FOLDER" ]] || { echo "Error: --sequence_folder is required"; exit 1; }
[[ -n "$PROMPTS_DIR"     ]] || { echo "Error: --prompts_dir is required"; exit 1; }
[[ -d "$SEQUENCE_FOLDER" ]] || { echo "Error: no such sequence folder: $SEQUENCE_FOLDER"; exit 1; }
[[ -d "$PROMPTS_DIR"     ]] || { echo "Error: no such prompts dir: $PROMPTS_DIR"; exit 1; }
SEQUENCE_FOLDER="$(cd "$SEQUENCE_FOLDER" && pwd)"
PROMPTS_DIR="$(cd "$PROMPTS_DIR" && pwd)"

# Volunteer flagged the clip as unusable — nothing to annotate.
if [[ -f "$(dirname "$PROMPTS_DIR")/BAD.json" ]]; then
    echo "[skip] volunteer flagged this exp bad: $(dirname "$PROMPTS_DIR")/BAD.json"
    exit 0
fi

# Object multi-view tracking only, unless the caller asked for hand.
if [[ "$HAND_SET" == "0" ]]; then
    PASSTHRU+=(--hand 0)
fi

# ---------- derive the annotated tool_masks dir (mirrors run_auto_annotator.sh) ----------
EXP_NAME="$(basename "$SEQUENCE_FOLDER")"
TASK_FOLDER="$(dirname "$SEQUENCE_FOLDER")"
TASK_NAME="$(basename "$TASK_FOLDER")"
VIDEOS_ROOT="$(dirname "$TASK_FOLDER")"
VIDEOS_FOLDER_NAME="$(basename "$VIDEOS_ROOT")"
BASE="$(dirname "$VIDEOS_ROOT")"
ANNOTATED_PATH="${BASE}/${VIDEOS_FOLDER_NAME}_annotated/${TASK_NAME}/${EXP_NAME}"
TOOL_MASKS_DIR="${ANNOTATED_PATH}/tool_masks"

echo "=========================================="
echo "exp        : $TASK_NAME/$EXP_NAME"
echo "sequence   : $SEQUENCE_FOLDER"
echo "prompts    : $PROMPTS_DIR ($(ls "$PROMPTS_DIR"/cam*.json 2>/dev/null | wc -l) camera(s))"
echo "tool_masks : $TOOL_MASKS_DIR"
echo "tool       : ${TOOL_NAME:-<from --tool_mesh>}"
echo "=========================================="

# ---------- Stage 0: volunteer prompts -> masks.h5 ----------
if [[ "$FORCE" == "1" ]]; then
    rm -f "$TOOL_MASKS_DIR/masks.h5"
fi
mkdir -p "$TOOL_MASKS_DIR"

MASK_ARGS=(
    --exp         "$SEQUENCE_FOLDER"
    --prompts_dir "$PROMPTS_DIR"
    --out_dir     "$TOOL_MASKS_DIR"
    --from_video
    --resume
)
# Reuse the job's fast scratch for the per-camera JPEG dump if the sbatch
# wrapper set one up; /tmp on a compute node is often small and shared.
[[ -n "${H5_SCRATCH_DIR:-}" ]] && MASK_ARGS+=(--tmp_dir "$H5_SCRATCH_DIR/va_frames")
[[ -n "$MAX_FRAMES" ]]        && MASK_ARGS+=(--max_frames "$MAX_FRAMES")

echo "[0/7] volunteer prompts -> masks.h5 ..."
python -u "$PROMPTS_TO_MASKS" "${MASK_ARGS[@]}"

if [[ ! -f "$TOOL_MASKS_DIR/masks.h5" || ! -f "$TOOL_MASKS_DIR/objects.yaml" ]]; then
    echo "Error: prompts_to_masks.py did not produce masks.h5 + objects.yaml in $TOOL_MASKS_DIR"
    exit 1
fi

if [[ "$MASKS_ONLY" == "1" ]]; then
    echo "[masks_only] stopping after mask generation."
    exit 0
fi

# ---------- resolve tool_name (run_auto_annotator requires it) ----------
# Priority: --tool_name > basename of the mesh's parent dir (keeps output paths
# stable per tool, same convention as the legacy mesh_map path) > the primary
# tool the volunteer picked, as recorded in objects.yaml.
if [[ -z "$TOOL_NAME" && -n "$TOOL_MESH" ]]; then
    TOOL_NAME="$(basename "$(dirname "$TOOL_MESH")")"
    echo "[tool] derived from mesh path: $TOOL_NAME"
fi
if [[ -z "$TOOL_NAME" ]]; then
    TOOL_NAME="$(python3 -c "
import yaml,sys
objs = yaml.safe_load(open('$TOOL_MASKS_DIR/objects.yaml')).get('objects') or []
print(objs[0] if objs else '')" 2>/dev/null || true)"
    # `|| true`: under `set -e` a false test as the last command of this block
    # would abort before the friendly error message below.
    [[ -n "$TOOL_NAME" ]] && echo "[tool] taken from volunteer objects.yaml: $TOOL_NAME" || true
fi
if [[ -z "$TOOL_NAME" ]]; then
    echo "Error: could not resolve a tool name. Pass --tool_name (see scripts/volunteer_exp_index.py)."
    exit 1
fi

# ---------- Stages 1-7: the existing auto pipeline, minus DINO ----------
RUN_ARGS=(
    --sequence_folder "$SEQUENCE_FOLDER"
    --skip_masks
    --tool_name "$TOOL_NAME"
)
[[ -n "$TOOL_MESH" ]] && RUN_ARGS+=(--tool_mesh "$TOOL_MESH")

echo "[1/7] handing off to run_auto_annotator.sh ..."
bash "$RUN_SCRIPT" "${RUN_ARGS[@]}" ${PASSTHRU[@]+"${PASSTHRU[@]}"}
