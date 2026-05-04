#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name remask_p
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/%j.err
#
# Re-run the auto annotation pipeline ONLY for exps that:
#   (1) have <annotated>/<exp>/mask_prompts.json   (re-annotated by
#       scripts/simple_mask_annotator.py + scripts/batch_redo_masks.py)
#   AND
#   (2) are labeled "visible" in <task>/frame0_visibility.yaml.
#
# Each eligible exp is re-processed FROM Stage 4 (fd_pose_solver) ONWARDS:
# the mask was just regenerated, so the object-pose tracker must be redone.
# Hand outputs (Stages 2, 3, 5) are PRESERVED — if result_hand_optimized.pkl
# / processed/mp/* / processed/mano_pose_solver/* already exist they will
# be reused (auto-resume), and only missing hand stages get re-run.
#
# What this means concretely:
#   For every eligible exp the wrapper does:
#     rm -rf <ann>/processed/fd_pose_solver       # Stage 4
#     rm -rf <ann>/processed/object_pose_solver   # Stage 6
#     rm -rf <ann>/processed/joint_pose_solver    # Stage 7
#   then invokes scripts/run_auto_annotator.sh as usual. Auto-resume in
#   that script then:
#     skips Stage 0 (h5 cached on /dev/shm/$H5_SCRATCH if found)
#     skips Stage 1 (DINO/SAM2 segmentation — masks are already in
#                    tool_masks/masks.h5 from batch_redo_masks.py)
#     skips Stages 2/3/5 if hand outputs exist; runs them otherwise
#     re-runs Stage 4 (fd_pose_solver) — outputs were deleted
#     re-runs Stage 6/7 (object_pose_solver, joint_pose_solver) —
#         ditto, depend on the re-tracked object poses
#
# Tool name resolution priority (high → low):
#   1. --force_tool NAME                     (CLI override)
#   2. tool_name from mask_prompts.json      (whatever the user typed
#                                              into simple_mask_annotator)
#   3. --tool_name_map_json (default-on)     JSON {key: tool_name}; key
#                                              matched as longest case-
#                                              insensitive substring of
#                                              "<task>/<exp>". Mesh
#                                              resolved in $MODELS_FOLDER/
#                                              <tool>/{textured_mesh ,
#                                              cleaned_mesh_10000 , mesh}.obj
#   4. --mesh_map_json PATH                  (legacy: gives full mesh path)
#   5. tool_keyword_mapping.yaml             (manual 词表)
#   6. heuristic match_tool_name.py
#
# Resume semantics:
#   --skip_already_redone         (DEFAULT ON) skip an exp if
#                                   processed/fd_pose_solver/fd_poses_merged_fixed.npy
#                                   is newer than mask_prompts.json — the
#                                   redo already ran successfully and the
#                                   prompts haven't changed since.
#   --no_skip_already_redone      always redo every eligible exp.
#   --force                       passes --force to run_auto_annotator
#                                   (redoes EVERY stage from scratch,
#                                   including hand). Use only if you want
#                                   a totally clean re-run.
#
# Usage:
#   sbatch scripts/sbatch_run_remask_pipeline.sh \
#       --videos_root /viscam/projects/robotool/data/videos_0102 \
#       [--hand 1]                            (default 1; 0 disables)
#       [--object_chunk_size 600]             (default 600)
#       [--fake_optimize]                     (recommended on long videos)
#       [--no_skip_already_redone]            (force redo every eligible)
#       [--force]                             (redo EVERY stage incl. hand)
#       [--force_tool NAME]
#       [--tool_name_map_json [PATH]]         (default: $TOOL_NAME_MAP_JSON_DEFAULT)
#       [--no_tool_name_map_json]
#       [--mesh_map_json [PATH]]              (legacy)
#       [--mapping_yaml PATH]
#       [--mapping_only]
#       [--start_frame 0] [--end_frame N]
#       [--object_chunk_overlap N]
#       [--hand_chunk_size N]
#       [--long_video_threshold N]
#       [--frame0_only]
#       [--mesh_scan_every N] [--dense_scan_every N]
#       [--seed_min_area N] [--seed_max_area N] [--seed_min_sim F]
#       [--dry_run]                           (list what would run, don't actually run)

set -u

# ---------- cluster paths (override via env if your layout differs) ----------
export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export HAND_ROOT="${HAND_ROOT:-/viscam/u/chenrq/crq_ws/robotool/HandReconstruction}"
export MODELS_FOLDER="${MODELS_FOLDER:-/viscam/u/chenrq/models}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
RUN_SCRIPT="$HOCAP_ROOT/scripts/run_auto_annotator.sh"
MATCH_SCRIPT="$HOCAP_ROOT/scripts/match_tool_name.py"

# SAM2 paths (kept in env even though Stage 1 will be skipped — downstream
# tools may still import sam2 utilities).
export SAM2_ROOT="${SAM2_ROOT:-/viscam/u/chenrq/crq_ws/robotool/sam2}"
export SAM2_CKPT="${SAM2_CKPT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt}"
export SAM2_VIDEO_CFG="${SAM2_VIDEO_CFG:-$HOCAP_ROOT/config/sam2_config/sam2.1_hiera_l.yaml}"
export SAM2_IMAGE_CFG="${SAM2_IMAGE_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"

# ---------- perf env ----------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export H5_COMPRESSION="lzf"
export PYOPENGL_PLATFORM="egl"

# ---------- /dev/shm scratch ----------
_JOBTAG="${SLURM_JOB_ID:-nonslurm_$$}"
H5_SCRATCH="/dev/shm/${USER}/${_JOBTAG}"
mkdir -p "$H5_SCRATCH"
cleanup_shm() {
    [[ -d "$H5_SCRATCH" ]] && rm -rf "$H5_SCRATCH" 2>/dev/null || true
}
trap cleanup_shm EXIT INT TERM HUP

# ---------- arg parsing ----------
VIDEOS_ROOT=""
CALIBRATION_YAML=""
HAND="1"                  # default: enable hand reconstruction (auto-resume preserves existing)
FAKE_OPTIMIZE=0
FORCE=0
SKIP_ALREADY_REDONE=1     # DEFAULT ON
FORCE_TOOL=""
MAPPING_YAML=""
MAPPING_ONLY=0

TOOL_NAME_MAP_JSON_DEFAULT="${TOOL_NAME_MAP_JSON_DEFAULT:-/viscam/u/chenrq/crq_ws/scripts/mesh_name_mapping.json}"
TOOL_NAME_MAP_JSON=""
TOOL_NAME_MAP_ONLY=0
TOOL_NAME_MAP_DISABLED=0

MESH_MAP_JSON_DEFAULT="${MESH_MAP_JSON_DEFAULT:-/viscam/u/kehanli/temp_scripts/dino_mesh_map_tool_mesh_full.json}"
MESH_MAP_JSON=""
MESH_MAP_ONLY=0
START_FRAME=0
END_FRAME=""
ROT_THRESH=15
TRANS_THRESH=0.03
TRACK_REFINE_ITER=10
OBJECT_CHUNK_SIZE=600
OBJECT_CHUNK_OVERLAP=""
HAND_CHUNK_SIZE=""
LONG_VIDEO_THRESHOLD=""
FRAME0_ONLY=0
DINO_MESH_SCAN_EVERY=""
DINO_DENSE_SCAN_EVERY=""
DINO_SEED_MIN_AREA=""
DINO_SEED_MAX_AREA=""
DINO_SEED_MIN_SIM=""
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --videos_root)        VIDEOS_ROOT="$2"; shift 2;;
        --calibration_yaml)   CALIBRATION_YAML="$2"; shift 2;;
        --hand)               HAND="$2"; shift 2;;
        --fake_optimize)      FAKE_OPTIMIZE=1; shift;;
        --force|--no_resume)  FORCE=1; shift;;
        --skip_already_redone)    SKIP_ALREADY_REDONE=1; shift;;
        --no_skip_already_redone) SKIP_ALREADY_REDONE=0; shift;;
        --force_tool)         FORCE_TOOL="$2"; shift 2;;
        --mapping_yaml)       MAPPING_YAML="$2"; shift 2;;
        --mapping_only)       MAPPING_ONLY=1; shift;;
        --tool_name_map_json)
            if [[ -n "${2:-}" && "${2:0:2}" != "--" ]]; then
                TOOL_NAME_MAP_JSON="$2"; shift 2
            else
                TOOL_NAME_MAP_JSON="$TOOL_NAME_MAP_JSON_DEFAULT"; shift
            fi
            ;;
        --tool_name_map_only)    TOOL_NAME_MAP_ONLY=1; shift;;
        --no_tool_name_map_json) TOOL_NAME_MAP_DISABLED=1; shift;;
        --mesh_map_json)
            if [[ -n "${2:-}" && "${2:0:2}" != "--" ]]; then
                MESH_MAP_JSON="$2"; shift 2
            else
                MESH_MAP_JSON="$MESH_MAP_JSON_DEFAULT"; shift
            fi
            ;;
        --mesh_map_only)      MESH_MAP_ONLY=1; shift;;
        --models_folder)      export MODELS_FOLDER="$2"; shift 2;;
        --start_frame)        START_FRAME="$2"; shift 2;;
        --end_frame)          END_FRAME="$2"; shift 2;;
        --rot_thresh)         ROT_THRESH="$2"; shift 2;;
        --trans_thresh)       TRANS_THRESH="$2"; shift 2;;
        --track_refine_iter)  TRACK_REFINE_ITER="$2"; shift 2;;
        --object_chunk_size)  OBJECT_CHUNK_SIZE="$2"; shift 2;;
        --object_chunk_overlap) OBJECT_CHUNK_OVERLAP="$2"; shift 2;;
        --hand_chunk_size)    HAND_CHUNK_SIZE="$2"; shift 2;;
        --long_video_threshold) LONG_VIDEO_THRESHOLD="$2"; shift 2;;
        --frame0_only)        FRAME0_ONLY=1; shift;;
        --mesh_scan_every)    DINO_MESH_SCAN_EVERY="$2"; shift 2;;
        --dense_scan_every)   DINO_DENSE_SCAN_EVERY="$2"; shift 2;;
        --seed_min_area)      DINO_SEED_MIN_AREA="$2"; shift 2;;
        --seed_max_area)      DINO_SEED_MAX_AREA="$2"; shift 2;;
        --seed_min_sim)       DINO_SEED_MIN_SIM="$2"; shift 2;;
        --dry_run)            DRY_RUN=1; shift;;
        -h|--help)            sed -n '2,84p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$VIDEOS_ROOT" ]]; then
    echo "Error: --videos_root is required"; exit 1
fi
VIDEOS_ROOT="$(cd "$VIDEOS_ROOT" && pwd)"
[[ -d "$VIDEOS_ROOT" ]] || { echo "Error: not a directory: $VIDEOS_ROOT"; exit 1; }
ANNOTATED_ROOT="$(dirname "$VIDEOS_ROOT")/$(basename "$VIDEOS_ROOT")_annotated"

echo "=========================================="
echo "REMASK PIPELINE"
echo "  job        : ${SLURM_JOB_ID:-<non-slurm>}   node: $(hostname)"
echo "  cpus/omp   : ${SLURM_CPUS_PER_TASK:-4}"
echo "  videos_root: $VIDEOS_ROOT"
echo "  annotated  : $ANNOTATED_ROOT"
echo "  models     : $MODELS_FOLDER"
echo "  scratch    : $H5_SCRATCH"
echo "  hand       : $HAND  fake_optim=$FAKE_OPTIMIZE  obj_chunk=$OBJECT_CHUNK_SIZE"
echo "  skip_already_redone: $SKIP_ALREADY_REDONE   force: $FORCE"
echo "=========================================="

# ---------- conda + ffmpeg ----------
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg not on PATH"; exit 1
fi

# ---------- discover calibration ----------
if [[ -z "$CALIBRATION_YAML" ]]; then
    CAL_FOLDER="$(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' | head -n 1)"
    if [[ -z "$CAL_FOLDER" ]]; then
        CALIBRATION_YAML="$(find "$VIDEOS_ROOT" -maxdepth 1 -type f -name 'realsense_calibration_*_global_aligned.yaml' | head -n 1)"
        [[ -z "$CALIBRATION_YAML" ]] && CALIBRATION_YAML="$(find "$VIDEOS_ROOT" -maxdepth 1 -type f -name 'realsense_calibration_*.yaml' ! -name '*_aligned.yaml' | head -n 1)"
    else
        ALIGNED="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name '*_global_aligned.yaml' | head -n 1)"
        if [[ -n "$ALIGNED" ]]; then
            CALIBRATION_YAML="$ALIGNED"
        else
            CALIBRATION_YAML="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name 'realsense_calibration_*.yaml' ! -name '*_aligned.yaml' | head -n 1)"
        fi
    fi
fi
if [[ -z "$CALIBRATION_YAML" || ! -f "$CALIBRATION_YAML" ]]; then
    echo "Error: could not find a calibration yaml under $VIDEOS_ROOT"
    echo "       Pass one explicitly via --calibration_yaml /abs/path.yaml"
    exit 1
fi
echo "calibration: $CALIBRATION_YAML"

# ---------- discover mapping yaml (词表) ----------
if [[ -z "$MAPPING_YAML" ]]; then
    if [[ -f "$VIDEOS_ROOT/tool_keyword_mapping.yaml" ]]; then
        MAPPING_YAML="$VIDEOS_ROOT/tool_keyword_mapping.yaml"
    fi
fi
if [[ -n "$MAPPING_YAML" && -f "$MAPPING_YAML" ]]; then
    echo "mapping    : $MAPPING_YAML  (mapping_only=$MAPPING_ONLY)"
fi

# ---------- tool_name_map auto-detect ----------
if [[ "$TOOL_NAME_MAP_DISABLED" == "1" ]]; then
    TOOL_NAME_MAP_JSON=""
    echo "tool_map   : <disabled by --no_tool_name_map_json>"
elif [[ -z "$TOOL_NAME_MAP_JSON" ]]; then
    if [[ -f "$TOOL_NAME_MAP_JSON_DEFAULT" ]]; then
        TOOL_NAME_MAP_JSON="$TOOL_NAME_MAP_JSON_DEFAULT"
        echo "tool_map   : $TOOL_NAME_MAP_JSON  (auto-detected default)"
    fi
fi

# ---------- discover tasks ----------
TASKS=()
while IFS= read -r -d '' d; do
    [[ "$(basename "$d")" == realsense_calibrate_* ]] && continue
    [[ "$(basename "$d")" == ref_pc* ]] && continue
    [[ "$(basename "$d")" == posts* ]] && continue
    [[ "$(basename "$d")" == "first_frame" ]] && continue
    TASKS+=("$d")
done < <(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
[[ "${#TASKS[@]}" -eq 0 ]] && { echo "Error: no task folders under $VIDEOS_ROOT"; exit 1; }
echo "tasks (${#TASKS[@]}):"; for t in "${TASKS[@]}"; do echo "  - $(basename "$t")"; done

# ---------- helper: get visibility label from <task>/frame0_visibility.yaml ----------
# echoes the label or empty string. Prints nothing on yaml absent / parse fail.
get_visibility_label() {
    local task_dir="$1"
    local exp_name="$2"
    local f0_yaml="${task_dir}/frame0_visibility.yaml"
    [[ -f "$f0_yaml" ]] || { echo ""; return; }
    F0_YAML="$f0_yaml" EXP_NAME="$exp_name" python3 - <<'PY' 2>/dev/null
import os, yaml, sys
try:
    data = yaml.safe_load(open(os.environ["F0_YAML"]).read()) or {}
    ann = (data.get("annotations") or {})
    print(ann.get(os.environ["EXP_NAME"], ""))
except Exception:
    print("")
PY
}

# ---------- helper: read tool_name from <ann>/mask_prompts.json ----------
get_prompt_tool_name() {
    local prompts_json="$1"
    [[ -f "$prompts_json" ]] || { echo ""; return; }
    PJ="$prompts_json" python3 - <<'PY' 2>/dev/null
import os, json, sys
try:
    d = json.load(open(os.environ["PJ"]))
    tn = (d.get("tool_name") or "").strip()
    print(tn)
except Exception:
    print("")
PY
}

# ---------- per-exp runner ----------
TOTAL=0; OK=0; SKIP=0; FAIL=0
SKIP_NOT_VISIBLE=0; SKIP_NO_PROMPTS=0; SKIP_ALREADY=0
FAILED=()

for TASK_DIR in "${TASKS[@]}"; do
    TASK_NAME="$(basename "$TASK_DIR")"
    ANN_TASK_DIR="${ANNOTATED_ROOT}/${TASK_NAME}"

    echo ""
    echo "##########################################"
    echo "# task: $TASK_NAME"
    echo "##########################################"

    # Iterate exp candidates
    EXP_NAMES=()
    while IFS= read -r d; do
        EXP_NAMES+=("$(basename "$d")")
    done < <(find "$TASK_DIR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort)
    [[ "${#EXP_NAMES[@]}" -eq 0 ]] && { echo "  (no exps under $TASK_NAME)"; continue; }

    for EXP_NAME in "${EXP_NAMES[@]}"; do
        EXP_DIR="${TASK_DIR}/${EXP_NAME}"
        ANN="${ANN_TASK_DIR}/${EXP_NAME}"
        [[ -d "$EXP_DIR" ]] || continue
        # Must have at least one cam*_rgb.mp4 to be a real exp.
        compgen -G "$EXP_DIR/cam*_rgb.mp4" >/dev/null 2>&1 || continue

        TOTAL=$((TOTAL+1))

        # ---- (1) require mask_prompts.json ----
        PROMPTS_JSON="${ANN}/mask_prompts.json"
        if [[ ! -f "$PROMPTS_JSON" ]]; then
            echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [skip: no mask_prompts.json]"
            SKIP=$((SKIP+1)); SKIP_NO_PROMPTS=$((SKIP_NO_PROMPTS+1)); continue
        fi

        # ---- (2) require visibility=visible ----
        LABEL="$(get_visibility_label "$TASK_DIR" "$EXP_NAME")"
        if [[ "$LABEL" != "visible" ]]; then
            echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [skip: visibility='${LABEL:-<unlabeled>}' (need 'visible')]"
            SKIP=$((SKIP+1)); SKIP_NOT_VISIBLE=$((SKIP_NOT_VISIBLE+1)); continue
        fi

        # ---- (resume) skip exps already redone after the latest prompts edit ----
        FD_MERGED="${ANN}/processed/fd_pose_solver/fd_poses_merged_fixed.npy"
        if [[ "$SKIP_ALREADY_REDONE" == "1" && "$FORCE" != "1" && -f "$FD_MERGED" ]]; then
            if [[ "$FD_MERGED" -nt "$PROMPTS_JSON" ]]; then
                echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [skip: fd_poses_merged_fixed.npy newer than prompts json]"
                SKIP=$((SKIP+1)); SKIP_ALREADY=$((SKIP_ALREADY+1)); continue
            fi
        fi

        # ---- resolve tool name + mesh ----
        # Priority:
        #   1. --force_tool
        #   2. tool_name field inside mask_prompts.json
        #   3. tool_name_map_json hit
        #   4. mesh_map_json hit (legacy)
        #   5. mapping_yaml + heuristic match_tool_name.py
        TOOL=""
        MESH=""
        MAPPED_VIA=""

        # 1) --force_tool wins outright.
        if [[ -n "$FORCE_TOOL" ]]; then
            TOOL="$FORCE_TOOL"
            MAPPED_VIA="force_tool"
        fi

        # 2) tool_name from mask_prompts.json
        if [[ -z "$TOOL" ]]; then
            P_TOOL="$(get_prompt_tool_name "$PROMPTS_JSON")"
            if [[ -n "$P_TOOL" ]]; then
                TOOL="$P_TOOL"
                MAPPED_VIA="prompts_json:$P_TOOL"
            fi
        fi

        # 3) tool_name_map_json
        if [[ -z "$TOOL" && -n "$TOOL_NAME_MAP_JSON" ]]; then
            MAPPED=$(TOOL_NAME_MAP_JSON="$TOOL_NAME_MAP_JSON" \
                     TASK_NAME="$TASK_NAME" \
                     EXP_NAME="$EXP_NAME" python3 - <<'PY'
import json, os, sys
mp = json.load(open(os.environ["TOOL_NAME_MAP_JSON"]))
hay = f"{os.environ['TASK_NAME']}/{os.environ['EXP_NAME']}".lower()
hits = [(k, v) for k, v in mp.items() if k.lower() in hay]
if not hits:
    sys.exit(0)
hits.sort(key=lambda kv: -len(kv[0]))
print(f"{hits[0][0]}\t{hits[0][1]}")
PY
            )
            if [[ -n "$MAPPED" ]]; then
                KEY="${MAPPED%%$'\t'*}"
                TOOL="${MAPPED##*$'\t'}"
                MAPPED_VIA="tool_name_map_json:$KEY -> $TOOL"
            elif [[ "$TOOL_NAME_MAP_ONLY" == "1" ]]; then
                echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [skip: no tool_name_map JSON match (--tool_name_map_only)]"
                SKIP=$((SKIP+1)); continue
            fi
        fi

        # 4) legacy mesh_map_json
        if [[ -z "$TOOL" && -n "$MESH_MAP_JSON" ]]; then
            MAPPED=$(MESH_MAP_JSON="$MESH_MAP_JSON" \
                     TASK_NAME="$TASK_NAME" \
                     EXP_NAME="$EXP_NAME" python3 - <<'PY'
import json, os, sys
mp = json.load(open(os.environ["MESH_MAP_JSON"]))
hay = f"{os.environ['TASK_NAME']}/{os.environ['EXP_NAME']}".lower()
hits = [(k, v) for k, v in mp.items() if k.lower() in hay]
if not hits:
    sys.exit(0)
hits.sort(key=lambda kv: -len(kv[0]))
print(f"{hits[0][0]}\t{hits[0][1]}")
PY
            )
            if [[ -n "$MAPPED" ]]; then
                KEY="${MAPPED%%$'\t'*}"
                MESH_PATH="${MAPPED##*$'\t'}"
                if [[ -f "$MESH_PATH" ]]; then
                    MESH="$MESH_PATH"
                    TOOL="$(basename "$(dirname "$MESH_PATH")")"
                    MAPPED_VIA="mesh_map_json:$KEY"
                fi
            elif [[ "$MESH_MAP_ONLY" == "1" ]]; then
                echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [skip: no mesh_map JSON match (--mesh_map_only)]"
                SKIP=$((SKIP+1)); continue
            fi
        fi

        # 5) heuristic match_tool_name.py
        if [[ -z "$TOOL" && -f "$MATCH_SCRIPT" ]]; then
            MATCH_ARGS=(
                --models_folder "$MODELS_FOLDER"
                --task_name "$TASK_NAME"
                --exp_name "$EXP_NAME"
            )
            [[ -n "$MAPPING_YAML" && -f "$MAPPING_YAML" ]] && MATCH_ARGS+=(--mapping_yaml "$MAPPING_YAML")
            [[ "$MAPPING_ONLY" == "1" ]] && MATCH_ARGS+=(--require_mapping)
            TOOL=$(python "$MATCH_SCRIPT" "${MATCH_ARGS[@]}" 2>/tmp/match_err.$$ || true)
            MATCH_RC=$?
            MATCH_ERR="$(cat /tmp/match_err.$$ 2>/dev/null)"; rm -f /tmp/match_err.$$
            if [[ $MATCH_RC -eq 3 ]]; then
                echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [skip: no mapping keyword (--mapping_only)]"
                SKIP=$((SKIP+1)); continue
            fi
            if [[ -z "$TOOL" ]]; then
                echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [FAIL: tool matcher returned empty -- $MATCH_ERR]"
                FAIL=$((FAIL+1)); FAILED+=("$TASK_NAME/$EXP_NAME:no_match"); continue
            fi
            MAPPED_VIA="${MAPPED_VIA:-heuristic}"
        fi

        if [[ -z "$TOOL" ]]; then
            echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [FAIL: tool name unresolved]"
            FAIL=$((FAIL+1)); FAILED+=("$TASK_NAME/$EXP_NAME:no_match"); continue
        fi

        # Mesh fallback if not already set: scan $MODELS_FOLDER/$TOOL/.
        if [[ -z "$MESH" ]]; then
            for cand in textured_mesh.obj cleaned_mesh_10000.obj mesh.obj; do
                if [[ -f "$MODELS_FOLDER/$TOOL/$cand" ]]; then
                    MESH="$MODELS_FOLDER/$TOOL/$cand"; break
                fi
            done
        fi
        if [[ -z "$MESH" ]]; then
            echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [FAIL: no mesh in $MODELS_FOLDER/$TOOL/]"
            FAIL=$((FAIL+1)); FAILED+=("$TASK_NAME/$EXP_NAME:no_mesh"); continue
        fi

        echo ""
        echo "------------------------------------------"
        echo "[$TOTAL] $TASK_NAME / $EXP_NAME"
        echo "  tool   : $TOOL"
        echo "  mesh   : $MESH"
        echo "  via    : $MAPPED_VIA"
        echo "  prompts: $PROMPTS_JSON"
        echo "------------------------------------------"

        # ---- delete object-pose stage outputs so auto-resume re-runs them ----
        # Hand outputs are PRESERVED (processed/mp/, processed/mano_pose_solver/,
        # result_hand_optimized.pkl). Stage 1 / tool_masks/masks.h5 also kept,
        # since the mask was just regenerated by batch_redo_masks.py.
        if [[ "$DRY_RUN" != "1" ]]; then
            for sub in fd_pose_solver object_pose_solver joint_pose_solver; do
                if [[ -d "${ANN}/processed/${sub}" ]]; then
                    rm -rf "${ANN}/processed/${sub}"
                    echo "  rm -rf ${ANN}/processed/${sub}"
                fi
            done
        else
            echo "  [dry_run] would clear processed/{fd_pose_solver,object_pose_solver,joint_pose_solver}/"
            continue
        fi

        # ---- assemble run_auto_annotator.sh args ----
        RUN_ARGS=(
            --sequence_folder "$EXP_DIR"
            --calibration_yaml "$CALIBRATION_YAML"
            --tool_name "$TOOL"
            --tool_mesh "$MESH"
            --models_folder "$MODELS_FOLDER"
            --h5_scratch_dir "$H5_SCRATCH"
            --start_frame "$START_FRAME"
            --rot_thresh "$ROT_THRESH"
            --trans_thresh "$TRANS_THRESH"
            --track_refine_iter "$TRACK_REFINE_ITER"
            --object_chunk_size "$OBJECT_CHUNK_SIZE"
        )
        [[ -n "$END_FRAME" ]]                && RUN_ARGS+=(--end_frame "$END_FRAME")
        [[ -n "$HAND" ]]                     && RUN_ARGS+=(--hand "$HAND")
        [[ -n "$HAND_CHUNK_SIZE" ]]          && RUN_ARGS+=(--hand_chunk_size "$HAND_CHUNK_SIZE")
        [[ -n "$OBJECT_CHUNK_OVERLAP" ]]     && RUN_ARGS+=(--object_chunk_overlap "$OBJECT_CHUNK_OVERLAP")
        [[ -n "$LONG_VIDEO_THRESHOLD" ]]     && RUN_ARGS+=(--long_video_threshold "$LONG_VIDEO_THRESHOLD")
        [[ "$FRAME0_ONLY" == "1" ]]          && RUN_ARGS+=(--frame0_only)
        [[ -n "$DINO_MESH_SCAN_EVERY" ]]     && RUN_ARGS+=(--mesh_scan_every  "$DINO_MESH_SCAN_EVERY")
        [[ -n "$DINO_DENSE_SCAN_EVERY" ]]    && RUN_ARGS+=(--dense_scan_every "$DINO_DENSE_SCAN_EVERY")
        [[ -n "$DINO_SEED_MIN_AREA" ]]       && RUN_ARGS+=(--seed_min_area    "$DINO_SEED_MIN_AREA")
        [[ -n "$DINO_SEED_MAX_AREA" ]]       && RUN_ARGS+=(--seed_max_area    "$DINO_SEED_MAX_AREA")
        [[ -n "$DINO_SEED_MIN_SIM" ]]        && RUN_ARGS+=(--seed_min_sim     "$DINO_SEED_MIN_SIM")
        [[ "$FORCE" == "1" ]]                && RUN_ARGS+=(--force)
        [[ "$FAKE_OPTIMIZE" == "1" ]]        && RUN_ARGS+=(--fake_optimize)

        if bash "$RUN_SCRIPT" "${RUN_ARGS[@]}"; then
            OK=$((OK+1))
            echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [done]"
        else
            RC=$?
            FAIL=$((FAIL+1)); FAILED+=("$TASK_NAME/$EXP_NAME:rc=$RC")
            echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [FAIL: run_auto_annotator rc=$RC]"
        fi

        if [[ -n "$H5_SCRATCH" && -d "$H5_SCRATCH" ]]; then
            rm -f "$H5_SCRATCH"/*.h5 "$H5_SCRATCH"/*.h5.full_backup_* 2>/dev/null || true
        fi
    done
done

echo ""
echo "=========================================="
echo "summary: total=$TOTAL  ok=$OK  skipped=$SKIP  failed=$FAIL"
echo "  skip breakdown:"
echo "    no mask_prompts.json   : $SKIP_NO_PROMPTS"
echo "    visibility != visible  : $SKIP_NOT_VISIBLE"
echo "    already redone (resume): $SKIP_ALREADY"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo "failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
echo "=========================================="

[[ "$FAIL" -gt 0 ]] && exit 1 || exit 0
