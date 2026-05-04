#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name hauto_v
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/%j.err
#
# Cluster sbatch wrapper that walks a videos_XXXX/ folder and runs
# scripts/run_auto_annotator.sh on every experiment in every task,
# honoring a manual frame-0 visibility annotation file if present.
#
# Layout assumed (from your data collection / DataCollection convention):
#   <videos_root>/                                  e.g. .../videos_0101
#     realsense_calibrate_*/                        ← calibration (skipped)
#       realsense_calibration_*.yaml
#       realsense_calibration_*_global_aligned.yaml ← preferred
#     <task_name>/                                  e.g. mallet_crush_nuts
#       frame0_visibility.yaml      (OPTIONAL — set by frame0_visibility_inspector.py)
#       <exp_name>/                                 e.g. 20260104_..._1
#         cam{0..7}_rgb.mp4
#         cam{0..7}_depth.mkv
#       <exp_name>/
#       ...
#     <another_task>/
#
# Frame-0 visibility filtering:
#   For each task:
#     · If `<task>/frame0_visibility.yaml` is ABSENT → run all exps.
#     · If yaml is PRESENT → only run exps the user has explicitly labeled
#       "visible". Exps labeled "not_visible" / "unsure" or NOT in the yaml at
#       all are skipped. The script prints a WARN listing un-annotated exps so
#       you can rerun frame0_visibility_inspector.py to label them and re-queue.
#
#   Override via --frame0_mode:
#     auto           : (default) yaml absent → all; yaml present → visible_only
#     visible_only   : same as auto-with-yaml — only label==visible
#     visible_unsure : rescue mode: visible OR unsure
#     all            : ignore the yaml entirely (run everything)
#
#   The yaml is produced by:
#     python scripts/frame0_visibility_inspector.py --videos_root <videos_root>
#
# Tool / mesh resolution per exp (priority high → low):
#   1. --force_tool NAME                         (overrides everything)
#   2. --tool_name_map_json [PATH]                (PREFERRED, default-on)
#                                                 JSON {key: tool_name}; key
#                                                 matched as longest case-
#                                                 insensitive substring of
#                                                 "<task>/<exp>". Mesh is then
#                                                 resolved in $MODELS_FOLDER/
#                                                 <tool>/ (textured_mesh.obj
#                                                 / cleaned_mesh_10000.obj /
#                                                 mesh.obj, in order). Default
#                                                 path: $TOOL_NAME_MAP_JSON_DEFAULT
#                                                 (= /viscam/u/chenrq/crq_ws/
#                                                 scripts/mesh_name_mapping.json).
#                                                 Auto-used if that file exists.
#   3. --mesh_map_json PATH                      (legacy: JSON gives full mesh
#                                                 path directly; tool_name =
#                                                 mesh's parent dir basename)
#   4. tool_keyword_mapping.yaml keyword hit     (manual 词表; see below)
#   5. heuristic auto-match (scripts/match_tool_name.py token overlap)
#
# JSON mesh map (--mesh_map_json):
#   File format (default path: $MESH_MAP_JSON_DEFAULT, currently
#                /viscam/u/kehanli/temp_scripts/dino_mesh_map_tool_mesh_full.json):
#     {
#       "largeplate_mallet_flatten_crush_largedough":
#           "/viscam/projects/robotool/data/tool_mesh_full/rubber_mallet/rubber_mallet.obj",
#       ...
#     }
#   Pass `--mesh_map_json` (no value) to use the default path, or
#   `--mesh_map_json /abs/path.json` to override.
#   Each key is matched as a case-insensitive substring of "<task>/<exp>".
#   The mesh path is used directly as --tool_mesh; the tool_name is derived
#   from the mesh's parent folder basename (so downstream paths stay stable
#   per tool, e.g. processed/fd_pose_solver/rubber_mallet/...).
#   Add --mesh_map_only to skip exps with no JSON match (instead of falling
#   back to mapping_yaml / heuristic).
#
# Mapping yaml (optional, recommended for clusters with curated vocab):
#   Default location: <videos_root>/tool_keyword_mapping.yaml
#   Override with --mapping_yaml /abs/path.yaml
#   Format:
#     mappings:
#       mallet:    rubber_mallet         # if "mallet" appears in exp_name → use rubber_mallet
#       largeplate: largeplate
#       woodenfork: big_wooden_fork
#   Pass --mapping_only to require a keyword hit (exps that don't match any
#   keyword are SKIPPED, never auto-matched). Useful for "only annotate exps
#   whose tool I've manually curated in the vocab".
#
# What runs per exp:
#   scripts/run_auto_annotator.sh
#     --sequence_folder <task>/<exp>
#     --calibration_yaml <discovered>
#     --tool_name <resolved>
#     --models_folder $MODELS_FOLDER
#     --object_chunk_size $OBJECT_CHUNK_SIZE
#     [--hand 1] [--fake_optimize] [--resume] ...
#
# Resume / force:
#   run_auto_annotator.sh now auto-resumes by default (every stage skips if
#   its primary output exists on disk: h5, DINO npz/masks.h5, meta.yaml,
#   fd_poses_merged_fixed.npy, result_hand_optimized.pkl, joint poses_o/m).
#   Re-running this sbatch is therefore safe and fast.
#
#     --force / --no_resume  →  forwarded as run_auto_annotator's --force,
#                                redo every stage from scratch.
#     --skip_existing        →  exp-level coarse skip (same as before): if
#                                the final done marker is present we skip the
#                                whole exp without invoking run_auto_annotator
#                                at all (saves python startup × N exps).
#     --resume               →  no-op now (kept for back-compat).
#
# Usage:
#   sbatch scripts/sbatch_run_auto_videos.sh \
#     --videos_root /viscam/projects/robotool/data/videos_0101 \
#     [--object_chunk_size 600]    (default 600)
#     [--fake_optimize]            (recommended on long videos)
#     [--hand 1]                   (0 disables hand)
#     [--force | --no_resume]      (redo every stage in run_auto_annotator)
#     [--skip_existing]            (exp-level skip: pre-check done marker)
#     [--frame0_mode visible_only|visible_unsure|all]  (auto by default)
#     [--force_tool NAME]          (override per-exp tool matching)
#     [--mapping_yaml PATH]        (词表 yaml; auto-detected at <videos_root>/tool_keyword_mapping.yaml)
#     [--mapping_only]             (only run exps whose name matches a词表 keyword)
#     [--tool_name_map_json [PATH]] (JSON {key: tool_name}; key matched as
#                                   substring of "<task>/<exp>"; mesh resolved
#                                   in $MODELS_FOLDER/<tool>/. Bare flag uses
#                                   $TOOL_NAME_MAP_JSON_DEFAULT. AUTO-USED if
#                                   default file exists.)
#     [--tool_name_map_only]       (only run exps with a tool_name_map hit)
#     [--no_tool_name_map_json]    (disable even if default file exists)
#     [--mesh_map_json [PATH]]     (LEGACY: JSON {key: mesh_path}, full path)
#     [--mesh_map_only]            (only run exps that match a JSON key)
#     [--start_frame 0] [--end_frame N]
#     [--object_chunk_overlap N]   (overlap between obj chunks; legacy)
#     [--hand_chunk_size N]        (hand chunked path window)
#     [--long_video_threshold N]   (auto-disable real optim for long videos)
#     [--frame0_only]              (DINO: only register on frame 0, no re-seed —
#                                   skips Phase 3-7 of dino_tool_segment.py;
#                                   much faster on slow GPUs but masks won't
#                                   re-seek if the tool drifts out of view)
#     [--mesh_scan_every N] [--dense_scan_every N]    (DINO seed strides)
#     [--seed_min_area N] [--seed_max_area N] [--seed_min_sim F]
#     [--dry_run]                  (list what would run, don't actually run)

set -u

# ---------- cluster paths (override via env if your layout differs) ----------
export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export HAND_ROOT="${HAND_ROOT:-/viscam/u/chenrq/crq_ws/robotool/HandReconstruction}"
export MODELS_FOLDER="${MODELS_FOLDER:-/viscam/u/chenrq/models}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
RUN_SCRIPT="$HOCAP_ROOT/scripts/run_auto_annotator.sh"
MATCH_SCRIPT="$HOCAP_ROOT/scripts/match_tool_name.py"

# SAM2 (Stage 2 DINO+SAM2). On viscam sam2 lives under robotool/, set explicitly.
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
HAND=""
FAKE_OPTIMIZE=0
# RESUME is now ON by default in run_auto_annotator.sh — every stage auto-
# skips when its output exists. Set FORCE=1 (--force) to redo all stages.
FORCE=0
SKIP_EXISTING=0
FRAME0_MODE="auto"        # auto | visible_only | visible_unsure | all
FORCE_TOOL=""
MAPPING_YAML=""           # path to tool_keyword_mapping.yaml; auto = <videos_root>/tool_keyword_mapping.yaml if present
MAPPING_ONLY=0            # if 1, --require_mapping (skip exps without keyword hit)

# --- TOOL NAME MAP (preferred path; default-on if file exists) ---
# JSON of {exp_key_substring: tool_name}. tool_name is then resolved against
# $MODELS_FOLDER/<tool>/{textured_mesh.obj | cleaned_mesh_10000.obj | mesh.obj}
# to find the actual mesh file. This is the "lookup-by-name then dig in
# models/" workflow (matches the legacy match_tool_name behaviour but uses
# JSON for explicit per-exp mapping instead of heuristic token matching).
#
# Default lookup path: /viscam/u/chenrq/crq_ws/scripts/mesh_name_mapping.json
# Auto-detected — if the file exists at this path, it's used without needing
# any flag. Override:
#   --tool_name_map_json /abs/path.json    explicit path
#   --tool_name_map_json                   bare flag → use $TOOL_NAME_MAP_JSON_DEFAULT
#   --no_tool_name_map_json                disable even if default exists
TOOL_NAME_MAP_JSON_DEFAULT="${TOOL_NAME_MAP_JSON_DEFAULT:-/viscam/u/chenrq/crq_ws/scripts/mesh_name_mapping.json}"
TOOL_NAME_MAP_JSON=""     # resolved at parse + auto-detect time
TOOL_NAME_MAP_ONLY=0      # if 1, skip exps that don't match any JSON key
TOOL_NAME_MAP_DISABLED=0  # set by --no_tool_name_map_json

# --- LEGACY MESH MAP (gives full mesh path directly; secondary priority) ---
MESH_MAP_JSON_DEFAULT="${MESH_MAP_JSON_DEFAULT:-/viscam/u/kehanli/temp_scripts/dino_mesh_map_tool_mesh_full.json}"
MESH_MAP_JSON=""          # resolved at parse time; "" = no JSON map active
MESH_MAP_ONLY=0           # if 1, skip exps that don't match any JSON key
START_FRAME=0
END_FRAME=""
ROT_THRESH=15
TRANS_THRESH=0.03
TRACK_REFINE_ITER=10
OBJECT_CHUNK_SIZE=600
OBJECT_CHUNK_OVERLAP=""
HAND_CHUNK_SIZE=""
LONG_VIDEO_THRESHOLD=""
FRAME0_ONLY=0             # forward to run_auto_annotator → DINO Phase 1 only, no re-seeding
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
        --resume)             shift;;          # ← no-op now (auto-resume is default)
        --force|--no_resume)  FORCE=1; shift;;
        --skip_existing)      SKIP_EXISTING=1; shift;;
        --frame0_mode)        FRAME0_MODE="$2"; shift 2;;
        --force_tool)         FORCE_TOOL="$2"; shift 2;;
        --mapping_yaml)       MAPPING_YAML="$2"; shift 2;;
        --mapping_only)       MAPPING_ONLY=1; shift;;
        --tool_name_map_json)
            # Optional value: explicit PATH or fall back to default.
            if [[ -n "${2:-}" && "${2:0:2}" != "--" ]]; then
                TOOL_NAME_MAP_JSON="$2"; shift 2
            else
                TOOL_NAME_MAP_JSON="$TOOL_NAME_MAP_JSON_DEFAULT"; shift
            fi
            ;;
        --tool_name_map_only)    TOOL_NAME_MAP_ONLY=1; shift;;
        --no_tool_name_map_json) TOOL_NAME_MAP_DISABLED=1; shift;;
        --mesh_map_json)
            # Legacy: JSON gives full mesh path directly. Optional value.
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
        -h|--help)            sed -n '2,72p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$VIDEOS_ROOT" ]]; then
    echo "Error: --videos_root is required"; exit 1
fi
VIDEOS_ROOT="$(cd "$VIDEOS_ROOT" && pwd)"
[[ -d "$VIDEOS_ROOT" ]] || { echo "Error: not a directory: $VIDEOS_ROOT"; exit 1; }

case "$FRAME0_MODE" in
    auto|visible_only|visible_unsure|all) ;;
    *) echo "Error: --frame0_mode must be auto / visible_only / visible_unsure / all"; exit 1;;
esac

echo "=========================================="
echo "job        : ${SLURM_JOB_ID:-<non-slurm>}   node: $(hostname)"
echo "cpus/omp   : ${SLURM_CPUS_PER_TASK:-4}"
echo "videos_root: $VIDEOS_ROOT"
echo "models     : $MODELS_FOLDER"
echo "scratch    : $H5_SCRATCH"
echo "frame0_mode: $FRAME0_MODE"
echo "chunk_size : $OBJECT_CHUNK_SIZE  fake_optim: $FAKE_OPTIMIZE  hand: ${HAND:-<unset>}"
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
        # Fallback: a *.yaml directly under videos_root (user's local layout)
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

# ---------- discover mapping yaml (optional 词表 for tool name) ----------
# Convention (matches batch_auto_annotator.sh): <videos_root>/tool_keyword_mapping.yaml
# Format:
#   mappings:
#     mallet:    rubber_mallet
#     dough:     mallet_crush_dough
#     plate:     largeplate
# Each kw is matched (case-insensitive substring) against an exp's name; first
# hit wins. The keyword hit takes precedence over the heuristic auto-match.
# With --mapping_only, an exp whose name matches no keyword is SKIPPED (not
# auto-matched) — useful for "only annotate exps whose tool I've curated".
if [[ -z "$MAPPING_YAML" ]]; then
    if [[ -f "$VIDEOS_ROOT/tool_keyword_mapping.yaml" ]]; then
        MAPPING_YAML="$VIDEOS_ROOT/tool_keyword_mapping.yaml"
    fi
fi
if [[ -n "$MAPPING_YAML" && -f "$MAPPING_YAML" ]]; then
    echo "mapping    : $MAPPING_YAML  (mapping_only=$MAPPING_ONLY)"
elif [[ -n "$MAPPING_YAML" ]]; then
    echo "Error: --mapping_yaml file not found: $MAPPING_YAML"; exit 1
else
    echo "mapping    : <none — auto-match heuristic only>"
fi

# --- tool_name_map auto-detect (default-on if default file exists) ---
if [[ "$TOOL_NAME_MAP_DISABLED" == "1" ]]; then
    TOOL_NAME_MAP_JSON=""
    echo "tool_map   : <disabled by --no_tool_name_map_json>"
elif [[ -z "$TOOL_NAME_MAP_JSON" ]]; then
    if [[ -f "$TOOL_NAME_MAP_JSON_DEFAULT" ]]; then
        TOOL_NAME_MAP_JSON="$TOOL_NAME_MAP_JSON_DEFAULT"
        echo "tool_map   : $TOOL_NAME_MAP_JSON  (auto-detected default)"
    else
        echo "tool_map   : <default $TOOL_NAME_MAP_JSON_DEFAULT not found; falling back to other resolvers>"
    fi
fi
if [[ -n "$TOOL_NAME_MAP_JSON" ]]; then
    if [[ ! -f "$TOOL_NAME_MAP_JSON" ]]; then
        echo "Error: --tool_name_map_json file not found: $TOOL_NAME_MAP_JSON"; exit 1
    fi
    n_keys=$(TM="$TOOL_NAME_MAP_JSON" python3 -c "import json,os; print(len(json.load(open(os.environ['TM']))))" 2>/dev/null) || {
        echo "Error: --tool_name_map_json is not valid JSON: $TOOL_NAME_MAP_JSON"; exit 1
    }
    echo "             ${n_keys} entries, tool_name_map_only=$TOOL_NAME_MAP_ONLY"
fi

# --- legacy --mesh_map_json (full mesh path; secondary) ---
if [[ -n "$MESH_MAP_JSON" ]]; then
    if [[ ! -f "$MESH_MAP_JSON" ]]; then
        echo "Error: --mesh_map_json file not found: $MESH_MAP_JSON"; exit 1
    fi
    n_keys=$(MM="$MESH_MAP_JSON" python3 -c "import json,os; print(len(json.load(open(os.environ['MM']))))" 2>/dev/null) || {
        echo "Error: --mesh_map_json is not valid JSON: $MESH_MAP_JSON"; exit 1
    }
    echo "mesh_map   : $MESH_MAP_JSON  (${n_keys} entries, mesh_map_only=$MESH_MAP_ONLY)"
fi
echo "=========================================="

# ---------- discover tasks ----------
TASKS=()
while IFS= read -r -d '' d; do
    [[ "$(basename "$d")" == realsense_calibrate_* ]] && continue
    TASKS+=("$d")
done < <(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
[[ "${#TASKS[@]}" -eq 0 ]] && { echo "Error: no task folders under $VIDEOS_ROOT"; exit 1; }
echo "tasks (${#TASKS[@]}):"; for t in "${TASKS[@]}"; do echo "  - $(basename "$t")"; done

# ---------- helper: read frame0_visibility.yaml -> filtered exp list ----------
# Usage: filter_exps_for_task <task_dir>
# Echoes one exp name per line. Honors $FRAME0_MODE.
# Also writes WARN lines to stderr listing un-annotated exps when a yaml is
# present (so you know what's pending review without silently dropping them).
filter_exps_for_task() {
    local task_dir="$1"
    local mode="$FRAME0_MODE"
    local f0_yaml="${task_dir}/frame0_visibility.yaml"

    # No yaml → no filter regardless of mode (nothing to filter against).
    if [[ ! -f "$f0_yaml" ]]; then
        mode="all"
    elif [[ "$mode" == "auto" ]]; then
        # yaml present + auto → strict visible-only (the user's intent: pick
        # only exps they've manually green-lit).
        mode="visible_only"
    fi

    F0_YAML="$f0_yaml" MODE="$mode" TASK_DIR="$task_dir" python3 - <<'PY'
import os, sys, yaml
from pathlib import Path
task_dir   = Path(os.environ["TASK_DIR"])
mode       = os.environ["MODE"]
yaml_path  = Path(os.environ["F0_YAML"])

# All candidate exp dirs (must have at least one cam*_rgb.mp4).
candidates = []
for sub in sorted(task_dir.iterdir()):
    if not sub.is_dir(): continue
    name = sub.name
    if name.endswith(".tar.gz") or name == "board_reference": continue
    if not list(sub.glob("cam*_rgb.mp4")): continue
    candidates.append(name)

if mode == "all" or not yaml_path.exists():
    for n in candidates:
        print(n)
    sys.exit(0)

try:
    data = yaml.safe_load(yaml_path.read_text()) or {}
except Exception as e:
    # Don't silently swallow un-annotated exps when the yaml is broken — fail
    # loud so the user fixes it instead of getting a partial run.
    sys.stderr.write(f"[ERROR] failed to parse {yaml_path}: {e}\n")
    sys.exit(1)

ann = (data.get("annotations") or {})

# Per-mode predicate. label == None means the exp isn't in the yaml at all.
def keeps(label):
    if mode == "visible_only":
        return label == "visible"
    if mode == "visible_unsure":
        return label in ("visible", "unsure")
    return True  # "all" handled above

kept, dropped_explicit, dropped_unannot = [], [], []
for n in candidates:
    label = ann.get(n)
    if keeps(label):
        kept.append(n)
    elif label is None:
        dropped_unannot.append(n)
    else:
        dropped_explicit.append((n, label))

# Headline summary
sys.stderr.write(
    f"[frame0:{mode}] kept={len(kept)}  "
    f"dropped_explicit={len(dropped_explicit)}  "
    f"unannotated={len(dropped_unannot)}\n"
)
# Detail: un-annotated exps under strict mode are skipped (treated like
# not-visible). Print them so the user can decide to label & re-queue.
if dropped_unannot:
    sys.stderr.write("[frame0]   skipped (NO label in yaml — re-run frame0_visibility_inspector.py):\n")
    for n in dropped_unannot:
        sys.stderr.write(f"[frame0]     - {n}\n")
if dropped_explicit:
    sys.stderr.write("[frame0]   skipped (explicit label):\n")
    for n, lbl in dropped_explicit:
        sys.stderr.write(f"[frame0]     - {n}  [{lbl}]\n")

for n in kept:
    print(n)
PY
}

# ---------- per-exp runner ----------
TOTAL=0; OK=0; SKIP=0; FAIL=0
FAILED=()
SKIPPED_NOT_VISIBLE=0

for TASK_DIR in "${TASKS[@]}"; do
    TASK_NAME="$(basename "$TASK_DIR")"
    F0_YAML="${TASK_DIR}/frame0_visibility.yaml"
    EXP_NAMES=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && EXP_NAMES+=("$line")
    done < <(filter_exps_for_task "$TASK_DIR")

    # Count not-visible / unannotated for the summary line
    if [[ -f "$F0_YAML" && ( "$FRAME0_MODE" == "visible_only" || "$FRAME0_MODE" == "auto" ) ]]; then
        ALL_CANDS=$(find "$TASK_DIR" -maxdepth 1 -mindepth 1 -type d \
                    \( ! -name "*.tar.gz" \) ! -name "board_reference" -printf "%f\n" | sort)
        ALL_COUNT=$(echo "$ALL_CANDS" | grep -c . || true)
        SKIPPED_NOT_VISIBLE=$((SKIPPED_NOT_VISIBLE + ALL_COUNT - ${#EXP_NAMES[@]}))
    fi

    echo ""
    echo "##########################################"
    echo "# task: $TASK_NAME"
    if [[ -f "$F0_YAML" ]]; then
        echo "#  frame0_visibility.yaml: present  (mode=$FRAME0_MODE)"
    else
        echo "#  frame0_visibility.yaml: NOT present  (running all exps)"
    fi
    echo "#  exps to run: ${#EXP_NAMES[@]}"
    echo "##########################################"

    [[ "${#EXP_NAMES[@]}" -eq 0 ]] && { echo "  (no exps to run for this task)"; continue; }

    ANNOTATED_ROOT="$(dirname "$VIDEOS_ROOT")/$(basename "$VIDEOS_ROOT")_annotated/${TASK_NAME}"

    for EXP_NAME in "${EXP_NAMES[@]}"; do
        EXP_DIR="${TASK_DIR}/${EXP_NAME}"
        TOTAL=$((TOTAL+1))

        # ---- skip_existing (exp-level coarse skip) ----
        ANNO="${ANNOTATED_ROOT}/${EXP_NAME}"
        if [[ -n "$HAND" && "$HAND" != "0" ]]; then
            DONE_MARKER="${ANNO}/result_hand_optimized.pkl"
        else
            DONE_MARKER="${ANNO}/processed/fd_pose_solver/fd_poses_merged_fixed.npy"
        fi
        if [[ "$SKIP_EXISTING" == "1" && -f "$DONE_MARKER" ]]; then
            echo ""
            echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [skip: $DONE_MARKER exists]"
            SKIP=$((SKIP+1)); continue
        fi

        # ---- resolve tool name + mesh ----
        # Priority:
        #   1. --force_tool                  (CLI override, all exps share NAME)
        #   2. --tool_name_map_json hit       (NEW; gives tool NAME, mesh resolved
        #                                     in $MODELS_FOLDER/<name>/. Default-on.)
        #   3. --mesh_map_json hit            (legacy; gives mesh PATH directly)
        #   4. tool_keyword_mapping.yaml hit  (manual 词表)
        #   5. heuristic match_tool_name.py   (token-overlap fallback)
        TOOL=""
        MESH=""
        MAPPED_VIA=""

        # 1) tool-name JSON map (preferred): JSON gives tool name; we resolve
        #    mesh by scanning $MODELS_FOLDER/<tool>/ for a known mesh filename.
        #    Longest-key wins (most specific match).
        if [[ -n "$TOOL_NAME_MAP_JSON" && -z "$TOOL" ]]; then
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
                NAME="${MAPPED##*$'\t'}"
                # Resolve mesh inside MODELS_FOLDER, same order as legacy.
                CAND_MESH=""
                for cand in textured_mesh.obj cleaned_mesh_10000.obj mesh.obj; do
                    if [[ -f "$MODELS_FOLDER/$NAME/$cand" ]]; then
                        CAND_MESH="$MODELS_FOLDER/$NAME/$cand"; break
                    fi
                done
                if [[ -n "$CAND_MESH" ]]; then
                    TOOL="$NAME"
                    MESH="$CAND_MESH"
                    MAPPED_VIA="tool_name_map_json:$KEY -> $NAME"
                else
                    echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [WARN] tool_name_map hit '$KEY' -> '$NAME' but no mesh in $MODELS_FOLDER/$NAME/"
                    if [[ "$TOOL_NAME_MAP_ONLY" == "1" ]]; then
                        echo "  [skip: tool_name_map_only and mesh missing]"
                        SKIP=$((SKIP+1)); continue
                    fi
                fi
            elif [[ "$TOOL_NAME_MAP_ONLY" == "1" ]]; then
                echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [skip: no tool_name_map JSON match (--tool_name_map_only)]"
                SKIP=$((SKIP+1)); continue
            fi
        fi

        # 2) legacy JSON mesh map: gives mesh PATH directly (parent dir = tool name).
        if [[ -n "$MESH_MAP_JSON" && -z "$TOOL" ]]; then
            MAPPED=$(MESH_MAP_JSON="$MESH_MAP_JSON" \
                     TASK_NAME="$TASK_NAME" \
                     EXP_NAME="$EXP_NAME" python3 - <<'PY'
import json, os, sys
mp = json.load(open(os.environ["MESH_MAP_JSON"]))
hay = f"{os.environ['TASK_NAME']}/{os.environ['EXP_NAME']}".lower()
hits = [(k, v) for k, v in mp.items() if k.lower() in hay]
if not hits:
    sys.exit(0)
# longest key wins (most specific)
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
                    MAPPED_VIA="json:$KEY"
                else
                    echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [WARN] json hit '$KEY' but mesh missing: $MESH_PATH"
                    if [[ "$MESH_MAP_ONLY" == "1" ]]; then
                        echo "  [skip: mesh_map_only and mesh missing]"
                        SKIP=$((SKIP+1)); continue
                    fi
                fi
            elif [[ "$MESH_MAP_ONLY" == "1" ]]; then
                echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [skip: no mesh_map JSON match (--mesh_map_only)]"
                SKIP=$((SKIP+1)); continue
            fi
        fi

        # 2) --force_tool / 3) match_tool_name.py heuristic — only if JSON
        #    didn't already fill TOOL+MESH.
        if [[ -z "$TOOL" || -z "$MESH" ]]; then
            if [[ -n "$FORCE_TOOL" ]]; then
                TOOL="$FORCE_TOOL"
                MAPPED_VIA="force_tool"
            elif [[ -f "$MATCH_SCRIPT" ]]; then
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
                MAPPED_VIA="heuristic"
            else
                echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [FAIL: no --force_tool / --mesh_map_json / $MATCH_SCRIPT]"
                FAIL=$((FAIL+1)); FAILED+=("$TASK_NAME/$EXP_NAME:no_matcher"); continue
            fi

            # Mesh fallback: scan $MODELS_FOLDER/$TOOL/ for a known filename.
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
        fi

        echo ""
        echo "------------------------------------------"
        echo "[$TOTAL] $TASK_NAME / $EXP_NAME"
        echo "  tool : $TOOL"
        echo "  mesh : $MESH"
        echo "  via  : $MAPPED_VIA"
        echo "------------------------------------------"

        if [[ "$DRY_RUN" == "1" ]]; then
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

        # /dev/shm defensive sweep: each exp's h5 is up to ~30 GB on long
        # videos. run_auto_annotator's own EXIT trap cleans up its h5 on
        # graceful exit, but SIGKILL (rc=137 from cgroup OOM) does NOT
        # invoke shell traps — the h5 persists and the next exp's h5 build
        # piles on top until tmpfs blows up. Wipe everything in
        # $H5_SCRATCH between exps so each invocation starts clean.
        if [[ -n "$H5_SCRATCH" && -d "$H5_SCRATCH" ]]; then
            rm -f "$H5_SCRATCH"/*.h5 "$H5_SCRATCH"/*.h5.full_backup_* 2>/dev/null || true
        fi
    done
done

echo ""
echo "=========================================="
echo "summary: total=$TOTAL  ok=$OK  skipped(done)=$SKIP  failed=$FAIL"
echo "         skipped by frame0 filter: $SKIPPED_NOT_VISIBLE"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo "failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
echo "=========================================="

# Exit non-zero if anything actually failed (for slurm requeue / chained jobs)
[[ "$FAIL" -gt 0 ]] && exit 1 || exit 0
