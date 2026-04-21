#!/bin/bash
# For every videos_XXXX/ under --data_root (or each --videos_root passed),
# if the calibration folder does NOT yet have <stem>_global_aligned.yaml,
# run a lightweight one-shot plane-snap alignment (tools/quick_plane_align.py).
#
# This does NOT run pairwise ICP / pose-graph optimization. It:
#   - merges all cached per-camera PLYs,
#   - fits ONE dominant plane (RANSAC),
#   - applies the rigid transform that puts that plane at z=0 to every cam.
# Expected runtime: seconds per session. Peak RAM: few hundred MB.
#
# If you need the heavy version (pairwise ICP + pose graph + per-cam refine),
# use scripts/run_full_auto.sh which calls tools/run_global_align_headless.py.
#
# Expects cached_pc/ to already exist for each session (run
# scripts/batch_cache_pc.sh first if not).
#
# Writes, per session that needs alignment:
#   <cal_folder>/<stem>_global_aligned.yaml
#   <cal_folder>/postalign_global.ply
#   <cal_folder>/calibration_snapshot.png   (matplotlib snapshot)
#
# Usage:
#   bash scripts/batch_global_align.sh --data_root /viscam/projects/robotool/data
#     [--force]     # redo even if _global_aligned.yaml exists
#     [--dry_run]   # show what would run, don't execute

set -u

DATA_ROOT=""
VIDEOS_ROOTS=()
FORCE=0
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --data_root)    DATA_ROOT="$2"; shift 2;;
        --videos_root)  VIDEOS_ROOTS+=("$2"); shift 2;;
        --force)        FORCE=1; shift;;
        --dry_run)      DRY_RUN=1; shift;;
        -h|--help)      sed -n '2,20p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

HOCAP_ROOT="/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation"
[[ -d "$HOCAP_ROOT" ]] || HOCAP_ROOT="/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation"

# Collect videos_* folders
if [[ -n "$DATA_ROOT" ]]; then
    if [[ ! -d "$DATA_ROOT" ]]; then
        echo "Error: --data_root not a dir: $DATA_ROOT"; exit 1
    fi
    while IFS= read -r -d '' d; do
        name="$(basename "$d")"
        [[ "$name" == *_annotated ]] && continue
        VIDEOS_ROOTS+=("$d")
    done < <(find "$DATA_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'videos_*' -print0 | sort -z)
fi

if [[ "${#VIDEOS_ROOTS[@]}" -eq 0 ]]; then
    echo "Error: no videos_* folders found. Pass --data_root or --videos_root."; exit 1
fi

if [[ "$DRY_RUN" == "0" ]]; then
    if [[ -f /home/ruoqu/miniconda3/etc/profile.d/conda.sh ]]; then
        source /home/ruoqu/miniconda3/etc/profile.d/conda.sh
    elif [[ -f /viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh ]]; then
        source /viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh
    else
        echo "Error: cannot find conda.sh"; exit 1
    fi
    conda activate hocap-annotation
fi

echo "=========================================="
echo "videos_roots (${#VIDEOS_ROOTS[@]}):"
for v in "${VIDEOS_ROOTS[@]}"; do echo "  - $v"; done
echo "force=$FORCE  dry_run=$DRY_RUN"
echo "=========================================="

OK_LIST=()
SKIP_LIST=()
FAIL_LIST=()

for VIDEOS_ROOT in "${VIDEOS_ROOTS[@]}"; do
    VIDEOS_ROOT="$(cd "$VIDEOS_ROOT" && pwd)"
    VNAME="$(basename "$VIDEOS_ROOT")"
    echo ""
    echo "### $VNAME"

    CAL_FOLDER="$(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' | head -n 1)"
    if [[ -z "$CAL_FOLDER" ]]; then
        echo "  [SKIP] no realsense_calibrate_* folder"
        FAIL_LIST+=("$VNAME:no_cal_folder"); continue
    fi
    ORIG_YAML="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name 'realsense_calibration_*.yaml' \
                 ! -name '*_global_aligned.yaml' ! -name '*_slider_aligned.yaml' \
                 ! -name '*_manual_aligned.yaml' ! -name '*_aligned.yaml' | head -n 1)"
    if [[ -z "$ORIG_YAML" ]]; then
        echo "  [SKIP] no original realsense_calibration_*.yaml in $CAL_FOLDER"
        FAIL_LIST+=("$VNAME:no_yaml"); continue
    fi
    STEM="$(basename "$ORIG_YAML" .yaml)"
    ALIGNED_YAML="${CAL_FOLDER}/${STEM}_global_aligned.yaml"
    CACHED_PC_DIR="${CAL_FOLDER}/cached_pc"
    POSTALIGN_PLY="${CAL_FOLDER}/postalign_global.ply"
    SNAPSHOT_PNG="${CAL_FOLDER}/calibration_snapshot.png"

    echo "  cal_folder: $CAL_FOLDER"
    echo "  orig_yaml : $(basename "$ORIG_YAML")"

    if [[ -f "$ALIGNED_YAML" && "$FORCE" == "0" ]]; then
        echo "  [skip] ${STEM}_global_aligned.yaml already exists (pass --force to redo)"
        SKIP_LIST+=("$VNAME"); continue
    fi

    if [[ ! -d "$CACHED_PC_DIR" ]] || [[ -z "$(ls "$CACHED_PC_DIR"/cam*_uncropped.ply 2>/dev/null)" ]]; then
        echo "  [FAIL] cached_pc missing or empty — run scripts/batch_cache_pc.sh first"
        FAIL_LIST+=("$VNAME:no_cached_pc"); continue
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [dry_run] would run global align -> $ALIGNED_YAML"
        OK_LIST+=("$VNAME"); continue
    fi

    echo "  [run] quick plane-snap alignment (single RANSAC plane fit)"
    if ! python "$HOCAP_ROOT/tools/quick_plane_align.py" \
            --cached_pc "$CACHED_PC_DIR" \
            --extrinsic_file "$ORIG_YAML" \
            --out_path "$CAL_FOLDER"; then
        echo "  [FAIL] quick_plane_align exited nonzero"
        FAIL_LIST+=("$VNAME:align"); continue
    fi

    if [[ ! -f "$ALIGNED_YAML" ]]; then
        echo "  [FAIL] align completed but $ALIGNED_YAML was not produced"
        FAIL_LIST+=("$VNAME:no_yaml_out"); continue
    fi

    if [[ -f "$POSTALIGN_PLY" ]]; then
        if python "$HOCAP_ROOT/scripts/render_calibration_snapshot.py" \
                --ply "$POSTALIGN_PLY" --out "$SNAPSHOT_PNG" >/dev/null 2>&1; then
            echo "  [snap] $SNAPSHOT_PNG"
        else
            echo "  [warn] snapshot render failed (yaml still produced)"
        fi
    fi

    echo "  [OK]  $ALIGNED_YAML"
    OK_LIST+=("$VNAME")
done

echo ""
echo "=========================================="
echo "Summary: ok=${#OK_LIST[@]}  skipped=${#SKIP_LIST[@]}  failed=${#FAIL_LIST[@]}"
if [[ "${#FAIL_LIST[@]}" -gt 0 ]]; then
    echo "Failed:"
    for f in "${FAIL_LIST[@]}"; do echo "  - $f"; done
fi
if [[ "${#SKIP_LIST[@]}" -gt 0 ]]; then
    echo "Already aligned (skipped):"
    for s in "${SKIP_LIST[@]}"; do echo "  - $s"; done
fi
echo "=========================================="
