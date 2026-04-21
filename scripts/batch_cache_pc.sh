#!/bin/bash
# Batch-generate cached per-camera point clouds for many videos_XXXX/ folders,
# so you can later run scripts/manual_align_viser.py on each of them.
#
# For every videos_XXXX under --data_root (or each --videos_root passed), this:
#   1) locates realsense_calibrate_*/realsense_calibration_*.yaml
#   2) picks a representative experiment (first non-*.tar.gz dir with cam*_rgb.mp4)
#   3) builds a 1-frame h5 from it
#   4) runs tools/00-0_align_cameras.py to produce <cal_folder>/cached_pc/
#   5) deletes the tiny h5
#
# Skips a videos_XXXX whose cached_pc already exists (unless --force).
# Prints a summary at the end with ready-to-copy manual_align_viser commands.
#
# Usage:
#   bash scripts/batch_cache_pc.sh --data_root /viscam/projects/robotool/data \
#        [--cal_frame_idx 0] [--representative_exp NAME] [--force] [--dry_run]
#
#   # or explicit list:
#   bash scripts/batch_cache_pc.sh \
#        --videos_root /.../videos_0101 --videos_root /.../videos_0115

set -u

DATA_ROOT=""
VIDEOS_ROOTS=()
CAL_FRAME_IDX=0
REPRESENTATIVE_EXP=""
FORCE=0
DRY_RUN=0
BASE_PORT=8080

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --data_root)          DATA_ROOT="$2"; shift 2;;
        --videos_root)        VIDEOS_ROOTS+=("$2"); shift 2;;
        --cal_frame_idx)      CAL_FRAME_IDX="$2"; shift 2;;
        --representative_exp) REPRESENTATIVE_EXP="$2"; shift 2;;
        --force)              FORCE=1; shift;;
        --dry_run)            DRY_RUN=1; shift;;
        --base_port)          BASE_PORT="$2"; shift 2;;
        -h|--help) sed -n '2,25p' "$0"; exit 0;;
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
        # skip output folders produced by the annotation pipeline
        [[ "$name" == *_annotated ]] && continue
        VIDEOS_ROOTS+=("$d")
    done < <(find "$DATA_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'videos_*' -print0 | sort -z)
fi

if [[ "${#VIDEOS_ROOTS[@]}" -eq 0 ]]; then
    echo "Error: no videos_* folders found. Pass --data_root or --videos_root."; exit 1
fi

if [[ "$DRY_RUN" == "0" ]]; then
    # only activate conda when we'll actually run python
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
echo "force=$FORCE  dry_run=$DRY_RUN  cal_frame_idx=$CAL_FRAME_IDX"
echo "=========================================="

OK_LIST=()     # "videos_name|cal_yaml|cached_pc_dir"
SKIP_LIST=()
FAIL_LIST=()

for VIDEOS_ROOT in "${VIDEOS_ROOTS[@]}"; do
    VIDEOS_ROOT="$(cd "$VIDEOS_ROOT" && pwd)"
    VNAME="$(basename "$VIDEOS_ROOT")"
    echo ""
    echo "### $VNAME"

    # find calibration folder + yaml
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
    CACHED_PC_DIR="${CAL_FOLDER}/cached_pc"
    echo "  cal_folder: $CAL_FOLDER"
    echo "  orig_yaml : $(basename "$ORIG_YAML")"

    # already cached?
    if [[ -d "$CACHED_PC_DIR" && "$FORCE" == "0" ]]; then
        n_ply=$(ls "$CACHED_PC_DIR"/cam*_uncropped.ply 2>/dev/null | wc -l)
        if [[ "$n_ply" -gt 0 ]]; then
            echo "  [skip] cached_pc already has $n_ply plys (pass --force to redo)"
            SKIP_LIST+=("$VNAME")
            OK_LIST+=("$VNAME|$ORIG_YAML|$CACHED_PC_DIR")
            continue
        fi
    fi

    # pick representative experiment
    REP_DIR=""
    while IFS= read -r -d '' task_dir; do
        t_name="$(basename "$task_dir")"
        [[ "$t_name" == realsense_calibrate_* ]] && continue
        if [[ -n "$REPRESENTATIVE_EXP" ]]; then
            cand="$task_dir/$REPRESENTATIVE_EXP"
            if [[ -d "$cand" ]] && ls "$cand"/cam*_rgb.mp4 >/dev/null 2>&1; then
                REP_DIR="$cand"; break
            fi
            continue
        fi
        while IFS= read -r -d '' d; do
            name="$(basename "$d")"
            [[ "$name" == *.tar.gz || "$name" == board_reference ]] && continue
            if ls "$d"/cam*_rgb.mp4 >/dev/null 2>&1; then
                REP_DIR="$d"; break
            fi
        done < <(find "$task_dir" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
        [[ -n "$REP_DIR" ]] && break
    done < <(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

    if [[ -z "$REP_DIR" ]]; then
        echo "  [FAIL] no experiment folder with cam*_rgb.mp4 found"
        FAIL_LIST+=("$VNAME:no_rep_exp"); continue
    fi
    echo "  rep_exp   : $(basename "$REP_DIR")"

    TINY_H5="${REP_DIR}/data00000000.h5"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [dry_run] would build h5 from $REP_DIR frame $CAL_FRAME_IDX"
        echo "  [dry_run] would run 00-0 with $ORIG_YAML -> $CACHED_PC_DIR"
        OK_LIST+=("$VNAME|$ORIG_YAML|$CACHED_PC_DIR")
        continue
    fi

    # step 1: tiny h5
    rm -f "$TINY_H5"
    if ! python "$HOCAP_ROOT/tools/00_convert_videos_to_h5.py" \
            --input_dir "$REP_DIR" \
            --output_file "$TINY_H5" \
            --start_frame "$CAL_FRAME_IDX" \
            --end_frame "$CAL_FRAME_IDX" > /dev/null 2>&1; then
        echo "  [FAIL] tiny h5 conversion"
        FAIL_LIST+=("$VNAME:h5"); rm -f "$TINY_H5"; continue
    fi

    # step 2: 00-0 cache pc
    if ! python "$HOCAP_ROOT/tools/00-0_align_cameras.py" \
            --h5_file "$TINY_H5" \
            --extrinsic_file "$ORIG_YAML" \
            --out_path "$CAL_FOLDER" \
            --frame_idx 0; then
        echo "  [FAIL] 00-0 cache pc"
        FAIL_LIST+=("$VNAME:00-0"); rm -f "$TINY_H5"; continue
    fi

    rm -f "$TINY_H5"

    n_ply=$(ls "$CACHED_PC_DIR"/cam*_uncropped.ply 2>/dev/null | wc -l)
    if [[ "$n_ply" -eq 0 ]]; then
        echo "  [FAIL] 00-0 completed but cached_pc is empty"
        FAIL_LIST+=("$VNAME:empty_pc"); continue
    fi
    echo "  [OK]  $n_ply cam plys in $CACHED_PC_DIR"
    OK_LIST+=("$VNAME|$ORIG_YAML|$CACHED_PC_DIR")
done

# ---------- summary ----------
echo ""
echo "=========================================="
echo "Summary: ok=${#OK_LIST[@]}  skipped=${#SKIP_LIST[@]}  failed=${#FAIL_LIST[@]}"
if [[ "${#FAIL_LIST[@]}" -gt 0 ]]; then
    echo "Failed:"
    for f in "${FAIL_LIST[@]}"; do echo "  - $f"; done
fi
echo ""
echo "Ready-to-run manual-align commands (change --port if collisions):"
port=$BASE_PORT
for row in "${OK_LIST[@]}"; do
    IFS='|' read -r vname yaml pcdir <<< "$row"
    aligned="${yaml%.yaml}_global_aligned.yaml"
    if [[ -f "$aligned" ]]; then
        initial_arg="--initial_yaml $aligned"
    else
        initial_arg=""
    fi
    echo ""
    echo "# $vname"
    echo "python scripts/manual_align_viser.py \\"
    echo "    --cached_pc $pcdir \\"
    echo "    --extrinsic_file $yaml \\"
    if [[ -n "$initial_arg" ]]; then
        echo "    $initial_arg \\"
    fi
    echo "    --port $port"
    port=$((port + 1))
done
echo "=========================================="
