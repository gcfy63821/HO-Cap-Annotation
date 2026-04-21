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
#
#   # Re-use another day's already-aligned yaml (same rig, cameras not moved):
#     [--copy_pair SRC_VNAME:DST_VNAME]      (repeatable)
#     [--copy_from SRC_VNAME --copy_to DST1,DST2,...]   (convenience)
#   Example: reuse videos_0101's hand-tuned alignment for 0102,0103:
#     --copy_from videos_0101 --copy_to videos_0102,videos_0103

set -u

DATA_ROOT=""
VIDEOS_ROOTS=()
FORCE=0
DRY_RUN=0
COPY_SRCS=()     # parallel arrays: COPY_SRCS[i] -> COPY_DSTS[i]
COPY_DSTS=()
COPY_FROM=""
COPY_TO=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --data_root)    DATA_ROOT="$2"; shift 2;;
        --videos_root)  VIDEOS_ROOTS+=("$2"); shift 2;;
        --force)        FORCE=1; shift;;
        --dry_run)      DRY_RUN=1; shift;;
        --copy_pair)
            pair="$2"
            src="${pair%%:*}"
            dst="${pair##*:}"
            if [[ -z "$src" || -z "$dst" || "$src" == "$dst" || "$src" == "$pair" ]]; then
                echo "Error: --copy_pair expects SRC:DST (got '$pair')"; exit 1
            fi
            COPY_SRCS+=("$src"); COPY_DSTS+=("$dst")
            shift 2;;
        --copy_from)    COPY_FROM="$2"; shift 2;;
        --copy_to)      COPY_TO="$2"; shift 2;;
        -h|--help)      sed -n '2,30p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

# Expand --copy_from + --copy_to into COPY_SRCS/COPY_DSTS entries.
if [[ -n "$COPY_FROM" || -n "$COPY_TO" ]]; then
    if [[ -z "$COPY_FROM" || -z "$COPY_TO" ]]; then
        echo "Error: --copy_from and --copy_to must be given together"; exit 1
    fi
    IFS=',' read -r -a _TO_LIST <<< "$COPY_TO"
    for t in "${_TO_LIST[@]}"; do
        t="${t// /}"
        [[ -z "$t" ]] && continue
        COPY_SRCS+=("$COPY_FROM"); COPY_DSTS+=("$t")
    done
fi

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
if [[ "${#COPY_DSTS[@]}" -gt 0 ]]; then
    echo "copy pairs:"
    for i in "${!COPY_DSTS[@]}"; do
        echo "  ${COPY_SRCS[$i]} -> ${COPY_DSTS[$i]}"
    done
fi
echo "=========================================="

# -------- helpers --------
# find_cal_folder <videos_root> -> prints calibration folder or empty
find_cal_folder() {
    find "$1" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' | head -n 1
}
# find_orig_yaml <cal_folder> -> prints non-derived calibration yaml path
find_orig_yaml() {
    find "$1" -maxdepth 1 -type f -name 'realsense_calibration_*.yaml' \
        ! -name '*_global_aligned.yaml' ! -name '*_slider_aligned.yaml' \
        ! -name '*_manual_aligned.yaml' ! -name '*_aligned.yaml' | head -n 1
}
# get_copy_src_vname <dst_vname> -> prints src vname if dst is a copy-target, else empty
get_copy_src_vname() {
    local dst="$1"
    for i in "${!COPY_DSTS[@]}"; do
        if [[ "${COPY_DSTS[$i]}" == "$dst" ]]; then
            echo "${COPY_SRCS[$i]}"
            return 0
        fi
    done
    echo ""
}

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

    # -------- copy-from-another-day path --------
    SRC_VNAME="$(get_copy_src_vname "$VNAME")"
    if [[ -n "$SRC_VNAME" ]]; then
        # Resolve src cal folder + src aligned yaml
        SRC_VIDEOS_ROOT=""
        if [[ -n "$DATA_ROOT" ]]; then
            SRC_VIDEOS_ROOT="${DATA_ROOT%/}/$SRC_VNAME"
        else
            # fall back: search among the provided videos_roots for a matching basename
            for v in "${VIDEOS_ROOTS[@]}"; do
                if [[ "$(basename "$v")" == "$SRC_VNAME" ]]; then SRC_VIDEOS_ROOT="$v"; break; fi
            done
        fi
        if [[ -z "$SRC_VIDEOS_ROOT" || ! -d "$SRC_VIDEOS_ROOT" ]]; then
            echo "  [FAIL] --copy source $SRC_VNAME not found next to dst"
            FAIL_LIST+=("$VNAME:src_missing"); continue
        fi
        SRC_CAL="$(find_cal_folder "$SRC_VIDEOS_ROOT")"
        if [[ -z "$SRC_CAL" ]]; then
            echo "  [FAIL] src $SRC_VNAME has no realsense_calibrate_* folder"
            FAIL_LIST+=("$VNAME:src_no_cal"); continue
        fi
        SRC_ORIG="$(find_orig_yaml "$SRC_CAL")"
        if [[ -z "$SRC_ORIG" ]]; then
            echo "  [FAIL] src $SRC_VNAME has no realsense_calibration_*.yaml"
            FAIL_LIST+=("$VNAME:src_no_yaml"); continue
        fi
        SRC_ALIGNED="${SRC_CAL}/$(basename "$SRC_ORIG" .yaml)_global_aligned.yaml"
        if [[ ! -f "$SRC_ALIGNED" ]]; then
            echo "  [FAIL] src $SRC_VNAME has no _global_aligned.yaml: $SRC_ALIGNED"
            FAIL_LIST+=("$VNAME:src_not_aligned"); continue
        fi

        echo "  [copy] from $SRC_VNAME ($(basename "$SRC_ALIGNED"))"
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  [dry_run] would copy $SRC_ALIGNED -> $ALIGNED_YAML"
            OK_LIST+=("$VNAME"); continue
        fi
        if ! python "$HOCAP_ROOT/tools/copy_aligned_yaml.py" \
                --src_yaml "$SRC_ALIGNED" \
                --dst_orig_yaml "$ORIG_YAML" \
                --dst_aligned_yaml "$ALIGNED_YAML"; then
            echo "  [FAIL] copy_aligned_yaml exited nonzero"
            FAIL_LIST+=("$VNAME:copy"); continue
        fi
        echo "  [OK]  $ALIGNED_YAML (copied from $SRC_VNAME)"
        OK_LIST+=("$VNAME"); continue
    fi
    # -------- end copy path --------

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
