#!/bin/bash
#
# Run scripts/cleanup_done_exps.sh against every videos_X_annotated/ folder
# under a data root. Convenience wrapper for the common cluster case where
# you have many date-stamped videos folders side by side.
#
# Layout assumed:
#   <data_root>/
#     videos_0101/                  ← raw recordings (untouched)
#     videos_0101_annotated/        ← processed (this script targets these)
#     videos_0102/
#     videos_0102_annotated/
#     ...
#
# Usage:
#   bash scripts/cleanup_all_videos.sh
#       (uses defaults: $DATA_ROOT or /viscam/projects/robotool/data)
#
#   bash scripts/cleanup_all_videos.sh --data_root /viscam/projects/robotool/data
#   bash scripts/cleanup_all_videos.sh --dry_run        (preview, don't delete)
#   bash scripts/cleanup_all_videos.sh --keep_ob_in_world --keep_dino_auto
#   bash scripts/cleanup_all_videos.sh --pattern 'videos_010*_annotated'  (glob filter)
#
# All flags after --data_root / --pattern are forwarded to cleanup_done_exps.sh.
# Examples of forwarded flags: --dry_run, --keep_ob_in_world, --keep_dino_auto,
# --quiet.

set -u

# ---------- defaults ----------
DATA_ROOT="${DATA_ROOT:-/viscam/projects/robotool/data}"
PATTERN="videos_*_annotated"

# Resolve cleanup_done_exps.sh next to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP="${SCRIPT_DIR}/cleanup_done_exps.sh"

# ---------- arg parsing (consume our own; pass rest through) ----------
PASS_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --data_root) DATA_ROOT="$2"; shift 2;;
        --pattern)   PATTERN="$2"; shift 2;;
        -h|--help)   sed -n '2,28p' "$0"; exit 0;;
        *)           PASS_ARGS+=("$1"); shift;;
    esac
done

# ---------- sanity ----------
if [[ ! -x "$CLEANUP" && ! -f "$CLEANUP" ]]; then
    echo "Error: cleanup_done_exps.sh not found at $CLEANUP"
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Error: --data_root not a directory: $DATA_ROOT"
    exit 1
fi

# Enumerate matching folders.
shopt -s nullglob
ROOTS=( "$DATA_ROOT"/$PATTERN )
shopt -u nullglob
# Keep only directories.
DIRS=()
for r in "${ROOTS[@]}"; do
    [[ -d "$r" ]] && DIRS+=("$r")
done

if [[ "${#DIRS[@]}" -eq 0 ]]; then
    echo "No directories matched: $DATA_ROOT/$PATTERN"
    exit 0
fi

echo "=========================================="
echo "data_root  : $DATA_ROOT"
echo "pattern    : $PATTERN"
echo "matched    : ${#DIRS[@]} directory(ies)"
for d in "${DIRS[@]}"; do echo "   - $(basename "$d")"; done
echo "forwarded  : ${PASS_ARGS[*]:-<none>}"
echo "=========================================="

# ---------- per-root cleanup ----------
N_OK=0; N_FAIL=0
FAILED=()

for d in "${DIRS[@]}"; do
    echo ""
    echo "##########################################"
    echo "# annotated_root: $(basename "$d")"
    echo "##########################################"
    if bash "$CLEANUP" --annotated_root "$d" "${PASS_ARGS[@]}"; then
        N_OK=$((N_OK + 1))
    else
        rc=$?
        N_FAIL=$((N_FAIL + 1))
        FAILED+=("$d (rc=$rc)")
    fi
done

echo ""
echo "=========================================="
echo "all-videos summary: ok=$N_OK  failed=$N_FAIL"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo "failed roots:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
echo "=========================================="
[[ "$N_FAIL" -gt 0 ]] && exit 1 || exit 0
