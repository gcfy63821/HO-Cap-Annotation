#!/bin/bash
# 对所有 02XX/03XX 的已完成任务批量跑 correct_prompt_consistency.py
# Usage: bash run_verify_correct_all.sh [--dry-run] [--workers N] [--role ROLE]

set -e
cd "$(dirname "$0")/.."

WORKERS=4
DRY_RUN=""
ROLE="primary_tool"
for arg in "$@"; do
    case $arg in
        --dry-run)       DRY_RUN="--dry-run" ;;
        --workers=*)     WORKERS="${arg#*=}" ;;
        --workers)       WORKERS="$2"; shift ;;
        --role=*)        ROLE="${arg#*=}" ;;
        --role)          ROLE="$2"; shift ;;
    esac
done

SCRIPT="volunteer_annotation/correct_prompt_consistency.py"

TASKS=(
    "videos_0202/spoon_press_sponge"
    "videos_0204/fork_flip_egg"
    "videos_0204/knife_spread_tomatosauce"
    "videos_0204/spatula_flip_egg"
    "videos_0204/spatula_spread_tomatosauce"
    "videos_0209/knife_spread_tomatosauce"
    "videos_0209/spatula_spread_tomatosauce"
    "videos_0209/spoon_spread_tomatosauce"
    "videos_0210/fork_flip_naan"
    "videos_0211/spatula_flip_naan"
    "videos_0212/fork_spread_tomatosauce"
    "videos_0213/fork_flip_egg"
    "videos_0213/spatula_flip_naan"
    "videos_0213/spatula_press_sponge"
    "videos_0216/fork_scoop_nuts"
    "videos_0216/ladle_scoop_nuts"
    "videos_0218/fork_dig_toysand"
    "videos_0218/rollingpin_roll_sand"
    "videos_0218/scooper_dig_toysand"
    "videos_0218/spoon_dig_toysand"
    "videos_0220/fork_spread_tomatosauce"
    "videos_0220/knife_open_box"
    "videos_0222/fork_shape_kineticsand"
    "videos_0222/knifeblade_open_box"
    "videos_0222/scooper_shape_kineticsand"
    "videos_0227/book_push_toy"
    "videos_0227/knifeblade_open_box"
    "videos_0227/mallet_push_toy"
    "videos_0227/pan_push_toy"
    "videos_0304/malletpestle_crush_nuts"
    "videos_0304/scooper_scoop_icecream"
    "videos_0304/spatula_lift_place_egg"
    "videos_0304/spoon_lift_place_egg"
    "videos_0309/pestle_crush_nuts"
    "videos_0309/spoon_dig_plantingpot"
)

echo "共 ${#TASKS[@]} 个任务 — role=$ROLE, workers=$WORKERS ${DRY_RUN}"
echo ""

run_one() {
    local task="$1"
    local slug="${task/\//_}"
    local logf="/tmp/correct_${slug}.log"
    echo "[START] $task"
    if conda run -n hocap-annotation python "$SCRIPT" \
        --task "$task" --role "$ROLE" $DRY_RUN \
        > "$logf" 2>&1; then
        local summary
        summary=$(grep -E "总计修正" "$logf" | tail -1)
        echo "[OK]    $task  $summary"
    else
        echo "[FAIL]  $task  (see $logf)"
    fi
}

export -f run_one
export SCRIPT DRY_RUN ROLE

printf '%s\n' "${TASKS[@]}" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}

echo ""
echo "全部完成。"
