#!/bin/bash
# 对本地有 depth PNG 的已完成任务重新跑自动标注（加入 depth_reproj）
# Usage: bash run_reannot_with_depth.sh [--workers N]

set -e
cd "$(dirname "$0")/.."

WORKERS=3
for arg in "$@"; do
    case $arg in
        --workers=*) WORKERS="${arg#*=}" ;;
        --workers)   WORKERS="$2"; shift ;;
    esac
done

SCRIPT="volunteer_annotation/template_auto_annotate.py"
FORCE="--overwrite"

# 任务列表：(task, keyword) 只包含本地有 depth PNG 的已完成任务
JOBS=(
    "videos_0202/spoon_press_sponge|bigwoodenspoon"
    "videos_0202/spoon_press_sponge|smallwoodenspoon"
    "videos_0204/fork_flip_egg|bigwoodenspatula"
    "videos_0204/fork_flip_egg|redrubberspatula"
    "videos_0204/knife_spread_tomatosauce|greenknife"
    "videos_0204/knife_spread_tomatosauce|orangeknife"
    "videos_0204/spatula_flip_egg|bigwoodenspatula"
    "videos_0204/spatula_flip_egg|curvedwoodenspatula"
    "videos_0204/spatula_flip_egg|redrubberspatula"
    "videos_0204/spatula_spread_tomatosauce|curvedwoodenspatula"
    "videos_0209/knife_spread_tomatosauce|greenknife"
    "videos_0209/knife_spread_tomatosauce|orangeknife"
    "videos_0209/spatula_spread_tomatosauce|redrubberspatula"
    "videos_0209/spoon_spread_tomatosauce|bigwoodenspoon1"
    "videos_0209/spoon_spread_tomatosauce|greenspoon"
    "videos_0210/fork_flip_naan|bigwoodenfork"
    "videos_0210/fork_flip_naan|greenfork"
    "videos_0211/spatula_flip_naan|bigwoodenspatula"
    "videos_0211/spatula_flip_naan|redrubberspatula"
    "videos_0212/fork_spread_tomatosauce|bigwoodenfork"
    "videos_0213/fork_flip_egg|bigwoodenfork"
    "videos_0213/fork_flip_egg|smallwoodenfork"
    "videos_0213/spatula_flip_naan|redrubberspatula"
    "videos_0213/spatula_press_sponge|curvedwoodenspatula"
    "videos_0213/spatula_press_sponge|redrubberspatula"
)

echo "共 ${#JOBS[@]} 个 (task, keyword) 组合 — 重跑加 depth_reproj"
echo "Workers: $WORKERS"
echo ""

run_one() {
    local task="$1" keyword="$2"
    local slug="${task/\//_}"
    local logf="/tmp/reann_depth_${slug}_${keyword}.log"
    echo "[START] $task  keyword=$keyword"
    if conda run -n hocap-annotation python "$SCRIPT" \
        --task "$task" --keyword "$keyword" $FORCE \
        > "$logf" 2>&1; then
        echo "[OK]    $task  keyword=$keyword"
    else
        echo "[FAIL]  $task  keyword=$keyword  (see $logf)"
        tail -5 "$logf"
    fi
}

export -f run_one
export SCRIPT FORCE

printf '%s\n' "${JOBS[@]}" | \
    xargs -P "$WORKERS" -I{} bash -c '
        IFS="|" read -r task keyword <<< "{}"
        run_one "$task" "$keyword"
    '

echo ""
echo "全部完成"
