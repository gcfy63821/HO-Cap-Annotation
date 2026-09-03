#!/bin/bash
# 批量跑所有 (task, keyword) 组合的自动标注
# Usage: bash run_auto_annotate_all.sh [--force] [--workers N]

set -e
cd "$(dirname "$0")/.."

FORCE=""
WORKERS=2
for arg in "$@"; do
    case $arg in
        --force) FORCE="--force" ;;
        --workers=*) WORKERS="${arg#*=}" ;;
    esac
done

MODEL_DIR="volunteer_annotation/color_models"
SCRIPT="volunteer_annotation/template_auto_annotate.py"
VERB_PATTERN="_flip_|_scoop_|_push_|_press_|_spread_|_open_|_roll_|_dig_|_shape_|_flatten_|_crush_"

# 收集所有 (task, keyword) 组合
declare -a JOBS=()
for task_dir in "$MODEL_DIR"/videos_0*/; do
    slug=$(basename "$task_dir")
    # 只处理 2-3月任务 (videos_02xx, videos_03xx)
    # videos_MMDD: positions 8-9 are MM (videos_ = 7 chars, then MM)
    month=$(echo "$slug" | cut -c8-9)
    if [[ "$month" != "02" && "$month" != "03" ]]; then continue; fi
    # slug: videos_0202_spoon_press_sponge → task: videos_0202/spoon_press_sponge
    # Handle videos_MMDD_N_task (e.g. videos_0105_1_scooper_scoop_nuts → videos_0105_1/scooper_scoop_nuts)
    if echo "$slug" | grep -qE '^videos_[0-9]{4}_[0-9]+_'; then
        date_part=$(echo "$slug" | cut -d_ -f1-3)
        task_part=$(echo "$slug" | cut -d_ -f4-)
    else
        date_part=$(echo "$slug" | cut -d_ -f1-2)
        task_part=$(echo "$slug" | cut -d_ -f3-)
    fi
    task="${date_part}/${task_part}"

    for meta_f in "$task_dir"*.meta.json; do
        [ -f "$meta_f" ] || continue
        fname=$(basename "$meta_f" .meta.json)
        # 跳过旧的长 keyword（含动词）
        if echo "$fname" | grep -qE "$VERB_PATTERN"; then
            continue
        fi
        JOBS+=("${task}|${fname}")
    done
done

echo "共 ${#JOBS[@]} 个 (task, keyword) 组合"
echo "Workers: $WORKERS  Force: ${FORCE:-no}"
echo ""

# 并行运行，每次最多 WORKERS 个
run_one() {
    local task="$1" keyword="$2"
    local slug="${task/\//_}"
    local logf="/tmp/auto_ann_${slug}_${keyword}.log"
    echo "[START] $task  keyword=$keyword"
    if conda run -n hocap-annotation python "$SCRIPT" \
        --task "$task" --keyword "$keyword" $FORCE \
        > "$logf" 2>&1; then
        local done_count=$(grep -c "^\[DONE\]\|already done\|Saved\|skipping" "$logf" 2>/dev/null || echo "?")
        echo "[OK]    $task  keyword=$keyword  (log: $logf)"
    else
        echo "[FAIL]  $task  keyword=$keyword  (see $logf)"
        tail -5 "$logf"
    fi
}

export -f run_one
export SCRIPT FORCE

# 用 xargs 并行
printf '%s\n' "${JOBS[@]}" | \
    xargs -P "$WORKERS" -I{} bash -c '
        IFS="|" read -r task keyword <<< "{}"
        run_one "$task" "$keyword"
    '

echo ""
echo "全部完成"
