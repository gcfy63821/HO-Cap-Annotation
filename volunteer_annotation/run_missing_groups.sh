#!/bin/bash
# 按 (task, keyword) 分组处理缺正点实验，每组共享一次 SAM2 加载
# 用法: bash run_missing_groups.sh [WORKERS] [--list FILE]
set -e
cd "$(dirname "$0")/.."

WORKERS=${1:-2}
shift || true
LIST_FILE="/tmp/missing_groups.txt"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --list) LIST_FILE="$2"; shift ;;
    esac
    shift
done

SCRIPT="volunteer_annotation/template_auto_annotate.py"

if [[ ! -f "$LIST_FILE" ]]; then
    echo "[ERROR] list file not found: $LIST_FILE"
    exit 1
fi

TOTAL=$(wc -l < "$LIST_FILE")
echo "共 $TOTAL 组 (task,keyword)  workers=$WORKERS"
echo ""

run_one() {
    local line="$1"
    local task="${line%%|*}"
    local rest="${line#*|}"
    local keyword="${rest%%|*}"
    local exp_list="${rest##*|}"
    local slug="${task/\//_}_${keyword}"
    local logf="/tmp/missing_group_${slug}.log"

    local n_exps=$(echo "$exp_list" | tr ',' '\n' | wc -l)
    echo "[START] $task  kw=$keyword  exps=$n_exps"

    if conda run -n hocap-annotation python "$SCRIPT" \
        --task "$task" --keyword "$keyword" \
        --exp-list "$exp_list" \
        --overwrite --first-frame-only \
        > "$logf" 2>&1; then
        summary=$(grep -E "生成:|未检测到:" "$logf" | tail -4 | tr "\n" " ")
        echo "[OK]    $task  kw=$keyword  exps=$n_exps  $summary"
    else
        echo "[FAIL]  $task  kw=$keyword  (see $logf)"
    fi
}

export -f run_one
export SCRIPT

cat "$LIST_FILE" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}

echo ""
echo "全部完成。"
