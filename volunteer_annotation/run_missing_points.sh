#!/bin/bash
# 只对缺正点的实验重新生成 first-frame 标注
# 用法: bash run_missing_points.sh [WORKERS] [--list FILE]
#   WORKERS: 并行 worker 数，默认 2
#   --list FILE: 实验列表文件，默认 /tmp/missing_points_exps.txt
#                格式每行: task/subtask|exp_name
set -e
cd "$(dirname "$0")/.."

WORKERS=${1:-2}
shift || true
LIST_FILE="/tmp/missing_points_exps.txt"
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
echo "缺正点实验: $TOTAL 个  workers=$WORKERS"
echo "列表: $LIST_FILE"
echo ""

run_one() {
    local line="$1"
    local task="${line%%|*}"
    local rest="${line#*|}"
    local exp="${rest%%|*}"
    local keyword="${rest##*|}"
    local slug="${task/\//_}"
    local logf="/tmp/missing_annot_${slug}_${exp}.log"

    if conda run -n hocap-annotation python "$SCRIPT" \
        --task "$task" --keyword "$keyword" --exp "$exp" \
        --overwrite --first-frame-only \
        > "$logf" 2>&1; then
        summary=$(grep -E "生成:|未检测到:" "$logf" | tail -3 | tr "\n" " ")
        echo "[OK]  $task  $exp  $summary"
    else
        echo "[FAIL] $task  $exp  (see $logf)"
    fi
}

export -f run_one
export SCRIPT

cat "$LIST_FILE" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}

echo ""
echo "全部完成。"
