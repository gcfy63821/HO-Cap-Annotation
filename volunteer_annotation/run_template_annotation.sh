#!/usr/bin/env bash
# 完整的 template → 自动标注 流程
#
# 用法：
#   bash run_template_annotation.sh \
#       --task videos_0106/spoon_scoop_nuts \
#       --template 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_10 \
#       [--keyword plasticspoon_scoop_almond_nuts]   # 省略则自动从 template 名提取
#       [--min-sim 0.65]
#       [--overwrite]
#       [--dry-run]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK=""
TEMPLATE=""
KEYWORD=""
MIN_SIM=0.65
THRESHOLD=0.85
OVERWRITE=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)     TASK="$2"; shift 2;;
        --template) TEMPLATE="$2"; shift 2;;
        --keyword)  KEYWORD="$2"; shift 2;;
        --min-sim)  MIN_SIM="$2"; shift 2;;
        --overwrite) OVERWRITE="--overwrite"; shift;;
        --dry-run)  DRY_RUN="--dry-run"; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$TASK" || -z "$TEMPLATE" ]]; then
    echo "Usage: $0 --task <task> --template <exp_name> [options]"
    exit 1
fi

cd "$SCRIPT_DIR/.."
ENV="hocap-annotation"

echo "=== Step 1: Extract template color model ==="
conda run -n "$ENV" python volunteer_annotation/template_color_extract.py \
    --task "$TASK" \
    --template-exp "$TEMPLATE"

# 自动提取 keyword（如果没有指定）
if [[ -z "$KEYWORD" ]]; then
    KEYWORD=$(conda run -n "$ENV" python3 -c "
import re, sys
name = '$TEMPLATE'
s = re.sub(r'^\d{8}_', '', name)
s = re.sub(r'_\d+$', '', s)
s = re.sub(r'_from_.+', '', s)
print(s)
")
    echo "Auto keyword: $KEYWORD"
fi

echo ""
echo "=== Step 2: Auto-annotate matching experiments ==="
conda run -n "$ENV" python volunteer_annotation/template_auto_annotate.py \
    --task "$TASK" \
    --keyword "$KEYWORD" \
    --min-sim "$MIN_SIM" \
    --threshold "$THRESHOLD" \
    $OVERWRITE $DRY_RUN

echo ""
echo "=== Done ==="
