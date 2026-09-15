#!/bin/bash
set -e
cd "$(dirname "$0")/.."

WORKERS=${1:-3}
shift || true
MAX_EXPS=0   # 0 = unlimited; set via --max-exps N
FIRST_FRAME_ONLY=0
SCRIPT="volunteer_annotation/template_auto_annotate.py"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-exps=*) MAX_EXPS="${1#*=}" ;;
        --max-exps)   MAX_EXPS="$2"; shift ;;
        --first-frame-only) FIRST_FRAME_ONLY=1 ;;
    esac
    shift
done

JOBS=(
    "videos_0202/spoon_press_sponge|smallwoodenspoon"
    "videos_0202/spoon_press_sponge|bigwoodenspoon_press_sponge_in_largeshallowcontainer"
    "videos_0202/spoon_press_sponge|smallwoodenspoon_press_sponge_in_largeshallowcontainer"
    "videos_0202/spoon_press_sponge|bigwoodenspoon"
    "videos_0204/fork_flip_egg|bigwoodenspatula"
    "videos_0204/fork_flip_egg|redrubberspatula"
    "videos_0204/knife_spread_tomatosauce|orangeknife"
    "videos_0204/knife_spread_tomatosauce|greenknife"
    "videos_0204/spatula_flip_egg|bigwoodenspatula"
    "videos_0204/spatula_flip_egg|redrubberspatula"
    "videos_0204/spatula_flip_egg|curvedwoodenspatula"
    "videos_0204/spatula_spread_tomatosauce|curvedwoodenspatula"
    "videos_0209/knife_spread_tomatosauce|orangeknife"
    "videos_0209/knife_spread_tomatosauce|greenknife"
    "videos_0209/spatula_spread_tomatosauce|redrubberspatula"
    "videos_0209/spoon_spread_tomatosauce|greenspoon"
    "videos_0209/spoon_spread_tomatosauce|bigwoodenspoon1"
    "videos_0210/fork_flip_naan|greenfork"
    "videos_0210/fork_flip_naan|bigwoodenfork"
    "videos_0211/spatula_flip_naan|bigwoodenspatula"
    "videos_0211/spatula_flip_naan|redrubberspatula"
    "videos_0212/fork_spread_tomatosauce|bigwoodenfork"
    "videos_0213/fork_flip_egg|bigwoodenfork"
    "videos_0213/fork_flip_egg|smallwoodenfork"
    "videos_0213/spatula_flip_naan|redrubberspatula"
    "videos_0213/spatula_press_sponge|redrubberspatula"
    "videos_0213/spatula_press_sponge|curvedwoodenspatula"
    "videos_0216/fork_scoop_nuts|greenfork"
    "videos_0216/fork_scoop_nuts|largeodenfork"
    "videos_0216/fork_scoop_nuts|smallwoodenfork"
    "videos_0216/ladle_scoop_nuts|ladle"
    "videos_0216/ladle_scoop_nuts|ladle_scoop_smallamount_peanuts_nuts_in_mediumshallowcontainer"
    "videos_0218/fork_dig_toysand|greenfork"
    "videos_0218/fork_dig_toysand|bigwoodenfork2"
    "videos_0218/fork_dig_toysand|bigwoodenfork_dig_toy"
    "videos_0218/fork_dig_toysand|bigwoodenfork"
    "videos_0218/rollingpin_roll_sand|rollingpin_small"
    "videos_0218/rollingpin_roll_sand|rollingpin_roll_small_sand_in_cuttingboard"
    "videos_0218/rollingpin_roll_sand|rollingpin_roll_large_sand_in_cuttingboard"
    "videos_0218/rollingpin_roll_sand|rollingpin_large"
    "videos_0218/rollingpin_roll_sand|rollingpin2"
    "videos_0218/rollingpin_roll_sand|rollingpin"
    "videos_0218/scooper_dig_toysand|icecreamscooper2"
    "videos_0218/scooper_dig_toysand|openscooper"
    "videos_0218/scooper_dig_toysand|icecreamscooper"
    "videos_0218/scooper_dig_toysand|icecreamscooper_dig_toy"
    "videos_0218/scooper_dig_toysand|openscooper_dig_toy"
    "videos_0218/scooper_dig_toysand|openscooper2"
    "videos_0218/spoon_dig_toysand|greenspoon"
    "videos_0218/spoon_dig_toysand|bigwoodenspoon"
    "videos_0218/spoon_dig_toysand|greenspoon_dig_toy"
    "videos_0220/fork_spread_tomatosauce|greenfork"
    "videos_0220/knife_open_box|greyplasticknife_open_cardboard_box_bluetape"
    "videos_0220/knife_open_box|pinkknife_open_cardboard_box_bluetape"
    "videos_0220/knife_open_box|greyplasticknife2"
    "videos_0220/knife_open_box|pinkknife"
    "videos_0220/knife_open_box|orangeknife"
    "videos_0220/knife_open_box|greenknife"
    "videos_0220/knife_open_box|greyplasticknife"
    "videos_0222/fork_shape_kineticsand|greenfork"
    "videos_0222/fork_shape_kineticsand|bigwoodenfork"
    "videos_0222/knifeblade_open_box|whiteknifeblade"
    "videos_0222/scooper_shape_kineticsand|icecreamscooper"
    "videos_0222/scooper_shape_kineticsand|measuringscooper"
    "videos_0227/book_push_toy|book"
    "videos_0227/book_push_toy|book_leftup"
    "videos_0227/book_push_toy|book_push_toy_rightup_faraway"
    "videos_0227/book_push_toy|book_rightup"
    "videos_0227/book_push_toy|book_push_toy_leftup_faraway"
    "videos_0227/knifeblade_open_box|greenknifeblade"
    "videos_0227/knifeblade_open_box|whiteknifeblade"
    "videos_0227/mallet_push_toy|mallet_rightup"
    "videos_0227/mallet_push_toy|mallet_push_toy_leftup_faraway"
    "videos_0227/mallet_push_toy|mallet"
    "videos_0227/mallet_push_toy|mallet_leftup"
    "videos_0227/mallet_push_toy|mallet_push_toy_rightup_faraway"
    "videos_0227/pan_push_toy|whitean_push_toy_leftup_faraway"
    "videos_0227/pan_push_toy|blackpan"
    "videos_0227/pan_push_toy|blackpan_push_toy_leftup_faraway"
    "videos_0227/pan_push_toy|whitepan_push_toy_rightup_faraway"
    "videos_0227/pan_push_toy|whitepan_rightup"
    "videos_0227/pan_push_toy|blackpan_leftup"
    "videos_0227/pan_push_toy|whitepan"
    "videos_0227/pan_push_toy|whitean"
    "videos_0304/malletpestle_crush_nuts|pestle"
    "videos_0304/scooper_scoop_icecream|icecreamscooper"
    "videos_0304/scooper_scoop_icecream|measuringscooper"
    "videos_0304/spatula_lift_place_egg|redrubberspatula"
    "videos_0304/spatula_lift_place_egg|bigwoodenspatula_lift_egg"
    "videos_0304/spoon_lift_place_egg|greenspoon"
    "videos_0304/spoon_lift_place_egg|bigwoodenspoon"
    "videos_0309/pestle_crush_nuts|pestle"
    "videos_0309/spoon_dig_plantingpot|greenspoon"
    "videos_0309/spoon_dig_plantingpot|smallwoodenspoon"
    "videos_0309/spoon_dig_plantingpot|yellowplasticspoon"
    "videos_0309/spoon_dig_plantingpot|bigwoodenspoon"
)

MAXEXPS_FLAG=""
[[ "$MAX_EXPS" -gt 0 ]] 2>/dev/null && MAXEXPS_FLAG="--max-exps $MAX_EXPS"
FIRST_FRAME_FLAG=""
[[ "$FIRST_FRAME_ONLY" -eq 1 ]] && FIRST_FRAME_FLAG="--first-frame-only"

echo "共 ${#JOBS[@]} 个 (task, keyword) — workers=$WORKERS${MAXEXPS_FLAG:+, max-exps=$MAX_EXPS}${FIRST_FRAME_FLAG:+, first-frame-only}"

run_one() {
    local task="${1%%|*}" keyword="${1##*|}"
    local slug="${task/\//_}"
    local logf="/tmp/reannot_overwrite_${slug}_${keyword}.log"
    echo "[START] $task  kw=$keyword"
    if conda run -n hocap-annotation python "$SCRIPT" \
        --task "$task" --keyword "$keyword" --overwrite $MAXEXPS_FLAG $FIRST_FRAME_FLAG \
        > "$logf" 2>&1; then
        summary=$(grep -E "生成:|跳过:" "$logf" | tail -3 | tr "\n" " ")
        echo "[OK]    $task  kw=$keyword  $summary"
    else
        echo "[FAIL]  $task  kw=$keyword  (see $logf)"
    fi
}

export -f run_one
export SCRIPT MAXEXPS_FLAG FIRST_FRAME_FLAG

printf '%s\n' "${JOBS[@]}" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}

echo ""
echo "全部完成。"
