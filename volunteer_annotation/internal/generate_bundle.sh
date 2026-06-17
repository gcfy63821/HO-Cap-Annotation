#!/usr/bin/env bash
# Parallel embedding-bundle generation: one precompute process per GPU, sharded
# by exp index, then merge the per-shard manifests into one manifest.json.
#
# Run on the GPU server (conda env `hocap-annotation` active):
#   bash generate_bundle.sh <SRC_ROOT> <BUNDLE_DIR> [GPU_LIST]
# e.g.
#   bash generate_bundle.sh /data/robotool/videos_0102 /data/robotool/_bundle 0,1,2,3
#
# Notes:
#   * --no_refmask: production-lean (the decode seam is already validated). Drop
#     it (and run a single --exp WITH refmask) if you want to re-validate.
#   * Idempotent: re-running skips nothing but overwrites per-exp files + manifest
#     fragments; safe to resume after interruption.
#   * Tune keyframes/thumbs by appending e.g. EXTRA="--keyframe_fracs 0,0.1,0.2".
set -euo pipefail

SRC=${1:?usage: generate_bundle.sh SRC BUNDLE [GPU_LIST]}
BUNDLE=${2:?usage: generate_bundle.sh SRC BUNDLE [GPU_LIST]}
GPUS=${3:-0}
EXTRA=${EXTRA:-}
HERE=$(cd "$(dirname "$0")" && pwd)

IFS=',' read -ra G <<< "$GPUS"
N=${#G[@]}
mkdir -p "$BUNDLE"
echo "[gen] src=$SRC bundle=$BUNDLE gpus=$GPUS shards=$N extra='$EXTRA'"

pids=()
for i in "${!G[@]}"; do
  echo "  -> shard $i/$N on GPU ${G[$i]}"
  CUDA_VISIBLE_DEVICES="${G[$i]}" python "$HERE/precompute_embeddings.py" \
    --all "$SRC" --bundle "$BUNDLE" --shard "$i/$N" \
    --no_merge --skip_existing --no_refmask $EXTRA \
    > "$BUNDLE/shard_$i.log" 2>&1 &
  pids+=($!)
done

fail=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then echo "  shard $idx done"; else echo "  shard $idx FAILED (see $BUNDLE/shard_$idx.log)"; fail=1; fi
done

python "$HERE/merge_manifests.py" --bundle "$BUNDLE"
echo "[gen] done -> $BUNDLE/manifest.json  (fail=$fail)"
exit $fail
