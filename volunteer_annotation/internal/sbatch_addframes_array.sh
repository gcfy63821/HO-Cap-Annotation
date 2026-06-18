#!/usr/bin/env bash
# ADD-FRAMES pass: compute extra annotatable keyframes (default the 50% and the
# LAST frame) for an EXISTING bundle, so volunteers can annotate the deformable
# manipulated-object at THREE frames (frame 0 + 50% + last). Additive — keeps the
# original 0/0.1/0.2 keyframes; only the new frames' embeddings are computed.
#
# Same workflow as sbatch_precompute_array.sh (this is a thin wrapper that adds
# --add_frames). Run on the cluster:
#     bash volunteer_annotation/internal/sbatch_addframes_array.sh \
#         --videos_root /viscam/projects/robotool/data/videos_0106
#     # all folders:
#     bash volunteer_annotation/internal/sbatch_addframes_array.sh
#     # different frames / fresh bundle / shard count:
#     FRACS=0.5,1.0 BUNDLE=/viscam/projects/robotool/_va_bundle_v2 \
#         bash volunteer_annotation/internal/sbatch_addframes_array.sh --videos_root ... --shards 4
#
# Resumable (skips exps whose new frames already exist). Submits the same sharded
# GPU array + dependent merge, logs to one file in slurm_outs.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
FRACS="${FRACS:-0.5,1.0}"   # the extra annotatable frames (50% + last)
exec bash "$HERE/sbatch_precompute_array.sh" --add_frames "$FRACS" "$@"
