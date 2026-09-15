#!/bin/bash
# 一键同步 auto_prompts 到 cluster 并启动 SAM2 稠密推理
# 用法:
#   bash sync_and_launch_cluster.sh videos_0204/spatula_flip_egg [--keyword redrubberspatula] [--max-exps 5]

set -e
TASK="${1:?usage: $0 <task> [--keyword kw] [--max-exps N] [--overwrite]}"
shift

CLUSTER="chenrq@scdt.stanford.edu"
CLUSTER_ROOT="/viscam/projects/robotool"
LOCAL_AP="/data/robotool/_va_bundle_v2_auto_prompts"
LOCAL_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$LOCAL_SCRIPT_DIR/../.." && pwd)"

# ── 1. 同步 auto_prompts（只同步该 task）──────────────────────────────────────
echo "[1/4] 同步 auto_prompts: $TASK"
ssh "$CLUSTER" "mkdir -p $CLUSTER_ROOT/_va_bundle_v2_auto_prompts/$TASK"
rsync -avz --progress \
    "$LOCAL_AP/$TASK/" \
    "$CLUSTER:$CLUSTER_ROOT/_va_bundle_v2_auto_prompts/$TASK/"

# ── 2. 同步推理脚本 ──────────────────────────────────────────────────────────
echo "[2/4] 同步推理脚本"
rsync -avz \
    "$LOCAL_SCRIPT_DIR/run_dense_masks_cluster.py" \
    "$CLUSTER:$CLUSTER_ROOT/src/"

# ── 3. 检查 cluster 上的 SAM2 和 conda env ───────────────────────────────────
echo "[3/4] 检查 cluster 环境"
ssh "$CLUSTER" "
    SAM2_CKPT=$CLUSTER_ROOT/src/HO-Cap-Annotation/../../../mesh_reconstruction/sam2/checkpoints/sam2.1_hiera_large.pt
    # Try common locations
    for p in \
        $CLUSTER_ROOT/../mesh_reconstruction/sam2 \
        /viscam/projects/robotool/src/mesh_reconstruction/sam2 \
        ~/mesh_reconstruction/sam2; do
        [ -f \"\$p/checkpoints/sam2.1_hiera_large.pt\" ] && echo \"SAM2 found: \$p\" && break
    done
    nvidia-smi -L 2>/dev/null || echo 'no GPU detected by nvidia-smi'
    conda env list 2>/dev/null | grep -E 'hocap|sam2|annotation' || echo 'no matching conda envs'
"

# ── 4. 启动推理（nohup + screen）────────────────────────────────────────────
echo "[4/4] 启动 cluster 推理"
LOGFILE="$CLUSTER_ROOT/_va_dense_masks/$(echo $TASK | tr '/' '_').log"

# Build extra args
EXTRA_ARGS="$@"

ssh "$CLUSTER" "
    mkdir -p $CLUSTER_ROOT/_va_dense_masks
    nohup conda run -n hocap-annotation python $CLUSTER_ROOT/src/run_dense_masks_cluster.py \
        --data-root  $CLUSTER_ROOT/data \
        --ap-root    $CLUSTER_ROOT/_va_bundle_v2_auto_prompts \
        --out-root   $CLUSTER_ROOT/_va_dense_masks \
        --task       $TASK \
        $EXTRA_ARGS \
        > $LOGFILE 2>&1 &
    echo 'PID: '\$!
    echo 'Log: $LOGFILE'
"
echo ""
echo "启动完成。查看进度:"
echo "  ssh $CLUSTER 'tail -f $LOGFILE'"
echo ""
echo "结果同步回本地:"
echo "  rsync -avz '$CLUSTER:$CLUSTER_ROOT/_va_dense_masks/$TASK/' /data/robotool/_va_dense_masks/$TASK/"
