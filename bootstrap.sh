#!/usr/bin/env bash
#
# bootstrap.sh — bring up GLM-5.2 on a 4-node GB10 / DGX Spark cluster (sm_121).
#
# Designed to be AGENT-RUNNABLE: an agent (Claude Code / pi / etc.) can execute
# this top-to-bottom, or a human can run it from the head node. Every external
# version is pinned. Edit the CONFIG block, then run from the HEAD node.
#
# It is intentionally idempotent-ish and verbose: each step announces itself and
# checks its result so you (or your agent) can see exactly where it stops.
#
# License: Apache-2.0. See LICENSE / NOTICE / CHANGES.md.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# CONFIG — edit these for your cluster
# ============================================================================
# Cluster RoCE rail IPs, rank 0 (head) first. The head must be the node you run
# this from. Worker nodes are reached over SSH (key-based, BatchMode).
NODES=(192.168.200.12 192.168.200.13 192.168.200.14 192.168.200.15)   # rank 0..3
SSH_USER="cosmicraisins"

# Weight distribution. This reference setup uses NFS (one host exports the weights
# read-only to all nodes). NFS is NOT required — replace steps 4-5 with whatever
# gets the weights onto every node (per-node local copies, another shared FS, ...).
NFS_HOST_RAIL="192.168.200.14"
WEIGHTS_RO_DIR="/srv/hf/hub"          # exported dir, mounted on every node
HF_HOME="/srv/hf"                     # HF_HOME on the NFS host

# HuggingFace repo IDs. Defaults pull the published weights for this stack; point
# them at your own repos only if you re-prune / rebuild the draft yourself.
PRUNED_REPO="CosmicRaisins/GLM-5.2-AWQ-INT4-15pct"        # published 15%-pruned AWQ
BASE_AWQ_REPO="cyankiwi/GLM-5.2-AWQ-INT4"                 # unpruned base, if you re-prune
MTP_DRAFT_REPO="CosmicRaisins/GLM-5.2-MTP-INT4-aligned"   # published MTP draft

# vLLM build: use eugr/spark-vllm-docker (NOT vendored here). Pin the ref.
SPARK_VLLM_DOCKER="$HOME/spark-vllm-docker"     # clone of eugr/spark-vllm-docker
VLLM_REF="ab666069935c1f23e8ef56038b4659ac9e8f19f8"   # post-0.23.0, GLM5.2 + indexer/MTP fix
IMAGE_TAG="vllm-node-tf5-glm52-b12x:probe"      # built/tagged by the harness

NCCL_VERSION="2.30.4"                  # the shm_broadcast-wedge fix; aarch64 wheel exists
RECIPE="4x-spark-cluster/glm52-awq-15pct-prod"  # SHORT name — see step 6 note

KERNEL_DST="\$HOME/glm-triton"         # per-node mount source for the Triton kernels
# ============================================================================

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✓ %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
on()   { ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_USER}@$1" "${@:2}"; }

# ----------------------------------------------------------------------------
say "Step 0 — preflight: nodes reachable, docker, arch"
for ip in "${NODES[@]}"; do
  on "$ip" "true" 2>/dev/null || die "cannot SSH to $ip (need key-based BatchMode SSH)"
  on "$ip" "docker info >/dev/null 2>&1" || die "docker not usable on $ip"
  cap=$(on "$ip" "nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1")
  case "$cap" in 12.*) ok "$ip  sm_$cap  docker OK" ;;
                 *) die "$ip reports compute_cap '$cap' — this stack targets sm_121 (12.x) only" ;;
  esac
done

# ----------------------------------------------------------------------------
say "Step 1 — build the vLLM image (eugr/spark-vllm-docker, pinned ref)"
# REQUIRED MODS (not vendored here — see README "Image build"):
#   * mods/glm52-sm12x-sparse  — installs kernels/ into the vLLM tree + patches
#       vllm/utils/deep_gemm.py and sparse_attn_indexer.py (the sm_121 DeepGEMM
#       bypass). Baked by Dockerfile.glm52-consolidated.
#   * mods/glm52-b12x-sparse   — pip install --no-deps b12x==0.23.0 + fused_indexer
#       patch. The decode path calls b12x first; WITHOUT it, cudagraph FULL crashes
#       (torch.full under capture) — use cudagraph_mode PIECEWISE instead.
# build-and-copy.sh below builds only the BASE image; apply both mods on top
# (RUN bash mods/<name>/run.sh, or launch-cluster.sh --apply-mod) before launch.
[ -d "$SPARK_VLLM_DOCKER" ] || die "clone eugr/spark-vllm-docker to $SPARK_VLLM_DOCKER first (NOT vendored here)"
if on "${NODES[0]}" "docker image inspect $IMAGE_TAG >/dev/null 2>&1"; then
  ok "image $IMAGE_TAG already present on head"
else
  say "    building @ $VLLM_REF — this is long (~1h first time); copies to all rails"
  ( cd "$SPARK_VLLM_DOCKER" && ./build-and-copy.sh --vllm-ref "$VLLM_REF" -t "$IMAGE_TAG" --tf5 --copy-to ) \
    || die "image build failed — verify cuda_available after build (torch must NOT be +cpu)"
  ok "image built + copied"
fi

# ----------------------------------------------------------------------------
say "Step 2 — NCCL $NCCL_VERSION (fixes the shm_broadcast warmup wedge)"
NCCL_DIR="$WEIGHTS_RO_DIR/nccl-$NCCL_VERSION"
if on "$NFS_HOST_RAIL" "test -f $NCCL_DIR/libnccl.so.2"; then
  ok "libnccl.so.2 already staged at $NCCL_DIR (NFS)"
else
  on "$NFS_HOST_RAIL" "
    set -e; python3 -m pip download --no-deps -d /tmp/ncclwheel nvidia-nccl-cu13==$NCCL_VERSION
    mkdir -p $NCCL_DIR
    cd /tmp/ncclwheel && unzip -o nvidia_nccl_cu13-*manylinux*aarch64.whl -d /tmp/ncclx
    cp /tmp/ncclx/nvidia/nccl/lib/libnccl.so.2 $NCCL_DIR/
  " || die "NCCL stage failed (need an aarch64 nvidia-nccl-cu13==$NCCL_VERSION wheel)"
  ok "libnccl.so.2 staged → recipe LD_PRELOADs it from $NCCL_DIR"
fi

# ----------------------------------------------------------------------------
say "Step 3 — deploy the Triton sparse-MLA kernels to every node"
# scp can choke on rsync version skew between GX10/GN100; cat|ssh is robust.
MLA="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla"
for ip in "${NODES[@]}"; do
  on "$ip" "mkdir -p $KERNEL_DST"
  for f in "$HERE"/kernels/*.py; do
    cat "$f" | on "$ip" "cat > $KERNEL_DST/$(basename "$f")"
  done
  n=$(on "$ip" "ls $KERNEL_DST/*.py | wc -l")
  ok "$ip  $n kernel files in $KERNEL_DST  (recipe ro-mounts them over $MLA/)"
done

# ----------------------------------------------------------------------------
say "Step 4 — fetch weights to the NFS host ($PRUNED_REPO)"
case "$PRUNED_REPO" in *"<your-hf-username>"*)
  cat <<EOF
   ! PRUNED_REPO is still a placeholder.
     Either (a) set PRUNED_REPO to your uploaded 15%-prune, or
            (b) re-create it: hf download $BASE_AWQ_REPO to the NFS host, then
                python3 $HERE/prune/awq_surgery.py build <src> $WEIGHTS_RO_DIR/glm52-awq-15pct 0.15
     and likewise build the MTP draft from mtp/ (see mtp/README or the retrospective).
EOF
  die "set PRUNED_REPO / MTP_DRAFT_REPO, or re-prune locally, then re-run from step 4" ;;
esac
on "$NFS_HOST_RAIL" "
  set -e; export HF_HOME=$HF_HOME HF_HUB_DISABLE_XET=0
  hf download $PRUNED_REPO  --local-dir $WEIGHTS_RO_DIR/glm52-awq-15pct
  hf download $MTP_DRAFT_REPO --local-dir $WEIGHTS_RO_DIR/glm52-mtp-int4-aligned
" || die "weight download failed (check HF auth + disk on $NFS_HOST_RAIL)"
ok "weights present under $WEIGHTS_RO_DIR"

# ----------------------------------------------------------------------------
say "Step 5 — mount NFS weights read-only on every node"
for ip in "${NODES[@]}"; do
  on "$ip" "mountpoint -q $HF_HOME || sudo mount -t nfs -o ro,vers=3 $NFS_HOST_RAIL:$HF_HOME $HF_HOME" \
    && ok "$ip  $HF_HOME mounted" || die "NFS mount failed on $ip (gn100 nodes may need an interactive sudo)"
done

# ----------------------------------------------------------------------------
say "Step 6 — launch the serving recipe"
# IMPORTANT: pass the SHORT recipe name (no 'recipes/' prefix, no '.yaml'), or
# run-recipe.sh silently falls through to the Ray path WITHOUT the triton mounts.
cp "$HERE/recipes/glm52-awq-15pct-prod.yaml" \
   "$SPARK_VLLM_DOCKER/recipes/4x-spark-cluster/glm52-awq-15pct-prod.yaml"
( cd "$SPARK_VLLM_DOCKER" && WEIGHTS_RO_DIR=$WEIGHTS_RO_DIR ./run-recipe.sh "$RECIPE" ) \
  || die "launch dispatch failed"

say "Step 7 — wait for readiness (~12 min load + ~10 min cudagraph warmup)"
echo "   poll:  curl -s http://localhost:8210/v1/models"
echo "   logs:  docker logs -f vllm_slot   (on the head node)"
echo
echo "   If warmup wedges (NCCL/Gloo), reap all nodes and relaunch:"
echo "     for ip in ${NODES[*]}; do ssh ${SSH_USER}@\$ip 'docker rm -f vllm_slot'; done"
echo "     then re-run step 6."
echo
ok "GLM-5.2 will serve an OpenAI-compatible API on :8210 as 'glm-5.2-15pct'"
