#!/usr/bin/env bash
# launch.sh — GLM-5.2 (recon image) TP=4 across our 4-node GB10 cluster.
# Adapted from CosmicRaisins/glm-5.2-gb10 launch.sh for OUR cluster:
#  - NODES = 100G/RoCE fabric IPs (head rank 0 first)
#  - bare `ssh <ip>` (head already routes to each worker's user via keys)
#  - RoCE *f1* HCAs/ifaces (our UP rails); recon image; kernels BAKED (no mounts)
#  - WEIGHTS_DIR=/opt/glm52 (head bind-mount + workers NFS-mount of the head's hub)
#   ./launch.sh [--dry-run|--stop]
set -uo pipefail

# ===== CONFIG =====
NODES=(10.78.0.1 10.78.0.2 10.78.0.3 10.78.0.4)   # rank 0..3, head first
IMAGE="vllm-glm52-recon-ref:latest"
NAME="vllm_slot"
PORT=8210
MASTER_PORT=29501
WEIGHTS_DIR="/opt/glm52"   # contains hub/{glm52-awq-15pct,glm52-mtp-int4-aligned,nccl-2.30.4}; .tritoncache local
# ==================

say(){ printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die(){ printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
DRYRUN=0; STOP=0
for a in "$@"; do case "$a" in --dry-run)DRYRUN=1;; --stop)STOP=1;; *)die "bad arg $a";; esac; done
NNODES="${#NODES[@]}"; HEAD="${NODES[0]}"

# run a shell line on a node: head = local, workers = ssh (bare; head's keys carry the user)
runon(){ local ip="$1"; shift; if [ "$ip" = "$HEAD" ]; then bash -c "$*"; else ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$ip" "$*" </dev/null; fi; }

if [ "$STOP" = 1 ]; then
  say "stopping '$NAME' on all $NNODES nodes"
  for ip in "${NODES[@]}"; do runon "$ip" "docker rm -f $NAME 2>/dev/null" && printf '   stopped on %s\n' "$ip"; done
  exit 0
fi

ENVV=(
  -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800"
  -e "LD_PRELOAD=/cache/huggingface/hub/nccl-2.30.4/libnccl.so.2"
  -e "HF_HOME=/cache/huggingface"
  -e "TRITON_CACHE_DIR=/cache/huggingface/.tritoncache"
  -e "HF_HUB_OFFLINE=1"
  -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  -e "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256"
  -e "GLM52_BIND_HOST_TRITON=1"
  -e "GLM52_MQA_LOGITS_TRITON=1"
  -e "GLM52_PAGED_MQA_TRITON=1"
  -e "GLM52_PAGED_MQA_TOPK_CHUNK_SIZE=8192"
  -e "GLM52_B12X_MLA=1"
  -e "TORCH_CUDA_ARCH_LIST=12.1a"
  -e "NCCL_NET=IB"
  -e "NCCL_IB_DISABLE=0"
  -e "NCCL_IB_HCA=rocep1s0f1"                          # rail 1 only; 2nd rail real but NCCL multi-rail GID resolution unsolved (TODO)
  -e "NCCL_SOCKET_IFNAME=enp1s0f1np1"                  # both fabric interfaces
  -e "GLOO_SOCKET_IFNAME=enp1s0f1np1"
  -e "NCCL_IB_GID_INDEX=3"
  -e "NCCL_CROSS_NIC=1"
  -e "NCCL_CUMEM_ENABLE=0"
  -e "NCCL_IGNORE_CPU_AFFINITY=1"
  -e "NCCL_DEBUG=WARN"
)
BASE=(
  --cap-add IPC_LOCK --ulimit memlock=-1:-1
  --network host --ipc host --shm-size 10gb --gpus all
  --device /dev/infiniband:/dev/infiniband
  -v "$WEIGHTS_DIR:/cache/huggingface"
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
)
SERVE=(
  vllm serve /cache/huggingface/hub/glm52-awq-15pct
  --served-model-name glm-5.2-15pct --host 0.0.0.0 --port "$PORT"
  --trust-remote-code --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  --enable-prefix-caching
  --speculative-config '{"model":"/cache/huggingface/hub/glm52-mtp-int4-aligned","method":"mtp","num_speculative_tokens":3,"attention_backend":"FLASHMLA_SPARSE"}'
  --tensor-parallel-size 4 --pipeline-parallel-size 1
  --max-model-len 131072 --max-num-seqs 1 --max-num-batched-tokens 4096
  --gpu-memory-utilization 0.90 --kv-cache-dtype fp8_ds_mla
  --distributed-executor-backend mp --compilation-config '{"cudagraph_mode":"FULL"}'
)

docker_run_cmd(){ # rank headless
  local rank="$1" headless="$2"
  local cmd=(docker run -d --name "$NAME" "${BASE[@]}" "${ENVV[@]}"
             -e "NODE_RANK=$rank" -e "MASTER_ADDR=$HEAD"
             "$IMAGE" "${SERVE[@]}"
             --nnodes "$NNODES" --node-rank "$rank" --master-addr "$HEAD" --master-port "$MASTER_PORT")
  [ "$headless" = 1 ] && cmd+=(--headless)
  local out="" t; for t in "${cmd[@]}"; do out+=" $(printf '%q' "$t")"; done; printf '%s' "${out# }"
}

say "GLM-5.2 recon launch: $NNODES nodes, head=$HEAD:$PORT, image=$IMAGE"
[ "$DRYRUN" = 1 ] && echo "   (dry-run)"
for ((rank=1; rank<NNODES; rank++)); do
  w="${NODES[$rank]}"; run="$(docker_run_cmd "$rank" 1)"; shell="docker rm -f $NAME 2>/dev/null; $run"
  if [ "$DRYRUN" = 1 ]; then printf '\n# worker %s rank %d\n%s\n' "$w" "$rank" "$shell"
  else printf '   worker %s rank=%d\n' "$w" "$rank"; runon "$w" "$shell" || die "worker launch failed on $w"; fi
done
run="$(docker_run_cmd 0 0)"; shell="docker rm -f $NAME 2>/dev/null; $run"
if [ "$DRYRUN" = 1 ]; then printf '\n# head %s rank 0\n%s\n' "$HEAD" "$shell"; exit 0; fi
printf '   head %s rank=0\n' "$HEAD"; runon "$HEAD" "$shell" || die "head launch failed"
say "launched — poll: curl -s localhost:$PORT/v1/models ; logs: docker logs -f $NAME"
