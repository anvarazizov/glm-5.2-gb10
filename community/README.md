# Porting CosmicRaisins/glm-5.2-gb10 to another 4× GB10 cluster — working config

We got the [CosmicRaisins GLM-5.2 stack](https://github.com/CosmicRaisins/glm-5.2-gb10)
(vLLM TP=4, MTP speculative decode, sparse-MLA Triton kernels, AWQ-INT4 15%-pruned) serving on a
**different** 4× GB10 / DGX Spark cluster. This documents the gaps we hit and how we closed them —
the goal is to make the recipe reproducible by anyone, not just on the author's exact machines.

## TL;DR of what's missing from the public repo

1. **The image build is not reproducible from public sources.** The README says the two required
   vLLM mods (`glm52-sm12x-sparse`, `glm52-b12x-sparse`) live in `eugr/spark-vllm-docker` — but they
   are **not** in that fork (checked `main` + `develop`), and `Dockerfile.glm52-consolidated` /
   `build-glm52-awq.sh` aren't published either. Only the **kernels** are public.
   → We reconstructed the mods from the public kernels: **[`build-recon-image.sh`](build-recon-image.sh)**.
2. **The base vLLM ref matters — a lot.** Building on a *newer* vLLM than the author's pinned
   `ab666069935c1f23e8ef56038b4659ac9e8f19f8` makes the real AWQ weights crash at
   `process_weights_after_loading` (`_k_scale.fill_` → `CUDA error: invalid argument`, async).
   Dummy weights load fine, so it's specific to real-weight processing on the wrong ref.
   **Build vLLM at exactly `ab66606 --tf5`** (`eugr/spark-vllm-docker build-and-copy.sh`).

## Reconstructing the image (the missing mods, from public kernels)

`build-recon-image.sh` layers on a `spark-vllm-docker --tf5` base built at the author's ref and:
- bakes the 8 public `kernels/*.py` into the vLLM tree + creates the missing
  `vllm/v1/attention/ops/deepseek_v4_ops/` package;
- patches `vllm/utils/deep_gemm.py` — on `is_device_capability_family(120)` routes
  `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` / `tf32_hc_prenorm_gemm` to the `_*_sm12x`
  fallbacks **before** the `_missing()` gate (signatures match 1:1; the paged one drops
  `schedule_metadata`/`clean_logits` — Triton self-schedules);
- patches `sparse_attn_indexer.py` — adds `and not is_device_capability_family(120)` to the
  `has_deep_gemm()` constructor gate;
- appends `from . import patch_flashmla_ops` to `mla/__init__.py` (that kernel **auto-applies on
  import**, rebinding flashmla→Triton + reporting sparse supported on sm12x);
- `pip install --no-deps b12x==0.23.0` (public on PyPI; enables `cudagraph_mode: FULL`).

Validate the wiring before a full launch (`docker run --gpus all … python3 -c` import check):
`flash_mla_sparse_fwd → flash_mla_sparse_fwd_triton`, `is_flashmla_sparse_supported → (True,None)`,
deep_gemm sm12x route present, fallbacks import, b12x present.

## Weights without the 378 GB download

The published 15%-prune is **deterministically reproducible** from `cyankiwi/GLM-5.2-AWQ-INT4` via
`prune/awq_surgery.py build <cyankiwi> <out> 0.15` (pure byte-level safetensors surgery, no GPU,
~20 min). If you already have the cyankiwi base, this beats the 378 GB download. The MTP draft
(`glm52-mtp-int4-aligned`, 5 GB) stays aligned to it. (The prune ratio is a free knob — `0.10`
gives a higher-quality 230-expert build, ~388 GB, may need a lower `max-model-len`.)

## Cluster-port gotchas (things hardcoded to the author's setup)

- **RoCE rails / GID index** — set `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` to YOUR active RoCEv2-IPv4
  HCAs (find with `show_gids` — pick the index whose type is `RoCE v2` and has your fabric IP).
  Multi-rail RoCE wants both rails on a real RDMA mesh **with peers' GIDs resolvable**; a 2nd rail
  that pings (ICMP) can still fail NCCL with `ibv_modify_qp ... remote GID ::` if NCCL can't
  correlate the 2nd subnet across peers. Single-rail works; dual-rail needs that GID resolution.
- **`gpu-memory-utilization`** — the recipe's `0.93` (→ 256k KV) is a knife-edge. On nodes with
  less free memory (ours run a monitoring stack + NFS server), `0.93` trips the boot free-mem guard
  (`Free memory X < desired`). Drop to `0.90` + a smaller `--max-model-len` (we used 131072).
- **No shared FS?** The weights need to be present on every node. We NFS-export the head's hub and
  mount it `ro,hard` on the workers (the recipe supports a shared mount — `WEIGHTS_DIR`). Loading
  378 GB over NFS is slow (one node's NVMe serves all readers); local copies are faster if disk allows.
- **Heterogeneous SSH users** — the stock `launch.sh` assumes one `SSH_USER`; if your nodes differ,
  reach workers via the head's keys (`ssh <ip>` with per-node config) and use an absolute
  `WEIGHTS_DIR` (we used `/opt/glm52`, head bind-mount + workers NFS-mount).

## Result

Serves on `:8210` as `glm-5.2-15pct`. Quality good (coherent code). Single-rail RoCE: **~9.4 t/s
decode** warm, MTP healthy (mean acceptance ~2.8/4, ~60% draft accept). The author's ~20 t/s uses
dual-rail RoCE — the inter-node allreduce bandwidth is the decode bottleneck on these boxes, so the
2nd rail is the lever to ~2×.
