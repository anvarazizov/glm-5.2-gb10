# CHANGES

This file states the modifications made to third-party Apache-2.0 source, as
required by Apache License 2.0 Section 4(b). The vendored kernels under
`kernels/` originate from the vLLM project (sm12x sparse-MLA / DeepGEMM-fallback
work carried by jasl). Their original `SPDX-License-Identifier: Apache-2.0` and
`Copyright contributors to the vLLM project` headers are retained unchanged.

## Modifications to `kernels/` (mine)

These changes were made by this project to get GLM-5.2's `glm_moe_dsa`
(DeepSeek-V3.2-style sparse-MLA) attention path running on **NVIDIA GB10 /
DGX Spark — sm_121 aarch64**, which the upstream Hopper-only `_flashmla_C`
extension does not support.

1. **V3.2 adaptation + monkeypatch** (`patch_flashmla_ops.py`,
   `flashmla_sparse.py`): adapted jasl's V4-package Triton wrappers into drop-in
   `flash_mla_sparse_fwd` / `flash_mla_with_kvcache` replacements matching the
   V3.2 `FlashMLASparseImpl` call signatures, and rebind them on sm12x so
   `GlmMoeDsa` routes to the portable Triton path instead of native `_flashmla_C`.

2. **int32 → int64 overflow fix** (`sm12x_sparse_mla_attn.py`, in
   `_gather_dequant_fp8ds_kernel`): under TP=4 with 24 heads/rank and T=2048
   gathered KV (~4.83 GB), `t * stride` overflows int32 at t≈1821, corrupting
   long-context prefill. Promoted the offending program-id/stride arithmetic to
   `tl.int64`. Found via a faithful single-GPU repro of the per-rank fp8_ds_mla
   paged path.

3. **Index upper-bound guards** (`sparse_mla_kernels.py`): added
   `(kv_index >= 0) & (kv_index < num_kv_rows)` bounds checks in the scalar,
   multihead, and d512-split sparse-MLA kernels to stop out-of-range gathers on
   prefill chunks beyond 2048 tokens.

4. **Fused gather-dequant-attn kernel** (`sm12x_sparse_mla_attn.py`, new
   `_fused_gather_dequant_attn_kernel` / `_fused_gather_dequant_attend`): a fused
   prefill path that splits the head dim 576 → NoPE 512 + RoPE 64 (avoiding a
   1024 pad), uses `BLOCK_N=32` with tensor-core `tl.dot` and online softmax, and
   never materializes the `[T, K, 576]` gathered tensor. Gated by
   `VLLM_SPARSE_MLA_FUSED` (set `=0` to fall back to the unfused path). This is
   what took cold-prefill from ~336 to ~508 tok/s and flattened the
   depth curve.

## Original, non-derivative files (not modified third-party code)

The following are **original works of this project**, not derived from the
vendored Apache-2.0 kernels, and are licensed under this repository's Apache-2.0
LICENSE with my own copyright:

- `prune/awq_surgery.py` — the data-free routed-expert prune.
- `mtp/*` — the separate-draft MTP reconstruction + verification scripts.
- `recipes/*` — the production serving recipe.
- `bootstrap.sh`, docs.
