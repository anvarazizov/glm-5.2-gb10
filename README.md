# GLM-5.2 on DGX Spark (GB10, sm_121)

Serves GLM-5.2 (744B/40B MoE, `GlmMoeDsa`) on a 4-node GB10 cluster at 256k
context with MTP speculative decode. Getting it running on sm_121 meant porting
the sparse-MLA attention off the Hopper-only `_flashmla_C` path and fixing several
sm_121-specific bugs (see `docs/retrospective.md`).

The 15% expert prune is data-free and coherence-checked, **not** benchmarked.
Treat quality as unverified.

## Requirements

4× GB10 / DGX Spark (sm_121, aarch64), a node-to-node RoCE fabric, and ~400 GB of
weights reachable from every node. Not portable to single-GPU, x86, or datacenter
Blackwell (sm_100).

## Run

Edit the CONFIG block in `bootstrap.sh` (node IPs, weights location, HF repo ids),
then run it from the head node. It verifies the cluster, builds the pinned vLLM
image, mounts the Triton kernels, installs NCCL 2.30.4, fetches the weights, and
launches. Serves an OpenAI-compatible API on `:8210` as `glm-5.2-15pct`.

The image build is not self-contained — it requires two `spark-vllm-docker` mods
that are not vendored here. See **Image build** below before running `bootstrap.sh`.

The serving recipe (`recipes/glm52-awq-15pct-prod.yaml`) also carries RoCE fabric
values (HCA + interface names, node IPs) hardcoded to my cluster — set those for
yours. The lines are marked `EDIT`.

## Image build — required vLLM mods (not vendored)

The `kernels/` here are the *implementations*. They do not wire themselves into
vLLM: two patch steps that live in my
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) fork are
required and are **not** vendored in this repo. Build the image at the pinned
ref, then apply both mods (bake them with `RUN bash mods/<name>/run.sh`, as
`Dockerfile.glm52-consolidated` does, or pass `--apply-mod mods/<name>` to
`launch-cluster.sh`):

- **`mods/glm52-sm12x-sparse`** — copies the `kernels/` files into the vLLM tree
  and patches `vllm/utils/deep_gemm.py` + `sparse_attn_indexer.py` in place. On
  capability family 120 it short-circuits `fp8_fp4_mqa_logits` /
  `fp8_fp4_paged_mqa_logits` / `tf32_hc_prenorm_gemm` to the `sm12x_*` fallbacks
  **before** the DeepGEMM `_missing()` gate, and rewrites the `SparseAttnIndexer`
  constructor gate so sm_121 never requires `has_deep_gemm()`. There is **no
  `deep_gemm` package shim** — the activation is this in-place wrapper patch.
- **`mods/glm52-b12x-sparse`** — `pip install --no-deps b12x==0.23.0` plus a
  `fused_indexer` score-mode patch. This provides the `b12x` package that the
  sparse-MLA **decode** path (`b12x_sparse_helpers.py`) calls first.

**cudagraph note (important):** my perf numbers use `cudagraph_mode: FULL` *with
b12x installed*. The b12x decode kernel is cudagraph-capture-safe. If b12x is
absent, `_fp8_flash_mla_kernel` silently falls back to the Triton
`flash_mla_with_kvcache` decode kernel, which does unconditional
`torch.full(..., device=...)` allocations that are illegal under graph capture —
so FULL capture crashes with `cudaErrorStreamCaptureInvalidated`. Without b12x,
run `cudagraph_mode: PIECEWISE`. (The `b12x` import warning is `warning_once`, so
it is emitted once during prefill warmup and then suppressed — decode falls back
silently.)

`build-glm52-awq.sh` builds the base image at the pinned ref;
`Dockerfile.glm52-consolidated` layers `glm52-sm12x-sparse` on top. Apply
`glm52-b12x-sparse` the same way for the `-b12x` image the recipe expects.

## Weights

- AWQ-INT4, 15%-pruned: https://huggingface.co/CosmicRaisins/GLM-5.2-AWQ-INT4-15pct
- MTP draft: https://huggingface.co/CosmicRaisins/GLM-5.2-MTP-INT4-aligned

`bootstrap.sh` pulls both.

## Contents

- `kernels/` — portable Triton sparse-MLA (vLLM/jasl, Apache-2.0, modified — `CHANGES.md`)
- `prune/awq_surgery.py` — the data-free 15% expert prune
- `mtp/` — separate-draft MTP reconstruction
- `recipes/` — the serving recipe
- `model-card/` — HuggingFace card for the pruned weights
- `docs/retrospective.md` — every fix, with attribution

## Performance

Measured on my 4× GB10 setup (TP=4, MTP k=3, llama-benchy generic corpus):

| Depth | Decode (tg) | Prefill (pp) |
|---|---|---|
| 0   | 20.2 t/s | 535 t/s |
| 8K  | 21.9 t/s | 517 t/s |
| 32K | 21.2 t/s | 476 t/s |

Numbers will vary with hardware and workload. In my tests decode is
memory-bandwidth-bound and prefill is bound by the sparse-MLA / indexer kernels
rather than the MoE GEMM (an NVFP4 MoE swap changed prefill by nothing) — but I
haven't stress-tested that conclusion broadly.

**MTP draft depth (k):** k=3 benchmarked best for me on a synthetic corpus, but
Z.ai recommends k=5, and I haven't compared 3 vs 5 in real-world usage yet. The
recipe ships k=3 — treat it as a starting point, not a settled answer.

## License

Apache-2.0 (this repo). Serves MIT weights: GLM-5.2 (Z.ai) → AWQ (cyankiwi) →
pruned here. See `NOTICE` and `ATTRIBUTION.md`.
