# GLM-5.2 on 4× GB10 — patch & fix retrospective

The sequence of fixes to serve GLM-5.2 (Z.ai — 744B/40B MoE, `GlmMoeDsa` =
DeepSeek-V3.2 sparse-MLA + lightning indexer + MTP) on a 4-node DGX Spark **GB10
(sm_121 aarch64)** cluster. Most of the foundation is open-source community work;
this repo adds the integration, the prefill bug fixes, the MTP reconstruction, and
the data-free prune.

## The chain of blockers → fixes

| # | Blocker | Fix | Whose work |
|---|---|---|---|
| 1 | Model class to parse the config | `GlmMoeDsaForCausalLM` / `glm_moe_dsa` | **vLLM** (v0.23+); arch by **Z.ai** |
| 2 | 4-bit base weights | `cyankiwi/GLM-5.2-AWQ-INT4` (compressed-tensors W4A16) | **cyankiwi** |
| 3 | DSA sparse-indexer hard-requires **DeepGEMM**, which rejects sm_120/121 (vLLM #41063) | `sm12x_deep_gemm_fallbacks` torch fallback, forward-ported onto v0.23 | kernels **jasl**; port = me |
| 4 | **The big one** — GLM routes to `flashmla_sparse` → native `_flashmla_C`, Hopper-only (no sm_121 build) | Ported **jasl's V4 Triton sparse-MLA kernels** (~3.5k lines) into drop-in V3.2 replacements + monkeypatched on sm12x | kernels **jasl**; V3.2 adaptation = me |
| 5 | NCCL `shm_broadcast` deadlock under load (2.29.7) | LD_PRELOAD **NCCL 2.30.4** aarch64 wheel | runbook **hazyumps**; wheel **NVIDIA** |
| 6 | Gloo `connectFullMesh` failure at dist-init | `GLOO_SOCKET_IFNAME` (rail-1 iface) | me (from the runbook) |
| 7 | NCCL silently over TCP (~12 vs 30+ tok/s) | `--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock` passthrough | **aidendle94 / local-inference-lab** raw-entrypoint pattern |
| 8 | Unified-memory OOM at KV alloc | Explicit `--kv-cache-memory-bytes` / `gmu` on the shared pool | me |
| 9 | **Prefill crash >2048 ctx** — incl. an **int32 overflow** in `_gather_dequant_fp8ds_kernel` (T=2048 KV ≈4.8 GB, `t*stride` > 2³¹) | int64 promotion + index bounds; found via a faithful local repro | **me** (original) |
| 10 | Long-ctx prefill slow + declining (~336→216) | **Fused gather-dequant-attn kernel** (split 576→512+64, BLOCK_N=32, tensor-core `tl.dot`, online softmax) — flat ~508 | **me** (original) |
| 11 | No MTP — cyankiwi dropped layer-78 | **Separate-draft MTP reconstruction** (`glm52-mtp-int4-aligned`), matched to target + verified vs the real loader; dodges vLLM #35041/#38494 | reconstruction = me; weights derive from **Z.ai** MTP / **0xSero** layer-78 |
| 12 | Full model doesn't fit 4 nodes with KV headroom | **15% data-free expert prune** — drop the highest `e_score_correction_bias` experts (least router-favored), 256→218 uniform, weight-only, no calibration | **me** (`awq_surgery.py`) |
| 13 | MTP `k` tuning | swept k=2..6 → **k=3** TG peak on generic corpus (accept saturates ~1.9/step) | me |
| 14 | Thinking leaked inline; no tool calls | `--reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice` | parsers **vLLM/Z.ai**; wiring = me |
| 15 | 256k vs boot free-mem guard knife-edge | `gmu 0.93` (0.935 trips the guard; 0.93 → 268k-token KV) | me |
| 16 | Launch silently fell to the Ray path (no triton mounts) | use the **short** recipe name → `launch-raw-entrypoint-nfs.py`; reap orphan workers between launches | me |
| 17 | "Is NVFP4 a faster door?" | Probe: `flashinfer_cutlass` FP4 MoE **runs** on sm_121 (where `flashinfer_b12x`/PR#40082 crashes) but gives **no prefill win** — prefill is attention/indexer-bound | b12x/oracle **vLLM**; conclusion = me |

## On the prune (it is NOT REAP)

REAP (CerebrasResearch) was investigated and **rejected**: its saliency needs a
calibration forward pass (data), the public repo doesn't handle `glm_moe_dsa`,
compressed-tensors INT4 can't be `save_pretrained`'d after load, and REAP emits
BF16 (the ~1.5 TB full model doesn't fit anything I have). So I did something
simpler and **fully data-free**: read each layer's learned
`e_score_correction_bias` and drop the highest-bias (least router-favored)
experts. No data, no forward passes. Using the router's own learned preferences
as a free saliency signal turned out to be both correct *and* elegant.

## What I learned about the bottleneck

Prefill on GLM-5.2/GB10 is **attention/indexer-bound**, not MoE-bound. I proved
it by swapping the MoE GEMM from INT4-Marlin to native-FP4 `flashinfer_cutlass`
(a kernel 2-3× faster in principle) and seeing **zero** prefill change (543 vs
515 tok/s, within noise). So the lever for faster prefill is the **Triton
sparse-MLA / lightning-indexer kernel**, not the quant format, not chunk size,
not cudagraph. Decode is bandwidth-bound on the 40B active params (~273 GB/s
LPDDR5X); MTP is the decode lever, and a stronger drafter (EAGLE-3 / tree) is the
path to more.

## Credits

See `../NOTICE` and `../ATTRIBUTION.md`.
