# GLM-5.2 AWQ-INT4 15% Expert Prune — logic & how we ran it

Date: 2026-06-23
Script: `prune/awq_surgery.py` from https://github.com/CosmicRaisins/glm-5.2-gb10
Purpose: turn the 256-expert `cyankiwi/GLM-5.2-AWQ-INT4` (which we already had at
`/opt/models/glm52-awq-int4`, 411 GB) into the **218-expert** build the MTP recipe serves —
**without downloading the 378 GB published prune**, and without any GPU/torch/calibration data.

## Why prune locally instead of downloading
The published `CosmicRaisins/GLM-5.2-AWQ-INT4-15pct` (378 GB) is **deterministically
reproducible** from the cyankiwi base + ratio 0.15. We already had the base, so generating it
locally was ~20 min of disk I/O vs hours of download — same bytes. (The MTP draft
`glm52-mtp-int4-aligned`, 4.8 GB, is separate and *was* downloaded; it's aligned to this prune.)

## The method (data-free, weight-only)
For each of the 75 MoE layers (L3..L77, 256 experts each):

1. Read the router's learned **`mlp.gate.e_score_correction_bias`** `[256]` (F32). This bias is
   what the router adds to each expert's score to balance routing — **a high positive bias means
   the router had to artificially *boost* that expert to get it used, i.e. it's the least
   intrinsically attractive.**
2. Sort experts by bias ascending; **keep the lowest-bias `n_keep`, drop the highest-bias**.
   `n_keep = 256 − round(256 × 0.15) = 256 − 38 = 218`.
3. Re-index survivors `0..217`.

The router's own learned preference is used as a **free saliency signal** — no forward pass, no
calibration data. (This is explicitly **not** REAP, which needs calibration data, can't handle
`glm_moe_dsa`, and emits BF16 — see the recipe's `docs/retrospective.md`.)

## What it touches (pure byte-level safetensors surgery — no dequant, no torch)
- **Routed experts**: each expert = 12 tensors (AWQ qweight/qzeros/scales for gate/up/down).
  Surviving experts are renamed `experts.{old}` → `experts.{new}`; dropped experts are skipped.
- **Router `gate.weight` `[256,H]` (BF16) + `gate.e_score_correction_bias` `[256]` (F32)**:
  row-sliced to the kept 218 rows (byte-exact row copy, verified by the `step1` hash harness).
- **Untouched (copied byte-for-byte)**: shared experts, dense layers, all attention, norms,
  embeddings, lm_head.
- **`config.json`**: `num_experts` and `n_routed_experts` → 218 (must stay uniform across layers).
- Regenerates `model.safetensors.index.json`; copies tokenizer + all other non-weight files.

Output: `~/.cache/huggingface/hub/glm52-awq-15pct` (~378 GB, 218 experts/layer).

## How we ran it (head node)
```bash
# fast correctness harness first (no write) — verifies the row-slice/reindex is byte-exact
python3 ~/awq_surgery.py step1  /opt/models/glm52-awq-int4
# → STEP1: PASS  (L3/L40/L77 PERMUTE + DROP gate/bias all OK)

# the build (~20 min I/O, detached)
python3 ~/awq_surgery.py build  /opt/models/glm52-awq-int4 \
        ~/.cache/huggingface/hub/glm52-awq-15pct  0.15

# structural validation of the output vs source
python3 ~/awq_surgery.py validate ~/.cache/huggingface/hub/glm52-awq-15pct \
        /opt/models/glm52-awq-int4
# → checks: every index name maps to a real shard, per-shard offsets contiguous, file sizes
#   match headers, gate rows == 218, expert tensor count == 218×12 per layer.
```

## Caveats
- **Quality is coherence-checked, not benchmarked** (author's words). Validate output quality
  before trusting it in production.
- Determinism depends on the source being exactly `cyankiwi/GLM-5.2-AWQ-INT4`. Ours came from
  that repo (June download → `/opt/models/glm52-awq-int4`), so the local prune == the published
  one, and the MTP draft stays aligned.
