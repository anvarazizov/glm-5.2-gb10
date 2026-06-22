# mtp/ — separate-draft MTP reconstruction

`cyankiwi/GLM-5.2-AWQ-INT4` drops GLM-5.2's native MTP layer (layer 78), so
in-model speculative decode isn't available. Rather than graft MTP back into the
target (which hits vLLM #35041 / #38494), I build a **separate INT4 MTP draft**
(`glm52-mtp-int4-aligned`) that vLLM loads via `--speculative-config`. The draft
is aligned to the pruned target on quantization, expert count, per-module quant,
and `DeepSeekMTP` parameter naming, then verified against the real loader.

## Pipeline (run in order)

1. **`glm52_dequant_mtp.py`** — dequantize the MTP source (GLM-5.2 native layer-78
   / 0xSero NVFP4 layer-78) to a working precision.
2. **`glm52_quant_mtp_int4.py`** — re-quantize to INT4 compressed-tensors matching
   the AWQ target's scheme.
3. **`_build_aligned.py`** — re-key / re-shape every tensor to the `DeepSeekMTP`
   layout vLLM's MTP loader expects, prune the draft's experts to match the
   target's count, and share embed/lm_head/final-norm with the target.
4. **`_verify_mtp.py`** — load the aligned draft through vLLM's real
   `DeepSeekMTP.load_weights` and assert every module maps cleanly (no
   missing/extra/shape-mismatched keys).

The output `glm52-mtp-int4-aligned` is what `recipes/glm52-awq-15pct-prod.yaml`
points `--speculative-config` at. Tune `num_speculative_tokens` (k) for your
traffic — k=3 measured best on a generic corpus; structured/code traffic often
sustains higher k.

## Attribution

The MTP *weights* derive from Z.ai's native GLM-5.2 MTP (MIT) and/or 0xSero's
NVFP4 layer-78 — if you redistribute the resulting draft, attribute that lineage
and confirm 0xSero's license for any bytes sourced there. The reconstruction
**logic** in these scripts is mine (Apache-2.0).
