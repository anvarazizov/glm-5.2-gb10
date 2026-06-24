#!/usr/bin/env bash
#
# build-recon-image.sh — reconstruct the CosmicRaisins GLM-5.2 sparse-MLA/MTP
# vLLM image from PUBLIC parts, layered on our existing vllm-glm52-tf5 base.
#
# The author's two image mods (glm52-sm12x-sparse, glm52-b12x-sparse) are NOT
# public. This script re-derives them from the public kernels + retrospective:
#   1. bake the 8 public Triton kernels into the vLLM tree
#   2. create vllm/v1/attention/ops/deepseek_v4_ops/ (missing in our base)
#   3. patch vllm/utils/deep_gemm.py — route the 3 DeepGEMM fns to sm12x fallbacks
#      on capability family 120 (before the _missing() gate)
#   4. patch sparse_attn_indexer.py — drop the has_deep_gemm() requirement on sm12x
#   5. trigger patch_flashmla_ops.apply() at startup (it auto-applies on import)
#   6. pip install --no-deps b12x==0.23.0 (public; enables FULL cudagraph decode)
#
# Run from the head. Produces vllm-glm52-recon:latest. Additive — touches nothing
# that is running.
set -euo pipefail
BASE_IMAGE="${BASE_IMAGE:-vllm-glm52-tf5:latest}"
OUT_IMAGE="${OUT_IMAGE:-vllm-glm52-recon:latest}"
KREPO="https://raw.githubusercontent.com/CosmicRaisins/glm-5.2-gb10/master/kernels"
WORK="$HOME/glm52-recon-build"
SP="/usr/local/lib/python3.12/dist-packages"
MLA="vllm/v1/attention/backends/mla"
OPS="vllm/v1/attention/ops/deepseek_v4_ops"

echo "== fetch public kernels =="
rm -rf "$WORK"; mkdir -p "$WORK/kernels"; cd "$WORK"
for f in sparse_mla_kernels sparse_mla_env sm12x_sparse_mla_attn patch_flashmla_ops \
         flashmla_sparse sm12x_deep_gemm_fallbacks sm12x_mqa b12x_sparse_helpers; do
  curl -fsSL "$KREPO/$f.py" -o "kernels/$f.py"
done
echo "   $(ls kernels | wc -l) kernel files"

# ---- deep_gemm.py patcher (idempotent) ----
cat > patch_deep_gemm.py <<'PY'
import re,sys
p=f"{sys.argv[1]}/vllm/utils/deep_gemm.py"; s=open(p).read()
if "GLM52_SM12X_DEEPGEMM_PATCH" in s: print("  deep_gemm already patched"); sys.exit(0)
JOBS=[
 ("fp8_fp4_mqa_logits","_fp8_mqa_logits_sm12x",
  "_f(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits)"),
 ("fp8_fp4_paged_mqa_logits","_fp8_paged_mqa_logits_sm12x",
  "_f(q, kv_cache, weights, context_lens, block_tables, max_model_len)"),
 ("tf32_hc_prenorm_gemm","_tf32_hc_prenorm_gemm_sm12x",
  "_f(x, fn, out, sqrsum, num_split)"),
]
for fn,imp,call in JOBS:
    m=re.search(rf"\ndef {fn}\(",s); assert m,fn
    anchor=s.index("\n    _lazy_init()",m.end())
    guard=("\n    from vllm.platforms import current_platform as _cp  # GLM52_SM12X_DEEPGEMM_PATCH"
           "\n    if _cp.is_device_capability_family(120):"
           f"\n        from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_deep_gemm_fallbacks import {imp} as _f"
           f"\n        return {call}")
    s=s[:anchor]+guard+s[anchor:]
open(p,"w").write(s); print("  deep_gemm patched (3 fns routed to sm12x)")
PY

# ---- sparse_attn_indexer.py patcher (idempotent) ----
cat > patch_indexer.py <<'PY'
import sys
p=f"{sys.argv[1]}/vllm/model_executor/layers/sparse_attn_indexer.py"; s=open(p).read()
old="if current_platform.is_cuda() and not has_deep_gemm():"
new="if current_platform.is_cuda() and not has_deep_gemm() and not current_platform.is_device_capability_family(120):  # GLM52"
if new in s: print("  indexer already patched")
elif old in s: open(p,"w").write(s.replace(old,new,1)); print("  indexer gate patched (skip DeepGEMM req on sm12x)")
else: print("  WARN: indexer gate anchor not found"); sys.exit(2)
PY

# ---- startup trigger: import patch_flashmla_ops when the mla backend package loads ----
cat > trigger.py <<'PY'
import sys
p=f"{sys.argv[1]}/vllm/v1/attention/backends/mla/__init__.py"
line="\ntry:\n    from . import patch_flashmla_ops  # noqa: F401  GLM52 auto-apply Triton sparse-MLA\nexcept Exception:\n    pass\n"
s=open(p).read() if __import__("os").path.exists(p) else ""
if "patch_flashmla_ops" in s: print("  trigger already present")
else: open(p,"a").write(line); print("  trigger appended to mla/__init__.py")
PY

echo "== write Dockerfile =="
cat > Dockerfile <<DOCKER
FROM ${BASE_IMAGE}
# 1. deepseek_v4_ops package (missing in base) + bake kernels
RUN mkdir -p ${SP}/${OPS} && touch ${SP}/${OPS}/__init__.py
COPY kernels/sm12x_deep_gemm_fallbacks.py ${SP}/${OPS}/sm12x_deep_gemm_fallbacks.py
COPY kernels/sm12x_mqa.py                 ${SP}/${OPS}/sm12x_mqa.py
COPY kernels/b12x_sparse_helpers.py       ${SP}/${OPS}/b12x_sparse_helpers.py
COPY kernels/sparse_mla_kernels.py        ${SP}/${MLA}/sparse_mla_kernels.py
COPY kernels/sparse_mla_env.py            ${SP}/${MLA}/sparse_mla_env.py
COPY kernels/sm12x_sparse_mla_attn.py     ${SP}/${MLA}/sm12x_sparse_mla_attn.py
COPY kernels/patch_flashmla_ops.py        ${SP}/${MLA}/patch_flashmla_ops.py
COPY kernels/flashmla_sparse.py           ${SP}/${MLA}/flashmla_sparse.py
# 2. source patches + startup trigger
COPY patch_deep_gemm.py patch_indexer.py trigger.py /tmp/
RUN python3 /tmp/patch_deep_gemm.py ${SP} && python3 /tmp/patch_indexer.py ${SP} && python3 /tmp/trigger.py ${SP}
# 3. b12x decode kernel (public PyPI) — enables FULL cudagraph; PIECEWISE works without it
RUN pip install --no-deps b12x==0.23.0 || echo "b12x install failed (run cudagraph_mode PIECEWISE)"
# 4. byte-compile sanity (import will be validated separately on a GPU)
RUN python3 -c "import py_compile,glob; [py_compile.compile(f,doraise=True) for f in glob.glob('${SP}/${MLA}/*.py')+glob.glob('${SP}/${OPS}/*.py')]" && echo "kernels compile OK"
DOCKER

echo "== docker build -> ${OUT_IMAGE} =="
docker build -t "${OUT_IMAGE}" "$WORK"
echo "== done: ${OUT_IMAGE} =="
