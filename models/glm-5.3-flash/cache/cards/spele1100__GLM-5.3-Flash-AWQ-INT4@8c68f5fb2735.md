---
license: apache-2.0
base_model:
- zai-org/GLM-5.3-Flash
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- quantized
- awq
- int4
- long-context
language:
- en
- zh
---
# GLM-5.3-Flash-AWQ-INT4 — Model Card

> ⏰ 2026-08-30 · 512K production bring-up guide (verified live)
> Previous: 2026-08-29 smoke test PASS / quant notes (merged below)


> ⚠️ **IMPORTANT DISCLAIMERS (read before use)**
> 1. **Max tested context = 512K (524288).** Do not exceed it — anything beyond 512K is untested and known to have issues (1M boots but retrieval is unreliable; treat >512K as broken until proven otherwise).
> 2. **This fork is unofficial and out-of-support.** It is a community patch on top of vLLM 0.28.1rc1.dev, NOT part of official vLLM — stock vLLM does not support this architecture properly, and the fork is provided as-is with no maintenance or support commitment. Expect instability; validate every build yourself.

## Model
- Base: GLM-5.3-Flash (glm5_next arch, 45 layers, hc_mult=4, 288 experts, inter=2048, h=4096)
- Quantization: AWQ INT4

## Required vLLM build
- Requires a **custom vLLM fork (sm89 long-context fixes)** — stock vLLM has a broken mHC fused-norm kernel on this arch (all-45-layer poisoning) and long-context CUDA-graph crashes.
- **Version matters — verify your build includes both the mHC kernel fix and the long-context graph fix before debugging anything else.**
- Env reference: Python 3.10, transformers 5.16.1, flashinfer + sm89-compatible indexer.

## Bring-up (512K production)

```bash
# 1. Verify your vLLM build includes the sm89 long-context fixes

# 2. Launch under tmux (NEVER bare — a dying shell takes the server with it)
tmux new -s vllm
<your serve script>
# detach: Ctrl-b d ; reattach: tmux attach

# 3. Health checks (loading ~46 safetensors shards takes several minutes)
curl -s http://localhost:<port>/v1/models | head
# smoke:
curl -s http://localhost:<port>/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm53-flash-awq","messages":[{"role":"user","content":"用一句話介紹自己"}],"max_tokens":2048}'
```

**Reference serve script:**

```bash
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export PYTHONPATH=<path to your vLLM fork>:$PYTHONPATH
exec vllm serve <model_path> \
  --served-model-name glm53-flash-awq \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 --port <port> \
  --max-model-len 524288 \
  --no-enable-prefix-caching \
  --hf-overrides '{"text_config": {"index_topk": null, "index_n_heads": null, "index_head_dim": null, "index_kpool": null}}' \
  --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}" \
  --gpu-memory-utilization 0.94 \
  --max-num-batched-tokens 4096 \
  --reasoning-parser glm47 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice
```

Key config notes:
- **Full-MLA mode**: the hf-overrides null out the sparse indexer (index_topk etc.) — sparse retrieval is OFF in production. An sm89 sparse-indexer port can pass correctness in chat-mode probes, but very-long-context (1M) sparse is blocked by an upstream persistent_topk smem bug.
- **Graph mode**: runs WITHOUT `--enforce-eager` — `FULL_DECODE_ONLY` cudagraph. (Enforce-eager was needed on older builds due to long-context CUDA-graph crashes; current fixes remove that requirement.)
- Prefix caching disabled; KV bf16 (fp8 unsupported, see below); TP8; ~34 GiB peak per GPU during load.

### Restart quirk (known gotcha)
If the server just died, **don't relaunch immediately** — wait ~1 min and clear `/dev/shm` residue (stale vLLM shm blocks cause the relaunch to fail). Then relaunch in the same tmux session.

## Verification checklist
1. `/v1/models` responds; loading ~46 shards takes minutes, not seconds.
2. Chat smoke: greeting + Traditional Chinese + simple math — coherent, no repetitive/scaled-token signature. **Use `/v1/chat/completions` or `llm.chat()`** — never raw `generate()` (see lessons below). max_tokens ≥ 2048 (reasoning eats budget).
3. Needle test at ~400–500K context: verify retrieval inside the 512K window.
4. Thinking field: expect `reasoning` (or `reasoning_content` if fronted by a proxy that normalizes it). Should appear once the model actually reasons.

## Context-length verdicts (verified 2026-08-30)
- **512K (524288): production-ready** — this is the verified config.
- **≤ ~600K: usable.**
- **1M: boots but retrieval is unreliable** — do not trust answers at 1M.
- **Nothing above 512K is supported.** 512K is the hard, tested ceiling of this release; longer contexts are known-broken territory.

## Known limits & lessons (2026-08-29/30)
- **Old vLLM builds = guaranteed garbage.** The mHC kernel fix must be present in your build — always verify before debugging.
- **Chat-model probes MUST use the chat template** (`llm.chat()` / `/v1/chat/completions`). Raw `generate()` degenerates on this model and looks like a model bug but isn't.
- **fp8 KV cache NOT supported on sm89 MLA** — vLLM init fails ("FP8 kv_data_type for MLA is only supported with the fa3 backend on SM90"). FA3-only. Keep KV bf16 (auto); do not retry.
- pe_dim=64: GLM has rope=0 — fp8_ds_mla packed layout (SM120 path) can never work; zero-pad rope trick is required (already in the fix branches).
- Chat template exposes reasoning inline in `content` on direct probes; use max_tokens ≥ 2048.
- vLLM serve hot path runs inside CUDA graph capture — python hooks won't fire; use offline eager mode for probing.
- Always launch under tmux; see restart quirk above.

## Validation history
- 2026-08-29 smoke: greeting / math / Traditional Chinese — coherent; HF reference parity layers 0–2 cos 0.9999+; effective GEMM cos 0.992, ratio 1.008 (post fold-fix).
- 2026-08-30: 512K production config verified live; context verdicts above.

## vLLM fork & build procedure (required — stock vLLM will NOT work)

The custom fork is mirrored at **https://github.com/spele1100/vllm-glm53**, branch **`sm89-longctx-fix-v2`**, verified commit **`f2935ce322`**.

> ⚠️ This is an **unofficial, experimental fork** — it is not part of official vLLM and receives **no support or maintenance guarantee**. It exists because stock vLLM does not properly support this architecture on sm89. Use at your own risk and always validate your build against the checklist below.

Patch history (5 commits on top of vLLM 0.28.1rc1.dev):
1. `cdc814dc4` — allow MLADimensions(256,0,256) in FA prefill; skip indexer weights when index_topk=null
2. `dc5fcfd3b` — enable GLM-5.3 sparse indexer + sparse MLA decode on Ada (portable MQA fallbacks, FA2 nope MLA with zero-padded rope, JIT toolchain fixes)
3. `1358bbd8c` — **mHC fused-norm correctness fix** (the all-45-layer poisoning bug)
4. `005c57e0d` — bound indexer MQA logits memory; fix long-context prefill crashes
5. `f2935ce32` — skip indexer checkpoint tensors when index_topk=null (no-indexer mode used in production)

### Build steps

```bash
git clone https://github.com/spele1100/vllm-glm53.git ~/llm/vllm-glm53
cd ~/llm/vllm-glm53
git checkout sm89-longctx-fix-v2
git rev-parse --short=10 HEAD   # must be f2935ce322 or newer

python3.10 -m venv ~/llm/venv-glm53-serve
source ~/llm/venv-glm53-serve/bin/activate
pip install -U pip wheel

# transformers pin
pip install 'transformers==5.16.1'

# FlashInfer: pip wheel 0.6.18 is broken on sm89/sm121 — build from source
git clone https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer
pip install -e /tmp/flashinfer --no-build-isolation
export FLASHINFER_DISABLE_VERSION_CHECK=1

# vLLM fork (editable, so PYTHONPATH pick-up works)
pip install -e ~/llm/vllm-glm53 --no-build-isolation
```

### Runtime env (must match the serve script)

```bash
N=$HOME/llm/venv-glm53-serve/lib/python3.10/site-packages/nvidia/cu13
export CUDA_HOME=$N
export PATH=$N/bin:$PATH
export PYTHONPATH=$HOME/llm/vllm-glm53:$PYTHONPATH
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONFAULTHANDLER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

### Build/run gotchas (learned the hard way)
- If vLLM init crashes in FlashInfer JIT, the version check masking is the usual suspect — keep `FLASHINFER_DISABLE_VERSION_CHECK=1`.
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` + `FULL_DECODE_ONLY` graph mode is the verified combo; do not mix with enforce-eager configs from older docs.
- After any branch/commit change, re-run the verification checklist above — config- and checkpoint-version mismatches are the #1 cause of gibberish output reports.
