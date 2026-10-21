---
license: other
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - nvfp4
  - modelopt
  - speculative-decoding
  - pplx-computer
---

# pplx-computer-qwen-3-8-27b-dspark2-20260819

Qwen3.8-27B (arch class `Qwen3_5ForConditionalGeneration`) quantized to NVFP4
(ModelOpt mixed-precision mirroring `RadixArk/Qwen3.8-27B-NVFP4`'s recipe,
FP8 KV cache) with one deliberate change: an **unquantized bf16 `lm_head`**
(DFlash2 refuses a quantized one — every public NVFP4 quant packs it to FP4),
plus baked-in **dspark2 MTP draft tensors** (`mtp.*` in `model-mtp.safetensors`,
referenced by `model.safetensors.index.json`).

Snapshot of the checkpoint serving on the pplx-computer DGX Spark fleet as of
2026-08-19. Verified perf on one GB10: **~30–35 tok/s single-stream decode,
84 tok/s aggregate @ 4 streams** (2.9× over no-spec NVFP4).

## How to run (DGX Spark / GB10)

Everything needed is in `serving/`. Stock vLLM can **not** serve this with
DFlash2 — it needs the pure-Python vLLM PR #52816 overlay plus a one-line
lm_head fix (see notes below), packaged here as two Dockerfiles.

```bash
# 0) Auth (private repo) — needs a perplexity-ai read token
hf auth login

# 1) Download this checkpoint + the DFlash2 draft (draft/) + the serving kit
hf download perplexity-ai/pplx-computer-qwen-3-8-27b-dspark2-20260819 \
  --local-dir ~/models/qwen38-dspark2-20260819

# 2) Build the patched vLLM image (pure-Python patches, no kernel rebuild)
cd ~/models/qwen38-dspark2-20260819/serving
docker build -f Dockerfile          -t vllm-dflash2:local     .
docker build -f Dockerfile.lmheadfix -t vllm-dflash2:lmheadfix .

# 3) Serve
MODEL_DIR=~/models/qwen38-dspark2-20260819 docker compose up -d

# 4) Verify (first start compiles graphs — allow up to ~10 min)
curl -sf localhost:8000/health && \
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "pplx-computer-qwen-3-8-27b-dspark2-20260819",
  "messages": [{"role": "user", "content": "Say hi in five words."}],
  "max_tokens": 64
}' | python3 -m json.tool
```

Tunables (env vars on `docker compose up`): `GPU_UTIL` (default 0.75 — raise
to ~0.9 on a dedicated box), `MAX_LEN` (262144), `MAX_SEQS` (4), `SERVED_NAME`
(the OpenAI-API model id; defaults to this repo's name — override if your
client expects a different id).

## Serve flags

```bash
vllm serve /models/qwen38 \
  --served-model-name pplx-computer-qwen-3-8-27b-dspark2-20260819 \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --async-scheduling \
  --speculative-config '{"method": "dflash", "model": "<model-dir>/draft", "num_speculative_tokens": 7}' \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

## Notes / gotchas

- **The DFlash2 draft is vendored at `draft/`** (a mirror of
  [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
  2026-08-19) and referenced by local path in the compose — the repo is fully
  self-contained, no external downloads at serve time. The draft is separate
  from the target weights: the in-checkpoint `mtp.*` tensors are the dspark2
  MTP module — MTP k=7 reaches similar acceptance but decodes ~2× slower
  (sequential draft passes), so DFlash2 is the recommended path.
- **bf16 lm_head is load-bearing.** DFlash2 requires an unquantized target
  `lm_head`; it is excluded via `exclude_modules` in both
  `hf_quant_config.json` and `config.json` (vLLM's modelopt loader keys on
  the exclude wildcards, not on absence from `quantized_layers`).
- **Why `Dockerfile.lmheadfix`:** vLLM gives an excluded lm_head an
  `UnquantizedLinearMethod`, which PR #52816's `isinstance` check wrongly
  rejects (it only accepts `UnquantizedEmbeddingMethod`). The second image
  layer relaxes that check.
- `--max-num-batched-tokens 8192` is required: with spec decode enabled,
  vLLM otherwise clamps scheduling to 2048 and prefill collapses.
