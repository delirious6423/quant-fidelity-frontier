---
library_name: transformers
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
tags:
- gemma4
- gemma-4
- nvfp4
- modelopt
- vllm
- quantization
- moe
---

# Gemma-4-26B-A4B-it-NVFP4

`NVFP4`-quantized `google/gemma-4-26B-A4B-it` for `vLLM`.

This is the **flat-key** variant: expert weights are stored under
`...experts.<n>.*` (no `moe.` namespace). Vanilla upstream `vllm`
`v0.20.1` and later load the checkpoint directly — **no runtime patch
required**. Earlier snapshots of this repo stored the same weights
under `...moe.experts.<n>.*` and relied on a bundled `gemma4_patched.py`
to disable a non-idempotent regex in `vllm/model_executor/models/gemma4.py`;
that workaround is no longer needed.

## Quantization profile

- Text experts: `NVFP4`
- Text self-attention: `NVFP4`
- Token embeddings: `NVFP4`
- Router: higher precision
- `lm_head`: higher precision
- Vision tower and vision embeddings: higher precision
- KV cache: `FP8`

## Included files

- `quantize_gemma4_moe.py` — quantization/export script, redistributed under `Apache-2.0`
- `gemma4_patched.py` — kept for historical reference only; not required since the flat-key snapshot

Attribution:

- `quantize_gemma4_moe.py` is based on the community release published by `bg-digitalservices` / `marioiseli`
- `gemma4_patched.py` is derived from `vLLM`'s Apache-2.0 Gemma 4 implementation

## Deployment

### Docker (recommended)

```bash
docker run -d \
  --name gemma4-a4b-nvfp4 \
  --gpus all \
  --ipc=host \
  --network host \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:v0.20.1 \
  --model Neural-ICE/Gemma-4-26B-A4B-IT-NVFP4 \
  --served-model-name gemma-4 \
  --host 0.0.0.0 \
  --port 8000 \
  --generation-config vllm \
  --quantization modelopt \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --moe-backend marlin \
  --trust-remote-code
```

### Native `vllm serve`

```bash
source .venv-vllm/bin/activate

VLLM_NVFP4_GEMM_BACKEND=marlin \
vllm serve Neural-ICE/Gemma-4-26B-A4B-IT-NVFP4 \
  --served-model-name gemma-4 \
  --host 0.0.0.0 \
  --port 8000 \
  --generation-config vllm \
  --quantization modelopt \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --moe-backend marlin \
  --trust-remote-code
```

### Why these flags matter

- `--quantization modelopt`: required for ModelOpt NVFP4 checkpoints
- `--moe-backend marlin`: required for the MoE expert path
- `VLLM_NVFP4_GEMM_BACKEND=marlin`: uses Marlin on the non-MoE NVFP4 path as well
- `--kv-cache-dtype fp8`: reduces KV memory pressure
- `--trust-remote-code`: required for Gemma 4

Official Gemma 4 `vLLM` recipe:

https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html

### Thinking and tool calling

To expose Gemma 4 reasoning parsing and tool calling on the server:

```bash
VLLM_NVFP4_GEMM_BACKEND=marlin \
vllm serve Neural-ICE/Gemma-4-26B-A4B-IT-NVFP4 \
  --served-model-name gemma-4 \
  --generation-config vllm \
  --max-model-len 32768 \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --moe-backend marlin \
  --trust-remote-code \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice
```

Important:

- `--reasoning-parser gemma4` parses Gemma 4 reasoning output
- actual thinking mode is activated per request with `chat_template_kwargs.enable_thinking=true`
- with the OpenAI Python client, pass `extra_body={"chat_template_kwargs": {"enable_thinking": True}}`

Tip: For text-only workloads, add `--limit-mm-per-prompt '{"image":0,"video":0}'` to skip multimodal encoder allocation. In practice, `--gpu-memory-utilization 0.75` to `0.85` is safer than hard-coding `0.90`.

## Text generation

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4",
    "messages": [
      {"role": "user", "content": "Explain quantum entanglement in simple terms."}
    ],
    "max_tokens": 512
  }'
```
