---
license: gemma
base_model: google/gemma-4-26B-A4B-it
tags: [gemma, gemma-4, moe, mixture-of-experts, quantized, nvfp4, vllm, reasoning, thinking]
library_name: vllm
---

# Gemma-4-26B-A4B-it-NVFP4A16

NVFP4A16 quantization of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it).

> **Status:** private until canonical re-eval lands at `thinking_token_budget=24576` / `max_gen_toks=49152`.

## Quant recipe

- **Quantizer:** [`omnimergekit/scripts/quantize_any.py --method nvfp4a16`](https://github.com/mann1x/omnimergekit/blob/main/scripts/quantize_any.py)
- **Engine:** nvidia-modelopt 0.45.0.dev78+g7a11fb240 with `NVFP4_DEFAULT_CFG`
- **Calibration:** 128 samples from `tatsu-lab/alpaca`, max_length=512
- **Excluded modules:** `*vision_tower*`, `*embed_vision*`, `*embed_audio*` (kept BF16 for vLLM's gemma4_mm loader)
- **Hardware:** vast.ai 2×RTX 3090 pod, host 53225 (Quebec, CA), env bit-matched to producer machine (transformers 5.8.0, torch 2.10.0+cu128, safetensors 0.8.0rc0)

## Usage with vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ManniX-ITA/Gemma-4-26B-A4B-it-NVFP4A16 \
    --port 8099 --gpu-memory-utilization 0.92 \
    --max-model-len 49152 --dtype bfloat16 --trust-remote-code \
    --reasoning-parser gemma4 \
    --default-chat-template-kwargs '{"enable_thinking": true}'
```

Per-request: `{ "thinking_token_budget": 24576 }` (mandatory; see vLLM #42250, #42434, #42635).

## VRAM

| GPU | runs? | mem (peak) |
|---|---|---|
| RTX 3090 24 GB | ✅ | ~13 GB weights + KV |

## Source

- BF16 base: [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
- Quant script: `omnimergekit/scripts/quantize_any.py`
- Quant date: 2026-05-14
