---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - nvidia
  - model-optimizer
  - nvfp4
  - fp4
  - quantization
  - vllm
  - tensorrt-llm
pipeline_tag: text-generation
quantization_config:
  quant_method: modelopt
  quant_algo: NVFP4
---

# Qwen3.8-27B — NVFP4 (W4A4) Quantized

This is the [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) model
quantized to **NVFP4** (NVIDIA FP4, W4A4) with an **FP8 KV cache**, produced with
[NVIDIA Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
(recipe `general/ptq/nvfp4_default-kv_fp8`).

- **Weights:** NVFP4 (E2M1), block size 16, dynamic per-block scale (FP8 E4M3)
- **KV cache:** FP8
- **Size:** ~20 GB (from ~52 GB BF16)
- **Format:** unified Hugging Face checkpoint (`quant_method: modelopt`)

## What was quantized

| Layer group | Count | Status |
|---|---|---|
| MLP `gate/up/down_proj` (all 64 layers) | 192 | NVFP4 |
| Linear-attention `in_proj_qkv`, `in_proj_z`, `out_proj` (48 layers) | 144 | NVFP4 |
| Full-attention `q/k/v/o_proj` (16 layers) | 64 | NVFP4 |
| Linear-attention `conv1d`, `in_proj_a`, `in_proj_b` | 144 | kept BF16 (numerically sensitive) |
| Vision encoder `model.visual.*` | ~112 | kept BF16 |
| `lm_head`, `embed_tokens`, MTP heads | 4 | kept BF16 |

Calibration: 512 samples (default `cnn_nemotron_v2_mix` dataset).

## Usage with vLLM

```bash
vllm serve kristianpaul/Qwen3.8-27B-NVFP4 \
  --dtype auto \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.45 \
  --enforce-eager
```

vLLM auto-detects the NVFP4 format from `config.json` and selects the
`FlashInferCutlassNvFp4LinearKernel`. Also loadable by TensorRT-LLM and SGLang.

> Note: this is a packed NVFP4 checkpoint. Plain
> `AutoModelForCausalLM.from_pretrained` will **not** load it — use an inference
> framework (vLLM / TensorRT-LLM / SGLang).

## Files

- `model-0000{1,2,3}-of-00003.safetensors` — packed NVFP4 weights + scales
- `config.json` — model config with `quantization_config`
- `hf_quant_config.json` — Model-Optimizer quantization config
- `.quant_summary.txt` — per-layer quantization summary
- tokenizer / chat template files

## License

Apache 2.0 (same as the base model).
