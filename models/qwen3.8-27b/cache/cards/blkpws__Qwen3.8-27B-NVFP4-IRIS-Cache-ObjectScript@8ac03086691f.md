---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
- objectscript
- intersystems
- iris
- cache
- finetune
- qwen3.8
---

# Qwen3.8-27B-NVFP4-IRIS-Cache-ObjectScript

Fine-tuned **Qwen3.8-27B** as an **InterSystems IRIS / Caché ObjectScript specialist**,
trained on a task-solving dataset (44% code task→solution, 16% real bugfix pairs,
31% short Q&A, 9% general anti-forgetting) with LR 1e-4 + grad-accum 2 (1200 steps,
eval loss 0.966 → 0.787 monotonic), merged to BF16, then re-quantized to **NVFP4**
(attn FP8 dynamic, MLP NVFP4) with domain-matched ObjectScript calibration.

Won an A/B against the previous v2 fine-tune on real ObjectScript coding prompts
(complete working code vs truncated prose).

## Model facts

| Property | Value |
| --- | --- |
| Base model | `Qwen/Qwen3.8-27B` (hybrid: 48 linear_attn + 16 self_attn layers) |
| **Parameters (BF16)** | **~27.8B** (the HF "18b" label below is the **quantized file size** in GB, not the parameter count) |
| **Default context window** | **262,144 tokens (256K)** — `max_position_embeddings: 262144` |
| Architecture | `Qwen3_5ForCausalLM`, hidden 5120, 64 layers, 24 heads / 4 KV, vocab 248,320 |
| Quantization | compressed-tensors: NVFP4 (W4A4 static, group 16) on MLP + lm_head, FP8 (W8A8 static) on attention |
| Quantized size on disk | ~18-20 GB (4-bit) |

**Why it shows "~18B" on Hub:** HuggingFace derives the number from the safetensors
**storage size** (~18-20 GB after 4-bit NVFP4 quantization) and displays it as "b".
The model is genuinely a **27B-class** model (Qwen3.8-27B, ~27.8B BF16 params);
4-bit quantization shrinks the *file size*, not the parameter count.

## Deployment

Serve with **vLLM** (`--quantization compressed-tensors`, `--max-model-len 262144`).
Served model name: `qwen3.8-27b-iris`. Context: **262144 (256K)**.

On a DGX Spark (GB10/SM121), keep `--gpu-memory-utilization` <= 0.85 and the freeze
guards (`--max-cudagraph-capture-size 4`, `--no-enable-flashinfer-autotune`).

## Generated with

- [llmcompressor](https://github.com/vllm-project/llm-compressor) 0.13.0 (NVFP4 + FP8 mixed)
- Custom peft + transformers Trainer (BF16 base + 4-bit qLoRA)

## Dialect

Trains on InterSystems IRIS / Caché ObjectScript (`.cls` / `.mac` / `.inc`).
