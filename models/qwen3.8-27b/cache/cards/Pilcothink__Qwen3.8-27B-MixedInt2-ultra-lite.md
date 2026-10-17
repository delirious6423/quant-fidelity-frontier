---
license: apache-2.0
library_name: transformers
base_model:
  - Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - qwen
  - autoround
  - quantized
  - int2
  - mixed-precision
  - vllm
  - conversational
  - 2-bit precision
  - auto-round
---

# Qwen3.8-27B-MixedInt2-ultra-lite

A mixed-precision AutoRound quantized version of Qwen/Qwen3.8-27B, optimized to significantly reduce memory requirements while preserving the quality of the original model.

## Base Model

Base model: `Qwen/Qwen3.8-27B`

This model is a quantized version of the original Qwen3.8-27B checkpoint. It is not a fine-tune, merge, or distillation.

## Quantization

Quantization was performed using Intel AutoRound with a custom mixed-precision quantization configuration.

The quantization recipe was designed to balance:

- Model quality
- Memory efficiency
- Inference performance
- vLLM compatibility

Some model components are intentionally retained at higher precision where appropriate.

Property | Value
--- | ---
Quantization framework | Intel AutoRound
Quantization type | Custom Mixed-Precision INT2
Group size | 32, 64, 128
Base model | Qwen/Qwen3.8-27B
Language layers | 64
Vision tower | Preserved at original precision

The detailed mixed-precision allocation strategy is not included in this model card.

## Evaluation

Evaluation was performed using AutoRound's evaluation interface with LM Evaluation Harness.

The following results compare the original Qwen3.8-27B model against Qwen3.8-27B-MixedInt2-ultra-lite.

Benchmark | Metric | Qwen3.8-27B | Qwen3.8-27B-MixedInt2-ultra-lite | Difference | Recovery Rate
--- | --- | --- | --- | --- | ---
MMLU | acc | 83.49% | 78.30% | -5.19 pp | 93.78%
GSM8K | exact_match (flexible) | 72.86% | 43.29% | -29.57 pp | 59.42%
ARC-Challenge | acc_norm | — | — | — | —
BoolQ | acc | — | — | — | —
HellaSwag | acc_norm | — | — | — | —
PIQA | acc_norm | — | — | — | —
WinoGrande | acc | — | — | — | —
Average | — | — | — | — | —

### MMLU Category Breakdown

MMLU Category | Qwen3.8-27B | Qwen3.8-27B-MixedInt2-ultra-lite | Difference | Recovery Rate
--- | --- | --- | --- | ---
Humanities | 77.39% | 73.33% | -4.06 pp | 94.75%
Other | 86.03% | 81.20% | -4.83 pp | 94.39%
Social Sciences | 90.74% | 85.99% | -4.75 pp | 94.77%
STEM | 83.03% | 75.36% | -7.67 pp | 90.76%

### GSM8K

Metric | Qwen3.8-27B | Qwen3.8-27B-MixedInt2-ultra-lite | Difference | Recovery Rate
--- | ---: | ---: | ---: | ---:
Flexible Exact Match | 72.86% | 43.29% | -29.57 pp | 59.42%
Strict Exact Match | 70.36% | 35.33% | -35.03 pp | 50.22%

> Recovery Rate represents benchmark performance relative to the original Qwen3.8-27B checkpoint. A recovery rate above 100% indicates that the quantized model scored higher than the original model in that particular evaluation. Benchmark preservation does not imply identical behavior for every prompt, multimodal workload, long-context workload, or generation setting.

## Usage

This checkpoint is intended for inference engines with AutoRound quantization support, including compatible versions of vLLM.



### vLLM
This PR is required to serve this model with vLLM. [vLLM PR #52890.](https://github.com/vllm-project/vllm/pulls)


As of August 24 at 9:30 PM KST, the model can now be served with vLLM. Previously, mixed 2-bit serving was not possible due to issues in both config.json and quantization_config.json, but these have now been fixed.


### Example Serving

```bash
vllm serve Pilcothink/Qwen3.8-27B-MixedInt2-ultra-lite \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --tool-call-parser qwen3_coder \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --mm-encoder-tp-mode data \
  --max-num-seqs 10
```

### Example with Serving 1M Context

```bash
vllm serve Pilcothink/Qwen3.8-27B-MixedInt2-ultra-lite \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --tool-call-parser qwen3_coder \
  --kv-cache-dtype fp8 \
  --max-model-len 1000000 \
  --max-num-batched-tokens 8192 \
  --mm-encoder-tp-mode data \
  --max-num-seqs 10 \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```


With MTP enabled, an error appears to occur due to incompatibility with the Humming kernel’s weight-loading path. MTP is currently not usable with Humming.


Additional reasoning and tool-calling options should be configured according to the vLLM version being used.

## Notes

- This is a quantized derivative of Qwen3.8-27B.
- The model uses a custom mixed-precision INT2 quantization recipe.
- The vision components are preserved at their original precision.
- This ultra-lite variant is designed to further reduce memory requirements compared with higher-bit quantized variants.
- Quantization may introduce small behavioral differences compared with the original checkpoint.

## Acknowledgements

- Base model: Qwen Team
- Quantization framework: Intel AutoRound

Please refer to the original Qwen3.8-27B model card for licensing, intended usage, limitations, and other information applicable to the base model.
