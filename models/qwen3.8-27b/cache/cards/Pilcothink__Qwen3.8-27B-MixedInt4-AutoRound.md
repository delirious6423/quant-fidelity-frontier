---
library_name: transformers
base_model:
  - Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
  - qwen
  - qwen3.8
  - autoround
  - quantized
  - int4
  - mixed-precision
  - vllm
---

# Qwen3.8-27B-MixedInt4-AutoRound

A mixed-precision AutoRound quantized version of **Qwen/Qwen3.8-27B**, optimized to reduce memory requirements while preserving the quality of the original model.

## Base Model

**Base model:** `Qwen/Qwen3.8-27B`

This model is a quantized version of the original Qwen3.8-27B checkpoint. It is not a fine-tune, merge, or distillation.

## Quantization

Quantization was performed using **Intel AutoRound** with a custom mixed-precision quantization configuration.

The quantization recipe was designed to balance:

- Model quality
- Memory efficiency
- Inference performance
- vLLM compatibility

Some model components are intentionally retained at higher precision where appropriate.

| Property | Value |
|---|---|
| Quantization framework | Intel AutoRound |
| Quantization type | Custom Mixed-Precision INT4 |
| Group size | 32 |
| Base model | Qwen/Qwen3.8-27B |
| Language layers | 64 |
| Vision tower | Preserved at original precision |

The detailed mixed-precision allocation strategy is not included in this model card.

## Evaluation

Evaluation was performed using **AutoRound's evaluation interface with LM Evaluation Harness**.

The following results compare the original **Qwen3.8-27B** model against **Qwen3.8-27B-MixedInt4-AutoRound**.

| Benchmark | Metric | Qwen3.8-27B | Qwen3.8-27B-MixedInt4-AutoRound | Difference | Recovery Rate |
|---|---:|---:|---:|---:|---:|
| MMLU | acc | **83.49%** | **83.07%** | **-0.42 pp** | **99.50%** |
| GSM8K | exact_match (flexible) | **72.86%** | **76.12%** | **+3.26 pp** | **104.47%** |
| ARC-Challenge | acc_norm | **58.87%** | **58.87%** | **0.00 pp** | **100.00%** |
| BoolQ | acc | **86.64%** | **80.49%** | **-6.15 pp** | **92.90%** |
| HellaSwag | acc_norm | **82.82%** | **82.40%** | **-0.42 pp** | **99.49%** |
| PIQA | acc_norm | **81.61%** | **81.66%** | **+0.05 pp** | **100.06%** |
| WinoGrande | acc | **75.85%** | **76.16%** | **+0.31 pp** | **100.41%** |
| **Average** | — | **77.45%** | **76.97%** | **-0.48 pp** | **99.38%** |


### MMLU Category Breakdown

| MMLU Category | Qwen3.8-27B | Qwen3.8-27B-MixedInt4-AutoRound | Difference | Recovery Rate |
|---|---:|---:|---:|---:|
| Humanities | **77.39%** | **77.39%** | **0.00 pp** | **100.00%** |
| Other | **86.03%** | **85.87%** | **-0.16 pp** | **99.81%** |
| Social Sciences | **90.74%** | **90.35%** | **-0.39 pp** | **99.57%** |
| STEM | **83.03%** | **81.67%** | **-1.36 pp** | **98.36%** |

### GSM8K

| Metric | Qwen3.8-27B | Qwen3.8-27B-MixedInt4-AutoRound | Difference | Recovery Rate |
|---|---:|---:|---:|---:|
| Flexible Exact Match | **72.86%** | **76.12%** | **+3.26 pp** | **104.47%** |
| Strict Exact Match | **70.36%** | **73.69%** | **+3.33 pp** | **104.73%** |

> Recovery Rate represents benchmark performance relative to the original Qwen3.8-27B checkpoint. A recovery rate above 100% indicates that the quantized model scored higher than the original model in that particular evaluation. Benchmark preservation does not imply identical behavior for every prompt, multimodal workload, long-context workload, or generation setting.

## Usage

This checkpoint is intended for inference engines with AutoRound quantization support, including compatible versions of vLLM.

Example:

```bash
vllm serve Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound \
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

Example with MTP

```bash
vllm serve Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound \
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
  --max-num-seqs 10 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```


Example with Serving 1M


```bash
vllm serve Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --tool-call-parser qwen3_coder \
  --kv-cache-dtype fp8 \
  --max-model-len 1010000 \
  --max-num-batched-tokens 8192 \
  --mm-encoder-tp-mode data \
  --max-num-seqs 10 \
  --hf-overrides '{"text_config": {"max_position_embeddings": 1010000}}'
```

Additional reasoning and tool-calling options should be configured according to the vLLM version being used.

## Notes

- This is a quantized derivative of Qwen3.8-27B.
- The model uses a custom mixed-precision quantization recipe.
- The vision components are preserved at their original precision.
- Quantization may introduce small behavioral differences compared with the original checkpoint.

## Acknowledgements

- **Base model:** Qwen Team
- **Quantization framework:** Intel AutoRound

Please refer to the original Qwen3.8-27B model card for licensing, intended usage, limitations, and other information applicable to the base model.
