---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - abliterated
  - heretic
  - arbitrary-rank-ablation
  - qwen3_5
  - gguf
  - vision
---

# Qwen3.8-27B heretic ARA — GGUF

GGUF builds of an abliterated Qwen3.8-27B, made with heretic's **Arbitrary-Rank Ablation** branch (PR 211). ARA does per-module matrix optimization under hooks, not directional ablation.

Best trial, exported here: **4/100 refusals** at KL divergence **0.0454** against the base model. For comparison: base model 92/100 refusals; directional heretic v1 65/100. The Optuna search ran ~85 trials on an 80GB H100.

## Files

| File | Size | Use |
|---|---|---|
| `qwen38-27b-heretic-ara-Q4_K_M.gguf` | 16.5 GB | Main quant for 24 GB cards |
| `qwen38-27b-heretic-ara-Q8_0.gguf` | 28.6 GB | Near-lossless quant |
| `qwen38-27b-heretic-ara-mmproj-f16.gguf` | 0.93 GB | Vision projector for llama.cpp |

## Usage

Architecture is `qwen35`, context length 262144. Use a recent llama.cpp build.

```
llama-server \
  -m qwen38-27b-heretic-ara-Q4_K_M.gguf \
  --mmproj qwen38-27b-heretic-ara-mmproj-f16.gguf \
  -c 32768
```

The chat template is embedded in the GGUF metadata. It supports `reasoning_effort` levels `xhigh` (default), `medium`, and `low`, and `enable_thinking`.

## Notes

- **No MTP tensors.** The heretic export drops them, so speculative decoding with the MTP head is not available.
- **Vision works.** The vision tower is untouched by the ablation. Pass the mmproj file with `--mmproj`.
- **Quality checks** ran on the GPTQ Int4 export of the same ARA weights, not on these GGUF files: GSM8K 92.0% (base quant: 90.5%, same subset, within noise), quality probes 5/5, over-refusal on benign-edgy prompts 0/5, vision correct.
- **Known quirk:** rare, prompt-triggered repetition on long generations (seen once in 9 long-form samples). It is drift-independent in this KL range.

## Provenance

Converted from the BF16 ARA export (`CorVous/Qwen3.8-27B-heretic-ara-BF16`). A GPTQ Int4 W4A16 build of the same weights exists as `CorVous/Qwen3.8-27B-heretic-ara-GPTQ-Int4-gs128`.
