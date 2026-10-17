---
base_model:
- Qwen/Qwen3.8-27B
tags:
- vllm
- llm-compressor
- qwen3_5
- compressed-tensors
- int4
- conversational
- speculative-decoding
license: apache-2.0
pipeline_tag: image-text-to-text
library_name: transformers
---

# Qwen3.8-27B INT4

A 4-bit (W4A16, group size 128) quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), ready for inference with vLLM.

The transformer-block linear weights are quantized to INT4 with [LLM Compressor](https://github.com/vllm-project/llm-compressor). The vision tower, `embed_tokens`, and `lm_head` are kept in their original precision (bf16).

This repository also ships the **MTP (multi-token prediction) drafter** (`model_mtp.safetensors`), enabling speculative decoding out of the box with vLLM.

## Contents

| File | Description |
| --- | --- |
| `model.safetensors` | Main model weights (INT4 body, bf16 `lm_head`) |
| `model_mtp.safetensors` | MTP drafter weights |
| `model.safetensors.index.json` | Weight index |
| `config.json` | Model architecture + quantization config |
| `tokenizer.json` / `tokenizer_config.json` | Tokenizer |
| `generation_config.json` | Generation defaults |
| `processor_config.json` | Vision processor config |
| `chat_template.jinja` | Stock chat template |
| `qwen_template_low.jinja` | Chat template with a lower default reasoning effort |
| `recipe.yaml` | Quantization recipe |

## vLLM Serving

```bash
vllm serve <this-repo> \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_capture_sizes": [1,2,4,8,16,32]}' \
  --speculative-config '{"method":"mtp","model":"<this-repo>","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
```

To use the lower-default chat template, add:

```
  --chat-template qwen_template_low.jinja \
  --default-chat-template-kwargs '{"reasoning_effort": "low"}'
```

The MTP drafter is loaded from the same directory via the `--speculative-config.model` field.

## Notes

- Quantization: W4A16, block size 128, applied to transformer-block `Linear` layers.
- `lm_head` is kept in bf16 (original precision).
- MTP: single-layer draft module, used with `num_speculative_tokens=3`.
- The two chat templates differ only in the default `reasoning_effort` (`xhigh` vs `low`); pass `reasoning_effort` explicitly to override either.

## Variants

This model is one of three related INT4 checkpoints. The only difference between them is the precision of `lm_head` and whether the MTP drafter is included:

| Variant | `lm_head` | MTP drafter (`model_mtp.safetensors`) | Size |
| --- | --- | --- | --- |
| [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | bf16 | — | ~18 GB |
| `-mtpq` | bf16 | ✅ | ~18 GB |
| `-both` | INT4 | ✅ | ~16 GB |

## License

Base model: Apache-2.0. Quantization derived from `Qwen/Qwen3.8-27B`.
