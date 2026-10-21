---
license: apache-2.0
base_model:
- RadixArk/Qwen3.8-27B-NVFP4
- Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- quantized
- NVFP4
- speculative-decoding
- dense-lm-head
---

# Qwen3.8-NVFP4-Hybrid-BF16

A variant of [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
with the `lm_head` restored to the **original BF16 tensor** from
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — hence
"Hybrid": NVFP4/FP8 body, BF16 output head.

## Why this exists

Several speculative-decoding drafters select candidate tokens by running
TopK directly over the target model's `lm_head` logits (for example
block-diffusion drafters in the DFlash2 family). Serving engines
implementing these drafters require a **dense FP16/BF16/FP32 `lm_head`**
and reject quantized heads at startup. The stock NVFP4 checkpoint
quantizes the head, which rules those drafters out.

This checkpoint makes NVFP4 serving compatible with dense-head drafters
while keeping the rest of the model byte-identical to the source
quantization.

## What was changed

Relative to `RadixArk/Qwen3.8-27B-NVFP4` (revision `554ebba`):

1. `lm_head.weight` replaced with the **pristine BF16 tensor from the
   upstream base model** — a transplant of the original weights, *not* a
   dequantization of the quantized head (dequantizing would bake the
   quantization error into the head; the original tensor has none).
2. `quantization_config` in `config.json` and the scheme in
   `hf_quant_config.json` updated to exclude `lm_head`, so loaders treat
   the head as an unquantized BF16 linear layer.

Everything else — all transformer weights, scales, tokenizer, chat
template — is unchanged from the source quantization.

## Quality

The output head carries the original full-precision weights, so this
variant's accuracy is greater than or equal to the source quantization
by construction. GSM8K spot-checks match the source checkpoint's
published qualification score within evaluation noise.

## Serving notes

- **KV cache dtype**: the checkpoint (like the source quantization)
  declares an FP8 KV-cache scheme, so `kv-cache-dtype: auto` will serve
  an FP8 KV cache. If you want BF16 KV, set it explicitly.
- **SGLang**: loads via the ModelOpt mixed-precision path; recent
  mainline includes drafters that require exactly this kind of dense
  head.
- **vLLM**: some versions strictly type-check the head's quantization
  method when a dense head is required; if you hit such a check with
  this checkpoint, it is a false positive of the type test rather than a
  quantized head — the head here is plain BF16.
- Memory: the BF16 head is larger than its quantized counterpart;
  expect slightly higher weight memory than the source checkpoint.

## Provenance and license

Derivative of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(Apache-2.0), by way of the
[RadixArk NVFP4 quantization](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
produced with NVIDIA Model Optimizer. See the upstream model card for
the base model's capabilities, limitations, and usage policies. This
repository is not affiliated with Qwen or RadixArk.
