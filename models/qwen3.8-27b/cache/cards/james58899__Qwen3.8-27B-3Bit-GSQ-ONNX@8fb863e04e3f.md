---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: onnxruntime
pipeline_tag: text-generation
tags:
- onnx
---

# Qwen3.8 27B (3-bit GSQ), ONNX

> [!Warning]
> The conversion scripts were written by Claude, and the conversion was run with them. Check it yourself before use; use it at your own risk.

A text-only ONNX conversion of
[ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ)
(revision `b5ce0b76f60020a875dee4f6ec9d934cca4121e4`, a 3-bit GSQ quantization of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)), with its weights as stored: every integer level and
scale is kept, nothing is re-quantized.

ONNX Runtime has no 3-bit kernel, so the 3-bit levels (symmetric, group 128) sit in 4-bit `MatMulNBits` blobs. Each
group of 128 is written as four blocks of 32 that share its scale, which is the block size the WebGPU prefill fast
path needs. The embedding and `lm_head` are 4-bit (group 64) in the source and stay that way.

## Contents

- `onnx/embed_tokens_q4f16.onnx`: the token embedding (`GatherBlockQuantized`, fp16 output).
- `onnx/decoder_model_merged_q4f16.onnx`: the decoder, with fp16 activations and state. Full-attention layers use
  `GroupQueryAttention` with a KV cache (`past_key_values.N.key` / `.value`). Linear-attention layers use
  `com.microsoft` `LinearAttention` (`gated_delta`) and `CausalConvWithState`, with a fixed-size state
  (`past_key_values.N.conv_state` / `.recurrent_state`) that starts as zeros.
- `config.json`: the text part of the source config (`Qwen3_5ForCausalLM`). There is no vision encoder and no MTP head.
- The tokenizer, chat template and generation config, copied from the source checkpoint.
- `scripts/`: the conversion scripts that produced the model
  (`python scripts/export.py <checkpoint dir> <out dir> --block-size 32`), and `verify.py`, which compares an export
  with a numpy port of the transformers forward pass.

Tested with ONNX Runtime 1.31 (CPU and WebGPU).

## License

[Apache 2.0](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ), same as the source checkpoint.
