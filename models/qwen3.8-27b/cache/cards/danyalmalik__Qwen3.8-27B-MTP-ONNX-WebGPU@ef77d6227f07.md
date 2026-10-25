---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
tags:
- onnx
- webgpu
- quick-inference
- speculative-decoding
---

# Qwen3.8-27B-MTP-ONNX-WebGPU

The multi-token prediction (MTP) layer that [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) ships in its checkpoint
(`mtp.*`), converted to ONNX for speculative decoding in a web browser with ONNX Runtime Web's WebGPU backend, as used
by Quick Inference. It drafts the next tokens for [amd/Qwen3.8-27B-fp16-ve-fp16-int4-k_quant-gs128-text-onnx](https://huggingface.co/amd/Qwen3.8-27B-fp16-ve-fp16-int4-k_quant-gs128-text-onnx), which checks them: the
output is exactly the model's own.

## What it computes

As the MTP layer of Qwen's checkpoint (vLLM's `Qwen3_5MultiTokenPredictor`): for a position t, the embedding of the
token at t + 1 and the model's final hidden state at t (after its final norm) are normalised, concatenated and mapped
to the hidden size, then one full-attention decoder layer (gated attention, its own KV cache at the
model's positions), a final norm and the model's output layer give the guess for t + 2. A chain of guesses feeds the
layer's own output hidden state back.

- Inputs: `input_ids` [1, n], `hidden` [1, n, hidden] (float16), `position_ids` [1, n], `last` (the row to guess
  from), `seqlens_k` and `total_sequence_length` (GroupQueryAttention's), `past_key_values.0.key/value` (a static
  cache, updated in place).
- Outputs: `token` [1, 1] (the guessed token id), `hidden_out` [1, 1, hidden] (for the next guess),
  `present.0.key/value`.

## Changes from the original

- The layer's matrices are quantised to 4 bits (round to nearest, block size 32).
- The output layer and the token embedding table are the 4-bit ones of the ONNX model it drafts for, keeping the first
  32,768 token ids only: guesses come from those (token ids past them are clamped for the embedding lookup). This makes
  it several times smaller and faster, for a few percent fewer kept guesses.
- Norm weights include the 1 of Qwen3.5's zero-centred RMSNorm.

These are modified files: their outputs are close to, but not identical to, the original layer's.

## Licence

Licensed under the same terms as the original model (Apache 2.0): see `LICENSE`.
