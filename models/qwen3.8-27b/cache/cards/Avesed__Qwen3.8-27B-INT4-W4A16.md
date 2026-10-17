---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
language:
- en
- zh
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- int4
- w4a16
- awq
- compressed-tensors
- vllm
- vision-language
---

# Qwen3.8-27B-INT4-W4A16

INT4 weight-only (**W4A16**) quantization of **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** — a dense hybrid model (GatedDeltaNet linear-attention + periodic full-attention) with a native vision tower and an MTP head.

At BF16 the base model needs ~56 GB and does not fit two 24 GB cards. This checkpoint is ~20 GB, so the weights fit on a single 24 GB GPU and leave room for a large KV cache at tensor-parallel 2.

## Quantization

- **AWQ** activation-aware scale search **+ symmetric INT4 weights, group_size 32, MSE observer** (llm-compressor), calibrated on 128 × 512-token Magicoder samples.
- **Quantized:** all MLP, the full-attention `self_attn`, and the GatedDeltaNet `out_proj`.
- **Kept bf16** (quality-sensitive): GatedDeltaNet `in_proj_a`/`in_proj_b`, `lm_head`, embeddings, vision tower, MTP head.
- Format: `compressed-tensors` (`pack-quantized`). Full recipe: [`recipe.yaml`](./recipe.yaml).

The **MTP head is included** (`mtp.safetensors`, bf16) and is listed in `quantization_config.ignore`, so vLLM loads it unquantized and speculative decoding works out of the box. Quantizers silently drop this head — if you re-quantize the base yourself, graft it back or spec-decode will accept 0 %.

## Validation

Served with vLLM tensor-parallel 2 on 2 × RTX 3090 and exercised over the OpenAI API:

- Chinese / English generation and translation: coherent.
- **Vision path** (image input through the native VL tower): working.
- `reasoning_effort` (`low` / `xhigh`) and `enable_thinking` pass through correctly.
- MTP speculative decoding: mean accept length ≈ 2.4 / 3.0 at K=2.

**GSM8K: 96.4 %** (241/250). Served with vLLM tp2 and scored over the OpenAI API — thinking
enabled, the base model's recommended sampling (`temperature=1.0, top_p=0.95, top_k=20`), default
`reasoning_effort`, `max_tokens=24576`, and the question suffixed with "Put your final numeric
answer inside `\boxed{}`". The same protocol scores the INT8-W8A8 build of this model at 96.8 %
(242/250), i.e. the two are indistinguishable at this sample size.

Note the protocol matters more than it looks: without the `\boxed{}` instruction this model often
states the answer in bold and then keeps talking, so a "last number in the answer" extractor scores
many correct answers wrong (the same 250 questions land at ~90 % that way). MMLU-Pro has not been
run on this checkpoint.

## Throughput

2 × RTX 3090, tensor-parallel 2, single stream, prefix caching disabled, no speculative decoding:

| Context | W4A16 decode | W4A16 prefill | W4A8 prefill |
|---|---|---|---|
| 4k | 69.5 tok/s | 2390 tok/s | 3963 tok/s |
| 16k | 66.5 tok/s | 2331 tok/s | 3801 tok/s |
| 32k | 63.4 tok/s | 2197 tok/s | 3436 tok/s |

Decode is int4-weight bandwidth-bound and is the same for W4A16 and W4A8; the int8 activation path buys **+61 % … +70 % prefill**.

## Usage (vLLM)

```bash
# W4A16 — int4 weights, fp16 activations
vllm serve Avesed/Qwen3.8-27B-INT4-W4A16 \
  --tensor-parallel-size 2 --trust-remote-code --reasoning-parser qwen3
```

On Ampere the same checkpoint can be served as **W4A8** (int4 weights + int8 dynamic activations — faster prefill and batched serving) with the [vllm-ampere-optimized](https://github.com/avesed/vllm-ampere-optimized) fork:

```bash
vllm serve Avesed/Qwen3.8-27B-INT4-W4A16 \
  --tensor-parallel-size 2 --marlin-input-dtype int8 --trust-remote-code --reasoning-parser qwen3
```

Add `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` to use the MTP head.

Sampling follows the base model: thinking `temperature=1.0, top_p=0.95, top_k=20`; non-thinking `temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5`.

Quantized with [vllm-ampere-optimized/quantize](https://github.com/avesed/vllm-ampere-optimized/tree/main/quantize).
