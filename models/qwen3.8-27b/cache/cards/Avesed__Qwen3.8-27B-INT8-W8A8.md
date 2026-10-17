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
- int8
- w8a8
- compressed-tensors
- vllm
- vision-language
---

# Qwen3.8-27B-INT8-W8A8

INT8 weight **and** activation (**W8A8**) quantization of **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** — a dense hybrid model (GatedDeltaNet linear-attention + periodic full-attention) with a native vision tower and an MTP head.

~30 GB, so it needs two 24 GB cards (tensor-parallel 2) but leaves a comfortable KV cache. Pick this over the [INT4-W4A16 build](https://huggingface.co/Avesed/Qwen3.8-27B-INT4-W4A16) when you want the fastest prefill and have the VRAM; pick INT4 when you want the fastest decode or a single-card fit.

## Quantization

- **Symmetric INT8 weights, per-output-channel, MSE observer** + **INT8 dynamic per-token activations** (llm-compressor), calibrated on 256 × 2048-token Magicoder samples.
- **Quantized:** all MLP, the full-attention `self_attn`, and the GatedDeltaNet `out_proj`.
- **Kept bf16** (quality-sensitive): GatedDeltaNet `in_proj_a`/`in_proj_b`, `lm_head`, embeddings, vision tower, MTP head.
- Format: `compressed-tensors` (`int-quantized`). Full recipe: [`recipe.yaml`](./recipe.yaml).

The **MTP head is included** (`mtp.safetensors`, bf16) and is listed in `quantization_config.ignore`, so vLLM loads it unquantized and speculative decoding works out of the box. Quantizers silently drop this head — if you re-quantize the base yourself, graft it back or spec-decode will accept 0 %.

## Evaluation

**GSM8K: 96.8 %** (242/250). Served with vLLM tp2 and scored over the OpenAI API — thinking enabled, the base model's recommended sampling (`temperature=1.0, top_p=0.95, top_k=20`), default `reasoning_effort`, `max_tokens=24576`, and the question suffixed with "Put your final numeric answer inside `\boxed{}`". The INT4-W4A16 build of the same model scores 96.4 % (241/250) under the identical protocol — the two are indistinguishable at this sample size, i.e. INT8 buys no measurable accuracy over INT4 here.

Also verified over the API on 2 × RTX 3090 (tp2): Chinese/English generation and translation coherent, **vision path working** (image input through the native VL tower), `reasoning_effort` and `enable_thinking` passthrough, and MTP speculative decoding at mean accept length ≈ 2.5 / 3.0 with K=2. MMLU-Pro has not been run.

## Throughput

2 × RTX 3090, tensor-parallel 2, single stream, 16k context, prefix caching disabled, MTP K=2:

| Build | decode | prefill |
|---|---|---|
| **INT8-W8A8** (this) | 73 tok/s | **4408 tok/s** |
| INT4-W4A16 | **90 tok/s** | 2276 tok/s |

INT8 weights are twice the bytes of INT4, so decode — which is weight-bandwidth-bound — is slower; the INT8 activation path makes prefill much faster.

## Usage (vLLM)

```bash
vllm serve Avesed/Qwen3.8-27B-INT8-W8A8 \
  --tensor-parallel-size 2 --trust-remote-code --reasoning-parser qwen3
```

Add `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` to use the MTP head.

Sampling follows the base model: thinking `temperature=1.0, top_p=0.95, top_k=20`; non-thinking `temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5`.

Quantized with [vllm-ampere-optimized/quantize](https://github.com/avesed/vllm-ampere-optimized/tree/main/quantize).
