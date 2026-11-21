---
license: mit
library_name: transformers
pipeline_tag: text-generation
base_model: zai-org/GLM-5.3-Flash
tags:
- glm
- moe
- int4
- fp8
- w4a8
- compressed-tensors
- auto-round
- sglang
---

# GLM-5.3-Flash W4AFP8 (AutoRound weights)

A 4-bit build of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B total / 18B active MoE) that fits on **2× H200** and runs the routed experts as **INT4 weights × FP8 activations** on Hopper tensor cores.

- **Routed-expert weights:** INT4, symmetric, group size 128. These are the weights of an [intel/auto-round](https://github.com/intel/auto-round) INT4 quantization of GLM-5.3-Flash (`--iters 200`, default calibration set), **repacked without any change in value** from the `auto_gptq` packing into the `compressed-tensors` `pack-quantized` format.
- **Activations:** FP8 E4M3, dynamic per token (no calibration needed).
- **Kept in 16-bit:** attention / indexer, shared experts, router, MLP of layers 0–2, embeddings, `lm_head`, and the non-expert parts of the MTP layer (its experts are INT4).
- **Size:** ~168 GB.

## Repack details

For every quantized linear layer:

| AutoRound (`auto_gptq`) | this repo (`compressed-tensors`) |
|---|---|
| `qweight` `int32 [K/8, N]` | `weight_packed = qweight.T` `int32 [N, K/8]` |
| `scales` `fp16 [K/128, N]` | `weight_scale = scales.T` `bf16 [N, K/128]` |
| `qzeros` (constant 8, symmetric) | dropped (`symmetric: true`) |
| – | `weight_shape = [N, K]` |

Dequantized values are bit-identical to the AutoRound checkpoint (verified per tensor). Relative weight error against the official FP8 checkpoint on sampled experts: `gate_proj` ≈ 0.102, `down_proj` ≈ 0.103 (round-to-nearest INT4 at the same group size: 0.111 and 0.118–0.185).

## Serving with SGLang (2× H200)

Tested with the SGLang nightly image `lmsysorg/sglang:11.11.26`.

```bash
python3 -m sglang.launch_server \
  --model-path beyoru/GLM-5.3-Flash-W4AFP8-AutoRound \
  --trust-remote-code --tp-size 2 \
  --mem-fraction-static 0.88 --context-length 131072 --chunked-prefill-size 16384 \
  --kv-cache-dtype bfloat16 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --disable-shared-experts-fusion \
  --mamba-full-memory-ratio 2.0 \
  --reasoning-parser glm45 --tool-call-parser glm47
```

Required / important flags:

- **`--disable-shared-experts-fusion` is required.** Without it SGLang fuses the BF16 shared expert into the INT4 routed-expert kernel and the output is garbage from the first token.
- The MoE runs through `CompressedTensorsW4AFP8MoE` (CUTLASS W4A8) by default.
- **Faster: Humming kernels** (`--quantization humming --moe-runner-backend humming`, SGLang with humming ≥ 0.1.18). humming 0.1.18 rejects `"group_size": null` in `input_activations`; set it to `0` (per token) in `config.json` for this backend.

### Thinking / non-thinking

The official GLM-5.3-Flash template always opens `<think>` and sets `Reasoning Effort: Max` by default. To serve without thinking, end the generation prompt with `<think></think>` **and** lower the effort (e.g. `reasoning_effort: "low"`), then pass `--default-chat-template-kwargs '{"enable_thinking": false}'` so the reasoning parser does not treat the answer as reasoning. With `<think></think>` but effort left at Max, the model occasionally still thinks and closes with a stray `</think>`.

## Throughput (2× H200, TP2)

Synthetic long-context workload, open loop: prompt p50 ≈ 14K / p90 ≈ 34K tokens, output p50 ≈ 50 tokens, ~25–30 % prefix-cache hits, non-thinking.

| Configuration | 1.0 req/s: TTFT p50 / p95 | TPOT p50 | 1.5 req/s: TTFT p95 |
|---|---|---|---|
| W4A16 (AutoRound, Marlin MoE) | 8.0 / 18.7 s | 312 ms | – (saturated) |
| W4AFP8, CUTLASS MoE | 2.5 / 6.4 s | 33 ms | 33 s |
| **W4AFP8, Humming MoE** | **1.6 / 4.4 s** | **17 ms** | **14 s** |
| Gemma-4-31B BF16 (reference, same GPUs) | 3.1 / 9.0 s | 136 ms | 52 s |

Single-stream decode with MTP: ≈ 175–210 tokens/s (2.8–3.2 accepted tokens per step).

## License and attribution

MIT, same as the base model ([zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)); see `LICENSE`. Expert weights come from an intel/auto-round INT4 quantization of that model. This repository only changes their storage format and the quantization config.
