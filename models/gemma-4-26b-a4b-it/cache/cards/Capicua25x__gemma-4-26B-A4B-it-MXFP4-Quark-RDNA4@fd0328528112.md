---
license: gemma
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
tags:
- mxfp4
- quark
- rdna4
- vllm
- speculative-decoding
- mtp
- moe
- long-context
---

# Gemma-4-26B-A4B-it — MXFP4 (Quark) for RDNA4, 262k window, native MTP-3 drafter

> **Release 2026-09-03.** Every number below was measured on this artifact on the hardware named, in one gate window.
> Nothing here is a claim about other stacks or other cards. Read the τ² row before choosing it for agentic work.

MXFP4 (OCP microscaling) quantization of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
made with **AMD Quark**, served by the RDNA4 vLLM port
[`capicua25x/vllm-rocm-rdna4`](https://hub.docker.com/r/capicua25x/vllm-rocm-rdna4) on two Radeon AI PRO R9700 (gfx1201,
32 GB each). **License headline: the weights are under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms), a
permissive but proprietary license that Google can update and enforce, i.e. revocable in practice; it is not open
source. Derivatives carry the same terms and the use-restriction notice (`NOTICE`). The Apache-2.0 models on this image
are [Ornith-1.5](https://huggingface.co/Capicua25x/Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4) and
[Muse-Glimmer-30B](https://huggingface.co/Capicua25x/Muse-Glimmer-30B-MXFP4-Quark-RDNA4).**

Why this model got a quant: a 26B mixture-of-experts with 4B active parameters, a 262,144-token native window, Google's
own **MTP draft head** (`google/gemma-4-26B-A4B-it-assistant`, served unquantized) and a sliding-window layout that keeps
the KV cache small. On two 32 GB cards it runs the full window with room for ~2.8 concurrent 262k requests, and its
per-token cost makes it the fastest general model in this collection.

## What is quantized

- **MXFP4 (fp4 e2m1 weights, group 32, e8m0 shared scales)** on the 30 dense MLP blocks and on **all 128 × 30 experts**
  (11,520 expert projections).
- **Kept in bf16:** every attention projection (`q/k/v/o_proj`), the routers, the vision tower, `lm_head`, norms and
  embeddings.
- **Activations:** the Quark export declares dynamic MXFP4 input quantization globally. On RDNA4 the dense MLPs run
  through the port's MXFP4×fp8 WMMA kernel; the experts run **weight-only (bf16 activations)** — the shipped
  `config.json` says so explicitly (`layer_quant_config["*experts*"].input_tensors = null`), because vLLM has no MoE
  kernel for MXFP4 activations on this hardware and emulation would dequantize every forward.
- Export: Quark real-quantized weights, `pack_method: reorder`, `quant_method: quark`.

## Serving config — what differs from the raw Quark export

The shipped `config.json` is the raw export plus three serving fixes; without them vLLM 0.28 cannot load the model:

| fix | why |
|---|---|
| `layer_quant_config: {"*experts*": {…, "input_tensors": null}}` | no MoE kernel for MXFP4 activations on RDNA4 (see above) |
| `self_attn.v_proj` added to `exclude` for the five global-attention layers (5, 11, 17, 23, 29) | Gemma-4 `attention_k_eq_v` layers have no `v_proj` on disk; Quark's exclude list only named q/k/o, and vLLM's fused `qkv_proj` requires one scheme for all three shards |
| `global_head_dim: 512`, `num_global_key_value_heads: 2` on `text_config` | transformers ≥ 5.15 writes these as `per_layer_config`, which vLLM 0.28 does not read; without the flat keys the global layers are built with the sliding-layer shape and weight loading fails |

Two port-side fixes ship in image tag **`0.28.0-rdna4-rc13`** (port commit `f5995c9c7`): `gelu_tanh` allowed on the
unfused Triton MoE experts, and the RDNA4 fp8 MXFP4 kernel declining GEMMs whose K is not a multiple of 128 (this model's
dense `down_proj`, K = 1056 per TP rank) so they fall through to the weight-only kernel instead of failing at load.

## Serving (vLLM on RDNA4 — 2× Radeon AI PRO R9700, TP2)

```bash
docker run --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add video --group-add render --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface capicua25x/vllm-rocm-rdna4:0.28.0-rdna4-rc13 \
  serve Capicua25x/gemma-4-26B-A4B-it-MXFP4-Quark-RDNA4 --port 8011 --trust-remote-code --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 --max-model-len 262144 --attention-backend TRITON_ATTN --moe-backend triton_unfused \
  --enable-prefix-caching --max-num-seqs 32 --max-num-batched-tokens 8000 --max-cudagraph-capture-size 128 --skip-mm-profiling \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --speculative-config '{"model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":3}'
```

- Boot: 11.98 GiB per card, KV cache 727,358 tokens (2.77× concurrency at 262k), ~5 min to ready on a warm cache.
- Thinking is a chat-template kwarg: send `"chat_template_kwargs": {"enable_thinking": true}` (the template injects the
  think token); `reasoning_effort` is inert on vLLM. The `gemma4` reasoning parser separates `reasoning_content`.
- The MTP drafter is Google's bf16 assistant head; it is not part of this repo (vLLM pulls it from the Hub).

## Measured throughput (2× R9700 TP2, MTP-3, apexia concurrency-bench v3, tokens/s per user)

| shape | c1 | c4 | c16 |
|---|---|---|---|
| 6k-token prompts, 400 out | **85.3** (2.34 accepted/step) | 61.4 (232 agg) | 37.2 (488 agg) |
| short prompts, 400 out | **98.9** (2.48 accepted/step) | – | – |

## Quality (this artifact, this hardware)

| suite | result |
|---|---|
| IFEval (80 prompts, chat template, no thinking) | inst-strict **0.9297** · prompt-strict **0.8875** |
| τ²-bench telecom (114 tasks, c6, thinking on) | **0.4649** (53/114) — weak at multi-turn policy work; for comparison Qwen3.8-27B-FP8 on this image scores 0.9386 and Muse-Glimmer 0.8421 |
| WhatsApp order-agent eval (37 cases, tool calls + DB effects) | **36 / 37**, 13.2 s average per turn chain |
| SQL-analyst regression suite (156 tests, Spanish, tool-free SQL generation) | **155 / 156**, 0 warnings |

## Long context: what is verified here

Needle-in-a-haystack (3 depths per size), prompt tokens as counted by the server:

| haystack | depth 10% | depth 50% | depth 90% |
|---|---|---|---|
| ~100k (105,016 tokens) | ✅ 223 s | ✅ 218 s | ✅ 164 s |
| ~200k (209,533 tokens) | ✅ 859 s | ✅ 858 s | ✅ 636 s |

Times are cold-prefill latencies to the answer; warm (prefix-cached) turns are interactive. Prefill on this hardware is
the known cost of Gemma-4's head-512 global layers.

## Credits

Google DeepMind for Gemma 4 and its MTP assistant head; AMD for Quark; the vLLM project. The RDNA4 port, the quant and
the measurements are by Capicua25x.
