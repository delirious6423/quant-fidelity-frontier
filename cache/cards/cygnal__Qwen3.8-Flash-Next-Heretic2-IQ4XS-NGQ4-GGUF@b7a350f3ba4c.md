---
library_name: llama-cpp
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: image-text-to-text
tags:
- heretic
- uncensored
- decensored
- abliterated
- gguf
- iq4_xs
- qwen4exp
- strix-halo
- gfx1151
- vulkan
- llama-cpp
- llama-flash-next
- benchmark
- halogen
---

# Qwen3.8-Flash-Next-Heretic2-IQ4XS-NGQ4-GGUF

IQ4_XS quantization of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) abliterated with [Heretic](https://github.com/p-e-w/heretic) v1.3.0+custom ([fork](https://github.com/timrohrbaugh/heretic)), with NGQ4-compressed n-gram prediction table. Built and benchmarked on AMD Strix Halo (Ryzen AI Max+ 395, gfx1151, 128 GB unified).

## Files

| File | Size | Description |
|------|------|-------------|
| `Qwen3.8-Flash-Next-Heretic2-IQ4XS-NGQ4.gguf` | 92 GB | Main model (IQ4_XS, NGQ4-compressed PLE table) |
| `mmproj-Heretic2-BF16.gguf` | 866 MB | Vision projector (multimodal/image input) |

## Quantization

Quantized from the Heretic2 BF16 safetensors → Q8_0 intermediate → IQ4_XS using the [llama-flash-next](https://github.com/nicoboss/llama.cpp/tree/qwen4exp) fork of llama.cpp (build 10656, commit 035e22731). The NGQ4 step compresses the 54 GB per-layer n-gram prediction table to ~27 GB by lowering precision on the indexer's token embedding, reducing the model from 116 GB to 92 GB with no measurable accuracy impact through 32K context.

**Byte-verified against the Heretic2 BF16 checkpoint.**

### Tensor types

| Tensor class | Type | Notes |
|---|---|---|
| Expert weights (ffn_down/up/gate_exps, shared experts, hc layers) | IQ4_XS | Bulk of parameters |
| Dense/trunk (non-expert weights) | Q8_0 | Attention projections, etc. |
| Output projection (output.weight) | Q6_K | Higher precision for output |
| Token embeddings (token_embd.weight) | BF16 | Small separate embedding [2560, 248320] |
| Per-layer token embeddings | IQ4_NL | NGQ4 compression (was BF16 in uncompressed IQ4_XS) |
| Norms (1D tensors) | F32 | |
| Indexer projections (indexer.k_proj, q_proj) | Q8_0 | |

## How to run

Requires the [llama-flash-next](https://github.com/nicoboss/llama.cpp/tree/qwen4exp) fork (qwen4exp branch) — standard llama.cpp cannot load the qwen4exp architecture.

```bash
# Text only
./llama-server \
  -m Qwen3.8-Flash-Next-Heretic2-IQ4XS-NGQ4.gguf \
  --alias qwen38-flash-next-heretic2 \
  -ngl 99 -fa on -c 65536 -np 1 --jinja \
  --host 127.0.0.1 --port 8096 \
  -dev Vulkan0

# With vision
./llama-server \
  -m Qwen3.8-Flash-Next-Heretic2-IQ4XS-NGQ4.gguf \
  --mmproj mmproj-Heretic2-BF16.gguf \
  --alias qwen38-flash-next-heretic2 \
  -ngl 99 -fa on -c 65536 -np 1 --jinja \
  --host 127.0.0.1 --port 8096 \
  -dev Vulkan0
```

Thinking mode is optional — disable it with `"chat_template_kwargs": {"enable_thinking": false}` in API requests to use the full token budget for output.

## Benchmarks — HumanEval+ (EvalPlus)

164 problems, `enable_thinking: false`, `temperature: 0.0`, `max_tokens: 1024`.

| Variant | Engine | Base pass@1 | Plus pass@1 | Mean s/prob | Median s/prob |
|:---|:---:|:---:|:---:|:---:|:---:|
| **This model (Heretic2 IQ4_XS NGQ4)** | **llama-server** | **84.1%** | **79.3%** | **4.16** | **3.31** |
| Q6_K_v7_v7 | llama-server | 88.4% | 82.9% | 11.26 | 10.02 |
| Q6_K | llama-server | 88.4% | 81.7% | 6.43 | 5.63 |
| Base flash-next IQ4_XS NGQ4 | llama-server | 82.3% | 78.0% | 4.82 | 3.21 |
| Halogen Official w4b (native + overlay) | Halogen | 82.3% | 78.0% | 13.82 | 10.66 |
| Uncensored IQ4_XS NGQ4 | llama-server | 81.7% | 78.7% | 4.91 | 4.31 |
| **Halogen Heretic2 IQ4_XS (BYO, no overlay)** | **Halogen** | **65.2%** | **61.0%** | **4.32** | **3.62** |

This model offers the best accuracy-to-speed ratio in the IQ4_XS tier: 84.1% base at 4.16s per problem. Only the Q6_K variants score higher, but they require 129 GB (won't coexist with other workloads on 128 GB) and take 50–170% longer per response.

## Origin story: Why this model exists

This GGUF was originally built as part of an apples-to-apples comparison between [llama-server](https://github.com/nicoboss/llama.cpp/tree/qwen4exp) and [Halogen Flash Server](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next) on AMD Strix Halo. Halogen's prefill throughput is dramatically faster — 2–4x at short contexts, approaching 4x at 16K+ tokens — so the question was whether it could replace llama-server as the default inference engine.

The answer turned out to be no, because of a critical limitation in Halogen's BYO (bring-your-own) GGUF workflow:

### The overlay problem

Halogen's official checkpoint ships with a **quality overlay sidecar** (`qwen38-flash-next-w4b.overlay.hgn`, 2.4 GiB, 741 activation-aware re-quantized tensors). The official model with overlay scores 82.3% / 78.0% on HumanEval+ — competitive with llama-server. But the overlay is built through an internal activation-aware quantization pipeline that isn't publicly available. There's no `halogen-quantize --build-overlay` command.

When you bring your own GGUF — any fine-tune, any abliteration, any custom variant — you run without the overlay. The result: **65.2% / 61.0%**, a 17-point collapse from the official model, using the exact same Heretic2 weights that score 84.1% on llama-server.

### The thinking mode problem

Halogen forces thinking mode on Qwen3.8-Flash-Next with no way to disable it. Every request burns ~1,049 hidden reasoning tokens before producing visible output. With `max_tokens=2048`, that leaves ~999 tokens for code. The official model with overlay compensates by producing better code in fewer tokens. Without the overlay, the BYO model truncates — 68% of HumanEval+ failures were SyntaxErrors from incomplete code that the model reported as `finish_reason=stop`.

llama-server doesn't have this problem: thinking mode is optional via `enable_thinking: false`, and the full token budget goes to output.

### The outcome

This Heretic2 IQ4_XS GGUF — originally built just for comparison — turned out to be a better model for production deployment than either the Halogen BYO setup (fast but 65% accuracy) or the Halogen official (82% accuracy but 3x slower per problem due to forced thinking). It runs on llama-server with full control over thinking mode, optional vision, and 84.1% HumanEval+ accuracy at 4.16s per problem.

**For the full technical assessment, see the companion write-up:** [Halogen on Strix Halo: Fast but Not Ready](https://huggingface.co/cygnal/Qwen3.8-Flash-Next-Heretic2-IQ4XS-NGQ4-GGUF/blob/main/halogen-assessment.md)

## Abliteration details

The Heretic2 abliteration was performed using [Heretic](https://github.com/p-e-w/heretic) v1.3.0+custom ([fork by timrohrbaugh](https://github.com/timrohrbaugh/heretic)) on the base [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) weights. Direction-removal method with per-layer direction index.

| Metric | Heretic2 | Original |
|:---|:---:|:---:|
| KL divergence | 0.0818 | 0 |
| Refusals (100 prompts) | 0/100 | 99/100 |

The abliteration has no measurable impact on code generation accuracy — Heretic2 IQ4_XS scores 84.1% vs base IQ4_XS at 82.3%, and the plus tests are within noise (79.3% vs 78.0%).

## Hardware tested

- **CPU:** AMD Ryzen AI Max+ 395
- **GPU:** AMD Radeon 8060S (gfx1151, RDNA 4 integrated)
- **Memory:** 128 GB unified (DDR5)
- **Backend:** Vulkan (ROCm also works but is slower for this architecture)
- **OS:** Ubuntu

## Throughput on Strix Halo (llama-server, Vulkan)

### Prefill (IQ4_XS standard, same architecture)

| Context | tok/s |
|:---:|:---:|
| 512 | 312 |
| 2K | 382 |
| 8K | 363 |
| 16K | 327 |

### Decode

~24 t/s serial (memory-bandwidth limited on unified memory).

## License

[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) (inherited from the base model).

## Credits

- **Base model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Alibaba Qwen team
- **Abliteration:** [Heretic](https://github.com/p-e-w/heretic) by p-e-w, [custom fork](https://github.com/timrohrbaugh/heretic) by timrohrbaugh
- **Quantization fork:** [llama-flash-next](https://github.com/nicoboss/llama.cpp/tree/qwen4exp) (qwen4exp branch) by nicoboss
- **Quantization & benchmarking:** [cygnal](https://huggingface.co/cygnal)
