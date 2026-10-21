---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
tags:
- Qwen3.8-27B
- abliterated
- Uncensored
- heretic
- TurboQuant
- gguf
- IQ4_XXS
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B-ULTIMATE-UNCENSORED-MTP-IQ4-GGUF-16GB

** ♥♥♥♥ This model is specially prepared for your graphics card with 16GB VRAM. **

This model is the IQ4_XS GGUF quantization of [AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16](https://huggingface.co/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16) (an uncensored / abliterated build of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)), produced via mradermacher's i1 GGUF pipeline ([mradermacher/Qwen3.8-27B-heretic-ara-i1-GGUF](https://huggingface.co/mradermacher/Qwen3.8-27B-heretic-ara-i1-GGUF)). The **16G** variant is tuned to fit comfortably on a 16GB VRAM GPU (e.g. RTX 4060 Ti 16GB) when run with a TurboQuant KV cache.

## Innovation

This model refers to the fully optimized [Qwen3.8-27B-i1-IQ4_XS](https://huggingface.co/mradermacher/Qwen3.8-27B-heretic-ara-i1-GGUF), which has restored the `attn_qkv` layer to pure `IQ4_XS`. Furthermore, we introduce a novel hybrid precision quantization strategy: **the FFN layer uses `IQ3_S`**, which achieves significantly smaller file sizes while maintaining core inference capabilities through support from TurboQuant KV caching. Additionally, we use AEON-7's AEON-ULTIMATE-UNCENSORED abliterated version of the base model for quantization, making it convenient for users to conduct in-depth research.

## Motivation

The original `llama.cpp` quantization heuristics upgrade `attn_qkv` layers to `q5_K` under certain conditions (e.g., `n_gqa >= 4`), causing noticeable file bloat. cHunter789's fix restores `attn_qkv` to pure `IQ4_XS`, saving ~375 MiB.

Taking this further: **FFN layers (ffn_down, ffn_up, ffn_gate) account for ~2/3 of total model parameters**, yet they have higher redundancy than attention layers. Downgrading them from `IQ4_XS` to `IQ3_S` is a natural next step — it yields substantial size reduction with minimal quality impact, especially when attention layers (which dominate inference quality) remain at `IQ4_XS`.

## Methodology

1. **Base model**: [AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16](https://huggingface.co/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16) — an uncensored / abliterated build of Qwen3.8-27B, optimized for inference
2. **Quantization tool**: llama.cpp with TurboQuant support, built from [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant/releases/tag/tqp-v0.3.0) (release **tqp-v0.3.0**)
3. **Quantization types**:
   - `attn_qkv`, `attn_k`, `attn_v`, `attn_output`, `output`: `IQ4_XS`
   - `ffn_down`, `ffn_up`, `ffn_gate`: `IQ3_S`
   - Other layers: default `IQ4_XS`
4. **Importance matrix (imatrix)**: sourced from mradermacher's [Qwen3.8-27B-heretic-ara-i1-GGUF](https://huggingface.co/mradermacher/Qwen3.8-27B-heretic-ara-i1-GGUF)

### Quantization Commands

```bash
llama-quantize.exe ^
  --imatrix Qwen3.8-27B-heretic-ara.imatrix.gguf ^
  Qwen3.8-27B-heretic-ara-BF16.gguf ^
  Qwen3.8-27B-heretic-IQ4_XXS.gguf ^
  IQ4_XS 8 ^
  --tensor-type "blk.*.ffn_down" iq3_s ^
  --tensor-type "blk.*.ffn_up" iq3_s ^
  --tensor-type "blk.*.ffn_gate" iq3_s
```

> **Note on TurboQuant**: This model is recommended to be used with llama.cpp (https://github.com/TheTom/llama-cpp-turboquant) that supports TurboQuant KV caching (release **tqp-v0.3.0**). TurboQuant allows the KV cache to use a separate, more compact quantization format (turbo4 / turbo3), dramatically reducing memory usage even when the model weights themselves remain at IQ4_XS. Of course, it is also possible to use vllm or other inference frameworks that support TurboQuant technology, but the author used llama.cpp for the test.

## Memory Performance (with TurboQuant KV Cache)

| Version | Context | KV Cache | VRAM Usage |
| :--- | :--- | :--- | :--- |
| `IQ4_XXS` (baseline) | 100K-110K | turbo4 | **15.1-15.3 GB** |
| `IQ4_XXS` + MTP | 80K | turbo4 | **~15.3 GB** |

### Key Takeaways

- **IQ4_XXS** reaches 100K-110K context before VRAM saturation
- **IQ4_XXS + MTP** (speculative decoding via `draft-mtp`): ~80K context, higher throughput thanks to multi-token prediction

### Inference Speed

Tested on **NVIDIA RTX 4060 Ti 16GB**:

| Scenario | Speed |
| :--- | :--- |
| IQ4_XXS | **20 tokens/s** |
| `IQ4_XXS` + MTP | **30 tokens/s** |

### Vision Support (Optional)

This model supports vision input when paired with the Vision Modality Projector (`mmproj-BF16.gguf`). If you load the vision module, the available context window will be reduced by approximately **20K** to accommodate the vision encoder's memory footprint.

## Caveats

- **TurboQuant is mandatory**: This model relies on TurboQuant KV cache for the listed memory figures. Standard llama.cpp builds without TurboQuant will consume significantly more VRAM.
- **FFN layers at IQ3_S**: While the quality impact is expected to be minimal for most tasks, some degradation may be observable in tasks that heavily depend on FFN-related capabilities (e.g., certain factual recall scenarios). The attention layers remain at IQ4_XS to preserve core inference quality.
- **Verification pending**: The perplexity / VRAM / speed figures below are carried over from the same 27B-class IQ4_XS / IQ4_XS-FFN-IQ3_S methodology and should be re-validated specifically for Qwen3.8.


## How to Use

You need a **TurboQuant-enabled** llama.cpp build from [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant/releases/tag/tqp-v0.3.0) (release **tqp-v0.3.0**).

> **Note**: This model is quantized from an **abliterated / uncensored (AEON-ULTIMATE-UNCENSORED)** base model ([AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16](https://huggingface.co/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16)). The base model removes content restrictions for research and development purposes.

**Recommended run command (16GB VRAM, ~100K context):**

```bash

set TURBO_AUTO_ASYMMETRIC=0

llama-turboquant-3.0\llama-server.exe  -m Qwen3.8-27B-uncensored-abliterated-heretic-IQ4-GGUF-16G.gguf -b 2048 -ub 2048 -c 102400 -ngl 999 --flash-attn on  -ctk turbo4 -ctv turbo4 --chat-template-file chat_template.jinja --host 0.0.0.0 --port 1234

```

> **Note**: You must first use set TURBO-AUTO_SYMETRIC=0, otherwise KV will automatically increase to Q8_0.

> **Note**: For visual support, please refer to the loading command in Qwen3.6 27B, but the context needs to be reduced by 20K.

**Run command with MTP (speculative decoding, ~80K context):**

```bash
llama-turboquant-3.0\llama-server.exe  -m Qwen3.8-27B-uncensored-abliterated-heretic-IQ4-GGUF-16G.gguf -b 2048 -ub 2048 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 2 -c 81920 -ngl 999 --flash-attn on  -ctk turbo4 -ctv turbo4 --chat-template-file chat_template.jinja --host 0.0.0.0 --port 1234
```

> **About MTP**: Enabling `--spec-type draft-mtp --spec-draft-n-max 2` turns on multi-token prediction (speculative decoding) for higher throughput. Because the draft model shares part of the KV cache budget, the usable context window is reduced to ~80K (`-c 81920`) on a 16GB GPU.


## Acknowledgments

- **[AEON-7](https://huggingface.co/AEON-7)** — for the `Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16` uncensored / abliterated base model
- **[llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)** — provides release tqp-v0.3.0, supports Qwen3.8, the TurboQuant KV cache implementation
