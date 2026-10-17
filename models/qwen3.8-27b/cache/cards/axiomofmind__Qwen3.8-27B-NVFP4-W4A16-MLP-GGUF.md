---
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- gguf
- llama.cpp
- qwen3.8
- nvfp4
- modelopt
- multimodal
- mtp
- speculative-decoding
Number of Parameters: 27B
---

# Qwen3.8-27B Selective NVFP4 MLP GGUFs

Two selective NVFP4 conversions of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for `llama.cpp`: an accuracy-oriented BF16-attention build and a smaller Q8-attention build.

Both were independently calibrated from the official Qwen BF16 checkpoint using NVIDIA Model Optimizer. In both variants, the 64-layer backbone MLP uses W4A16 NVFP4 while embeddings, the output head, MTP weights, and vision components remain at higher precision. The variants differ in how their attention weights are stored.

## Files

| File | Purpose | Size |
| --- | --- | ---: |
| `Qwen3.8-27B-NVFP4-W4A16-MLP-BF16attn-MTP.gguf` | Accuracy-oriented variant with BF16/F32 attention and embedded MTP/NextN weights | 30.06 GB |
| `Qwen3.8-27B-NVFP4-MLP-Q8attn-BF16head-MTP.gguf` | Compact variant with Q8_0 attention, BF16/F32 output head, and embedded MTP/NextN weights | 23.29 GB |
| `mmproj-Qwen3.8-27B-BF16.gguf` | BF16 vision projector for image input | 0.93 GB |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | [Qwen3.8-27B DFlash 2 Q4_K_M drafter](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) | 1.14 GB |
| `qwen38_27b_nvfp4_w4a16_mlp_only.yaml` | ModelOpt recipe for the BF16-attention variant | — |
| `qwen38_27b_nvfp4_mlp_fp8_attn_w4a16.yaml` | ModelOpt recipe for the compact variant; calibrated FP8 attention weights are stored as Q8_0 in GGUF | — |

Either model GGUF works by itself for text. Add the same `mmproj` file for image input with either variant.

### Choosing a variant

| Variant | Best for | Trade-off |
| --- | --- | --- |
| BF16 attention, 30.06 GB | Maximum retained attention precision | Higher VRAM and storage use |
| Q8_0 attention, 23.29 GB | Lower VRAM and faster model loading | Attention weights use 8-bit block quantization |

## Quantization layout

In both files, `W4A16 MLP` means the large feed-forward weights use 4-bit NVFP4 while their inference activations remain 16-bit. The `gate_proj`, `up_proj`, and `down_proj` weights were quantized in each of the 64 backbone layers, producing 192 NVFP4 tensors.

The 30.06 GB variant retains the attention paths at BF16/F32. For the 23.29 GB variant, softmax-attention and the large linear-attention projections were calibrated as FP8 by Model Optimizer and stored as Q8_0 during GGUF conversion. FP8 activation scales are not used by `llama.cpp`; `Q8attn` describes the final GGUF weight storage.

| Component | BF16-attention variant | Q8-attention variant |
| --- | --- | --- |
| Backbone MLP `gate_proj`, `up_proj`, `down_proj` | NVFP4 W4A16 | NVFP4 W4A16 |
| Self-attention Q/K/V/O weights | BF16/F32 | Q8_0 |
| Large linear-attention projection weights | BF16/F32 | Q8_0 |
| Smaller linear-attention state/conv components | BF16/F32 | BF16/F32 |
| Token embeddings and `lm_head` | BF16/F32 | BF16/F32 |
| MTP/NextN weights | BF16/F32, embedded | BF16/F32, embedded |
| Vision tower/projector | Shared BF16 `mmproj` | Shared BF16 `mmproj` |
| KV cache | Selected at runtime | Selected at runtime |

## Running with embedded MTP

A recent CUDA build of `llama.cpp` with native NVFP4 and Qwen3.8 support is required. The following starts the built-in web UI and uses the MTP/NextN weights already contained in the selected model GGUF:

```bash
llama-server \
  -m Qwen3.8-27B-NVFP4-W4A16-MLP-BF16attn-MTP.gguf \
  --mmproj mmproj-Qwen3.8-27B-BF16.gguf \
  --ctx-size 32768 \
  --flash-attn on \
  --n-gpu-layers -1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.75 \
  --reasoning on \
  --reasoning-effort xhigh \
  --reasoning-preserve \
  --jinja \
  --ui
```

For the compact variant, replace the model filename after `-m` with `Qwen3.8-27B-NVFP4-MLP-Q8attn-BF16head-MTP.gguf`; all other options remain the same. Remove `--mmproj` for text-only use. Increase context size according to available VRAM.

### Thinking controls

Qwen3.8 uses thinking mode by default. Its default reasoning effort is `xhigh`, and preserved thinking is enabled by default. The launch command above makes those defaults explicit.

| llama.cpp flag | Effect |
| --- | --- |
| `--reasoning on` | Enable thinking |
| `--reasoning off` | Disable thinking for direct responses |
| `--reasoning auto` | Let the model's chat template select the mode |
| `--reasoning-effort xhigh` | Thorough reasoning; Qwen3.8 default |
| `--reasoning-effort medium` | Balance reasoning depth and speed |
| `--reasoning-effort low` | Prefer shorter, faster reasoning |
| `--reasoning-budget N` | Limit thinking to `N` tokens; `-1` is unrestricted |
| `--reasoning-preserve` | Retain reasoning context across conversation history |
| `--no-reasoning-preserve` | Do not retain earlier reasoning blocks |

`--reasoning-format` only controls how thinking is returned by the API. Its choices include `none`, `deepseek`, and `deepseek-legacy`; the default `auto` mode normally selects the appropriate parser.

## Recommended DFlash 2 acceleration

These GGUFs can also be used as targets for the separate [Qwen3.8-27B DFlash 2 Q4_K_M drafter](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF). The drafter is not required to run either model.

DFlash 2 requires a compatible `llama.cpp` build; this combination was tested using [DFlash 2 PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342).

```bash
llama-server \
  -m Qwen3.8-27B-NVFP4-W4A16-MLP-BF16attn-MTP.gguf \
  -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.8-27B-BF16.gguf \
  --ctx-size 32768 \
  --flash-attn on \
  --n-gpu-layers -1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --spec-type draft-dflash \
  --spec-draft-n-max 7 \
  --spec-draft-ngl all \
  --cache-type-k-draft q8_0 \
  --cache-type-v-draft q8_0 \
  --reasoning on \
  --reasoning-effort xhigh \
  --reasoning-preserve \
  --jinja \
  --ui
```

One local CUDA run with the BF16-attention target at 32,768 context used the default `xhigh` thinking effort, the Q4_K_M drafter, and Q8_0 target/draft KV caches. It generated 4,628 tokens at 103.18 tokens/s, with 51.68% draft acceptance (3,625 of 7,014 draft tokens) and a mean accepted length of 4.62. This is a single-system result, not a universal performance claim; speed and acceptance vary with hardware, prompt, sampling, context, and runtime build.


## License and attribution

The source model is released under Apache-2.0. The upstream license is included as `LICENSE`. Use of these files remains subject to the source model's terms and applicable law.

- Source model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- NVIDIA Model Optimizer: [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- GGUF runtime and converter: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
