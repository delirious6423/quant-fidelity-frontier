---
library_name: mlx
license: other
license_name: gemma-terms-of-use
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
tags:
- mlx
- mlx-lm
- safetensors
- quantized
- 3-bit
- gemma4
base_model: google/gemma-4-26B-A4B-it
---

# Gemma 4 26B-A4B IT MLX 3-bit

This repository contains a 3-bit MLX quantization of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it), optimized for local text generation on Apple Silicon.

It was created for experiments with fully local dictation cleanup and formatter-style workloads, including use in GemmaFlow.

## What this is

- **Format:** MLX safetensors
- **Base model:** `google/gemma-4-26B-A4B-it`
- **Model type:** Gemma 4 26B-A4B mixture-of-experts language model
- **Quantization:** 3-bit affine quantization, group size 32
- **Router layers:** kept at 8-bit, group size 64
- **Repository size:** about 12.6 GB of safetensors
- **Observed local memory:** about 12.8 GB peak for a short `mlx-lm` generation test on Apple Silicon
- **Primary use case:** local text generation, rewriting, and dictation cleanup

This is **not** a fine-tune. It is a quantized derivative of the upstream Gemma 4 model.

## How this differs from the common MLX 4-bit version

The closest common reference point is [`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit).

| Feature | This repo | `mlx-community/gemma-4-26b-a4b-it-4bit` |
|---|---:|---:|
| Quantization | 3-bit affine, selected router layers at 8-bit | 4-bit |
| Safetensors size | ~12.6 GB | ~15.6 GB |
| HF pipeline | `text-generation` | `image-text-to-text` |
| Intended focus | Lower-memory local text generation | Higher-fidelity/multimodal MLX usage |
| Vision support | Not included/tested | Includes VLM-style processor metadata |

In practice, this 3-bit version is meant to reduce disk and memory pressure. The tradeoff is that it may lose some quality or stability compared with the 4-bit version. If you want maximum fidelity or multimodal/image input, use the 4-bit VLM-oriented model instead. If you want a smaller local text model for rewriting/dictation cleanup, this repo is the lighter option.

## Use with `mlx-lm`

Install `mlx-lm`:

```bash
pip install -U mlx-lm
```

Run a quick text-generation smoke test:

```bash
python -m mlx_lm generate --model nsalerni/gemma-4-26b-a4b-it-mlx-3bit --prompt "Rewrite this clearly: i was gonna say lets move the meeting to tuesday actually make that thursday" --max-tokens 80 --temperature 0.0
```

For dictation cleanup, use an instruction-style prompt:

```bash
python -m mlx_lm generate --model nsalerni/gemma-4-26b-a4b-it-mlx-3bit --prompt "You are a dictation cleanup engine. Rewrite the user's dictated text clearly. Output only the rewritten text. User text: i was gonna say lets move the meeting to tuesday actually make that thursday" --max-tokens 80 --temperature 0.0
```

## Use with GemmaFlow

In GemmaFlow, add this Hugging Face model path in the formatter model settings:

```text
nsalerni/gemma-4-26b-a4b-it-mlx-3bit
```

This model uses the Gemma 4 A4B/MoE architecture, so the app needs a GemmaFlow build with Gemma 4 MoE MLX support. Older builds that only support dense Gemma 4 text models may fail with errors about unhandled `experts` or `router` keys.

## Notes and limitations

- This repo is intended for local MLX text generation, not hosted inference.
- This is a quantized model, so outputs may differ from the original BF16 model and from 4-bit MLX variants.
- Short local tests showed good memory behavior for dictation-cleanup-style prompts, but you should benchmark your own target prompts and context lengths.
- The upstream Gemma license and acceptable use policy still apply.

## License

This model is a quantized derivative of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it). Please follow the [Gemma Terms of Use](https://ai.google.dev/gemma/docs/gemma_4_license) and the upstream model card.
