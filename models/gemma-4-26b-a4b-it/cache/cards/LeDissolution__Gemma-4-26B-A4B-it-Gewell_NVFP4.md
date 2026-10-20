---
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
- nvidia/Gemma-4-26B-A4B-NVFP4
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- gemma
- gemma-4
- gewell
- nvfp4
- nvidia
- moe
---

# Gemma 4 26B A4B IT — Gewell NVIDIA NVFP4

[NVIDIA's Gemma 4 26B A4B NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4)
repacked for [Gewell](https://github.com/LeDissolution/gewell),
a single-GPU inference engine for NVIDIA Blackwell (`sm_120`/`sm_120a`).
Includes the BF16 vision tower and the
[official MTP assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant).

The routed experts use NVIDIA's NVFP4 weights and scales. Attention, shared
MLP, router, embeddings and normalization weights remain BF16.

## Contents

| File | Contents | Size |
|---|---|---:|
| `weights.gwt` | NVIDIA NVFP4 text weights in Gewell format | 17.19 GiB |
| `manifest.json` | Model and tensor metadata | 5.35 MiB |
| `tokenizer.json` | Tokenizer | 30.68 MiB |
| `vision.safetensors` | BF16 vision tower and projector | 1.07 GiB |
| `assistant.safetensors` | BF16 MTP assistant | 0.78 GiB |

## Download and run

[Build Gewell](https://github.com/LeDissolution/gewell/blob/main/docs/build.md),
then download the bundle:

```bash
MODEL_DIR=./gemma4-26b-nvfp4
hf download LeDissolution/Gemma-4-26B-A4B-it-Gewell_NVFP4 --local-dir "$MODEL_DIR"
```

Start the OpenAI-compatible server with image input and MTP enabled:

```bash
build/gewell \
  --vision "$MODEL_DIR/vision.safetensors" \
  --assistant "$MODEL_DIR/assistant.safetensors" \
  --mtp-depth 3 \
  serve-http \
  --model-dir "$MODEL_DIR" \
  --model gemma-4-26b-a4b \
  --max-batch 1 \
  --kv-cache-gpu-mib 8192
```

The API is available at `http://127.0.0.1:6311/v1`. This example fits a
96 GB RTX PRO 6000 Blackwell. Omit `--vision` for text-only serving;
omit `--assistant` and `--mtp-depth` to disable MTP.

See the [HTTP API](https://github.com/LeDissolution/gewell/blob/main/docs/http-api.md)
for requests, image input, tools and structured output.

The [BF16 bundle](https://huggingface.co/LeDissolution/Gemma-4-26B-A4B-it-Gewell_BF16)
is also available.

## License

Apache 2.0. Text weights come from NVIDIA's Gemma 4 26B A4B NVFP4;
vision and assistant weights come from Google Gemma 4 26B A4B IT and
Google Gemma 4 26B A4B IT Assistant.
