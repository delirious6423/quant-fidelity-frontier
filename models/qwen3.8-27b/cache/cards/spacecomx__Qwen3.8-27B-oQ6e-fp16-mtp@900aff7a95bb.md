---
language:
  - multilingual
license: apache-2.0
library_name: mlx
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - mlx
  - safetensors
  - qwen3_5
  - oq
  - quantized
  - 6-bit
  - multimodal
  - mtp
---

# Qwen3.8-27B-oQ6e-fp16-mtp

An oQ mixed-precision MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), packaged for Apple Silicon.

## Quantization

- **Base quantization:** 6-bit affine, group size 64
- **Selective higher precision:** the token embeddings, LM head, and selected linear-attention / MLP projections remain 8-bit; see `config.json` for the complete per-module configuration
- **Format:** MLX safetensors
- **MTP:** retained (`mtp_num_hidden_layers: 1`); speculative decoding requires runtime support

## Chat template

`chat_template.jinja` uses the [Froggeric Qwen Fixed Chat Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) template, version `qwen3.8-froggeric-v22.2`.

This is an intentional behavioral change from the upstream Qwen template. It adds compatibility handling for system/developer messages, thinking-history fields, tool-call serialization, and tool responses. It does **not** alter the model weights or improve the underlying model capability. The template is intended for compatible Jinja-capable runtimes such as oMLX/MLX, llama.cpp, LM Studio, and vLLM. Review the upstream template project for its supported options and runtime-specific setup.

## Usage

Use a current MLX-compatible runtime that supports MLX safetensors, the quantization metadata in `config.json`, and the included `chat_template.jinja`.

oMLX discovers models from subdirectories of its configured model directory:

```bash
omlx serve --model-dir /path/to/models
```

The server exposes an OpenAI-compatible API on port `8000` by default:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B-oQ6e-fp16-mtp",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ]
  }'
```

For Qwen reasoning mode, use `temperature: 0.6`, `top_p: 0.95`, `top_k: 20`, and `min_p: 0`. For non-thinking mode, use `temperature: 0.7`, `top_p: 0.8`, `top_k: 20`, and `min_p: 0`.

Avoid greedy decoding while thinking is enabled; Qwen documents that it can cause repetitive output. The checkpoint retains its multi-token-prediction head, but speculative decoding requires runtime support.

The included `chat_template.jinja` is the Froggeric fixed template. Use a runtime that honors external Jinja chat templates and per-module quantization metadata. Compatibility outside oMLX, including `mlx-lm` and `mlx-vlm`, depends on the installed runtime version and has not been validated for this release.

## License and attribution

Apache 2.0, inherited from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). A copy of
the license is included as `LICENSE`. Quantization does not change the licensing terms — all
credit for the underlying model belongs to the Qwen team. This repository distributes a quantized derivative. The included
chat template is derived from the Apache-2.0-licensed [Froggeric Qwen Fixed Chat Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates).
