---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-serve
- qwen3_5
- apple-silicon
- vision-language
- mtp
- speculative-decoding
---

# Qwen3.8-27B, MLX 8-bit for mlx-serve

8-bit MLX conversion of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), built for **[mlx-serve](https://mlxserve.com)**, the native Zig MLX server for Apple Silicon ([mlxserve.com](https://mlxserve.com) · [github.com/ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve)).

31.2 GB on disk. Text, vision, tool calling and thinking all work, and the checkpoint's **native MTP head is included and served**, worth up to 3.3x on decode (numbers below).

Same recipe as the [4-bit build](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit), one knob apart. Pick this one if it fits; pick 4-bit for a 32 GB Mac.

## Which one

Measured on these weights, quantization noise per matmul-read layer (RMS error over the layer's own RMS, median across attention, MLP and lm_head at layers 0 / 20 / 42 / 63):

| | on-disk | weight noise | decode (MTP on) |
|---|---|---|---|
| 4-bit | 18.2 GB | **9.25%** | 75.3 tok/s |
| 8-bit | 31.2 GB | **0.71%** | 53.6 tok/s |

13x less noise for 1.7x the bytes and about 1.4x the decode time. On a Mac that holds it, 8-bit is the one to run.

## What mlx-serve is

A single native binary that runs any LLM on your Mac. No Python, no cloud, no Electron. It speaks **OpenAI-compatible and Anthropic-compatible HTTP** on the same port, so Claude Code, the OpenAI SDK, Continue, Cursor, Open WebUI and anything else that talks one of those wires just works against `http://localhost:11234`. Beyond text it also generates images, video, music, speech with voice cloning, and 3D models, all natively on MLX in the same server.

It ships with **MLX Core**, a signed and notarized macOS menu-bar app with chat, agent mode, MCP tool calling, model browsing and downloads.

- Site: **[mlxserve.com](https://mlxserve.com)**
- Source and releases: **[github.com/ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve)**
- Download the app: **[MLX Core.app](https://github.com/ddalcu/mlx-serve/releases/latest)**

## Speed on mlx-serve

M4 Max (128 GB), macOS 26.5, temperature 0, median of 3, taken from the server's own reported decode rate. This is mlx-serve against itself with the checkpoint's MTP head on and off, not a comparison against another engine:

| workload | MTP on (default) | MTP off (`--no-mtp`) | |
|---|---|---|---|
| write a class from scratch | **53.6 tok/s** | 16.4 tok/s | 3.27x |
| explanatory prose | **26.4 tok/s** | 16.4 tok/s | 1.61x |

Code is where multi-token prediction pays: indentation, closing brackets and repeated identifiers are all predictable enough to draft several ahead. Prose accepts fewer drafts and gains proportionally less. Drafted tokens are verified against the trunk, so the output is the model's own either way.

## Serving

```bash
mlx-serve --model ddalcu/Qwen3.8-27B-MLX-Serve-8bit --serve
```

Or pull it from the model browser in MLX Core and hit Load.

- **Thinking** is on by default. Turn it off per request with `"enable_thinking": false`, or tune the depth with `"reasoning_effort": "xhigh" | "medium" | "low"` (Qwen3.8's own vocabulary, default `xhigh`). Reasoning comes back as `reasoning_content` on both the OpenAI and Anthropic APIs.
- **Tools** use Qwen3.8's XML call format (`<tool_call><function=name><parameter=key>`). mlx-serve parses, repairs and schema-coerces it into standard OpenAI `tool_calls`.
- **Images** work on the chat APIs. Send `image_url` content as usual.
- **Context** is 262,144 tokens natively; mlx-serve sizes the window to your machine at load.

## Conversion

- Affine 8-bit, group size 64, on every matmul-read weight: attention q/k/v/o (the q projection carries the fused output gate), the MLPs, the GatedDeltaNet in/out projections, and `lm_head`.
- Kept bf16: `embed_tokens` (a gather-read table, not a matmul), the whole vision tower, `mtp.fc`, and every norm, bias, conv and SSM state.
- The MTP head ships with the model. mlx-serve finds it in the trunk shards and loads it with no flags.

Three things had to be fixed to make the raw checkpoint serve correctly on MLX, noted here because anyone converting this model themselves hits all three:

1. **The norms are delta-encoded.** Qwen3.8 stores the decoder layer norms, the final norm and q/k_norm zero-centered: the reference layer computes `1 + w`. The `+1` is baked into this conversion. `linear_attn.norm` and the vision tower are not delta-encoded and are left alone. Miss this and the model emits pure gibberish from the first token.
2. **The depthwise conv1d needs transposing.** PyTorch ships `[C, 1, K]`, MLX reads `[C, K, 1]`.
3. **The Conv3d patch embed is channels-last in MLX.** PyTorch's `[out, C, T, H, W]` becomes `[out, T, H, W, C]`. The shapes are the same size either way so nothing errors, the tower just reads the colour channels as spatial and describes a striped image. It still gets shapes and positions right, which makes this one easy to miss.

Weights, config, tokenizer and chat template are otherwise verbatim from the base repo.
