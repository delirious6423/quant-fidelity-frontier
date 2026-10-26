---
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
library_name: llama-cpp
tags:
- qwen3.8
- qwen3.8-27b
- qwen35
- nvfp4
- gguf
- llama-cpp
- multimodal
- vision
- hybrid attention
- linear-attention
language:
- multilingual
---

# Qwen3.8-27B-NVFP4 (GGUF)

NVFP4 GGUF quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), ready for [llama.cpp](https://github.com/ggerganov/llama.cpp) and other GGUF-compatible runtimes.

## Model Overview

Qwen3.8-27B is a native vision-language (image-text-to-text) causal language model with a separate vision encoder, built on the Qwen3.5 architectural foundation. Its defining trait is a **hybrid stack**: three Gated DeltaNet (linear/recurrent attention) layers for every one Gated Attention (full softmax attention) layer, each followed by a standard dense SwiGLU FFN.

| Property                              | Value                                                                                                                        |
| :------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------- |
| **Type**                              | Causal Language Model with Vision Encoder                                                                                    |
| **architecture**                      | `qwen35` (`general.architecture = qwen35`)                                                                                   |
| **Parameters**                        | 27B                                                                                                                          |
| **Layers**                            | 64                                                                                                                           |
| **Hidden dimension**                  | 5120                                                                                                                         |
| **Layer layout**                      | 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))                                                              |
| **Full-attention interval**           | every 4th layer (`full_attention_interval = 4`, 48 recurrent + 16 full-attention layers)                                     |
| **Gated DeltaNet (linear attention)** | 48 value heads, 16 QK heads, head dim 128; conv kernel 4; state size 128; group count 16; time-step rank 48; inner size 6144 |
| **Gated Attention (full attention)**  | 24 Q heads, 4 KV heads, head dim 256 (partial RoPE, rotary dim 64 = 25% of head dim)                                         |
| **RoPE**                              | interleaved M-RoPE, sections `[11, 11, 10]`, `rope_freq_base = 1e7`                                                          |
| **FFN**                               | dense SwiGLU, intermediate dim 17408 (gate/up/down per layer)                                                                |
| **Output gating**                     | swish-gated attention output (`attn_gate` present in every layer)                                                            |
| **Context length**                    | 262,144 tokens natively (upstream reports 1,000,000 via YaRN scaling in vLLM/SGLang/TokenSpeed)                              |
| **Vocabulary**                        | 248,320 tokens (padded), BPE (`tokenizer.ggml.pre = qwen35`), NFC normalization on                                           |
| **Special tokens**                    | BOS/PAD `248044`, EOS `248046`, image `248056`, video `248057`, vision start/end `248053`/`248054`                           |
| **RMS norm eps**                      | 1e-6                                                                                                                         |
| **Vision encoder**                    | 27 blocks, hidden 1152, 16 heads, patch 16, spatial merge 2, temporal patch 2, projects to 5120 (shipped in `mmproj.gguf`)   |

## Files

| File                     |  Size | Notes                 |
| :----------------------- | ----: | :-------------------- |
| `qwen3.8-27b-nvfp4.gguf` | 15 GB | NVFP4 quantized model |
| `mmproj.gguf`            | 1 GiB | image/video input     |

Text-only runs work without `mmproj.gguf`; multimodal input needs it.

## Usage

### llama-cli

```bash
# Text-only
./llama-cli \
  -m qwen3.8-27b-nvfp4.gguf \
  -c 32768 \
  --temp 1.0 --top-k 20 --top-p 0.95 \
  -p "Explain quantum computing in simple terms."

# With vision
./llama-cli \
  -m qwen3.8-27b-nvfp4.gguf \
  --mmproj mmproj.gguf \
  -i photo.jpg \
  --temp 1.0 --top-k 20 --top-p 0.95 \
  -p "Describe this image."
```

### llama-server (OpenAI-compatible API)

```bash
./llama-server \
  -m qwen3.8-27b-nvfp4.gguf \
  --mmproj mmproj.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 32768 \
  --temp 1.0 --top-k 20 --top-p 0.95
```

The chat template is embedded in the GGUF and accepts `image` / `video` content parts, so standard OpenAI `image_url` payloads work through `llama-server`.

## Generation Parameters

Defaults baked into the GGUF (`general.sampling.*`) and upstream `generation_config.json`: `temperature = 1.0`, `top_k = 20`, `top_p = 0.95`, `do_sample = true`.

Upstream-recommended settings:

| Mode                    | Temp | Top-P | Top-K | Min-P | Presence Pen. | Repetition Pen. |
| :---------------------- | :--- | :---- | :---- | :---- | :------------ | :-------------- |
| Thinking (default)      | 1.0  | 0.95  | 20    | 0.0   | 0.0           | 1.0             |
| Instruct (non-thinking) | 0.7  | 0.80  | 20    | 0.0   | 1.5           | 1.0             |

Raising `presence_penalty` toward 2.0 further suppresses repetition at the cost of occasional language mixing and a small quality drop.

## Thinking Mode

Thinking is **on by default**: the model emits a reasoning block delimited by `<think>\n…\n</think>\n\n` before the final answer.

- **Disable per request:** pass `enable_thinking: false` in the chat-template kwargs, e.g. `--chat-template-kwargs '{"enable_thinking": false}'` or via the API body.
- **Preserved thinking:** on by default — reasoning traces from earlier turns are retained in context for consistency across long agentic sessions. Turn off with `preserve_thinking: false`.
- **Reasoning effort:** upstream exposes `xhigh` (default), `medium`, `low`. Support depends on the serving layer, not the GGUF itself.
- Allow generous output length: upstream suggests up to 262,144 tokens for reasoning and 131,072 for the final answer within a 1M context.

## Key Features

- **Hybrid linear + full attention:** 48 Gated DeltaNet layers interleaved with 16 Gated Attention layers (1 : 3 ratio) gives near-linear scaling of recurrent state while keeping periodic full attention for retrieval quality.
- **Dense 27B, not MoE:** straightforward deployment — no expert offloading, routing, or load-balancing concerns.
- **Native vision-language:** images and video via the bundled `mmproj.gguf` projector (patch-16, 2×2 spatial merge, 2-frame temporal patch).
- **262K native context** with M-RoPE (`rope_freq_base = 1e7`), extensible upstream with YaRN.
- **Flexible reasoning control:** per-request thinking toggle, preserved thinking across turns, tunable reasoning effort.

## Acknowledgements

- Original model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — Qwen Team

```bibtex
@misc{qwen38,
    title  = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url    = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month  = {August},
    year   = {2026}
}
```
