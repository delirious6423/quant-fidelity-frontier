---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- gguf
- deepseek41
- text-generation
- speculative-decoding
- image-text-to-text
- vision
---

# DeepSeek V4.1 Flash 1017 for Razor

[DeepSeek V4.1 Flash 1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) packed to run locally on a single 128 GB Mac with Razor.

> [!WARNING]
> **Razor is not open source yet.** The runtime that loads these files has not been publicly released. You can download the weights now, but you can't run them until Razor is published.

> [!IMPORTANT]
> These files only run in Razor on Apple Silicon. llama.cpp and other GGUF runtimes can't load them.

## Files

- **`DeepSeek-V4.1-Flash-1017-TRL-R2.gguf`** (349.3 GB): the model. The routed experts are compressed to about 2 bits per weight, with more bits on the experts that matter most for chat and non-English text; everything else keeps the checkpoint's original precision. SHA-256 `ced50f57dfd543f6272296828145bff15fc20abea01fba1b92b82758543805e6`.
- **`DeepSeek-V4.1-Flash-1017-DSpark.gguf`** (8.6 GB): an optional small draft model that speeds up generation through speculative decoding.
- **`DeepSeek-V4.1-Flash-1017-Vision.gguf`** (972 MB): an optional vision sidecar that lets the model see images. It holds the checkpoint's vision encoder and image-to-text projection, copied unchanged in BF16. SHA-256 `33c59a9dcaa9e4c6839b13c387858c30914401d68d1f205a6578957c84e1c2db`.

The model is larger than 128 GB of RAM. Razor streams the experts it needs from the internal SSD into a memory cache, so a fast internal SSD matters.

## Requirements

- Apple Silicon Mac with 128 GB of unified memory. Razor was developed and tested on an M5 Max.
- macOS 15 or later.
- About 360 GB of free disk space for all three files.

## Run

Download:

```sh
hf download flaviodelgrosso/deepseek-v4.1-flash-1017-gguf --local-dir ~/.razor/models
```

Serve, once Razor is available:

```sh
razor serve --model ~/.razor/models/DeepSeek-V4.1-Flash-1017-TRL-R2.gguf \
  --speculation opportunistic --dspark ~/.razor/models/DeepSeek-V4.1-Flash-1017-DSpark.gguf
```

Add `--vision ~/.razor/models/DeepSeek-V4.1-Flash-1017-Vision.gguf` to enable image input.

Razor serves OpenAI- and Anthropic-compatible APIs, so coding agents and chat clients can connect to it directly.

## Images

With the vision sidecar loaded, user messages can include PNG or JPEG images: OpenAI `image_url` and Responses `input_image` parts with base64 data URLs, Anthropic `image` blocks with a base64 source, or `/image <path>` in `razor chat`. Remote image URLs are not fetched.

Images are prepared the way DeepSeek's reference code does it, and each one takes up to 1,024 prompt tokens. Razor's image encoder matches DeepSeek's reference code to a cosine similarity of at least 0.999. Encoding a large image takes about 1.6 s on an M5 Max. Follow-up questions about the same image reuse the earlier work instead of encoding it again.

## Quality

On 48 held-out prompts, this is how often the compressed model picks the same next token as the full-precision model:

| Category | Agreement |
|---|---:|
| Coding | 97.8 % |
| Reasoning | 96.0 % |
| Chat | 88.2 % |
| Multilingual | 79.8 % |

Code and reasoning are nearly lossless. Non-English text loses the most.

## Speed

Measured on an Apple M5 Max with 128 GB (2,048-token prose prompt, 256 generated tokens, automatic expert cache):

- **About 36 tokens/s** on ordinary text (27.3–27.5 ms per token) once the expert cache has warmed up; about 34 tokens/s on the first request after loading, while the cache still misses experts.
- **71–79 tokens/s** with the DSpark draft model on predictable text. On short, varied prose it gives little or no gain.

Sustained generation runs a few percent slower as the Mac heats up.

## License

MIT, as upstream. The model weights are © DeepSeek ([license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE)).
