---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- gguf
- deepseek41
- text-generation
- speculative-decoding
---

# DeepSeek V4.1 Flash 1017 for Razor

[DeepSeek V4.1 Flash 1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) packed to run locally on a single 128 GB Mac with Razor.

> [!WARNING]
> **Razor is not open source yet.** The runtime that loads these files has not been publicly released. You can download the weights now, but you can't run them until Razor is published.

> [!IMPORTANT]
> These files only run in Razor on Apple Silicon. llama.cpp and other GGUF runtimes can't load them.

## Files

- **`DeepSeek-V4.1-Flash-1017-TRL-R2.gguf`** (349.3 GB): the model. The routed experts are compressed to about 2 bits per weight, and everything else keeps the checkpoint's original precision.
- **`DeepSeek-V4.1-Flash-1017-DSpark.gguf`** (8.6 GB): an optional small draft model that speeds up generation through speculative decoding.

The model is larger than 128 GB of RAM. Razor streams the experts it needs from the internal SSD into a memory cache, so a fast internal SSD matters.

## Requirements

- Apple Silicon Mac with 128 GB of unified memory. Razor was developed and tested on an M5 Max.
- macOS 15 or later.
- About 360 GB of free disk space for both files.

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

Razor serves OpenAI- and Anthropic-compatible APIs, so coding agents and chat clients can connect to it directly.

## Quality

On 48 held-out prompts, this is how often the compressed model picks the same next token as the full-precision model:

| Category | Agreement |
|---|---:|
| Coding | 97.4 % |
| Reasoning | 96.1 % |
| Chat | 88.4 % |
| Multilingual | 78.9 % |

Code and reasoning are nearly lossless. Non-English text loses the most.

## Speed

Measured on an Apple M5 Max with 128 GB:

- **About 32 tokens/s** on ordinary text.
- **Up to about 58 tokens/s** with the DSpark draft model on predictable text. On short, varied prose it gives little or no gain.

## License

MIT, as upstream. The model weights are © DeepSeek ([license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE)).
