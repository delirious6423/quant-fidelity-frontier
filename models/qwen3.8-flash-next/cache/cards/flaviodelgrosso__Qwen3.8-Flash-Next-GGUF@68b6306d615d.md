---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- qwen4exp
- text-generation
---

# Qwen3.8 Flash Next Q4-D6 for Razor

[Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized to run fast and fully in memory on a single 128 GB Mac with Razor.

> [!WARNING]
> **Razor is not open source yet.** The runtime this file is built for has not been publicly released. You can download the weights now, but you can't run them with Razor until it is published.

## File

**`Qwen3.8-Flash-Next-Q4-D6.gguf`** (175 GB)

- The expert weights use 4 bits and the rest of the model uses about 6 bits. That makes generation about 30% faster than a plain Q4 build, with quality the same within measurement noise.
- It includes the built-in draft head for speculative decoding, so no separate draft model is needed.
- About 69 GiB of it is loaded into memory. The rest is a large lookup table that stays on disk.

## Requirements

- Apple Silicon Mac with 128 GB of unified memory. Razor was developed and tested on an M5 Max.
- macOS 15 or later.
- About 175 GB of free disk space.

## Run

Download:

```sh
hf download flaviodelgrosso/qwen3.8-flash-next-gguf Qwen3.8-Flash-Next-Q4-D6.gguf --local-dir ~/.razor/models
```

Serve, once Razor is available:

```sh
razor serve --model ~/.razor/models/Qwen3.8-Flash-Next-Q4-D6.gguf --speculation opportunistic
```

Razor serves OpenAI- and Anthropic-compatible APIs, so coding agents and chat clients can connect to it directly.

## Speed

Measured on an Apple M5 Max with 128 GB:

| Setup | Tokens/s |
|---|---:|
| Single session | ~68 |
| Single session, speculative decoding | ~95 |
| 8 sessions together | ~159 |

## License

[Qwen Community License 1.0](LICENSE), as upstream. Commercial use is subject to its conditions.
