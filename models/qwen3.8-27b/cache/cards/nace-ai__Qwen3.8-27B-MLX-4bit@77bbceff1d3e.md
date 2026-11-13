---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: mlx
tags:
- kapitsa
- apple-silicon
- quantized
---

# Qwen3.8-27B, 4-bit MLX, for Kapitsa

Qwen3.8-27B in 4-bit MLX, run by [Kapitsa](https://github.com/nace-ai/kapitsa), the local inference engine from [Nace.AI](https://www.nace.ai/) for Apple silicon Macs.

| Writes | Reads a prompt | Memory | Smallest Mac measured | Measured on |
| :---: | :---: | :---: | :---: | :---: |
| **144 tok/s** | **1,126 tok/s** | **20.8 GiB** | **M5 Pro, 48 GB** | MacBook Pro M5 Max, 128 GB |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-4bit/resolve/main/assets/speed-dark.svg">
  <img alt="Bar chart of Qwen3.8-27B writing speed in tokens per second. M5 Pro 48 GB: Kapitsa 76.6, LM Studio 17.2, MLX 29.7, Ollama 44.0, llama.cpp 32.2; M5 Max 128 GB: Kapitsa 144, MLX 56.3, Ollama 71.4, llama.cpp 47.6." src="https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-4bit/resolve/main/assets/speed-light.svg">
</picture>

## Run it

```bash
brew install nace-ai/tap/kapitsa
# chat page and API at http://127.0.0.1:8000
kapitsa serve --model nace-ai/Qwen3.8-27B-MLX-4bit
# or start the opencode coding agent on it
kapitsa opencode --model nace-ai/Qwen3.8-27B-MLX-4bit
```

## Fit

| Variant | This repo's download | Smallest Mac measured | Memory in use at peak |
| --- | ---: | ---: | ---: |
| mlx-4bit-mtp | 16.3 GB | 48 GB | 20.8 GiB |

## Speed

| Mac | Chip | Memory | Writes (tok/s) | Reads a 4K prompt (tok/s) | Peak memory (GiB) | Kapitsa |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MacBook Pro | M5 Max | 128 GB | 144 | 1,126 | 20.8 | `87d7dd5` |
| MacBook Pro | M5 Pro | 48 GB | 76.6 | 749 | 16.6 | `d132a2cc` |

| How measured | |
| --- | --- |
| Writes | a 512-token coding answer at temperature 0.6 |
| Reads | a cold prompt of about 4K tokens |
| Peak memory | the server's peak while running both |
| Prompt | Kapitsa's drafting speed depends on the text, so another prompt gives another number |
| Measured | 10 November 2026 or earlier; each row's build is in the last column of the Speed table |

## Against other engines

| Engine (M5 Max, same 4-bit model) | Version | What ran | Writes (tok/s) | Kapitsa is | Measured |
| --- | --- | --- | ---: | ---: | --- |
| **Kapitsa** | `87d7dd5` | MLX 4-bit, MTP head, KFlash (default) | **144** | | 2026‑11‑05 |
| Ollama (fastest setup) | 0.40.11 | MLX runner, 4-bit, MTP | 71.4 | 2.0x faster | 2026‑10‑25 |
| MLX (fastest setup) | mlx-vlm 0.7.10 | mlx-vlm, MLX 4-bit, MTP | 56.3 | 2.6x faster | 2026‑10‑25 |
| llama.cpp (fastest setup) | b11871 | GGUF Q4_K_M, MTP, n-max 3 | 47.6 | 3.0x faster | 2026‑10‑25 |

## Quality

| Test | This variant | Reference | Verdict |
| --- | ---: | ---: | --- |
| GSM8K (100 questions) | 0.940 (n=100) | bf16 0.940 (n=100) | matches |

## Files

| Variant | Format | Size | Repo |
| --- | --- | ---: | --- |
| `mlx-3bit-mtp` | mlx 3-bit | 12.9 GB | [nace-ai/Qwen3.8-27B-MLX-3bit](https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-3bit) |
| `mlx-4bit-mtp` | mlx 4-bit | 16.3 GB | [nace-ai/Qwen3.8-27B-MLX-4bit](https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-4bit) |
| `mlx-8bit-mtp` | mlx 8-bit | 29.8 GB | [nace-ai/Qwen3.8-27B-MLX-8bit](https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-8bit) |
| `gguf-q2_k` | gguf 2.96-bit | 10.9 GB | [nace-ai/Qwen3.8-27B-GGUF](https://huggingface.co/nace-ai/Qwen3.8-27B-GGUF) |
| `gguf-q3_k_m` | gguf 3.91-bit | 13.5 GB | [nace-ai/Qwen3.8-27B-GGUF](https://huggingface.co/nace-ai/Qwen3.8-27B-GGUF) |
| `gguf-q4_k_m` | gguf 4.85-bit | 16.8 GB | [nace-ai/Qwen3.8-27B-GGUF](https://huggingface.co/nace-ai/Qwen3.8-27B-GGUF) |
| `gguf-q8_0` | gguf 8.5-bit | 29.1 GB | [nace-ai/Qwen3.8-27B-GGUF](https://huggingface.co/nace-ai/Qwen3.8-27B-GGUF) |
| `nvfp4` | nvfp4 4-bit | 20.6 GB | [nace-ai/Qwen3.8-27B-NVFP4](https://huggingface.co/nace-ai/Qwen3.8-27B-NVFP4) |

## License

apache-2.0, as the base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
