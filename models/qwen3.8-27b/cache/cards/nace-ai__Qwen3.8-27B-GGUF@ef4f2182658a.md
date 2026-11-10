---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
tags:
- kapitsa
- apple-silicon
- quantized
---

# Qwen3.8-27B, Q4_K_M GGUF, for Kapitsa

Qwen3.8-27B in Q4_K_M GGUF, run by [Kapitsa](https://github.com/nace-ai/kapitsa), the local inference engine from [Nace.AI](https://www.nace.ai/) for Apple silicon Macs.

| Writes | Reads a prompt | Memory | Smallest Mac measured | Measured on |
| :---: | :---: | :---: | :---: | :---: |
| **52.3 tok/s** | **395 tok/s** | **20.7 GiB** | **M5 Pro, 48 GB** | MacBook Pro M5 Pro, 48 GB |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/nace-ai/Qwen3.8-27B-GGUF/resolve/main/assets/speed-dark.svg">
  <img alt="Bar chart of Qwen3.8-27B writing speed in tokens per second. M5 Pro 48 GB: Kapitsa 52.3." src="https://huggingface.co/nace-ai/Qwen3.8-27B-GGUF/resolve/main/assets/speed-light.svg">
</picture>

## Run it

Download one `.gguf` file from this repo (see Files), then:

```bash
brew install nace-ai/tap/kapitsa
# converts on first use; chat page and API at http://127.0.0.1:8000
kapitsa serve --model /path/to/file.gguf
```

## Fit

| Variant | This repo's download | Smallest Mac measured | Memory in use at peak |
| --- | ---: | ---: | ---: |
| gguf-q4_k_m | 16.8 GB | 48 GB | 20.7 GiB |

## Speed

| Mac | Chip | Memory | Writes (tok/s) | Reads a 4K prompt (tok/s) | Peak memory (GiB) | Kapitsa |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MacBook Pro | M5 Pro | 48 GB | 52.3 | 395 | 20.7 | `d132a2cc` |

| How measured | |
| --- | --- |
| Writes | a 512-token coding answer at temperature 0.6 |
| Reads | a cold prompt of about 4K tokens |
| Peak memory | the server's peak while running both |
| Prompt | Kapitsa's drafting speed depends on the text, so another prompt gives another number |
| Measured | 10 November 2026 or earlier; each row's build is in the last column of the Speed table |

## Quality

| Test | This variant | Reference | Verdict |
| --- | ---: | ---: | --- |
| GSM8K (100 questions) | 0.960 (n=100) | bf16 0.940 (n=100) | matches |

## Files

| File | Quant | Size | Measured on this card | Note |
| --- | --- | ---: | :---: | --- |
| `qwen3.8-27b-Q2_K.gguf` | Q2_K | 10.9 GB | no | made with an importance matrix |
| `qwen3.8-27b-Q3_K_M.gguf` | Q3_K_M | 13.5 GB | no | made with an importance matrix |
| `qwen3.8-27b-Q4_K_M.gguf` | Q4_K_M | 16.8 GB | yes | made with an importance matrix |
| `qwen3.8-27b-Q8_0.gguf` | Q8_0 | 29.1 GB | no | made without an importance matrix |

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
