---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- sushi
- exl3
- apple-silicon
- qwen4_exp
- moe
---

<p align="center"><img src="assets/sushi-horizon.png" alt="sushi" width="100%"></p>

# Qwen3.8-Flash-Next-Sushi-4bpw

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) packed for
[sushi](https://github.com/beamivalice/sushi), a native inference engine for Apple Silicon.
**Target: a 96 GB Mac.** It does not fit on a 64 GB Mac; use
[Sushi-3bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-3bpw) there.

| part | how it is stored |
|---|---|
| routed experts (48 layers x 512, plus the MTP layer) | EXL3, made with our own converter |
| dense part (attention, GDN, shared experts, hyper-connections, indexer, embedding, lm_head) | 8-bit affine; the same in both packs |
| n-gram embedding table | bf16 as released, one file `ngram_table.bin`, read from the SSD and never loaded into GPU memory (Sushi-3bpw ships a 4-bit copy) |
| vision tower (Qwen3-VL ViT) | included: image and video input |
| MTP head | included |
| context | up to 1,048,576 tokens (`config.json` sets YaRN x4 over the native 262,144) |
| disk | ~159 GiB: 63.7 GiB of weights + 95.4 GiB n-gram table |

**sushi only.** This pack needs sushi v1.0.0 or later on Apple Silicon with macOS 26.2 or later. It does not load
in transformers, vLLM, mlx-lm or exllamav3.

How to run it (install, launch flags, memory and context per Mac): [sushi usage](https://github.com/beamivalice/sushi/blob/main/docs/usage.md).

## Quality

<p align="center"><img src="assets/sushi-4bpw-kld.png" alt="Sushi-4bpw KLD" width="100%"></p>

<p align="center"><img src="assets/kld-chart.png" alt="KLD vs size" width="100%"></p>

KLD against the bf16 checkpoint: 16 prompts x 512 tokens, scored to the first EOS (7186 positions), KV cache at kv8,
every pack run by sushi: Sushi-3bpw and Sushi-4bpw on a later build than the rest, Sushi-2bpw on v1.0.4 with bf16 KV. Lower is better. The n-gram table is
not counted: it stays on the SSD.

| pack | weights in GPU memory (GiB) | KLD | top-1 agreement |
|---|---|---|---|
| **Sushi-4bpw** | **63.68** | **0.0592** | **92.89%** |
| oMLX oQ5e | 83.97 | 0.0625 | 92.40% |
| mlx-serve [mixed-4-8bit](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit) | 70.13 | 0.0818 | 91.39% |
| **Sushi-3bpw** (4-bit n-gram table) | **49.33** | **0.1036** | **90.31%** |
| Sushi-2.6bpw (4-bit n-gram table) | 43.95 | 0.1355 | 89.08% |
| oMLX oQ4e | 69.21 | 0.1370 | 88.87% |
| mlx-serve [iQ-MLX 3.3bpw](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw) | 50.60 | 0.1987 | 86.28% |
| Vontra 4-bit g32 (TensorFold) | 75.60 | 0.2074 | 85.01% |
| Sushi-2bpw (4-bit n-gram table) | 34.97 | 0.2080 | 85.94% |

At about the same memory, Sushi-4bpw has less than half oQ4e's KLD (0.059 vs 0.137), and it beats oQ5e (0.059 vs 0.063)
with 20 GiB less. At about 50 GiB, Sushi-3bpw has about half the KLD of mlx-serve's iQ-MLX 3.3bpw (0.104 vs 0.199).
Each pack is measured with the n-gram table it ships: 4-bit g32 in Sushi-2bpw, Sushi-2.6bpw, Sushi-3bpw and mlx-serve's
packs, 4- or 5-bit in oMLX's, bf16 in Sushi-4bpw. The oMLX and mlx-serve packs were measured as published; the oMLX
n-gram tables were repacked into sushi's file unchanged.

## Hard questions

<p align="center"><img src="assets/eval-card.png" alt="LLMProbe Hard 50: 90.0%" width="100%"></p>

LLMProbe's Hard 50 set (MMLU-Pro, OlympiadBench, LiveBench, NIST Juliet): 45 correct, 5 incorrect, 0 partial, 90.0%.
llmprobe 0.6.12, xhigh effort, 12k reasoning / 16k output tokens, 8-bit KV and MTP.

## Choosing the n-gram table

Sushi-3bpw and Sushi-4bpw ship different copies of the same n-gram table, and either copy works with either pack. Use
the 4-bit table to save 66 GiB of disk, or the bf16 table for a small quality gain. The table is read from the SSD, so
GPU memory is the same either way.

| table | size on disk | ships with | KLD with Sushi-3bpw | KLD with Sushi-4bpw |
|---|---|---|---|---|
| 4-bit, group size 32 | 29.8 GiB | Sushi-3bpw | 0.1036 | 0.0654 |
| bf16 | 95.4 GiB | Sushi-4bpw | 0.1006 | 0.0592 |

To swap, download the other pack's `ngram_table.bin` into this pack's folder, replacing the current one:

```
hf download beamster/Qwen3.8-Flash-Next-Sushi-3bpw ngram_table.bin --local-dir <this pack's folder>
```

Then set `ngram_table` in `config.json` to match the file:
`"ngram_table": {"file": "ngram_table.bin", "bits": 4, "group_size": 32}` for the 4-bit table.

## Credits and license

- **Qwen team**: the base model, [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). This
  pack is a derivative under the **Qwen Community License 1.0** (`LICENSE`), including its conditions on large
  commercial services.
- **turboderp**: the EXL3 format.
- **ddalcu**: [mlx-serve](https://github.com/ddalcu/mlx-serve). sushi is a fork of it.
- **David Tai**: his work on --mtp-typical
