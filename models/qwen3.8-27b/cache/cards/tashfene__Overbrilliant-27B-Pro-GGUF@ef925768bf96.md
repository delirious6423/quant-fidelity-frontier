---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
- en
pipeline_tag: text-generation
library_name: gguf
tags:
- gguf
- llama.cpp
- 16gb-gpu
- qwen3.8
- gated-deltanet
- lora
- code
- function-calling
- mtp
---

# Overbrilliant-27B-Pro

**A 12 GB Qwen3.8-27B that calls tools better than the original and matches it on hard coding.**

Overbrilliant-27B-Pro is the higher-headroom sibling of [Overbrilliant-27B-Air](https://huggingface.co/tashfene/Overbrilliant-27B-Air-GGUF). It is built for 16 GB graphics cards and Macs with 24 GB or more. It keeps the model's multi-token-prediction (MTP) layer for roughly 2× faster decoding and protects the recurrent memory layers with extra precision.

![Memory needed at a 32k-token context](assets/memory.png)

## Highlights

- **Better tool calling than the original.** BFCL v3: 90.0% vs 86.2% for bf16 (+3.8 points, p < 0.001, paired McNemar test).
- **Hard coding at parity.** LiveCodeBench: 57.5% vs 60.9% (−2.6, p = 0.46). Over 334 paired problems the difference is −3.0 ± 2.0 (p ≈ 0.17), half the gap of standard quantizations.
- **Everyday coding at parity.** HumanEval+ 92.7 / MBPP+ 75.4 vs 92.1 / 78.3 for bf16 (−1.8 combined, p = 0.12).
- **Fast.** MTP speculative decoding at depth 3: 32.7 tokens/s on an AMD Ryzen AI Max (Strix Halo), about 2× the speed without MTP.
- **Fits a 16 GB GPU.** 14.6 GiB at a 32k context with MTP and an 8-bit KV cache.

## Which edition should I use?

| Your machine | Edition | Context | Notes |
|---|---|---|---|
| **16 GB Mac** (M1–M4) | **[Air](https://huggingface.co/tashfene/Overbrilliant-27B-Air-GGUF)** | up to 32k | 10.2 GiB measured at 32k; close memory-heavy apps. 64k reaches the GPU limit. |
| **24 GB+ Mac** | **[Pro](https://huggingface.co/tashfene/Overbrilliant-27B-Pro-GGUF)** | 32k–64k | Drop the MTP flags on a Mac (no speed gain on Metal). |
| **16 GB graphics card** (e.g. RTX 4080, 5070 Ti, RX 7800 XT) | **[Pro](https://huggingface.co/tashfene/Overbrilliant-27B-Pro-GGUF)** | up to 32k | 14.6 GiB measured at 32k with MTP; 64k leaves almost no headroom. |
| **12 GB graphics card** | [Air](https://huggingface.co/tashfene/Overbrilliant-27B-Air-GGUF) | 16k–32k | About 9.9–10.2 GiB by calculation; not yet tested. |
| 8 GB or less | — | — | Too small for a 27B model. |

## Memory by context length

![Memory by context length](assets/context.png)

| Context | Air (KV q4_0) | Pro (MTP, KV q8_0) |
|---|---|---|
| 8k | ~9.8 GiB | ~13.7 GiB |
| 16k | ~9.9 GiB | ~14.0 GiB |
| 32k | **10.2 GiB** (measured) | **14.6 GiB** (measured) |
| 64k | ~10.7 GiB (at the 16 GB Mac limit) | **15.9 GiB** (measured) |

Values marked ~ are calculated from the measured KV-cache cost of the model's 16 attention layers; its 48 Gated DeltaNet layers use a fixed-size state that does not grow with context.

## Benchmarks

![Benchmarks against the original model](assets/benchmarks.png)

| Model | Size | BFCL v3 (600) | LiveCodeBench (200, 8k) | HumanEval+ / MBPP+ |
|---|---|---|---|---|
| Qwen3.8-27B bf16 (original) | 54.7 GB | 86.2 | 60.9 | 92.1 / 78.3 |
| unsloth UD-Q2_K_XL (starting point) | 9.8 GB | 87.2 | 55.0 | 90.2 / 77.0 |
| **Overbrilliant-27B-Pro** | **12.0 GB** | **90.0** | **57.5** | **92.7 / 75.4** |
| [Overbrilliant-27B-Air](https://huggingface.co/tashfene/Overbrilliant-27B-Air-GGUF) | 9.5 GB | 89.0 | 57.5 | — |

![LiveCodeBench by difficulty](assets/difficulty.png)

## Quick start (llama.cpp)

```bash
llama-server -m Overbrilliant-27B-Pro.gguf \
  --lora adapter/Overbrilliant-27B-Pro-heal-lora.gguf \
  -ngl 99 -fa on -c 32768 -ctk q8_0 -ctv q8_0 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3
```

`--spec-type draft-mtp` uses the model's built-in MTP layer; no separate draft model is needed. On Apple Metal we measured no speed gain from MTP, so Mac users can drop those two flags (or use the Air edition on 16 GB Macs).

## Files

| File | Size | Purpose |
|---|---|---|
| `Overbrilliant-27B-Pro.gguf` | 11.0 GiB | Base weights (with built-in MTP layer) |
| `adapter/Overbrilliant-27B-Pro-heal-lora.gguf` | 0.17 GiB | Healing adapter (LoRA, rank 64) |

## How it was made

1. **Starting point:** unsloth's dynamic 2.9-bit quantization (UD-Q2_K_XL) of Qwen3.8-27B.
2. **Protected recurrent memory:** the 144 projection tensors of the 48 Gated DeltaNet layers (attn_qkv, attn_gate, ssm_out) were replaced by Q5_K/Q6_K versions quantized from bf16 with unsloth's importance matrix. Every other tensor is byte-identical to the starting point.
3. **Healed:** a rank-64 LoRA on the feed-forward layers, trained by on-policy distillation from the bf16 model on the quantized model's own outputs.

The two changes fix different weaknesses: healing fixes tool calling, the protected recurrent layers fix long reasoning, and only the combination brings hard problems back to the original's level.

## Limitations

- One greedy run per problem; "no significant difference" is not proof of equality.
- BFCL measured on a 600-task subset with our own AST checker, not the official harness.
- Memory was measured with the external MTP draft model; the built-in MTP path uses less.

## Credits

- **Model:** Qwen3.8-27B by the Qwen team (Apache-2.0).
- **Base quantization:** UD-Q2_K_XL by [unsloth](https://huggingface.co/unsloth) (Apache-2.0).
- **Overbrilliant-27B-Pro:** Tashfeen Ahmed.

## Citation

```bibtex
@misc{ahmed2026healmemory,
  title  = {Heal the Memory, Not the Bits: Running Qwen3.8-27B on a 16 GB Mac},
  author = {Ahmed, Tashfeen},
  year   = {2026},
  note   = {Overbrilliant-27B-Air and Overbrilliant-27B-Pro}
}
```
