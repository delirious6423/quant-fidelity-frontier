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
- apple-silicon
- metal
- 16gb
- qwen3.8
- gated-deltanet
- lora
- code
- function-calling
---

# Overbrilliant-27B-Air

**A 27-billion-parameter model that runs on a 16 GB Mac at a 32k-token context.**

Overbrilliant-27B-Air is a 10 GB build of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) designed for the most common local-AI machine there is: the 16 GB Apple Silicon laptop. In paired tests against the full 55 GB bf16 model it is **significantly better at tool calling** and **statistically indistinguishable on competitive programming**, while needing about **10.2 GiB** of memory with a 32,768-token context.

![Memory needed at a 32k-token context](assets/memory.png)

## Highlights

- **Fits a 16 GB Mac.** 10.2 GiB at 32k context (measured with llama.cpp on Metal), under the ~10.7 GB a 16 GB Mac gives the GPU by default. No `sysctl` tweaks needed.
- **Better tool calling than the original.** BFCL v3: 89.0% vs 86.2% for bf16 (+2.8 points, p = 0.006, paired McNemar test).
- **Hard coding at parity.** LiveCodeBench: 57.5% vs 60.9% for bf16 (−2.1, p = 0.56): no significant difference. Medium problems: 72.0% vs 68.0%.
- **Long context that works.** Correctly answered a retrieval question over a 29,495-token prompt.

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

| Model | Size | Memory at 32k | BFCL v3 (600) | LiveCodeBench (200, 8k) |
|---|---|---|---|---|
| Qwen3.8-27B bf16 (original) | 54.7 GB | — | 86.2 | 60.9 |
| unsloth UD-Q2_K_XL (starting point) | 9.8 GB | ~10.9 GiB | 87.2 | 55.0 |
| [Overbrilliant-27B-Pro](https://huggingface.co/tashfene/Overbrilliant-27B-Pro-GGUF) | 12.0 GB | 14.6 GiB | **90.0** | 57.5 |
| **Overbrilliant-27B-Air** | **9.5 GB** | **10.2 GiB** | **89.0** | **57.5** |

![LiveCodeBench by difficulty](assets/difficulty.png)

All comparisons are paired, per problem, against the bf16 model with greedy decoding.

## Quick start (llama.cpp)

Download the model and the adapter, then:

```bash
llama-server -m Overbrilliant-27B-Air.gguf \
  --lora adapter/Overbrilliant-27B-Air-heal-lora.gguf \
  -ngl 99 -fa on -c 32768 -ctk q4_0 -ctv q4_0 --jinja
```

The adapter is part of the model: run it with `--lora`. Runtimes that cannot load a LoRA file will run the base alone, which scores lower (see *How it was made*). Close memory-heavy apps for the full 32k context; use `-c 16384` if you need more headroom.

## Files

| File | Size | Purpose |
|---|---|---|
| `Overbrilliant-27B-Air.gguf` | 8.67 GiB | Base weights |
| `adapter/Overbrilliant-27B-Air-heal-lora.gguf` | 0.19 GiB | Healing adapter (LoRA, rank 32) |

## How it was made

Qwen3.8-27B is a hybrid model: 48 Gated DeltaNet layers keep a fixed-size recurrent memory, and 16 attention layers keep a KV cache. Aggressive quantization damages long reasoning mainly through the recurrent layers, whose errors persist across every later token.

1. **Starting point:** unsloth's dynamic 2.9-bit quantization (UD-Q2_K_XL).
2. **Trimmed for a 16 GB Mac:** removed the unused multi-token-prediction block (−0.33 GiB) and stored the output projection at Q3_K (−0.15 GiB).
3. **Healed:** a rank-32 LoRA, trained by on-policy distillation from the bf16 model on the quantized model's own outputs, corrects both the feed-forward layers and the recurrent (Gated DeltaNet) projections. Training sees the exact quantized weights that ship.
4. **4-bit KV cache:** tested in isolation, it cost nothing measurable on LiveCodeBench.

Healing the recurrent layers recovers as much hard-coding accuracy as giving them 2 GB of extra precision, which is what makes a 16 GB build possible. Details are in the accompanying paper, *Heal the Memory, Not the Bits*.

## Limitations

- One greedy run per problem; "no significant difference" is not proof of equality.
- Tool calling was measured on a 600-task BFCL v3 subset with our own AST checker, not the official harness.
- Accuracy was evaluated with llama.cpp on CUDA; memory and the long-context check on Apple Metal. We have not yet run it on a physical 16 GB Mac, and do not report speed.
- HumanEval+/MBPP+ have not been run for this exact build.

## Credits

- **Model:** Qwen3.8-27B by the Qwen team (Apache-2.0).
- **Base quantization:** UD-Q2_K_XL by [unsloth](https://huggingface.co/unsloth) (Apache-2.0).
- **Overbrilliant-27B-Air:** Tashfeen Ahmed.

## Citation

```bibtex
@misc{ahmed2026healmemory,
  title  = {Heal the Memory, Not the Bits: Running Qwen3.8-27B on a 16 GB Mac},
  author = {Ahmed, Tashfeen},
  year   = {2026},
  note   = {Overbrilliant-27B-Air and Overbrilliant-27B-Pro}
}
```
