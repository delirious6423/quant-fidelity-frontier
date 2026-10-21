---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- qwen
- imatrix
- quantized
- 16gb-vram
- llama.cpp
- lm-studio
- ollama
- omenator
---

# Qwen3.8-27B · 4.2 BPW GGUF for 16 GB GPUs

**11.88 GB · 4.189 effective bits per weight · fully VRAM-resident on a single 16 GB GPU · 12.77 tok/s measured decode**

A calibration-driven, importance-weighted quantization of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced with **Omenator AI** (hence its nickname, the *Omenator build*). Every parameter of the 27B model is kept — precision is reallocated, nothing is pruned — and the result fits entirely in 16 GB of VRAM with no CPU offload.

## Why this exists

Typical 4-bit GGUF builds of 27B-class models land around 16–20 GB. On a 16 GB card that means partial CPU offload, and interactive speed collapses. This build holds the model to **11.88 GB**, leaving VRAM headroom for the KV cache — so the whole thing actually *lives* on the GPU and stays usable. That opens up the entire 16 GB class: RTX 5080 / 5070 Ti laptops (HP Omen and similar), 16 GB desktop cards, and workstation GPUs that couldn't comfortably hold a 27B before.

## Specs

| | |
|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — 27B dense, Apache-2.0 |
| Native context (base model) | 262,144 tokens |
| Format | GGUF, importance-matrix mixed precision (per-tensor-class type overrides) |
| Model file | `Qwen3.8-27B-4.2BPW-16GB.gguf` |
| Size on disk | **11.88 GB** |
| Effective precision | **4.189 bits per weight** (whole-model) |
| Target hardware | 16 GB VRAM GPUs |
| Toolchain | Omenator AI (calibration + importance analysis + candidate search) |

## Measured performance

Reference machine: HP Omen, NVIDIA Blackwell GPU, 16 GB VRAM. All layers on GPU, no CPU offload.

| Metric | Result |
|---|---|
| Decode speed | **12.77 tok/s** |
| VRAM residency | Full — model + context in GPU memory |

Raw runs and the exact test configuration are in [`benchmarks.json`](./benchmarks.json). The GGUF itself is standard llama.cpp format — it runs on any llama.cpp backend with ~12 GB+ of free VRAM, not only Blackwell; the numbers above are simply what the reference machine measured.

## Quality

Estimated **~2.6% degradation** relative to a near-lossless reference, based on the perplexity / KL-divergence measurements the Omenator pipeline runs on held-out text during candidate selection.

> **This is a build-time estimate, not a completed downstream benchmark suite.** No claims are made yet about specific eval scores. Independent evaluations are very welcome.
## Run it

### llama.cpp

```bash
llama-server -m Qwen3.8-27B-4.2BPW-16GB.gguf -ngl 99 -c 32768
```

Raise `-c` as far as your VRAM allows; enabling flash attention and quantizing the KV cache (`q8_0`) stretches the maximum context further.

### Ollama

```bash
ollama run hf.co/hitsfmdj/Qwen3.8-27B-GGUF-4.2BPW-16GB
```

Or, with the files downloaded, import locally using the included `Modelfile`:

```bash
ollama create qwen3.8-omenator -f Modelfile
```

### LM Studio

Search for `Qwen3.8-27B-GGUF-4.2BPW-16GB` in Discover, download, and set GPU offload to maximum.

## What's in this repo

| Path | What it is |
|---|---|
| `Qwen3.8-27B-4.2BPW-16GB.gguf` | The model — the only file you need to run it |
| `tokenizer/` | Tokenizer artifacts from the base model (already embedded in the GGUF; included for transparency) |
| `calibration/` | Calibration artifacts and importance analysis used to build this quantization |
| `benchmarks.json` | Measured size, speed, and VRAM results behind the numbers above |
| `Modelfile` | Optional — for local Ollama import (`ollama create`); sane defaults (16K context, temp 0.7) |
| `*.json` | Precision-allocation metadata, build configuration, and checksums |

Everything besides the `.gguf` documents how this build was produced, so the result is reproducible rather than a mystery file.

## How it was made

Omenator AI ran the full pipeline against the original Qwen3.8-27B weights:

1. **Calibration** on a diverse corpus produces an activation-aware importance matrix.
2. **Importance analysis** classifies tensors: sensitive tensors are protected at higher precision, tolerant bulk is packed lower — an i-matrix GGUF mix with per-tensor-class type overrides.
3. **Candidate builds** are measured on real hardware: actual file BPW, perplexity and KL divergence against a near-lossless reference, VRAM use, tokens/s, and maximum practical context.
4. **Hard acceptance rules** (max quality loss, max VRAM, minimum context and performance) reject failing candidates; the best surviving candidate is what you're looking at.

All 27B parameters are retained — this is precision reallocation, not pruning or distillation.

## Credits

Base model by the **Qwen team**, released under Apache-2.0 — all base-model terms apply. Quantized and released by **[hitsfmdj](https://huggingface.co/hitsfmdj)**. Built with **Omenator AI**.
