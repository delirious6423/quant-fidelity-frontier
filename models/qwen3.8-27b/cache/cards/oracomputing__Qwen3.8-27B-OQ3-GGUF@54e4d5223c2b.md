---
license: other
license_name: ora-custom-model-license
license_link: LICENSE
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
  - quantized
  - gguf
  - llama.cpp
  - qwen3_5
---

<div align="center">
  <img src="assets/ora-logo-extended-white.svg" alt="Ora" width="520"/>
  <br/>
  <a href="https://www.oracomputing.com/en"><b>Website</b></a>
  &nbsp;·&nbsp;
  <a href="https://www.oracomputing.com/en/blog"><b>Blog</b></a>
  &nbsp;·&nbsp;
  <a href="https://www.oracomputing.com/en/contact"><b>Contact</b></a>
</div>

# Qwen3.8-27B-OQ3 (GGUF)

> [!IMPORTANT]
> **~5.3× smaller** than the original 16-bit [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), with **95.4% accuracy retention**.

**OQ3** is a Post-Training Quantization (PTQ) of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with a calibration-based per-tensor format allocation. The name encodes the model's **3.026 bits-per-weight full-model average** (body 2.967 bpw, Q2_K embedding / Q4_K output head).

## Files

| file | size | full avg BPW | body BPW |
| --- | ---: | ---: | ---: |
| `Qwen3.8-27B-OQ3.gguf` | 10.54 GB | 3.026 | 2.967 |
| `mmproj-F16.gguf` | 0.93 GB | — | — |

The main file embeds the model's **multi-token-prediction (MTP) head** (Q6_K/Q8_0, ~0.36 GB) as an optional speculative draft — llama.cpp ignores it unless you pass the MTP flags, so the live-decode footprint and the 3.026 BPW figure are unchanged. `mmproj-F16.gguf` is the optional **vision projector** for image/video input.

## Benchmarks

Scores versus the original 16-bit [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Higher is better.

| Model | MMLU-Pro | AIME-25 | Average | Full BPW | Body BPW | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B (bf16) | 76.71 | 95.83 | 86.27 | 16.00 | 16.00 | 100.0% |
| **OQ3 (this repo)** | **73.94** | **90.42** | **82.18** | **3.026** | **2.967** | **95.4%** |
| Escha W2 | 71.24 | 87.50 | 79.37 | 3.020 | 2.485 | 92.1% |

**Note**: The **same setup** was applied to all models, so scores in this table are comparable. The quantized rows are at a **matched total byte budget** (10.17 vs 10.15 GB of live weights). MMLU-Pro is 5-shot CoT generative over 12,032 questions; AIME-25 is 8 seeds × 30 problems with 65,536 max new tokens.

## Usage (llama.cpp)

No manual download needed — llama.cpp pulls the model (and the vision projector) straight from this repo:

```bash
# OpenAI-compatible server with web UI:
llama-server -hf oracomputing/Qwen3.8-27B-OQ3-GGUF

# Chat in the terminal:
llama-cli -hf oracomputing/Qwen3.8-27B-OQ3-GGUF
```

Or with the [llama.app](https://llama.app) launcher: `llama serve -hf oracomputing/Qwen3.8-27B-OQ3-GGUF`. To download the files explicitly instead:

```bash
hf download oracomputing/Qwen3.8-27B-OQ3-GGUF Qwen3.8-27B-OQ3.gguf mmproj-F16.gguf --local-dir .
```

then point llama.cpp at the local file with `-m Qwen3.8-27B-OQ3.gguf` in place of `-hf ...`.

### Faster decoding with the built-in MTP head (recommended)

Qwen3.8 ships a multi-token-prediction head, embedded in this GGUF as block 64. On llama.cpp **b10502 or newer**, enable it for a substantial decode speedup (community-measured +33–145% depending on hardware) — no separate draft model needed:

```bash
llama-server -hf oracomputing/Qwen3.8-27B-OQ3-GGUF \
  -c 131072 -ngl 999 -fa on \
  --spec-type draft-mtp --spec-draft-n-max 3 --parallel 1
```

`--spec-draft-n-max 3` is a good default; the optimum is hardware-dependent, so try values 1–4. Measured on this file: draft acceptance ≈ 0.78. Older llama.cpp builds simply ignore the MTP tensors and run the model normally.

### Vision (image / video input)

With `-hf`, the `mmproj-F16.gguf` vision projector from this repo is downloaded and paired automatically. When loading from local files, pass it explicitly:

```bash
llama-server -m Qwen3.8-27B-OQ3.gguf --mmproj mmproj-F16.gguf \
  -c 131072 -ngl 999 -fa on --image-min-tokens 1024
```

`--image-min-tokens 1024` is recommended for grounding tasks. Vision and MTP flags can be combined.

**Best practices** (same as [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)):

- **Sampling parameters**:
  - Thinking mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
  - Instruct (non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
  - If you hit endless repetition, raise `presence_penalty` between 0 and 2 (higher values can mix languages and slightly hurt quality).
- **Adequate output length**: allocate generous token budgets for reasoning-heavy tasks — up to 262,144 tokens for reasoning content and 131,072 for the final response, within the model's native 262K context.
