---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: gguf
quantized_by: fraQtl
inference: false
pipeline_tag: text-generation
tags:
  - gguf
  - quantized
  - 2-bit
  - moe
  - gemma-4
  - gemma-4-26b-a4b
  - llama.cpp
  - imatrix
  - on-device
  - edge
  - nano
  - fraqtl
  - conversational
language:
  - en
---

# Gemma-4-26B-A4B-it · **Nano** (2-bit GGUF) — fraQtl

**Run a 26B-class model on-device.** Nano is **~6.7 GB at a true 2.12 bpw** — small enough to fit where the full model (~50 GB) and even the 4-bit QAT (14.4 GB) can't. Built for the edge — phones, laptops, air-gapped boxes — anywhere **size is the binding constraint**. It still solves grade-school math at **86% (zero-shot)** and keeps Gemma-4's reasoning ("thinking") mode intact.

> **Nano is our size-first tier** — a deliberate trade: the smallest usable footprint, and honest about the cost. It *performs* on tasks but its token-level fidelity to the original is loose (disclosed below) — the real price of true 2-bit. Need near-original fidelity? Use the **Hi-Fi** tier. *(Ladder: Hi-Fi → SmartEdge → Nano.)*

> Every number on this page is measured on this exact file. Where a number is soft or pending, we say so.

## Model overview

A post-training 2-bit quantization of Google's Gemma-4-26B-A4B-it (26B-param MoE, 3.8B active, 256K context), built with fraQtl's **importance-driven bit allocation**: bits are spent where they matter most, not uniformly. Fragile tensors are protected; the bulk MoE expert weights carry the low-bit tier.

## Size — using everyone's *published* numbers

| artifact | bits | size on disk | source |
|---|---|---:|---|
| full precision (bf16) | 16-bit | ~50 GB | — |
| Google official QAT | 4-bit | 14.4 GB | Google card (published) |
| standard community 2-bit (`IQ2_XXS`) | "2-bit" | 9.66 GB | published community-quant size |
| standard community 2-bit (`Q2_K`) | "2-bit" | 11 GB | published community-quant size |
| **fraQtl Nano (this file)** | **2-bit** | **~6.7 GB** | this file (2.117 bpw × params) |

**≈31% smaller than a standard `IQ2_XXS`, ≈2.15× smaller than the official 4-bit QAT, ≈7.5× smaller than bf16.** No quality claim is made against those — this is size only, from published sizes.

## Evaluation (measured on this file)

| benchmark | Nano 2-bit | reference | note |
|---|---:|---:|---|
| **GSM8K** (0-shot, chat-templated, 100 Q) | **86%** | — | 0 degenerate generations |
| **MMLU-Pro** (full 1000, **reasoning on**) | **65%** | Google base 82.6% | reasoning-matched → **~79% of full-model** |

*Reasoning-matched: MMLU-Pro is evaluated with Gemma-4's thinking mode enabled, the same mode Google's published number uses. Direct-answer (no-reasoning) scores are intentionally not reported — MMLU-Pro is built to require reasoning, so forcing a bare letter understates every model.*

### Fidelity (KLD-vs-fp16) — disclosed, this is the 2-bit cost
| slice | Nano 2-bit (KLD ↓ / top-1 ↑) | Hi-Fi Q4 (for reference) |
|---|---:|---:|
| code + math | **1.97** / 60% | 0.09 / near-full |
| general | **2.91** / 44% | 0.13 / near-full |

Nano's token-distribution fidelity is **loose** — expected for a true 2-bit, and disclosed plainly (same way the Hi-Fi card discloses its own small losses). Task performance above is the practical evidence it stays capable; the KLD tells you it is *not* a drop-in replacement for fidelity-critical work. For that, use Hi-Fi.

### Calibration — these are a floor, not a ceiling
This build used **generic** importance calibration (imatrix on generic text). **Task-aware calibration** — tuning the allocation to a target task distribution — was **not** applied here, and is expected to lift task scores at the same file size. So today's numbers are a generic-calibration **floor**. Task-aware editions are a planned follow-up.

### Sample outputs (unedited, greedy)
- **Why is the sky blue?** → *"Rayleigh scattering… shorter wavelengths (blue/violet) are scattered more intensely by gas molecules."*
- **17 × 24?** → *10×24=240, 7×24=168, 240+168 = **408**.*
- **Three South American countries + capitals** → Brazil–Brasília, Argentina–Buenos Aires, Colombia–Bogotá.

## Usage

```bash
pip install llama-cpp-python==0.3.33
hf download fraQtl/Gemma-4-26B-A4B-it-Nano-GGUF Gemma-4-26B-A4B-it-fraQtl-2bit.gguf --local-dir .
```
```python
from llama_cpp import Llama
llm = Llama(model_path="Gemma-4-26B-A4B-it-fraQtl-2bit.gguf", n_ctx=4096, n_gpu_layers=-1)
out = llm.create_chat_completion(
    messages=[{"role": "user", "content": "What is 17 times 24?"}], temperature=0.0)
print(out["choices"][0]["message"]["content"])
```
CLI: `llama-cli -m Gemma-4-26B-A4B-it-fraQtl-2bit.gguf -cnv`. Use the chat template (the model degenerates on raw prompts).

## Method
Importance-driven per-tensor allocation over the bf16 base: the router, token embeddings, attention, and the fragile `ffn_down` expert projections are protected at higher bits; the bulk `ffn_gate`/`ffn_up` expert weights carry the low-bit tier. imatrix-calibrated. Standard **llama.cpp GGUF** — runs anywhere llama.cpp runs.

## Limitations
- Post-training 2-bit — **use when size/memory is the binding constraint.** For fidelity-critical use, use Hi-Fi (Q4).
- Loose token-distribution fidelity (KLD 2–3, disclosed) — not a drop-in for the full model.
- Generic-calibration numbers; task-aware calibration expected to raise them.

## License
Apache 2.0 for the quantization artifacts and receipts in this repository. Weights derive from `google/gemma-4-26B-A4B-it` and remain subject to Google's Gemma terms.

---
*More from fraQtl: [huggingface.co/fraQtl](https://huggingface.co/fraQtl) — importance-driven bit allocation, KV-cache compression, and serving.*
