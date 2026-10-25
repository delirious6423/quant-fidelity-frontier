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

# Gemma-4-26B-A4B-it · **Nano** (2-bit GGUF) · fraQtl

**Gemma-4-26B-A4B-it at ~6.7 GB (2.12 bpw): GSM8K 86% (0-shot, 100 Q), MMLU-Pro 65% with reasoning on (Google base 82.6%). KLD to fp16: 1.97 code+math, 2.91 general. The full model is ~50 GB; the 4-bit QAT is 14.4 GB.**

Nano is the size-first tier (ladder: Hi-Fi → SmartEdge → Nano): the smallest footprint, for phones, laptops and air-gapped boxes where size is the binding constraint. Gemma-4's reasoning ("thinking") mode is kept. For near-original token-level fidelity, use the **Hi-Fi** tier. Every number on this page is measured on this exact file.

## Model overview

A post-training 2-bit quantization of Google's Gemma-4-26B-A4B-it (26B-param MoE, 3.8B active, 256K context), built with fraQtl's **importance-driven bit allocation**: fragile tensors are protected at higher bits; the bulk MoE expert weights carry the low-bit tier.

## Size (published numbers)

| artifact | bits | size on disk | source |
|---|---|---:|---|
| full precision (bf16) | 16-bit | ~50 GB | n/a |
| Google official QAT | 4-bit | 14.4 GB | Google card (published) |
| standard community 2-bit (`IQ2_XXS`) | "2-bit" | 9.66 GB | published community-quant size |
| standard community 2-bit (`Q2_K`) | "2-bit" | 11 GB | published community-quant size |
| **fraQtl Nano (this file)** | **2-bit** | **~6.7 GB** | this file (2.117 bpw × params) |

≈31% smaller than a standard `IQ2_XXS`, ≈2.15× smaller than the official 4-bit QAT, ≈7.5× smaller than bf16. Size only, from published sizes; the other artifacts were not evaluated here.

## Evaluation (measured on this file)

| benchmark | Nano 2-bit | reference | note |
|---|---:|---:|---|
| **GSM8K** (0-shot, chat-templated, 100 Q) | **86%** | n/a | 0 degenerate generations |
| **MMLU-Pro** (full 1000, **reasoning on**) | **65%** | Google base 82.6% | reasoning-matched → **~79% of full-model** |

Reasoning-matched: MMLU-Pro is evaluated with Gemma-4's thinking mode enabled, the same mode Google's published number uses. Direct-answer (no-reasoning) scores are not reported; MMLU-Pro is built to require reasoning, so a bare-letter protocol understates every model.

### Fidelity (KLD vs fp16)

| slice | Nano 2-bit (KLD ↓ / top-1 ↑) | Hi-Fi Q4 (for reference) |
|---|---:|---:|
| code + math | **1.97** / 60% | 0.09 / near-full |
| general | **2.91** / 44% | 0.13 / near-full |

KLD 1.97 and 2.91 against 0.09 and 0.13 for Hi-Fi Q4: this is the 2-bit cost. Nano is not a drop-in replacement for fidelity-critical work; for that, use Hi-Fi.

### Sample outputs (unedited, greedy)

- **Why is the sky blue?** → *"Rayleigh scattering… shorter wavelengths (blue/violet) are scattered more intensely by gas molecules."*
- **17 × 24?** → *10×24=240, 7×24=168, 240+168 = **408**.*
- **Three South American countries + capitals** → Brazil–Brasília, Argentina–Buenos Aires, Colombia–Bogotá.

## Scope

- GSM8K is a 100-question 0-shot screen; MMLU-Pro is the full 1000 with reasoning on.
- Calibration is generic imatrix on generic text. Task-aware calibration (allocation tuned to a target task distribution) was not applied to this file; task-aware editions are a planned follow-up.
- KLD / top-1 are measured against fp16 on the code+math and general slices.

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

Importance-driven per-tensor allocation over the bf16 base: the router, token embeddings, attention, and the fragile `ffn_down` expert projections are protected at higher bits; the bulk `ffn_gate`/`ffn_up` expert weights carry the low-bit tier. imatrix-calibrated. Standard **llama.cpp GGUF**; runs anywhere llama.cpp runs.

## Limitations

- Post-training 2-bit; use when size/memory is the binding constraint. For fidelity-critical use, use Hi-Fi (Q4).
- Token-distribution fidelity: KLD 2–3 vs fp16 (table above). Not a drop-in for the full model.
- Generic-calibration numbers; task-aware calibration was not applied.

## License

Apache 2.0 for the quantization artifacts and receipts in this repository. Weights derive from `google/gemma-4-26B-A4B-it` and remain subject to Google's Gemma terms.

---
*More from fraQtl: [huggingface.co/fraQtl](https://huggingface.co/fraQtl): importance-driven bit allocation, KV-cache compression, and serving.*
