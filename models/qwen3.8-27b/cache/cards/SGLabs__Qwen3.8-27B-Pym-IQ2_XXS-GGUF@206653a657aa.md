---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - quantized
  - pym
  - imatrix
  - dense
  - text-generation
  - endpoints_compatible
base_model: Qwen/Qwen3.8-27B
quantized_by: SGLabs
---

# Qwen3.8-27B-Pym-IQ2_XXS

A ~16 GB imatrix-calibrated, mixed-precision GGUF of **Qwen3.8-27B** (a `qwen35` hybrid
SSM+attention architecture) — SGLabs' first dense-model Pym quant.

**Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — a multimodal
(`image-text-to-text`), 27B dense model with 256K context, agentic-coding and reasoning
strengths, and a native MTP head.

This quant: dense FFN to 2–3 bits; attention, SSM, norms, embeddings, and the **MTP head
preserved at Q8** (so speculative decoding stays on). Runs on llama.cpp — AMD (ROCm) and
Apple Silicon. No CUDA required.

## Precision map

- `ffn_gate`, `ffn_up` → **IQ2_XXS**
- `ffn_down` → **IQ3_XXS**
- attention · SSM · norms · embeddings · output · **MTP block** → **Q8_0**

## Usage

```bash
# with MTP speculative decoding (recommended):
llama-server -m Qwen3.8-27B-Pym-IQ2_XXS.gguf -ngl 999 -c 32768 \
  --spec-type draft-mtp --spec-draft-n-max 2 -np 1
```

## File / compatibility

| | |
|---|---|
| File | `Qwen3.8-27B-Pym-IQ2_XXS.gguf` |
| Size | 15.99 GB (14.9 GiB) · 4.68 BPW |
| Architecture | `qwen35` · 65 blocks (64 + MTP) |

Mixed-precision quant: the filename carries the dominant standard token (`IQ2_XXS`) for
general compatibility; loaders requiring a uniform `general.file_type` may report it as custom.

---

*Quantized by SGLabs.*
