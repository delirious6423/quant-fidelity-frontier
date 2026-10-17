---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
tags:
- gguf
- quantized
- text-generation
---

# Qwen3.8-27B — KERN

KERN uses per-tensor quantization to offer four size and quality tiers. `eq` denotes an approximate tier: individual tensors use different quantization formats. All files include Q3_K MTP weights.

| Quant | GB | Mean KLD ↓ | 28Q (% of BF16) |
|---|---:|---:|---:|
| [KERN-eqQ4](Qwen3.8-27B-KERN-eqQ4-MTP.gguf) | 14.730 | 0.030233 | 90.48% |
| [KERN-eqQ3_M](Qwen3.8-27B-KERN-eqQ3_M-MTP.gguf) | 13.044 | 0.046237 | 85.71% |
| [KERN-eqQ3_S](Qwen3.8-27B-KERN-eqQ3_S-MTP.gguf) | 11.591 | 0.080428 | 71.43% |
| [KERN-eqIQ2_M](Qwen3.8-27B-KERN-eqIQ2_M-MTP.gguf) | 9.580 | 0.164745 | 80.95% |

Scores are relative to BF16’s 28Q result (BF16 = 100%), using the lower primary result where multiple runs exist. The repeat run is excluded. This is a screening metric, not a general capability rating.

**KERN-eqQ3_S** offers a sub-0.1 mean KLD option at 11.59 GB. **KERN-eqIQ2_M** is the smallest option at 9.58 GB, with measured mean KLD **0.164745**.

## Stock comparisons

| Control | GB | Mean KLD ↓ | 28Q (% of BF16) |
|---|---:|---:|---:|
| BF16 | 54.658 | 0 (reference) | 100.00% |
| Stock Q4_K_M | 16.730 | 0.016437 | 85.71% |
| Stock Q3_K_S | 12.257 | 0.088238 | 80.95% |
| Stock IQ2_M | 10.187 | ≈0.1551 | 85.71% |


<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/andrew-paul/Qwen3.8-27B-GGUF/resolve/main/assets/28cell-size-light.png">
  <img src="https://huggingface.co/andrew-paul/Qwen3.8-27B-GGUF/resolve/main/assets/28cell-size-dark.png" alt="KERN 28Q scores relative to BF16 versus size">
</picture>

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/andrew-paul/Qwen3.8-27B-GGUF/resolve/main/assets/kld-light.png">
  <img src="https://huggingface.co/andrew-paul/Qwen3.8-27B-GGUF/resolve/main/assets/kld-dark.png" alt="KERN and stock control mean KLD">
</picture>

## Usage

Use a llama.cpp build with Qwen MTP support:

```bash
./llama-server -m Qwen3.8-27B-KERN-eqQ3_S-MTP.gguf -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0 --jinja --spec-type draft-mtp --spec-draft-n-max 3
```

File size excludes KV cache and runtime overhead.

Verify downloads with `sha256sum -c SHA256SUMS`. File identities are in [release-manifest.json](release-manifest.json), with tensor maps in [recipes](https://huggingface.co/andrew-paul/Qwen3.8-27B-GGUF/tree/main/recipes).
