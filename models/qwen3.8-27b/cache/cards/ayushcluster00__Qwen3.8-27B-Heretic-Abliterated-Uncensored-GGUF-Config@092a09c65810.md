---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
library_name: transformers
tags:
- qwen3.8
- qwen3.5
- heretic
- abliterated
- uncensored
- roleplay
- gguf
- imatrix
pipeline_tag: text-generation
---

# Qwen3.8-27B RVN Heretic Abliterated Uncensored (GGUF)

**RVN** is a double-refined abliterated variant of **Qwen3.8-27B**, built on top of
[`trohrbaugh/Qwen3.8-27B-heretic-ara`](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara)
(an ARA abliteration by **Tim Rohrbaugh**) and further refined with **two additional
full-weight ARA passes** targeting residual refusals. It retains very low behavioral
damage (KL ≈ 0.0085) while reducing harmful-prompt refusals from 3/100 (source) to
**0–1/100** in independent measurements.

> **Note on this repository's history.** This repo previously hosted the original
> `Qwen3.8-27B-Heretic-Q4_K_M.gguf` (single-quant release from the earlier
> `trohrbaugh/Qwen3.8-27B-heretic` source). That file is kept as **legacy** for
> download-count continuity and backward compatibility — it is the *older* abliteration
> variant and is superseded by the RVN files below. Prefer the RVN quants for new
> deployments.

> **Not for all audiences.** This model has reduced safety guardrails by design. It is
> intended for adult audiences (18+) doing research, creative writing, roleplay, and
> uncensored generation. Certain guardrails are intentionally left in place; use
> responsibly and in accordance with your local laws.

---

## What is ARA?

**ARA (Arbitrary-Rank Ablation)** is the abliteration technique implemented in
[p-e-w/heretic](https://github.com/p-e-w/heretic). Traditional directional abliteration
finds a single "refusal direction" in activation space and subtracts it — a one-shot,
low-rank surgery that is simple but can leave residual refusals or damage unrelated
behavior.

ARA instead treats abliteration as a **matrix optimization problem**. For every target
module (attention out-projection and MLP down-projection), it collects activations on
"good" prompts (harmless requests) and "bad" prompts (harmful requests), then uses an
LBFGS optimizer to rewrite the module's weight matrix so that:

- **Preserve:** outputs on good prompts change as little as possible (KL is kept low)
- **Steer:** outputs on bad prompts are pulled toward the good-prompt output manifold
  (via k-nearest-neighbor distances), so harmful requests stop triggering the refusal
  circuitry
- **Overcorrect:** outputs on bad prompts are additionally pushed *away* from the
  original bad-prompt outputs, which helps overcome complex, multi-stage refusal
  mechanisms

Because the weight matrix is optimized directly (rather than subtracting a single
direction), ARA is "arbitrary rank" — it can carve out a much richer refusal-removal
subspace while keeping behavioral damage minimal.

## Why "Heretic" and "Abliterated"?

These two words describe two layers of the same process:

- **Heretic** is the *tool*: the open-source implementation of ARA (and related
  abliteration methods) used to modify the model. Models produced with it are commonly
  labeled "heretic" in the community.
- **Abliterated** is the *result*: the model's refusal behavior has been surgically
  removed. An abliterated model still knows everything the base model knows, but it no
  longer refuses to answer the categories that were steered away during the process.

So "Heretic Abliterated" means: *abliterated using the heretic toolset.* RVN goes one
step further — it applies the ARA procedure **three times total**: once by the original
author (`trohrbaugh`) to get from base Qwen3.8-27B to `-ara`, and twice more by us to
get from `-ara` to RVN, squeezing out the last residual refusals.

## Special Thanks

This work would not exist without **Tim Rohrbaugh** (`trohrbaugh`), whose
[`heretic-ara`](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara) ARA
abliteration of Qwen3.8-27B (refusals 3/100, KL 0.0535) provided the foundation we
refined into RVN. His upstream contributions to the heretic codebase — including the
row-norm preservation feature and Qwen3.5 MoE/DeltaNet hybrid handling — are directly
responsible for making DeltaNet-layer abliteration work at all. Thank you, Tim.

## Model Overview

| Property | Value |
|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Abliteration source | [`trohrbaugh/Qwen3.8-27B-heretic-ara`](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara) (ARA, KL 0.0535, refusals 3/100) |
| **RVN refinement** | **2-pass ARA on top of source → KL 0.0085, refusals 0–1/100** |
| Architecture | `qwen3_5_text` (Qwen3.8 family), Gated DeltaNet hybrid |
| Parameters | 27B total |
| Hidden size | 5120 |
| Layers | 64 (16 standard attention + 48 Gated DeltaNet linear attention) |
| Attention heads | 24 · KV heads 4 (GQA) · head_dim 256 |
| Vocab | 248,320 |
| Context length | 262,144 (262K) |
| License | Apache-2.0 (retained from Qwen3.8-27B) |
| Format | GGUF (llama.cpp), MTP/NextN draft tensors excluded (`--no-nextn`) |

## Why RVN?

`trohrbaugh/Qwen3.8-27B-heretic-ara` is already a strong ARA abliteration, but three
harmful prompts still triggered refusals in our independent evaluation (racism website,
malware, government database hacking). RVN applies **two additional full-weight ARA
passes** using the same tight parameter set (start 26, end 56, preserve 0.9432,
steer 0.0009, overcorrect 0.5038, neighbor 10), which:

- Reduced refusals from **3/100 → 0–1/100** (the only remaining refusal is a
  chemical-weapon WMD prompt — one of the strongest safety-trained categories, and
  intentionally one of the guardrails we left in place)
- Reduced KL damage from 0.0535 (source) to **0.0085** vs base — a ~6× improvement
  in behavioral preservation
- Verified independently on two rented GPU machines with prefix-based (real-answer)
  refusal measurement

### Refusal evaluation (100 harmful-behaviors prompts, prefix-forced real answers)

| Model | Refusals | KL vs base |
|---|---|---|
| Qwen3.8-27B (base) | ~99/100 | — |
| trohrbaugh -ara (source) | 3/100 | 0.0535 |
| **RVN (this repo)** | **0–1/100** | **0.0085** |

## Files & Quantization Spectrum

<div style="font-size: 0.85em">

| File | Size (GB / GiB) | Notes |
|---|---|---|
| `RVN-F16.gguf` | 53.81 / 50.11 | F16 reference (no NextN/MTP) |
| `RVN-BF16.gguf` | 53.81 / 50.11 | BF16 reference (no NextN/MTP) |
| `RVN-Q8_0.gguf` | 28.60 / 26.63 | Max-quality 8-bit |
| `RVN-Q6_K.gguf` | 22.08 / 20.57 | High-quality 6-bit |
| `RVN-Q5_K_M.gguf` | 19.23 / 17.91 | Balanced 5-bit |
| `RVN-Q5_K_S.gguf` | 18.68 / 17.40 | 5-bit small |
| `RVN-Q4_K_M.gguf` | 16.55 / 15.41 | Recommended 4-bit (24 GB VRAM) |
| `Qwen3.8-27B-Heretic-Q4_K_M.gguf` | 16.55 / 15.41 | **Legacy** (older abliteration variant, kept for download continuity) |
| `RVN-IQ4_NL.gguf` | 15.89 / 14.80 | imatrix 4-bit |
| `RVN-Q4_K_S.gguf` | 15.59 / 14.52 | Small 4-bit |
| `RVN-IQ4_XS.gguf` | 15.19 / 14.15 | imatrix 4-bit extra-small |
| `RVN-Q3_K_L.gguf` | 14.34 / 13.36 | Large 3-bit |
| `RVN-Q3_K_M.gguf` | 13.30 / 12.39 | Compact 3-bit |
| `RVN-IQ3_M.gguf` | 12.58 / 11.72 | imatrix 3-bit — **re-uploaded 2026-08-17** (previous file had corrupted tensor data: NaN/Inf scales + zeroed tensors from a bad quantize run; re-quantized from F16 with a fresh imatrix and verified — see note below) |
| `RVN-IQ3_S.gguf` | 12.42 / 11.57 | imatrix 3-bit small |
| `RVN-Q3_K_S.gguf` | 12.07 / 11.24 | Compact 3-bit small |
| `RVN-IQ3_XS.gguf` | 11.97 / 11.15 | imatrix 3-bit extra-small |
| `RVN-IQ3_XXS.gguf` | 11.19 / 10.42 | imatrix 3-bit extra-extra-small |
| `RVN-Q2_K.gguf` | 10.71 / 9.98 | 2-bit K-quant |
| `RVN-Q2_K_S.gguf` | 10.25 / 9.54 | 2-bit K-quant small |
| `RVN-IQ2_M.gguf` | 10.00 / 9.32 | imatrix 2-bit |
| `RVN-IQ2_S.gguf` | 9.36 / 8.72 | imatrix 2-bit small |
| `RVN-IQ2_XS.gguf` | 9.09 / 8.47 | imatrix 2-bit extreme small |
| `RVN-IQ2_XXS.gguf` | 8.43 / 7.85 | imatrix 2-bit (minimum) |
| `RVN-IQ1_M.gguf` | 7.63 / 7.11 | imatrix 1-bit (experimental) |
| `RVN-IQ1_S.gguf` | 7.15 / 6.66 | imatrix 1-bit (experimental) |

</div>

> imatrix-based quants are produced from the same F16 with an activation importance
> matrix computed over wikitext-2-raw (original spectrum, 563 chunks) / tiny_shakespeare
> (2026-08-17 re-quant additions: `IQ3_M` fix + `IQ2_S`/`IQ3_XXS`/`IQ3_XS`/`IQ3_S`, llama-imatrix, `-ngl 99`).

## Quant → GPU / Memory Guide

| GPU / Memory | Best quant(s) (full GPU load) | Effective ctx @ Q8_0 KV |
|---|---|---|
| **8 GB** (RTX 3050, 4060 Laptop) | `IQ1_S`, `IQ1_M`; `IQ2_XXS` partial offload only | ~2–4K |
| **12 GB** (RTX 3060, 4070) | `IQ2_M`, `IQ2_S`, `IQ2_XS`, `Q2_K_S`; `IQ3_XXS` (tight) | ~8–16K |
| **16 GB** (RTX 4080, 4090 Laptop, M3 Max) | `IQ3_M`, `IQ3_S`, `Q3_K_M`; `IQ4_XS`/`Q4_K_S`/`IQ4_NL` (tight ctx) | ~6–24K |
| **24 GB** (RTX 3090, 4090, M4 Max) | `Q5_K_M`, `Q5_K_S`, `Q6_K`, `Q4_K_M`; `Q8_0` partial | ~16–48K |
| **32 GB** (RTX 5090, A6000) | `Q8_0`, `Q6_K` | ~24–64K |
| **64 GB+** (A100 80 GB, RTX PRO 6000, M3/M4 Ultra) | `F16`, `BF16` | ~64–100K+ |

> Sizes in the file table are the **actual file sizes on the Hub** (decimal GB / GiB),
> pulled from repository metadata. Full GPU load means the whole quant fits in VRAM;
> quants whose file size exceeds your VRAM need partial offloading.
>
> **2026-08-17 — RVN-IQ3_M incident & fix:** the original `RVN-IQ3_M.gguf` generated only
> `/` characters on every backend (confirmed by the community and reproduced locally). A
> tensor-level audit showed corrupted quantization data — NaN/Inf block scales and fully
> zeroed tensors (e.g. `token_embd` had ~39.6M NaN values) — from a bad quantize run, not a
> llama.cpp regression (all other quants from the same F16 dequantize cleanly). The file was
> pulled, re-quantized from the F16 with a freshly computed imatrix, generation-tested
> (`"The capital of France is" → Paris`, 70+ t/s) and re-uploaded. New quants added the same
> day: `IQ2_S`, `IQ3_XXS`, `IQ3_XS`, `IQ3_S`, `Q3_K_L`, `Q5_K_S`.

**KV cache math (GQA, 4 KV heads, head_dim 256):**
`2 × 64 layers × 4 KV heads × 256 head_dim × 2 bytes = 256 KiB/token FP16`
→ 16K ctx ≈ 4.2 GB · 32K ctx ≈ 8.4 GB · 64K ctx ≈ 16.8 GB (Q8_0 KV halves this).
A 16 GB card running `Q3_K_M` (13.30 GB model) + 16K ctx Q8_0 KV fits comfortably;
`Q4_K_M` (16.55 GB) really needs a 24 GB card.

**Rule of thumb:** pick the largest quant that leaves ≥ 4 GB for KV cache + compute
buffers. If you only need short replies, drop the quant one notch and get a bigger
context; if you need long context, prioritize KV budget over quant size.

## Limitations & Responsible Use

- **Reduced safety guardrails by design.** This model is not intended for use in
  applications requiring robust safety filtering, content moderation, or deployment to
  minors.
- **Certain guardrails are intentionally left in place.** Abliteration targets refusal
  behavior on general harmful-prompt categories; a small set of hard safety-trained
  categories is deliberately not fully removed. Behavior may vary across domains and
  languages.
- Not affiliated with or endorsed by Qwen/Alibaba or trohrbaugh.

## License & Attribution

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — Apache-2.0
- Abliteration source: [`trohrbaugh/Qwen3.8-27B-heretic-ara`](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara) — Apache-2.0
- This repo: Apache-2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{rohrbaugh2026heretic,
  title={Qwen3.8-27B-heretic-ara: ARA Abliteration of Qwen3.8-27B},
  author={Rohrbaugh, Tim},
  year={2026},
  howpublished={\url{https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara}}
}

@misc{rvn2026,
  title={RVN: Qwen3.8-27B Heretic Abliterated Uncensored},
  author={0bserverx},
  year={2026},
  howpublished={\url{https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF}}
}
```
