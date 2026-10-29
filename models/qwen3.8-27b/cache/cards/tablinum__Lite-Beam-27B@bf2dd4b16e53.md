---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
language:
- en
tags:
- lite-beam
- qwen
- qwen3.8
- gguf
- llama.cpp
- quantized
- 2-bit
- 1-bit
- engram
- retrieval-augmented
- speculative-decoding
pretty_name: Lite-Beam 27B
---

<img src="assets/banner.png" alt="Lite-Beam 27B: Qwen3.8-27B in 9.26 GB, with a gated engram" width="100%">

# Lite-Beam 27B

**Qwen3.8-27B in 9.26 GB (2-bit) or 6.73 GB (1-bit) GGUFs that run on small consumer GPUs. A per-model gated engram
lifts knowledge far past where the quant starts, and the 1-bit build gains the most.** With Tablinum-2, the 1-bit
build scores **+43.0** on SimpleQA (only 7 wrong of 200) and **+13.5** on AA-Omniscience dev, up from −14.5. Both are
the best results of any build we tested, including a larger 10.94 GB build. Apache-2.0.

This is a quantized derivative of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), not a new base
model. The weights are built with Tablinum's calibration set, importance matrix and quantizer (the Lite-Beam method),
on per-tensor type layouts credited below. Two files: **IQ1_M (6.73 GB, 1-bit)**, the recommended build when
paired with the engram, and **IQ2_S (9.26 GB, 2-bit)**, the stronger build without retrieval. Use the `v1` injection
prompt (below) with either.

**Quick links:** [Files](#files) · [Run locally](#run-locally) · [Tablinum-2 engram](https://huggingface.co/datasets/tablinum/tablinum-2) · [Lite-Beam 1.0 (501B)](https://huggingface.co/tablinum/Lite-Beam-1.0-501B-A23B-IQS_XXS)

## The result

<img src="assets/results.png" alt="Lite-Beam 27B engram scorecard: closed-book vs Tablinum-2 engram on AA-Omniscience, SimpleQA, gated AA dev, GPQA and AIME" width="100%">

| Knowledge (held-out) | build | closed-book | + engram | Δ | correct / incorrect with engram |
|---|---|---:|---:|---:|---:|
| SimpleQA-verified (dev 200) | **IQ1_M 6.73 GB** | −14.0 | **+43.0** | **+57.0** | 93 / **7** |
| | IQ2_S 9.26 GB | −25.5 | +38.5 | +64.0 | 99 / 22 |
| AA-Omniscience (test 400), plain prompt | IQ1_M | **−7.2** | −3.5 | +3.7 | 72 / 86 |
| | IQ2_S | −19.8 | **−1.0** | **+18.8** (93 better / 36 worse, p = 5·10⁻⁷) | 115 / 119 |

The index is 100 × (correct − incorrect) / n, so a wrong answer costs as much as a right one earns. The engram
mostly helps by **replacing confident wrong answers with grounded ones**. The 1-bit build starts better calibrated
(it declines instead of guessing: 71 wrong vs 149 on AA closed-book) and, with the `v1` prompt, keeps that discipline
when passages are injected.

These test-split arms use the Software + Law tables with the plain injection prompt on AA, and the Wikipedia table on
SimpleQA. They are self-judged at low reasoning effort, so the absolute values are not comparable with published AA
numbers; the paired deltas are the result. The release route is the per-domain gate with the `v1` prompt (below).

### Per-domain gate (AA dev 200)

Each table is kept for a model only if it helps that model in a paired dev test. `v1` is the injection prompt that
says: *use a passage only if it states the answer to this exact question; otherwise rely on your own knowledge or say
you don't know.*

**IQ1_M (6.73 GB):**

| domain | closed-book | engram, plain | engram, `v1` | better / worse (`v1`) | gate |
|---|---:|---:|---:|---:|---|
| Software | −23.5 | **+47.1** | +44.1 | 20 / 1 (p < .001) | on |
| Law | 0.0 | +15.2 | **+21.2** | 10 / 3 | on, `v1` |
| STEM | −15.2 | −9.1 | **+15.2** | 13 / 4 (p = .049) | on, `v1` only |
| Finance | −12.1 | **+6.1** | +3.0 | 8 / 3 | on |
| Health | −36.4 | **−3.0** | −6.1 | 11 / 4 | on |
| Humanities | 0.0 | −2.9 | +2.9 | 3 / 3 | flat |
| **all 200** | **−14.5** | **+9.0** | **+13.5** | | best per domain ≈ **+15** |

**IQ2_S (9.26 GB):** closed-book −32.5 → +1.5 with the plain prompt; Health +6.1 with `v1`; gated ≈ +3.

| build | closed-book | gated engram | gain |
|---|---:|---:|---:|
| **IQ1_M 6.73 GB** | **−14.5** | **≈ +15** | +30 |
| IQ2_S 9.26 GB | −32.5 | ≈ +3 | +35 |
| reference 10.94 GB build (not released) | −22.5 | ≈ +6 | +28 |

The 1-bit build starts highest because it guesses less, gains about as much from the engram, and with `v1` is the
only build where STEM tables help. The simple route, v2 tables with `v1` on every domain, scores +13.5 with no
per-domain tuning. STEM and Humanities remain coverage-limited (the answer is in the top 3 passages for few
questions), and better sources for both are being built for Tablinum-2.

### Reasoning (pending)

GPQA Diamond and AIME 2026 are running at low effort with a 32K budget (an 8K cap zeroed every answer that hit it).
For reference, Qwen's published scores for the full-precision model:

| benchmark | Qwen3.8-27B (published) | Lite-Beam 27B 9.26 GB |
|---|---:|---:|
| GPQA Diamond | 89.2 | running (32K) |
| AIME 2026 | — | running (32K) |
| HLE (no tools) | 30.8 | not run |
| IFBench | 79.5 | not run |
| LiveCodeBench v6 | 90.3 | not run |

Published values are Qwen's, on their harness. Ours will be added as they finish.

## What changed

| | IQ2_S (recommended) | IQ1_M (1-bit) |
|---|---:|---:|
| size | **9,259,511,744 bytes (8.62 GiB)** | **6,729,166,784 bytes (6.27 GiB)** |
| bpw | **2.71** | **1.97** |
| KLD vs BF16, chat (assistant tokens) | **0.121** | 0.418 |
| KLD vs BF16, wikitext | **0.174** | 0.475 |
| same top-1 token, chat | — | 85.8% |
| imatrix / calibration | Tablinum | Tablinum |
| per-tensor layout | ISTA-DASLab GSQ-RCO IQ2_S | Unsloth UD-IQ1_M |

The same per-tensor layout, built with the stock quantizer and imatrix, scores 0.237 KLD on the chat set, so our
calibration and quantizer roughly halve it at identical bytes. KLD is measured only on assistant-turn tokens: an
instruct model's system and user tokens are untrained and swamp the mean otherwise.

IQ1_M KLD was measured on the CPU; IQ2_S on the GPUs. The 1-bit build keeps 86% top-1 agreement with BF16, but its
closed-book knowledge drops more, which the engram is there to recover.

**IQ2_S map:** attention IQ3_S / IQ3_XXS, FFN mostly IQ2_XXS–IQ3_XXS with a few IQ1 bands, output IQ4_XS, token
embeddings IQ2_S, DeltaNet gates F32/BF16. The exact per-tensor maps are in the two manifests.

## Files

| file | bytes | size | use |
|---|---:|---:|---|
| `Lite-Beam-27B-IQ2_S.gguf` | 9,259,511,744 | **8.62 GiB** | llama.cpp |
| `Lite-Beam-27B-IQ1_M.gguf` | 6,729,166,784 | **6.27 GiB** | llama.cpp, smallest machines |
| `LICENSE` | — | — | Apache-2.0 |
| `NOTICE` | — | — | attribution |
| `quantization_manifest.json` | — | — | IQ2_S: exact map, KLD, hash |
| `quantization_manifest_IQ1_M.json` | — | — | IQ1_M: exact map, KLD, hash |
| `benchmarks/` | — | — | graded arms and scripts |

SHA-256:
- `Lite-Beam-27B-IQ2_S.gguf` `f08759290b3b117d3f8a7f33914148ec13b0186c6c791a5b86eaad53c362ac63`
- `Lite-Beam-27B-IQ1_M.gguf` `a2c792dc720efd4077cd1cf4c903208a924947c653b2f3188472eb979ea0ca42`

## Run locally

Reference rig: RTX A2000 12 GB + RTX 2060 6 GB, split 7:2.

```bash
llama-server -m Lite-Beam-27B-IQ2_S.gguf --alias lite-beam-27b \
  -ngl 99 -ts 7,2 -fa on -np 4 -c 57344 -ctk q8_0 -ctv q8_0 \
  --jinja --host 127.0.0.1 --port 18371
```

| speed (A2000 + 2060) | t/s |
|---|---:|
| prompt 512 | 361 |
| prompt 4096 | 430 |
| generation | 14.6 |
| generation with EAGLE-4 draft (1 slot) | **23.6** (19.8–27.4) |

**Faster single-stream:** the EAGLE-4 draft for Qwen3.8-27B works with this quant, at +63% generation speed on our
rig and close to the reference acceptance rate. It needs a llama.cpp fork with EAGLE-4 support; see
[`quimmedes/Qwen3.8-27B-EAGLE4-GGUF`](https://huggingface.co/quimmedes/Qwen3.8-27B-EAGLE4-GGUF). The draft is under
the Eagle 4 source-available license, not this repo's, and is not bundled.

## Tablinum-2 engram

The gated retrieval tables, routes and code are in
**[`tablinum/tablinum-2`](https://huggingface.co/datasets/tablinum/tablinum-2)**. Tables whose
sources do not allow redistribution are shipped as exact build recipes, not text. The engram runs locally; nothing
is sent to a remote endpoint.

## What you need

- About **10 GB of VRAM across one or two GPUs** for the 9.26 GB file with a 57K context, or about **8 GB** for the
  1-bit file. CPU offload works but is slow.
- A recent **llama.cpp** with Qwen3.5/3.8 (`qwen35`) support and `--jinja`.
- Optional: the Tablinum-2 tables (sizes vary per domain) and its retrieval code.
- For maximum accuracy use the BF16 / FP8 checkpoint. This is the 2-bit local build.

## License and attribution

Qwen3.8-27B is by the Qwen team (Alibaba Cloud), Apache-2.0. This GGUF is a modified, quantized derivative produced
by Lightspeed / Tablinum. The per-tensor type layout follows
[ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) (Apache-2.0); the
tensor data is our own quantization. The IQ1_M build follows the per-tensor type layout of
[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-IQ1_M (Apache-2.0); again the tensor data
is our own. Include the Apache-2.0 text and the NOTICE with any redistribution. No
endorsement by Qwen, ISTA-DASLab or Unsloth is implied.

## Citation

```bibtex
@misc{litebeam27b2026,
  title  = {Lite-Beam 27B},
  author = {Lightspeed / tablinum},
  year   = {2026},
  note   = {Mixed 2-bit GGUF of Qwen/Qwen3.8-27B with a per-model gated engram},
  url    = {https://huggingface.co/tablinum/Lite-Beam-27B}
}
```
