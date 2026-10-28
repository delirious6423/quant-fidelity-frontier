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

**Qwen3.8-27B in a 9.26 GB mixed 2-bit GGUF that runs on two small consumer GPUs. A per-model gated engram pushes
its knowledge scores past where the quant starts.** On SimpleQA it goes from −25.5 to **+38.5**. On the
AA-Omniscience held-out test it goes from −19.8 to **−1.0**: 93 questions better, 36 worse, p = 5·10⁻⁷. Apache-2.0.

This is a quantized derivative of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), not a new base
model. The weights are built with Tablinum's calibration set, importance matrix and quantizer (the Lite-Beam method),
on a per-tensor type layout credited below. A 1-bit build is in testing and will be added to this repo when its
pipeline finishes.

**Quick links:** [Files](#files) · [Run locally](#run-locally) · [Tablinum-2 engram](https://huggingface.co/datasets/tablinum/tablinum-2) · [Lite-Beam 1.0 (501B)](https://huggingface.co/tablinum/Lite-Beam-1.0-501B-A23B-IQS_XXS)

## The result

<img src="assets/results.png" alt="Lite-Beam 27B engram scorecard: closed-book vs Tablinum-2 engram on AA-Omniscience, SimpleQA, gated AA dev, GPQA and AIME" width="100%">

| Knowledge (held-out) | Lite-Beam 27B | + engram | Δ | paired better / worse |
|---|---:|---:|---:|---:|
| AA-Omniscience index (test 400) | −19.8 | **−1.0** | **+18.8** | 93 / 36 (p = 5·10⁻⁷) |
| AA correct / incorrect | 70 / 149 | 115 / 119 | | |
| SimpleQA-verified index (dev 200) | −25.5 | **+38.5** | **+64.0** | 110 / 12 (p = 5·10⁻²¹) |
| SimpleQA correct / incorrect | 8 / 59 | 99 / 22 | | |

The index is 100 × (correct − incorrect) / n, so a wrong answer costs as much as a right one earns. The engram
mostly helps by **replacing confident wrong answers with grounded ones**: incorrect answers fall from 149 to 119 on
AA and from 59 to 22 on SimpleQA.

These knowledge arms use the Software + Law tables on AA and the Wikipedia table on SimpleQA. They are self-judged at
low reasoning effort, so the absolute values are not comparable with published AA numbers. The paired deltas are the
result. The per-domain gated route (below) is the release route; its test-split run is in progress.

### Per-domain gate (AA dev 200)

Each table is kept for a model only if it helps that model in a paired dev test. These are the gate results for
Qwen3.8-27B, measured on the 10.94 GB reference build; the gate for this exact 9.26 GB file is running.

| domain | closed-book | engram v2 | better / worse | gate |
|---|---:|---:|---:|---|
| Software | −20.6 | **+35.3** | 16 / 2 (p = .001) | **on**: software docs + wiki |
| Law | −33.3 | **+15.2** | 14 / 2 (p = .004) | **on**: case law + wiki |
| Health | −33.3 | **+6.1** | 13 / 4 (p = .049) | **on**: health corpus + guidelines + wiki |
| Finance | −24.2 | **+3.0** | 11 / 3 (p = .057) | on, provisional: finance papers + CRS + wiki |
| STEM | −6.1 | −6.1 | 7 / 7 | off: closed-book |
| Humanities | −17.6 | −26.5 | 3 / 6 | off: closed-book (prompt-level gate in test) |
| **all 200** | **−22.5** | **+4.5** | **64 / 24** | gated route ≈ **+6** |

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

| | Lite-Beam 27B (9.26 GB) |
|---|---:|
| size | **9,259,511,744 bytes (8.62 GiB)** |
| bpw | **2.71** |
| KLD vs BF16, chat (assistant tokens) | **0.121** |
| KLD vs BF16, wikitext | **0.174** |
| imatrix / calibration | Tablinum (chat-rendered, no mid-sequence EOS) |
| per-tensor layout | ISTA-DASLab GSQ-RCO IQ2_S layout |

The same per-tensor layout, built with the stock quantizer and imatrix, scores 0.237 KLD on the chat set, so our
calibration and quantizer roughly halve it at identical bytes. KLD is measured only on assistant-turn tokens: an
instruct model's system and user tokens are untrained and swamp the mean otherwise.

**Map:** attention IQ3_S / IQ3_XXS, FFN mostly IQ2_XXS–IQ3_XXS with a few IQ1 bands, output IQ4_XS, token
embeddings IQ2_S, DeltaNet gates F32/BF16. The exact per-tensor map is in `quantization_manifest.json`.

## Files

| file | bytes | size | use |
|---|---:|---:|---|
| `Lite-Beam-27B-IQ2_S.gguf` | 9,259,511,744 | **8.62 GiB** | llama.cpp |
| `Lite-Beam-27B-IQ1_*.gguf` | — | ~6 GB | in testing |
| `LICENSE` | — | — | Apache-2.0 |
| `NOTICE` | — | — | attribution |
| `quantization_manifest.json` | — | — | exact map, KLD, hashes |
| `benchmarks/` | — | — | graded arms and scripts |

SHA-256 (`Lite-Beam-27B-IQ2_S.gguf`):
`f08759290b3b117d3f8a7f33914148ec13b0186c6c791a5b86eaad53c362ac63`

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

- About **10 GB of VRAM across one or two GPUs** for the 9.26 GB file with a 57K context. CPU offload works but is slow.
- A recent **llama.cpp** with Qwen3.5/3.8 (`qwen35`) support and `--jinja`.
- Optional: the Tablinum-2 tables (sizes vary per domain) and its retrieval code.
- For maximum accuracy use the BF16 / FP8 checkpoint. This is the 2-bit local build.

## License and attribution

Qwen3.8-27B is by the Qwen team (Alibaba Cloud), Apache-2.0. This GGUF is a modified, quantized derivative produced
by Lightspeed / Tablinum. The per-tensor type layout follows
[ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) (Apache-2.0); the
tensor data is our own quantization. Include the Apache-2.0 text and the NOTICE with any redistribution. No
endorsement by Qwen or ISTA-DASLab is implied.

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
