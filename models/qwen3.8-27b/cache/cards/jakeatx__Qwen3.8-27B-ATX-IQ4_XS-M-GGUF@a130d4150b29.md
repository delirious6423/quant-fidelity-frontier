---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: llama.cpp
pipeline_tag: text-generation
tags:
- gguf
- qwen3.8
- rtx-3090
- long-context
- speculative-decoding
- mtp
---

# Qwen3.8-27B ATX-IQ4_XS-M (GGUF)

A speed-oriented quantization of Qwen3.8-27B for a single RTX 3090 / 3090 Ti (24 GB), designed to fit a populated 200K-token context with an 8-bit key cache and the model's own MTP speculative head, and to decode faster per speculative round than both Unsloth's Q3_K_XL and Q4_K_M on that card.

| file | GPU-resident weights | notes |
|---|---:|---|
| `Qwen3.8-27B-ATX-4-XS.gguf` | 13.9 GiB | the file keeps the working name; see the note below |

**On the name.** ATX-IQ4_XS-M reads as: base format IQ4_XS on the bulk tensors, upgrade pattern M. In llama.cpp's vocabulary the XS in IQ4_XS is part of the format's name (the 4.25 bits-per-weight super-block layout of the IQ4 codebook), not a mix size, while the S/M/L suffix on the K-quants describes how many tensors are lifted above the base format. This file lifts the same tensors Q4_K_M does (Q5_0, Q6_K and Q8_0 on attention K/V, attention output, GDN output and FFN down), so it is an M-pattern mix on an IQ4_XS base: 4.56 bits per weight, between Q4_K_S (4.58) and Q4_K_M (4.82) in size. It was developed under the working name ATX-4-XS, which the GGUF filename, the tensor map and the run logs keep.

The same recipe applied to Jackrong's Qwopus3.8-27B-Flash fine-tune, with a model-specific importance matrix and a paired speed comparison against its Q4_K_S, is at [jakeatx/Qwopus3.8-27B-Flash-ATX-IQ4_XS-M-GGUF](https://huggingface.co/jakeatx/Qwopus3.8-27B-Flash-ATX-IQ4_XS-M-GGUF).

## Recipe

Built from Unsloth's Qwen3.8-27B BF16 GGUF with Unsloth's importance matrix (`imatrix_unsloth.gguf` from their UD-IQ4_XS release) and a per-tensor type map:

| tensors | format | share of weight bytes |
|---|---|---:|
| everything not listed below | IQ4_XS | ~75% |
| attn_output, ssm_out, ffn_down in the layers Unsloth's tier ladder upgrades first | Q5_0 | ~17% |
| attention K/V projections, output head | Q6_K | |
| the eight attention K/V tensors Q4_K_M keeps at Q8_0 (V in layers 11, 27, 31, 51, 55, 59, 63; K in 31) | Q8_0 | |
| MTP draft layer (blk.64) | Q5_0 | |
| GDN alpha / beta vectors (96 tiny, 0.01 GiB total) | Q8_0 | |

```
llama-quantize --imatrix imatrix_unsloth.gguf --tensor-type-file tensor_types_ATX-4-XS.txt \
               --token-embedding-type q4_K Qwen3.8-27B-BF16.gguf Qwen3.8-27B-ATX-4-XS.gguf iq4_xs
```

Why this mix: on SM86 the fastest weight format per tensor at speculative verification widths 1-5 is IQ4_XS, not the 2-3 bit codebook types (IQ3_S, IQ3_XXS, IQ2_S are instruction-bound and slower despite fewer bytes); Q5_0 is about 16% cheaper than Q5_K; Q8_0 is the only format near the memory roof. Extra bits go where Unsloth's tier ladder puts them (attention V/K, attention output, GDN output, FFN down), not where the imatrix shows the largest outliers; the two correlate only weakly (0.14).

## Measured (RTX 3090 Ti, 350 W, TurboQuant+ fork with SM86 kernel work, MTP-3, q8_0 K / turbo3 V)

| | ms per speculative round 64K / 100K | prefill tok/s 64K / 100K | 45-prompt protocol | max context |
|---|---:|---:|---|---:|
| Q3_K_XL (11.7 GiB) | 46 / 53 | 1055 / 903 | 18 / 71 (reference) | 262K |
| Q4_K_M (14.6 GiB) | 52 / 57 | 1063 / 910 | reference | ~230K (est.) |
| **ATX-IQ4_XS-M (13.9 GiB)** | **42.2 / 48.1** | **1170 / 993** | 17 / 70 | 245K (measured) |

Sampled decode at temperature 1 / top-k 20 / top-p 0.95, 256-token generations: about 93-108 tok/s at 16K, 60-79 at 64K, 44-72 at 100K depending on workload (coding and agentic at the top, retrieval at the bottom). Quality is expected between Q3_K_XL and Q4_K_M: every tensor Q3_K_XL holds below 4 bits is at 4.25 bpw or higher here, while the bulk tensors carry a quarter-bit less than Q4_K_M. A matched Terminal-Bench / SciCode / GDPval comparison is in progress.

## Run it

The largest measured window is 245,760 tokens (22.1 GiB ready with a 240K prompt). Use `-c 204800` for 200K at 20.9 GiB if you want margin.

Runtime with the kernel and memory work: https://github.com/JakeATX/llama-cpp-qwen-ampere, branch `main` at `fdfea8123` (TheTom's TurboQuant+ branch as of 2026-09-03 plus the SM86 work). Build with `-DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES=86`.

```
GGML_Q8_TURBO3_MMA_FUSED=1 llama-server -m Qwen3.8-27B-ATX-4-XS.gguf \
  -c 245760 -b 4096 -ub 1024 -t 8 -tb 8 -ngl 99 -fa on -ctk q8_0 -ctv turbo3 \
  --parallel 1 --jinja --fit off \
  --cache-prompt --cache-ram 8192 --ctx-checkpoints 24 --checkpoint-min-step 10240 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.45 \
  --spec-draft-type-k q8_0 --spec-draft-type-v turbo3
```

The three cache flags are what make a long conversation usable rather than merely possible. `--cache-prompt` keeps the conversation's KV cache in the slot between turns, so a new turn on a 200K conversation pays only for the new tokens instead of a 100-second re-prefill. `--ctx-checkpoints` matters specifically for this model: 48 of its layers are recurrent, and a recurrent state cannot be rewound, so when you edit or regenerate a turn the server needs a saved state from before the edit point; it keeps up to 24 of them, at least 10,240 tokens apart (and one at every user turn regardless), in host RAM. Measured on this model, a snapshot is 150 MiB plus 1.5 KiB per token of position, because the MTP drafter's own single-layer KV cache is saved with the recurrent state: about 165 MiB at 10K, 495 MiB at 240K. 24 at 10,240 spacing covers the whole 245,760 window for about 7.8 GiB with at most ten seconds of replay after an edit; denser spacing multiplies that RAM (60 at 4,096 is about 20 GiB at the deep end). `--cache-ram` is a separate host-RAM budget for parking a whole conversation's KV (with its checkpoints) when another conversation takes the slot; a populated 200K conversation is about 7.4 GB, so 8 GiB holds one, and it only does work when you switch between chats. None of this touches VRAM. Budget about 16 GB of host RAM for it at the deep end (up to 8 GiB of snapshots plus the 8 GiB park space) on top of the model's own mapping; on a 32 GB machine keep the desktop light, or drop the count to 12 at 20,480 spacing for half the snapshot RAM.

**Optional disk tier (this fork only).** Add `--cache-disk-path /fast-nvme/llama-cache --cache-disk-limit 65536` and conversations that fall out of the RAM cache, or never fit it (a full 200K-245K session is 7-9 GB), are written to disk and restored from there on return, with only the new tokens processed. Restores run at about 1.5 GB/s, so a 200K conversation is back in a few seconds instead of a 100-second re-prefill; the restored state is exact and the index survives a restart.

Single-user configuration (`--parallel 1`, one request at a time; a second slot would need its own ~7 GiB KV cache at 200K). Ready VRAM with a populated 200K prompt is 21,372 MiB (20.9 GiB), peak 21,422 MiB, against a usable ceiling of about 23.0 GiB; at that depth the file prefills at about 690 tok/s and decodes at about 65 tok/s. In a 288-turn agent session on the merged runtime that grew the context from 100K to 245K (120K generated tokens, tool results between turns, EOS never suppressed), decode averaged 52 tok/s: about 60 tok/s at 110K, 47 tok/s at 245K, draft acceptance 76%. ATX-IQ4_XS-M's 245K is measured: a 245,760-token window with a 240,000-token prompt loads at 22,600 MiB ready, 22,634 MiB peak. The Q4_K_M cell marked (est.) is extrapolated from the runtime's ~37 MiB per 1,000 tokens. The file also loads on stock TurboQuant+ and on mainline llama.cpp (without the turbo3 cache there), just with less context headroom.

## Credits

Qwen team for Qwen3.8; Unsloth for the BF16 GGUF, the imatrix, and the dynamic-quant tier ladder this recipe follows; TheTom for TurboQuant+.

<!-- qwen38-perian-lineage:start -->
## Qwen3.8 Perian project lineage

This repository is retained in the
[Qwen3.8 Perian checkpoints collection](https://huggingface.co/collections/jakeatx/qwen38-perian-checkpoints-6aa33bb7acf0eba80edece40).
Its exact position in the lineage is: **Baseline/comparator quant retained with the project checkpoints. It is not a Perian-pruned or Perian-QLoRA artifact.**

The final [Qwen3.8 Perian GGUF release](https://huggingface.co/jakeatx/Qwen3.8-Perian-GGUF)
combines three reductions and one post-training stage:

- depth: **48 to 32 transformer layers**;
- routed-expert width: **384 to 288 experts per layer**;
- PLE n-gram capacity: **320,001,446 to 160,000,768 rows** (50%, about
  25.60B parameters removed), using activation-aware bigram and
  frequency-ranked trigram selections validated on a document-disjoint
  5M-token holdout;
- **rank-32 QLoRA** on **12,558 normalized traces** spanning math/STEM
  reasoning, coding/debugging, agentic tool use, retrieval, and general
  multi-step reasoning. The trace mixture draws from several frontier-model
  families, including Fable 5, GLM 5.2, Kimi K3, Claude Opus 4.7,
  Qwen3.8-Max, and GPT-5.6-Sol. The final merged milestone was trained through
  **9,336,692 supervised assistant tokens**.

Earlier checkpoints in this collection do not inherit later stages merely by
being listed beside them; the stage statement above is authoritative for this
artifact.
<!-- qwen38-perian-lineage:end -->
