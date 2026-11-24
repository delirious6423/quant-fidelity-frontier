---
base_model: Qwen/Qwen3.8-Flash-Next
quantization: gptq
tags:
  - auto-round
  - w4a16
  - intel-arc
  - intel-xpu
  - battlemage
  - code
pipeline_tag: text-generation
---

# Qwen3.8-Flash-Next (W4A16 Auto-Round GPTQ, group 64)

A 4-bit quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), tuned for coding
and built for two **Intel Arc Pro B70 (Battlemage)** cards with the
[intel-arc-inference-server](https://github.com/urakozz/intel-arc-inference-server) engine.

Quantized with [Auto-Round](https://github.com/intel/auto-round), tuned on `nvidia/OpenCodeInstruct` through the chat
template, to keep code reasoning and tool calls accurate.

## Quantization configuration

| Parameter | Value |
| --- | --- |
| **Method** | Auto-Round (tuned) |
| **Scheme** | W4A16 (4-bit weights, 16-bit activations), symmetric |
| **Format** | `auto_round:auto_gptq` |
| **Group size** | 64 |
| **Calibration data** | `nvidia/OpenCodeInstruct` (chat template), 128 samples x 2,048 tokens |
| **Iterations** | 200 |
| **Batch** | 1, gradient accumulation 8 |
| **Seed** | 42 |

Kept in bf16 (exactly the source's tensors): the routers, hyper-connections, `shared_expert_gate`, the gated
delta-net `in_proj_a` / `in_proj_b`, the sparse-attention indexer, the per-layer embedding (PLE) table, norms, the
embedding, `lm_head` (the engine builds its int8 head at load), the vision tower, and the MTP head's attention and
shared expert. The MTP head's routed experts are RTN int4 g64.

The export is about 165 GB, most of it the bf16 PLE table; the routed experts and linears are int4. `ple-int8/`
(51.8 GB) is the same PLE table as int8 rows with bf16 scales, the form the engine reads.

## Performance on two Intel Arc Pro B70

intel-arc-inference-server, two cards, the experts that do not fit VRAM in system RAM.

Prefill, t/s:

| setup | 32k-token prompt | 128k-token prompt |
| --- | ---: | ---: |
| two cards `pp` | **3,415** | **2,918** |
| two cards `tp` | 2,234 | 1,980 |

Generation, t/s:

| setup | code | prose | tool calls |
| --- | ---: | ---: | ---: |
| plain, two cards `pp` | 54-59 | 48-49 | 52-63 |
| plain, two cards `tp` | 59-66 | 49-55 | 54-70 |
| MTP, two cards `pp` | 72-89 | 45-59 | 78-105 |
| MTP, two cards `tp` | **86-89** | **54-69** | **73-108** |

## Running on two Intel Arc Pro B70

Needs at least 128 GB of system RAM (the experts that do not fit VRAM, the n-gram table and the prefix cache take
about 104 GB).

```bash
hf download urakozz/Qwen3.8-Flash-Next-W4A16-g64-AutoRound-GPTQ
./build/src/cli/b70-serve urakozz/Qwen3.8-Flash-Next-W4A16-g64-AutoRound-GPTQ --parallel pp
```

MTP and the prefix cache are on by default; `--parallel tp` for the faster single session. Setup and flags: the
[intel-arc-inference-server README](https://github.com/urakozz/intel-arc-inference-server).


## Quantisation provenance

- tool: auto-round at `958d6eb49b1edbd028770fad648f5d764f01200b` (our fork, branch `local-patches-v2`: intel/auto-round main `e7178d8aa99889b38e8c0ce89adf99bffc12cca6` + the cached-OpenCodeInstruct commit; version string 0.17.0)
- recipe: `tools/quantize/quantize_qwen4exp.sh` - W4A16, group 64, symmetric, format
  auto_round:auto_gptq, tuned: 200 iterations on 128 x 2048 rows of `nvidia/OpenCodeInstruct:apply_chat_template`, batch 1;
  kept bf16: routers, hyper-connections, shared_expert_gate, GDN in_proj_a / in_proj_b, the QSA indexer, the
  PLE layer, norms, the embedding, vision, the MTP head's attention and shared expert; the MTP head's routed
  experts RTN int4 g64 (not tuned: transformers does not build the head); `lm_head` kept bf16 (the engine builds its int8 head at load); qwen4exp_recipe.json has the call
- source: `Qwen/Qwen3.8-Flash-Next` at `812d96e10a0dd82b69d749e68693698cf6c36daa`; made 2026-11-23T21:30Z
- checked by `tools/quantize/qwen4exp/check.py` (with the int8 PLE file `812d96e10a0dd82b69d749e68693698cf6c36daa-w4g64-ple-int8`) and
  `tools/quantize/check_gptq_export.py --lm-head keep --source`
