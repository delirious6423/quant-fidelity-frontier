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

The export is about 229 GB, the largest part the bf16 PLE table (102 GB); the routed experts and linears are int4. `ple-int8/`
(51.8 GB) is the same PLE table as int8 rows with bf16 scales, the form the engine reads.

## Performance on two Intel Arc Pro B70

intel-arc-inference-server, two cards, the experts that do not fit VRAM in system RAM.

Prefill, t/s:

| setup | 32k-token prompt | 128k-token prompt |
| --- | ---: | ---: |
| two cards `pp` | **3,415** | **2,918** |
| two cards `tp` | 2,234 | 1,980 |

A short prompt (43 tokens) gets its first token in 0.30 s on `pp` and on `tp`

Generation, t/s:

| setup | code | prose | tool calls |
| --- | ---: | ---: | ---: |
| plain, two cards `pp` | 54-59 | 48-49 | 52-63 |
| plain, two cards `tp` | 59-66 | 49-55 | 54-70 |
| MTP, two cards `pp` | 72-89 | 45-59 | 78-105 |
| MTP, two cards `tp` | **84-92** | **49-70** | **74-114** |
| MTP, two cards `tp`, PCIe 5.0 x16 + DDR5 4+ channels (estimate) | ~125-155 | | |

Measured on a PCIe 3.0 workstation with DDR4-2133 and peer-to-peer between the cards; the estimate row assumes
PCIe 5.0 x16 to each card and 4 or more DDR5 channels; a CPU with AVX-512 should help further.

## Running on two Intel Arc Pro B70

Needs at least 128 GB of system RAM (the experts that do not fit VRAM, the n-gram table and the prefix cache take
about 104 GB).

```bash
# about 133 GB: the engine reads ple-int8/, so files 4-23 (96 GB of the bf16 PLE table) can stay on the Hub
hf download urakozz/Qwen3.8-Flash-Next-W4A16-g64-AutoRound-GPTQ \
  --exclude "model-0000[4-9]-*" --exclude "model-0001?-*" --exclude "model-0002[0-3]-*"
./build/src/cli/b70-serve urakozz/Qwen3.8-Flash-Next-W4A16-g64-AutoRound-GPTQ --parallel tp --prefix-split-last
```

MTP and the prefix cache are on by default. Context: 65,536 by default; `--max-len 131072` or `262144` for longer,
with fewer experts kept in VRAM.

Or in a container from Docker Hub (the host needs only the `xe` driver):

```bash
docker run --rm --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
    -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 \
    urakozz/intel-arc-inference-server urakozz/Qwen3.8-Flash-Next-W4A16-g64-AutoRound-GPTQ --parallel tp --prefix-split-last
``` Setup and flags: the
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
