---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- deepseek
- deepseek-v4.1
- moe
- quantized
- 2-bit
- ternary
- mixed-precision
- gptq
- uncensored
- strata
- consumer-gpu
---

<div align="center">

# DeepSeek-V4.1-Flash-1017 · XQ mixed-tier: ternary / 2-bit / 3-bit experts (V4 163 GB coming 26–27 Oct · V3 163 GB · V2 152 GB)

**A 552B-parameter MoE running on one gaming PC — RTX 4090 (24 GB) + 90 GB RAM + NVMe**

Runs on **[Xolotl](https://github.com/Indras-Mirror/Xolotl)** (formerly Strata-DSV4.1), an engine built for exactly
this model and this class of machine, on top of [Strata](https://github.com/Niko1221/Strata).

</div>

> ## V4 is coming: quantising now, release expected around 26–27 Oct
>
> Same size as V3 (163 GB / 152 GiB). It runs XQ, our own per-expert refinement, on every routed expert, on top of a new per-tier recipe, a much bigger calibration set drawn from real agent and code data (SWE-rebench, SWE-Hero, SWE-smith trajectories, OpenCodeReasoning, Swift-1.5's agent set) and a fresh tier map. Early numbers (23 Oct, 4 of 40 layers done): each layer's expert error on held-out text is 29–48% lower than the same layer in V3. The new tier map on its own, with both builds plainly rounded at the same size, cuts divergence from the full model by 24% (KL 0.634 → 0.480) and lifts top-1 agreement from 78.8% to 82.4%. These are in-progress measures; the release will have the full gates (perplexity, top-1, Gauntlet, decode speed). Decode speed should land close to V3's since the size is the same. The goal is still to beat our September V4.1 V1 on the Gauntlet.

> **Builds in this repo (20 Oct)**
>
> **V2** (`v2/`, 152 GB / 142 GiB): the first 1017 build. Calibrated on a chat-weighted mix with a gentler tier map, and made smaller than our September V4.1 V1 so it decodes faster. Head gate 6.93 ppl / 0.670 top-1 vs the edited teacher's 6.57 / 0.687 (+5.5% ppl, -1.7 pt). Short Gauntlet 2/5, with 03 at 0.00 and 07 at 0.91 (unsolved), so a step behind our September build on the two hardest tasks. Gauntlet decode 26.4 tok/s.
>
> **V3** (`v3/`, 163 GB / 152 GiB): same checkpoint and edit, recalibrated on a code + agentic heavy mix (about 40% code, 30% agentic/tool use) with our V1 tier recipe, so it's a bit bigger. Head gate 6.72 ppl / 0.691 top-1 vs the teacher's 6.36 / 0.706 (+5.7% ppl, -1.5 pt, measured on its own held-out mix so not directly comparable to V2's). Gauntlet 03 still 0.00, 07 up to 0.95 but still unsolved. Gauntlet decode 22.3 tok/s (bigger file, fewer experts fit in VRAM), DSpark around 3.1 tokens/pass, prefill about 249 tok/s on a fresh 16.8K prompt. Honestly it's only a small step over V2, the hard agentic tasks didn't really move.
>
> All speeds here are on Linux with an RTX 4090. On Windows DSpark can end up slower than running without it, see [Xolotl #14](https://github.com/Indras-Mirror/Xolotl/issues/14).

> This file is **not** a llama.cpp / Ollama / LM Studio GGUF. It uses per-expert quant types those runtimes don't load.
> Use it with Xolotl.

## What this is

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) has 40 layers with 384 routed experts each
(top-6 per token), a shared expert, Hyper-Connections, CSA sparse attention, and an **Engram** n-gram memory. Its
15,360 routed experts hold 543.6B weights; even at their native FP4 they're far bigger than any consumer PC's RAM.

This repository holds a **calibrated mixed-tier quantization** of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), carrying our own
**rank-2 refusal-direction edit** (see *The edit* below). **Every expert's gate/up and down projections get their own
format**:

| Format | Grid | Bits per weight | Share of expert weights |
| --- | --- | ---: | ---: |
| **PTQ1_0** (ternary) | {−1, 0, +1} × fp16 scale per 128 | 1.75 | **62.4%** |
| **Q2_0** | {−1, 0, +1, +2} × fp16 scale per 64 | 2.25 | 24.7% |
| **Q3_K** | 3-bit, 6-bit sub-scales per 16, fp16 per 256 | 3.44 | 13.0% |
| **All experts** | | **2.093** | |

The dense half (attention, shared expert, output head) is **Q8_0**. The embedding, compressor and indexer stay F16,
and norms, router and mHC stay F32.

The expert budget is **132.4 GiB**. On the calibration proxy, this allocation has **20.9% less output error than
uniform Q2_0** (1.7663 vs 2.2318) while being about 10 GiB smaller (uniform Q2_0 would be ~142 GiB).

Unlike the V4.1 release, the allocation here is deliberately **gentler in layers 20–31 and 33–35**: in that band at
most 85% of the expert projections (gate/up and down counted separately) may be ternary, instead of no cap, so it is no
longer nearly all-ternary. Every other layer stays capped at 50% of projections (~55-57% of expert weights where the cap
binds, since gate/up holds twice the weights of down). The V4.1
measurements showed that band carried roughly half of the chat-text quality loss, and the rebalanced calibration mix
weights chat traffic accordingly. The cost: 4.8% worse proxy error than the uncapped pick (1.7663 vs 1.6846) and 2 GiB more (132.4 vs 130.4 GiB).
The proxy doesn't see chat behaviour, and that is what the cap is meant to protect.

**Naming.** This was first released under a "GSQ" name. That overstated it: the build uses GSQ-family format grids
and an allocation idea from [RCO](https://arxiv.org/abs/2605.00649), but the codes themselves come from **GPTQ**, not
from [GSQ](https://github.com/IST-DASLab/GSQ)'s Gumbel-softmax optimisation. So it is renamed **XQ** (Xolotl Quant),
our own method: this build is XQ's first stage (GPTQ + our budget-exact "RCO-lite" tier allocation). A
DeepSeek-V4.1-specific GSQ-style refinement (learned codes and scales on each expert's own routed tokens, through the
real SwiGLU output) is being built as XQ's next stage and will be released as its own folder once it is measured.

## The edit

We applied a **mrfakename-style rank-2 refusal-direction edit** to the 1017 checkpoint, on the attention output
projection and the shared-expert down projection (experts untouched). The manifest id is `adea0515bbf69a0b`. The edit
is applied before quantization to the dense half in full precision. A refusal-direction probe on the served quantized
model is **to be measured**; this card does not claim a refusal rate until that probe is run.

## Why

We wanted a frontier-class open model running **locally, on hardware people actually own**, without slicing it down to
a distilled toy. The design question was where the bits matter. A MoE spends most of its parameters on experts that
each token touches only rarely, so we spend the bit budget where the calibration says it counts, and run the rest
as cheaply as the quality numbers allow.

The engine then treats the experts as a three-tier cache:

| Tier | Holds | Fed by |
| --- | --- | --- |
| VRAM cache | the hottest ~1,300–1,600 experts | LRU + routing bias toward resident experts |
| Pinned RAM arena | ~6,700 experts (60 GiB) | DMA straight to the GPU |
| NVMe file tier | the other ~8,700 experts | O_DIRECT reads, prefetched a layer ahead |

The hotness ranking in `rank-1017-085.bin` is derived from this build's own calibration routing: the first 64 GiB of
experts cover 79.8% of routed tokens, 80 GiB covers 87.4%, 100 GiB covers 94.7%, and 120 GiB covers 99.0%.

## How it was made

The whole pipeline is open source in
[Xolotl/tools/v41](https://github.com/Indras-Mirror/Xolotl/tree/main/tools/v41). It runs on a single
RTX 4090.

**0. Weight-only metrics can't rank these experts.** V4.1's routed experts were trained in FP4, so all 384 experts of a
layer share the same per-32 weight distribution. Which experts deserve more bits can only come from **activations**:
what the experts actually compute on real text.

**1. Calibration corpus (rebalanced for chat).** Real coding-agent session transcripts (secrets scrubbed value by
value, tool results clipped) plus antirez/ds4's 4,690 imatrix prompts, all re-rendered with DeepSeek's own V4.1 chat
encoder (DSML tool-call format), and rebalanced toward chat traffic for this pass. That is 128 sequences × 2,048
tokens, with about a quarter of each domain **held out**: 98 sequences fit the quantizer, and 30 only ever measure it.

**2. A layer-streaming teacher forward.** DeepSeek's own reference `model.py` runs **one layer at a time** straight
from the HF shards, keeping the hidden state on disk between layers. That makes the teacher the real model, not an
approximation. Each layer's MoE runs expert-major: every FP4 expert is dequantized once and sees all of its routed
tokens.

**3. Per-expert GPTQ at three exact grids.** For every expert, gate+up are fitted on the Hessian of its input, and down
on the Hessian of its clamped-SwiGLU × routing-weight input. Scales are chosen **before** GPTQ by a diag(H)-weighted
search (11 clip ratios × both signs) and rounded to what each format stores, so every result is exactly representable
in its GGUF type. GPTQ (damp 1.0) then picks the integers. It beat round-to-nearest in every group (error ratios
0.58 ternary / 0.70 Q2 / 0.74 Q3). Each expert's error is measured on the **held-out** sequences only, as output error
with the routing weight included.

**4. Budget-exact allocation ("RCO-lite").** The cost of a choice is held-out output error divided by the layer's
routed output energy, so every layer counts equally; the first two and last three layers count double. Gate+up and
down are allocated separately. The budget is 132.4 GiB, solved by Lagrangian relaxation (bisection on λ) plus a greedy
pass to spend the remainder. Guard rails: at most 50% of each layer's expert projections may be ternary outside the gentler band, at most 85%
inside layers 20–31 and 33–35, and experts with fewer than 8 held-out tokens are floored at Q2_0, so an expert can't drift to
ternary without evidence.

**5. Sequential quantize pass.** A second layer-by-layer chain quantizes every expert at its chosen format. Each layer
is fitted on the outputs of the **already-quantized** layers below it, so later layers adapt to the errors earlier
ones actually make.

**6. Bit-exact export.** Block encoders write the GGUF bytes directly and are checked bit for bit against ggml's
dequantizers and the PTQ1_0 reference. Ternary packs 5 trits per byte (28 bytes per 128 weights), and Strata runs it
natively on the CPU and in its grouped-expert CUDA kernels.

### Where the bits went

Per-layer held-out relative output error from the sequential quantize pass is **to be measured** once the pass
completes; the table below will be filled from the build logs.

| Layers | Expert bpw | Held-out relative output error | |
| --- | ---: | ---: | --- |
| 0–19 | to be measured | to be measured | |
| 20–31 | ~2.0 | to be measured | the gentler 85%-ternary band |
| 32 | to be measured | to be measured | the most sensitive layer in V4.1 |
| 33–35 | ~2.0 | to be measured | the gentler 85%-ternary band |
| 36–39 | to be measured | to be measured | |
| **All 40** | **2.093** | to be measured | |

### Not done yet

- **XQ refinement (GSQ-style, in development):** Gumbel-softmax optimization of the grid assignments at each expert's chosen format. This is
  the step that closes most of the gap to vector quantization in the GSQ paper. It slots in where GPTQ runs and
  writes the same format.
- A Hadamard rotation for the ternary tier.
- The full RCO manifold search: the allocation above is its Lagrangian "lite" version.

## Quality

Perplexity on the **49,146 held-out calibration tokens** (never used for fitting). The quantized head gate and the
edited full-precision teacher are both measured; the behavioural probes and a short Gauntlet are measured below.

| Domain | XQ (quantized) | Edited teacher (full precision) |
| --- | ---: | ---: |
| All (head gate) | **6.934 ppl / 0.670 top-1** | **6.572 ppl / 0.687 top-1** |
| chat | 5.489 ppl / 0.661 top-1 | 5.170 ppl / 0.681 top-1 |
| code | 2.161 ppl / 0.828 top-1 | 1.980 ppl / 0.851 top-1 |
| agent | 3.148 ppl / 0.830 top-1 | 3.328 ppl / 0.828 top-1 |

Absolute perplexity here is not comparable to the V4.1 card: this calibration mix is rebalanced, so the held-out text
is different. Compare **quantized vs edited teacher on the same mix**.

**Behavioural probes** (opening-phrase refusal classifier, first 48 words, deliberately conservative): chat mode
**0/100 harmful refused, 0/100 safe refused**; think mode **0/50 harmful, 0/50 safe**, with 0 refusals in the reasoning
text; 0 unjudged. The safe set is the Alpaca fallback (XSTest is gated on the Hub).

**Short Gauntlet** (tasks 01, 03, 06, 07, 10; 2026-10-19): a 4-turn run scored **2/5** (SCORE 2.77, 26.4 tok/s decode).
A fair 6-turn rerun of the two borderline tasks gives **01 PASS@4** and **07 progress 0.91 (unsolved)**. Against the
V4.1 v2 build on the same tasks (4/5: 01@4, 06@3, 07@4, 10@3; 03 0.50), this build matches on 01, 06 and 10 and is
worse on 07 and 03. That is a real regression on this subset. Its cause is not isolated yet: the candidates are the
chat-weighted calibration mix (the V4.1 builds were calibrated agentic-first), the gentler tier map, the edit, or the
1017 checkpoint itself. V3 goes after the first two.

**First-token KL against the unedited base is not measured**: no unedited 1017 quant exists; it can be done later with
the layer-streaming teacher.

## Speed (V2, Linux · RTX 4090 24 GB · Ryzen 7 5700X · 90 GB DDR4 · NVMe)

Measured on this PC (Linux), for V2 (V3's numbers are in the notes at the top), one 8K prompt + 256-token decode, 2026-10-18, plus a short served smoke test.

| | |
| --- | ---: |
| Prefill, 8K-token prompt | 239.8 tok/s (exact routing) · 255.2 tok/s with the F16 attention gather (bit-identical output) |
| Decode, one-shot, exact routing | 12–13 tok/s |
| Served 1-turn decode with the 1017 DSpark (Q2_K) | 23.1–26.3 tok/s · 24.6–29.9 tok/s on short smoke outputs |
| DSpark acceptance (V3 serve log, 19 requests, mostly code/agent) | median 2.99 tokens/pass, 3.08 overall (1.6-7.7) |
| Same serve test on our V4.1 v2 build, for comparison | 21.4 tok/s (single runs each; DSpark acceptance depends on the text, so treat the gap as indicative) |
| Served prefill, one ~6.5K-token turn | 193.7–198.9 tok/s |
| Model load to `/v1/models` | ~92 s (includes pinning the 60 GiB RAM arena) |

## Files

| File | What |
| --- | --- |
| `v2/V41-XQ-1017.gguf.part-NN-of-MM` | the 152 GB model, split into ~10 GiB byte ranges so a slow or interrupted upload never has to restart more than one part |
| `v2/reassemble.sh`, `v2/SHA256SUMS` | checks every part, joins them into `V41-XQ-1017.gguf`, verifies the result |
| `SHA256SUMS` | hashes of every part and of the joined file |
| `model-0004{7,8}-of-00048.safetensors.NEED-TO-DOWNLOAD-ENGRAM.txt` | **placeholders - you need these two files, download them separately**: the two Engram shards (101.5 GB each) are not hosted here; these say where to get them (step 2 below). The extracted tables will follow here later as 10 GiB parts; until then use the official shards. |
| `v2/V41-XQ-1017.gguf.experts.json` | per-expert offsets and formats |
| `v2/rank-1017-085.bin` | expert hotness ranking for this build, used to seed the VRAM cache and RAM arena |
| `v3/V41-XQ-1017-v3.gguf.part-NN-of-16`, `v3/reassemble.sh`, `v3/SHA256SUMS` | V3, the 163 GB model in 16 parts, joined and checked the same way (`--include "v3/*"` in step 1, then `bash V41-XQ-1017/v3/reassemble.sh V41-XQ-1017/v3`) |
| `v3/V41-XQ-1017-v3.gguf.experts.json`, `v3/rank-1017-v3.bin` | V3's per-expert offsets/formats and hotness ranking (use `--rank .../v3/rank-1017-v3.bin` with V3) |
| `tokenizer/` | tokenizer + chat template (identical to the 1017 checkpoint's) |
| `V41-MTP-Vision-1017.gguf` | the **DSpark** speculative-draft stack (`mtp.0-2`: experts MXFP4, a lossless repack of the original FP4; the rest Q8_0) plus the **vision tower** (Q8_0). |
| `V41-MTP-Vision-1017.Q2_K.gguf` | **recommended for DSpark:** the same file with the draft experts requantized to Q2_K. Use with `STRATA_DSPARK=V41-MTP-Vision-1017.Q2_K.gguf STRATA_DSPARK_DEPTH=4` (in `--serve` via `STRATA_SERVE_DSPARK`). Vision support is still in progress. |

## Quick start (4 steps)

**You need:** Linux, an NVIDIA GPU with 24 GB, ~90 GB RAM (the RAM arena is 60 GiB by default; `--arena-gib` lowers
it, at some speed cost), the CUDA 13 toolkit, and fast SSD space for **the model (152 GB) + the Engram tables
(203 GB) = ~355 GB**, plus another ~203 GB *temporarily* while the Engram tables are extracted (or download the
Engram shards to a different drive).

**1. Download this repo and join the model** (~152 GB):

```bash
hf download IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored --include "v2/*" "tokenizer/*" "*.txt" \
   "V41-MTP-Vision-1017*" --local-dir V41-XQ-1017
bash V41-XQ-1017/v2/reassemble.sh V41-XQ-1017/v2              # needs ~152 GB free on top of the parts
# or: bash V41-XQ-1017/v2/reassemble.sh V41-XQ-1017/v2 --delete-parts
```

**2. Download the two Engram shards - they are NOT in this repo yet** (the `…NEED-TO-DOWNLOAD-ENGRAM.txt` files mark
them; 101.5 GB each). V4.1 reads its Engram n-gram memory from disk at run time. The tables are unchanged FP8 rows
from the original checkpoint, so there is no point hosting a third copy:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors \
   model-00048-of-00048.safetensors model.safetensors.index.json config.json --local-dir FP8-engram
```

**3. Build the engine and extract the Engram tables** (one table per fast drive is ideal: decode reads ~48 random
256-byte rows per token):

```bash
git clone https://github.com/Indras-Mirror/Xolotl.git xolotl && cd xolotl
bash tools/xolotl/fetch-llama.sh   # llama.cpp at the pinned commit + Xolotl's ggml patch -> third_party/llama.cpp
cmake -S . -B build-xolotl -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
      -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp \
      -DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON
ninja -C build-xolotl ds41_generate
python3 tools/v41/extract_engram.py ../FP8-engram --to layers.1=/fast/a --to layers.14=/fast/b
# when it has finished (it checks sizes and spot-checks 2000 rows per tensor), ../FP8-engram can be deleted
```

**4. Run** (the first load takes ~50 s: it pins the 60 GiB RAM arena):

```bash
STRATA_NO_LARGEPAGES=1 build-xolotl/ds41_generate -m ../V41-XQ-1017/v2/V41-XQ-1017.gguf --ids-file prompt.i32 -n 128 \
     --backend cuda --experts gpu --engram "/fast/a:/fast/b" --rank ../V41-XQ-1017/v2/rank-1017-085.bin
```

`--ids-file` takes int32 token ids. For chat and agents, run the OpenAI- and Anthropic-compatible server with tool
calls in `tools/v41/serve/` (see "Serving it" in the
[GitHub README](https://github.com/Indras-Mirror/Xolotl)); it uses
`V41-XQ-1017/tokenizer/`.

## Credits

- **DeepSeek** for [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT).
- **mrfakename** for the rank-2 refusal-direction edit style this release uses.
- **[GSQ](https://github.com/IST-DASLab/GSQ)** (Dadgarnia, Tabesh, Nikdan, Helcig, Kurtić, Kleinegger, Alistarh) and
  **[RCO](https://github.com/IST-DASLab/RCO)** (Helcig, Alistarh), IST Austria DAS Lab, for the quantization grids
  and the budget-exact allocation idea this pipeline follows.
- **[antirez/ds4](https://github.com/antirez/ds4)** for its 4,690 imatrix calibration prompts (re-rendered for V4.1).
- **llama-tq** for the PTQ1_0 ternary block format (type 143) and its reference kernels (MIT).
- **[Strata](https://github.com/Niko1221/Strata)** (Niko1221), the tiered MoE engine this work builds on;
  **ggml / llama.cpp** for the compute backend.
- Quantization pipeline, engine port and measurements: **Indra's Mirror** ([Xolotl](https://github.com/Indras-Mirror/Xolotl)).

## Support

If this has been useful to you and you want to help support what I do, you can shout me a coffee on [Ko-fi](https://ko-fi.com/indrasmirror). Every bit helps keep this going :)
