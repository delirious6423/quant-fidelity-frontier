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

# DeepSeek-V4.1-Flash-1017 · XQ mixed-tier: ternary / 2-bit / 3-bit / native FP4 experts

**A 552B-parameter MoE running on one gaming PC: RTX 4090 (24 GB) + 90 GB RAM + NVMe**

Runs on **[Xolotl](https://github.com/Indras-Mirror/Xolotl)** (formerly Strata-DSV4.1), an engine built for exactly
this model and this class of machine, on top of [Strata](https://github.com/Niko1221/Strata).

**Builds:** **V4** (`v4/`, 163 GB, recommended) · V3 (`v3/`, 163 GB) · V2 (`v2/`, 152 GB)

</div>

> ### Already downloaded V4? Grab the new rank file
>
> On 25 Oct (about 20:10 UTC on the 24th) I added **`v4/rank-1017-v4.bin`**, V4's own expert hotness ranking. Before that the instructions pointed at `v4/rank-1017-v3.bin` (V3's ranking), which still works fine but leaves speed on the table. The new one is faster and no worse on quality (86.9% top-1 / KL 0.306 on the held-out rows vs 86.8% / 0.316 with the old one). It's 123 KB, no need to touch the 163 GB model:
>
> ```bash
> hf download IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored v4/rank-1017-v4.bin --local-dir V41-XQ-1017
> # then use --rank V41-XQ-1017/v4/rank-1017-v4.bin (and pull the latest Xolotl for the faster CPU kernels)
> ```

> This file is **not** a llama.cpp / Ollama / LM Studio GGUF. It uses per-expert quant types those runtimes don't load.
> Use it with Xolotl.

## V4 (25 Oct): the best build so far

V4 is the first build with our XQ refinement on every routed expert, a new tier map and a much bigger calibration set drawn from real agent and code data (SWE-rebench, SWE-Hero, SWE-smith trajectories, OpenCodeReasoning, Swift-1.5's agent set). It also adds a fourth tier: the 1,056 hottest experts are kept **exactly as DeepSeek trained them**, in their native FP4 (MXFP4, a lossless repack), and the bytes for that come out of the cold experts, which go ternary more often. Same file size as V3.

DeepSeek-V4.1-Flash has 40 layers with 384 routed experts each (top-6 per token), so 15,360 routed experts holding 543.6B weights, plus a shared expert, Hyper-Connections, CSA sparse attention and an **Engram** n-gram memory. Every expert's gate/up and down projections get their own format:

| Format | Grid | Bits per weight | Share of V4's expert weights |
| --- | --- | ---: | ---: |
| **PTQ1_0** (ternary) | {−1, 0, +1} × fp16 scale per 128 | 1.75 | **57.7%** |
| **Q2_0** | {−1, 0, +1, +2} × fp16 scale per 64 | 2.25 | 22.7% |
| **Q3_K** | 3-bit, 6-bit sub-scales per 16, fp16 per 256 | 3.44 | 12.7% |
| **MXFP4** (native, untouched) | DeepSeek's own FP4 + E8M0 scale per 32 | 4.25 | 6.9% |
| **All experts** | | **2.25** | |

The dense half (attention, shared expert, output head) is Q8_0. The embedding, compressor and indexer stay F16, and norms, router and mHC stay F32.

When you serve it with Xolotl's `ds41-engine.sh`, the dense half gets re-quantised to Q6_K at load by default (`STRATA_DENSE_REQUANT=q6_K`, set it empty to keep Q8_0). That frees about 1.7 GB of VRAM for more experts (decode +11% after a 6.5K prompt, more on longer ones), and when we measured it on our earlier V4.1 build it cost next to nothing (KL to the FP8 model 0.2281 to 0.2306, top-1 agreement 85.87% to 85.76%, on 100 fixtures). All the speed and Gauntlet numbers below were run with it on; the agreement numbers in the quality table were measured with the dense half at Q8_0.

### Quality

Held-out text the quantiser never fitted on, compared against the full-precision edited model:

| | V4 | V3 |
| --- | ---: | ---: |
| Top-1 agreement with the full model (7 held-out rows, Python reference path) | **85.7%** | 81.5% |
| KL divergence from the full model (same rows) | **0.343** | 0.471 |
| Top-1 agreement / KL, same rows through the real engine | **86.9% / 0.306** | 82.9% / 0.405 |
| Engine perplexity on those rows (full model 4.98) | **5.04** | 5.24 |

V4's own head gate (57,337 held-out tokens) is **6.369 ppl / 0.705 top-1 against the full model's 6.341 / 0.707**, so +0.4% ppl and -0.2 pt. By domain V4 agrees with the full model 95.1% of the time on code, 93.0% on agent turns, 89.7% on reasoning and 66.1% on long agentic traces (where the full model itself only gets 42% of next tokens right). A Wikitext-2 number (to compare with other quants) is coming.

### Gauntlet v2

The core 7 tasks, 10 turns each, the same 32K thinking budget per turn for every model, hidden tests graded after every turn:

| | Solved | Score | Avg turns | Tokens |
| --- | ---: | ---: | ---: | ---: |
| **V4** | **6/7** | **6.50** | 3.7 | 55,114 |
| DeepSeek API (full model) | 5/7 | 6.41 | 4.8 | 106,792 |
| Our September V4.1 build | 4/7 | 6.20 | 4.0 | 104,648 |

V4 solved everything except 03, which sits at 0.50 for every model I've run (the API included). 07, which the September build never solved, went at turn 4. It's one run per model, so I wouldn't read too much into it edging the API, but it's going toe to toe with it, which honestly I didn't expect. No sign of the overthinking the earlier builds had either: no step came near the thinking cap.

### Speed

Good news, V4 has caught up. At release it was slower than V3, because the hottest experts are now the fattest ones, so fewer of them fit in VRAM and more of each token's experts come from RAM. I spent the last day on the engine and it's climbing nicely. Served with DSpark on my PC (Linux, RTX 4090, Ryzen 7 5700X, 90 GB DDR4), `rank-1017-v4.bin` + `--tier-batch-pcie 0.25`, 512-token replies, two runs each (25 Oct evening):

| | Decode after an 8K prompt | after a 20K prompt | Prefill (8K / 20K) |
| --- | ---: | ---: | ---: |
| **V4 now, dense half Q8_0** | **21.9-22.4 tok/s** | **21.8-23.8 tok/s** | 237-250 / 309-321 tok/s |
| V4 now, dense half Q6_K (`ds41-engine.sh`'s default) | 22.7-23.3 tok/s | 24.2-24.3 tok/s | 237-243 / 364-383 tok/s |
| V4, morning of 25 Oct (Q6_K) | 17.5 tok/s | 19.6 tok/s | 192 / 271 tok/s |
| V4, V3's rank file (first release, Q6_K) | 15.6 tok/s | 16.8 tok/s | 199 / 280 tok/s |

A short follow-up turn (~180 new tokens) on a long chat comes back in about 7-8 s. For reference V3 did 22.3 tok/s across the Gauntlet and our September build 23.4.

None of this touches the model or costs quality, it's all engine work in Xolotl: the route bias now refreshes as the conversation goes (that one actually nudged quality up a bit), expert reads go straight from NVMe into pinned memory with no extra copy, each layer's reads go out all at once, the RAM experts staged for big prompts run on the fast MMQ kernels, and the CPU kernels for the Q2_0 and ternary tiers are a lot faster. Needs Xolotl from 25 Oct or newer. Decode on my box is mostly bound by RAM bandwidth now, so I think the next real jump is a RAM upgrade, but I'm still working on prefill and I'll update this table as it lands.

All speeds here are on Linux. On Windows DSpark can end up slower than running without it, see [Xolotl #14](https://github.com/Indras-Mirror/Xolotl/issues/14).

### How V4 was made

The same layer-streaming pipeline as the earlier builds (details under *Older builds* below), with three changes: a much bigger calibration set (1.58M fitting tokens across agent, long agentic, code, reasoning, chat, web, Chinese and multilingual text, plus 57K held-out tokens that only ever measure), a fresh tier map at a 142.4 GiB expert budget, and the XQ refinement: after GPTQ, each expert's codes and scales are refined on that expert's own routed tokens through the real SwiGLU output, and the refinement is kept only where it beats GPTQ on a validation slice. The hottest experts by the calibration routing stay in DeepSeek's native FP4. The XQ refinement stage itself isn't published (yet); the GPTQ + allocation pipeline is open source in [Xolotl/tools/v41](https://github.com/Indras-Mirror/Xolotl/tree/main/tools/v41).

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

The engine treats the experts as a three-tier cache: the hottest ones in VRAM (about 750-950 of V4's, fewer when the DSpark draft is loaded), about 5,600 in a pinned 64 GiB RAM arena, and the rest read from NVMe with O_DIRECT, prefetched a layer ahead. Misses from the RAM arena are either copied to the GPU over PCIe or computed on the CPU (`--tier-batch-pcie` sets the split).

## Files

| File | What |
| --- | --- |
| `v4/V41-XQ-1017-v4.gguf.part-NN-of-16`, `v4/reassemble.sh`, `v4/SHA256SUMS` | **V4**, the 163 GB model in 16 parts of ~10 GiB, joined and checked by `reassemble.sh`. Needs Xolotl from 21 Oct or newer |
| `v4/rank-1017-v4.bin` | **V4's hotness ranking (use this one)**, seeds the VRAM cache and RAM arena |
| `v4/V41-XQ-1017-v4.gguf.experts.json` | V4's per-expert offsets and formats |
| `v4/rank-1017-v3.bin` | the ranking V4 first shipped with (V3's); still works, slower |
| `v3/...`, `v2/...` | the older builds, same layout (`rank-1017-v3.bin` for V3, `rank-1017-085.bin` for V2) |
| `tokenizer/` | tokenizer + chat template (identical to the 1017 checkpoint's) |
| `V41-MTP-Vision-1017.Q2_K.gguf` | **recommended DSpark draft**: the speculative-draft stack with its experts in Q2_K, plus the vision tower. Use with `STRATA_SERVE_DSPARK=V41-MTP-Vision-1017.Q2_K.gguf` (one-shot: `STRATA_DSPARK=... STRATA_DSPARK_DEPTH=4`) |
| `V41-MTP-Vision-1017.gguf` | the same draft stack with MXFP4 experts (a lossless repack of the original FP4) |
| `model-0004{7,8}-of-00048.safetensors.NEED-TO-DOWNLOAD-ENGRAM.txt` | **placeholders - you need these two files, download them separately**: the two Engram shards (101.5 GB each) are not hosted here; these say where to get them (step 2 below) |
| `SHA256SUMS` | hashes of every part and of the joined files |

## Quick start (4 steps)

**You need:** Linux, an NVIDIA GPU with 24 GB, ~90 GB RAM (the RAM arena is 60 GiB by default; `--arena-gib` lowers
it, at some speed cost), the CUDA 13 toolkit, and fast SSD space for **the model (163 GB) + the Engram tables
(203 GB) = ~366 GB**, plus another ~203 GB *temporarily* while the Engram tables are extracted (or download the
Engram shards to a different drive).

**1. Download this repo and join the model** (~163 GB):

```bash
hf download IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored --include "v4/*" "tokenizer/*" "*.txt" \
   "V41-MTP-Vision-1017*" --local-dir V41-XQ-1017
bash V41-XQ-1017/v4/reassemble.sh V41-XQ-1017/v4              # needs ~163 GB free on top of the parts
# or: bash V41-XQ-1017/v4/reassemble.sh V41-XQ-1017/v4 --delete-parts
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
STRATA_NO_LARGEPAGES=1 build-xolotl/ds41_generate -m ../V41-XQ-1017/v4/V41-XQ-1017-v4.gguf --ids-file prompt.i32 -n 128 \
     --backend cuda --experts gpu --engram "/fast/a:/fast/b" --rank ../V41-XQ-1017/v4/rank-1017-v4.bin
```

`--ids-file` takes int32 token ids. For chat and agents, run the OpenAI- and Anthropic-compatible server with tool
calls in `tools/v41/serve/` (see "Serving it" in the
[GitHub README](https://github.com/Indras-Mirror/Xolotl)); it uses
`V41-XQ-1017/tokenizer/`.

## Older builds (V2, V3)

<details>
<summary>V2 and V3: notes, quality, speed and how they were made</summary>

**Builds in this repo (20 Oct)**

**V2** (`v2/`, 152 GB / 142 GiB): the first 1017 build. Calibrated on a chat-weighted mix with a gentler tier map, and made smaller than our September V4.1 V1 so it decodes faster. Head gate 6.93 ppl / 0.670 top-1 vs the edited teacher's 6.57 / 0.687 (+5.5% ppl, -1.7 pt). Short Gauntlet 2/5, with 03 at 0.00 and 07 at 0.91 (unsolved), so a step behind our September build on the two hardest tasks. Gauntlet decode 26.4 tok/s.

**V3** (`v3/`, 163 GB / 152 GiB): same checkpoint and edit, recalibrated on a code + agentic heavy mix (about 40% code, 30% agentic/tool use) with our V1 tier recipe, so it's a bit bigger. Head gate 6.72 ppl / 0.691 top-1 vs the teacher's 6.36 / 0.706 (+5.7% ppl, -1.5 pt, measured on its own held-out mix so not directly comparable to V2's). Gauntlet 03 still 0.00, 07 up to 0.95 but still unsolved. Gauntlet decode 22.3 tok/s (bigger file, fewer experts fit in VRAM), DSpark around 3.1 tokens/pass, prefill about 249 tok/s on a fresh 16.8K prompt. Honestly it's only a small step over V2, the hard agentic tasks didn't really move.

All speeds here are on Linux with an RTX 4090. On Windows DSpark can end up slower than running without it, see [Xolotl #14](https://github.com/Indras-Mirror/Xolotl/issues/14).


### V2's formats and allocation

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

### V2 quality

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

### Speed (V2, Linux · RTX 4090 24 GB · Ryzen 7 5700X · 90 GB DDR4 · NVMe)

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

### How V2 and V3 were made

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


</details>

## Credits

- **DeepSeek** for [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT).
- **[ISTA-DASLab](https://huggingface.co/ISTA-DASLab)** for their GSQ-RCO GGUFs of the unedited 1017, which Xolotl also runs.
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
