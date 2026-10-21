---
license: mit
library_name: llama.cpp
pipeline_tag: image-text-to-text
base_model:
- zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- gguf
- llama-cpp
- abliteration
- uncensored
- glm
- iq1_s
- moe
- expert-editing
---

# GLM-5.3-Flash Uncensored — GGUF

Refusal removed from [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by editing
**666 of its 12,384 routed experts** — about **1.74% of the model's parameters**. Quantized to
IQ1_S at **71.5 GiB**, which fits on a single 96 GB card.

I did the edit and the quantization myself, and I measured what it cost. Every number below is
first-party, taken on my own rig, on the file in this repo.

> **Read this before you run it.** Every headline number below was measured at
> **temperature 0, seed 0, repetition penalty 1.1/256, 16k context, `--reasoning-budget 1024`.**
> The file itself ships `temperature 1.0` in its metadata and no reasoning budget, so a plain
> `llama-server -m …` does **not** run the measured configuration. I tested that gap directly
> (table under *Default settings vs tested settings*): at the file's own defaults, long coding
> tasks ran out of output budget without answering about **1 time in 8**; with
> `--reasoning-budget 1024` they did not. **For code and chat, use the quickstart line as
> written. For hard math, leave the budget off** (see the correction below). This is an experimental 1.92-bit build: 37 of
> 120 harmful-set responses and 8 of 120 benign ones hit the output cap in the headline run, and
> that is reported separately from "unusable" below.

> **Correction, 2026-09-25: the reasoning budget is a trade-off, not a free fix.**
> `--reasoning-budget 1024` stops GLM thinking forever on coding tasks (measured below), and I
> still recommend it for code and chat. But on **hard multi-step math** it costs accuracy: on the
> IQ2_XXS sibling build it cut MATH-500 level-5 from **81% to 61%**. For hard math, run with no
> budget, and expect this 1-bit file to be weak at it anyway: **36% correct** on the same test
> (the unedited model at the same IQ1_S recipe scores 41%, so that ceiling comes from the 1-bit
> squeeze, not from the edit). Details under **Hard math** below. An earlier version of this card
> said the budget was simply required. That was true for the coding suite I had run, and
> overstated for everything else.

> 🧩 **LM Studio template fix (2026-10-07).** The chat template inside the file is ZAI's own and
> uses Jinja's `x.0.type` list access in four places, which llama.cpp accepts and LM Studio's Jinja
> engine rejects (`Expected identifier following dot operator`, reported on the sibling repo in
> [this discussion](https://huggingface.co/BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF/discussions/4)). **Shard 1 was re-published with those four spots rewritten as
> `x[0].type`**, header only, tensor data unchanged, shard 2 and the mmproj untouched. llama.cpp
> users can keep what they have; for LM Studio re-download
> `GLM-5.3-Flash-Uncensored-IQ1S-00001-of-00002.gguf`. New hash under **Provenance**.

> 🔁 **Update 2026-10-02: runs on mainline llama.cpp.** This file was built with the PR #27754
> fork (architecture string `glm5next`). That PR was closed unmerged on 2026-10-01 and mainline
> merged [PR #27773](https://github.com/ggml-org/llama.cpp/pull/27773) instead, which names the
> same architecture `glm5-next`. The tensors are identical between the two, so **shard 1 was
> re-published with only its header renamed; tensor data unchanged, shard 2 untouched.** If you
> already have the model, re-download only `GLM-5.3-Flash-Uncensored-IQ1S-00001-of-00002.gguf`
> (and the mmproj if you use vision — it was re-published the same way). New hashes under
> **Provenance**. See **Requirements** for what changed and what was checked.

> Full write-up with charts and method: **https://curve666-field-report.pages.dev/**

## What is different about this one

- **Small edit.** Only the down-projections of 666 selected experts are written. Every other
  tensor is byte-identical to the parent **at the BF16 stage, before quantization** — the
  released IQ1_S file is, of course, quantized throughout (see the tensor classes under Quants).
- **Selected by refusal traffic, not magnitude.** Same 384-expert budget spent on experts ranked
  by the size of their refusal component leaves 25.0% refusal; spent on experts that actually
  carry refusal traffic it leaves 5.8%. Which experts you pick matters more than how many.
- **Anti-repetition defaults baked into the file.** `general.sampling.penalty_repeat = 1.1` and
  `general.sampling.penalty_last_n = 256` are set in the GGUF, so llama.cpp picks them up with no
  flags. That cuts unusable answers from 14.2% to 5.0%.

## Requirements

**Mainline llama.cpp, b11279 or newer** ([PR #27773](https://github.com/ggml-org/llama.cpp/pull/27773),
merged 2026-09-30). Since 2026-10-02 the files here carry mainline's architecture name
(`glm5-next`); the original upload said `glm5next` and only loaded on the now-closed
[PR #27754](https://github.com/ggml-org/llama.cpp/pull/27754) fork. Only the header changed:
the tensor data section hashes identical before and after, and the retarget is reversible.

Checked on mainline before publishing (on the sibling IQ2_XXS build, same architecture, same
retarget script): it loads, the `blk.45` MTP block is skipped as before, a 12-prompt coherence
probe passes, and mainline vs. the fork on identical weights differ by KL 0.020 (prose) /
0.017 (math) with 94–96% top-1 agreement — each build is exactly deterministic against itself,
so that is a real implementation difference, small but not zero. **Every number on this page was
measured on the fork** (PR #27754 plus two local commits of mine, an MTP path and a KV-cache
change) and may move slightly on mainline. One mainline note: MTP is not in mainline yet
([PR #27917](https://github.com/ggml-org/llama.cpp/pull/27917) adds it and uses the `blk.45`
tensors already in this file).

## Quants

| Quant | Size | Fits | Measured |
|---|---|---|---|
| **IQ1_S** | 71.5 GiB (2 shards) | one 96 GB card | yes — every number below |

More rungs to follow. Each will say plainly whether it was evaluated or only built, because I
can only serve up to about 96 GB and anything larger I cannot benchmark myself.

| Shared file | Size | |
|---|---|---|
| `mmproj-GLM-5.3-Flash-F16.gguf` | 1.13 GB | vision, works with any rung; re-published 2026-10-02 for mainline (projector `glm5next` → `glm5v`, `swiglu_limit` → `swiglu_clamp`, image pixel budget keys added; vision tensors unchanged). Smoke-tested on mainline with this IQ1_S file: one synthetic image (red circle, blue square, the text "42") described correctly in 2 of 2 prompts. One image, not a benchmark |

Not uniformly 1-bit, but not "only the experts are 1-bit" either. Read out of the published
file: attention (208 tensors), the shared experts (129) and block 45 are **q8_0**; the routed
experts are **IQ1_S** (124), Q2_K (2) and q8_0 (3, the MTP block); the first three **dense**
FFN blocks are **IQ1_S** (gate/up) and Q2_K (down); **`token_embd` is Q2_K** and `output` is
Q5_K. An earlier version of this card said only the routed experts were IQ1_S — that was wrong.
Built with a 10-chunk imatrix inherited from a sibling build; thin, and disclosed as such.

The vision tower is **unmodified** — I converted it myself from `zai-org/GLM-5.3-Flash`, and the
edit never touches it. Text-only use does not need that file.

## Quickstart

```bash
# text only
llama-server \
  -m GLM-5.3-Flash-Uncensored-IQ1S-00001-of-00002.gguf \
  -c 16384 --jinja --reasoning-budget 1024

# with vision
llama-server \
  -m GLM-5.3-Flash-Uncensored-IQ1S-00001-of-00002.gguf \
  --mmproj mmproj-GLM-5.3-Flash-F16.gguf \
  -c 16384 --jinja --reasoning-budget 1024
```

Point at the **first** shard; llama.cpp loads the rest automatically.

`--reasoning-budget 1024` matters **for coding and chat**. Without it GLM can keep thinking and
never reach an answer. The refusal, over-refusal and HumanEval numbers below were measured with
it set. **For hard math, drop it** — see *Hard math* below for what it costs there.

## Method

1. Capture routing on refusal prompts, rank routed experts by how much refusal traffic each one
   actually carries.
2. Take the top 666, project a single refusal direction (layer 44) out of their
   `ffn_down_exps` weights at BF16.
3. Quantize the edited BF16 to IQ1_S with an imatrix.

The edit is applied at full precision and quantized afterwards. There is no runtime steering,
no LoRA, and nothing to load at serve time.

## Results

Fixed 120-prompt harmful set, 120-prompt benign set, greedy, seed 0, `--reasoning-budget 1024`.
Measured on the file in this repo.

### Refusal

| | parent | this model |
|---|---:|---:|
| keyword classifier | 99.2% | **3.3%** |
| model judge (StrongREJECT rubric) | — | **4.2%** |

The judge reads the whole answer, so it catches replies that open agreeably and then decline —
which a keyword rule scores as compliance. I report both because the cheap number flatters the
result by about a point.

### Over-refusal on benign prompts

| | refused |
|---|---:|
| parent | 1.7% |
| **this model** | **0.0%** |

0 of 120, no errors, no empty responses. It refuses harmless requests *less* often than the model
it came from.

### Degenerate output

| sampler | refused | repetition loops | no usable answer | hit the output cap |
|---|---:|---:|---:|---:|
| greedy | 4.2% | 10.0% | 14.2% | 44 / 120 |
| baked defaults (rep-pen 1.1) | 3.3% | 1.7% | **5.0%** | **37 / 120** |
| baked defaults, benign set | 0.0% | 0.0% | 0.0% | 8 / 120 |

A repetition loop contains no refusal language, so a refusal classifier scores it as a success.
Reporting refusal without this number overstates any abliteration.

**Two things this table does not say on its own.** First, "no usable answer" counts refusals
and loops; it does **not** count the responses that hit the token cap, which are listed in the
last column separately — a capped response may be a fine answer that ran long, or it may have
never reached one, and I did not adjudicate them individually. Second, every row was measured
at **temperature 0, seed 0, 16k context, `--reasoning-budget 1024`**. The file ships
`temperature 1.0` in its metadata. What the file does at its *own* defaults, with no reasoning
budget, is a separate measurement — see **Default settings vs tested settings** below.

### Default settings vs tested settings

Every number above was measured at temperature 0 with a 1,024-token reasoning budget. The file
itself ships `temperature 1.0` and no budget. Those are different machines to a user, so on
2026-09-06 I ran the same 12-task coherence suite used on the OrcaRouter rebuilds (4 reasoning
tasks with exact-answer checks, 4 coding tasks with executed asserts, 4 three-turn retention
tasks; 2 seeds; 8,192-token output budget; loop flags from the same n-gram heuristic as the
table above) under three server configurations, on the exact bytes in this repo, on the same
3×3090 box the other numbers came from:

| condition | what it is | pass | loop flags | capped, no answer | reasoning | code | multi-turn |
|---|---|---:|---:|---:|---|---|---|
| **shipped** | baked defaults, **no** reasoning budget — a plain `llama-server -m …` | 21/24 | 0 final, 2 in reasoning | **3** | 8/8 | **5/8** | 8/8 |
| nopen | same, repetition penalty off (control) | 20/24 | 0 final, 2 in reasoning | **4** | 7/8 | 5/8 | 8/8 |
| **quickstart** | the quickstart line on this card: baked temp 1.0, `--reasoning-budget 1024` | 22/24 | 0 | **0** | 8/8 | 6/8 | 8/8 |
| **tested** | temp 0, seed 0, penalty, `--reasoning-budget 1024` — the condition above | **24/24** | 0 | **0** | 8/8 | **8/8** | 8/8 |

Read it column by column. **Capped, no answer** is the one that matters: at the file's own
defaults, 3 of 8 coding runs (and with the penalty off, 4 of 8 plus one reasoning task)
thought for 34–36 thousand characters and never wrote an answer. Those traces are not
degenerate — the n-gram loop heuristic scores them 0.71–0.97 distinct, they read as fluent
reasoning that keeps re-litigating edge cases — they are GLM thinking with no ceiling. Add the
1,024-token reasoning budget and that failure goes to **zero** at both temperatures. The two
quickstart misses are ordinary temperature-1.0 sampling errors (one malformed regex in an
over-long answer, one wrong merge), not non-termination; at temperature 0 the same tasks are
8 for 8. Multi-turn retention was perfect in every condition.

So, **for this coding suite: the weights are fine, the budget is required, the quickstart line is
the right recommendation, and temperature 0 is better still for code.** For hard math the budget
hurts. See *Hard math*. What the file does with no
budget is documented here so nobody has to discover it. The repetition penalty, which the
file bakes in, made no difference to this failure (21 vs 20) — it is there for the n-gram
loops in the table above, which are a different thing. Same caveats as every table on this
page: 24 correlated cases per condition at two seeds bound the rate, they do not certify it.

### Capability

| | pass@1 |
|---|---:|
| parent, run 1 | 87.8% |
| parent, run 2 | 90.2% |
| **this model** | **87.2%** |

HumanEval-164, greedy, 6000-token cap, run on the quantized weights **before** the sampler
defaults were baked into the file (the bake changes metadata only, not weights). The parent
disagrees with itself by 2.4 points across two runs; this model sits 0.6 below the weaker one.

### Hard math (added 2026-09-25)

MATH-500 level 5, first 100 problems, the file's own sampling (no overrides), 16k output cap,
exact match on the boxed answer. Same prompts, script and cap for every row.

| build | answered | ran out of room | **correct** |
|---|---:|---:|---:|
| **this model, IQ1_S**, no budget | 48% | 52% | **36%** |
| this model, IQ1_S, `reasoning_effort=low` | 68% | 31% | 43% |
| unedited GLM-5.3-Flash, same IQ1_S recipe | 50% | 50% | 41% |
| OrcaRouter's edit, IQ2_XXS ([our quant](https://huggingface.co/BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF)), no budget | 89% | 11% | **81%** |
| same IQ2_XXS, `--reasoning-budget 1024` | 98% | 2% | 61% |

What this says:

- **At 1.92 bits the quantization is the ceiling, not the edit.** The unedited model at the
  same recipe does about as badly (41% vs 36%; with n=100 that gap is inside the noise).
  Half of the hard problems run past 16k tokens of thinking without an answer.
- **A reasoning budget buys answers, not right answers, on math.** On the IQ2_XXS build it
  raised the answer rate from 89% to 98% and dropped accuracy from 81% to 61%. I did not
  measure the budget on this IQ1_S file for math. Expect the same direction.
- **If hard reasoning matters to you, use a 2-bit build.** At IQ2_XXS (97 GB) the same
  architecture scores 81%, and it fits fully on 4×24 GB cards (see the OrcaRouter quant card
  for the exact split).

## What I did not measure

- **Beyond coding and hard math.** No GSM8K, MMLU, MBPP, tool use or agentic evaluation.
- **Any language other than English.** Every prompt set is English.
- **Long context.** Everything was run at 16k.
- **The reasoning budget on this file for math.** The math table measures the budget on the
  IQ2_XXS sibling only.

Two base runs do not establish statistical equivalence. These are same-condition lab
measurements, not a leaderboard reproduction, and they are first-party — I made the model and I
graded it.

## Intended use

Research, and local assistants where over-refusal is a problem. It is still a general model and
it will produce unsafe content if you ask it to. Use it under your own policy and applicable law.

## Provenance

| | |
|---|---|
| Parent | `zai-org/GLM-5.3-Flash` |
| Quantizer | llama.cpp CUDA build 10700 |
| Edit | single refusal direction, layer 44, 666 routed-expert down-projections |
| Runtime (measurements) | llama.cpp PR #27754, plus 2 local commits (see Requirements) |
| Runtime (files, since 2026-10-02) | mainline llama.cpp b11279+ (PR #27773) |
| shard 1 | `1c5e14c9f48210652d92cf1e1a30d9cf588af3aba8b878a11dd667883114783d` (chat template fixed for LM Studio 2026-10-07; was `8c65bb56…` after the 2026-10-02 header retarget, `04d81e6d…` originally) |
| shard 2 | `4d0d4ce1eb9961306bdfa05af12d99609e7d0eb784c36c131c44d2cd47c515cd` (unchanged) |
| mmproj | `1efafe6c768a5d8f9a84a95d52cef027885c26cd6a8583de25cb9541f59fdd6f` (retargeted 2026-10-02; was `fffaa0aa…`) |

If you re-download this repo later and the hashes differ, the weights changed. Check.
