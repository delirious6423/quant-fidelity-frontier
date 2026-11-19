---
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
license: mit
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - deepseek
  - moe
  - expert-pruning
  - code
  - gsq
  - rco
---

# DeepSeek-V4.1-Flash-1017 · CoderPlus240 (GGUF)

**ISTA-DASLab's DeepSeek-V4.1-Flash Coder with 48 more experts per layer put back: 240 of 384 routed experts per layer
instead of 192.** Built to fill a single 96 GB GPU exactly, with every expert resident, so it keeps the Coder's speed and
gets back much of what the Coder's pruning cost.

- 82.6 GB of routed experts plus about 7 GB of other weights. Everything sits on one 96 GB card, nothing on the CPU.
- No new quantization and no retraining. Every byte comes from ISTA-DASLab's
  [GSQ-RCO IQ2_XXS (2.00 bpw) build](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF);
  this repository only changes which experts are kept.
- Runs on Strata's `strata-dsv41` DeepSeek engine (see [Running it](#running-it)). It has not been tested on llama.cpp.

All credit for the quantization (GSQ), the Coder's expert selection (RCO) and the base files goes to
[ISTA-DASLab](https://github.com/IST-DASLab), and for the model to DeepSeek. This is an independent derivative, not an
ISTA-DASLab release.

---

## What it is

DeepSeek-V4.1-Flash-1017 has 40 transformer layers, each with 384 routed experts; the router picks 6 per token per layer.
ISTA-DASLab's [Coder](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF) removes half of them
(192 kept per layer), chosen with RCO to change the model's outputs as little as possible on code, agentic tool use, long
context and vision. That brings the experts down to 66 GB, which leaves about 18 GB of a 96 GB card unused.

CoderPlus240 spends that unused memory on experts:

| | Experts kept per layer | How they were chosen | Routed experts | On a 96 GB card |
|---|---:|---|---:|---|
| ISTA IQ2_XXS (2.00 bpw), unpruned | 384 | none removed | 132 GB | does not fit (about 40% of experts on the CPU, in RAM or on the SSD) |
| ISTA Coder | 192 | RCO: KL-optimal on ISTA's coding/agent calibration data | 66 GB | fits, ~18 GB free |
| **CoderPlus240** | **240** | **ISTA's 192, plus the 48 removed experts the router uses most** | **82.6 GB** | **fits, ~1.5 GB free** |

### Why add experts back

ISTA's 192 experts per layer were chosen to minimise KL divergence, not by how often they are used. Measured against the
unpruned build's routing, the experts the Coder keeps receive only about 62% of the router's picks. The other ~38% go to
experts the Coder removed; it has to route those tokens to its second choices.

CoderPlus240 adds back the removed experts the router most wanted. Share of the unpruned build's router picks that land on
experts the model still has:

| Routing profile | Coder (192) | **CoderPlus240 (240)** |
|---|---:|---:|
| Strata's shipped profile for the IQ2_XXS build: mean over layers | 62.0% | **82.4%** |
| the same, lowest layer | 55.1% | 79.8% |
| Local use (coding-agent sessions, benchmarks, prompts): mean | 59.3% | **86.5%** |
| the same, lowest layer (layer 23) | 30.5% | 88.1% |

The layers differ a lot. The Coder's coverage of the local routing ranges from 30.5% (layer 23) to 71.3% (layer 3).
That suggests the next step is a different expert count per layer, which the GGUF format cannot express (it stores
one expert count for the whole model); see [Limitations](#limitations).

### How the 48 were chosen

In each layer separately:

1. Start from the 192 experts ISTA-DASLab's Coder keeps (their published `kept-experts.json`).
2. Score each of the 192 removed experts by its share of that layer's router picks in two routing profiles of the
   unpruned IQ2_XXS build, added together with equal weight:
   - the hot-expert profile Strata ships for that build;
   - the same profile after local use on the machine below: coding-agent sessions (KannaCode's benchmark tasks), speed
     benchmarks and a set of mixed test prompts, about 483,000 tokens.
3. Keep the 48 highest-scoring removed experts. 192 + 48 = 240.

The count of 48 is what fills a 96 GB card with about 1.5 GB of headroom (one expert in every layer costs 344 MB). It is
the same in every layer.

The full selection is in `expert-selection/CoderPlus240.kept-experts.json`: per layer the 240 kept base-model expert ids
(ascending, the order they are stored in), which of them are ISTA's, and the 48 added back.

### What was changed in the files

Only five tensors per layer are cut, all along their expert axis: `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` (the
experts), `ffn_gate_inp` (the router's rows for them) and `exp_probs_b` (their routing biases).
`deepseek41.expert_count` is set to 240. Every other tensor and every metadata value is the IQ2_XXS build's,
byte for byte, and every quantization type is unchanged.

This was checked byte for byte on 2,008 tensors and experts: ISTA's kept experts are identical to the same experts in
ISTA's Coder files, the added experts are identical to the IQ2_XXS build's, and all unsliced tensors are unchanged.

---

## Files

| File | Size | |
|---|---:|---|
| `DeepSeek-V4.1-Flash-1017-CoderPlus240-IQ2_XXS-0000N-of-00004.gguf`, N = 1-4 | 88.7 GB | the model, ISTA's older four-shard layout |
| `expert-selection/CoderPlus240.kept-experts.json` | 46 KB | the kept experts per layer, ISTA's and the added ones |
| `magma/expert-profile-240.bin` | 41 KB | a starting hot-expert profile for Strata's engine (uint32 [43][240]) |

Also needed, and not duplicated here:

- The **Engram tables**: `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (110.6 GB), the same file as in
  [ISTA-DASLab's release](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF). They stay on the SSD
  and are read a few rows per token.
- From [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF): the DSpark
  drafter `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` (8 GB), the vision projector `mmproj-BF16.gguf` (1 GB), and the
  5.8 MB metadata shard `UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf`, which Strata's packer reads
  (`--engram-meta`). These are what every ISTA-DAS set uses in Strata.

Quant types in the shards (routed experts by type, as in ISTA's IQ2_XXS build): IQ2_XXS 30.7 GB, IQ1_M 37.2 GB,
IQ2_XS 14.7 GB; other weights MXFP4 7.2 GB, Q5_K 5.3 GB, BF16 1.3 GB, Q8_0 1.0 GB, F32 0.2 GB.

---

## Results

Machine: one **RTX PRO 6000 Blackwell 96 GB** (PCIe 4.0 x16), **Ryzen 9 3900X** (12 cores), **62 GB RAM**, NVMe SSD.
Engine: Strata's `strata-dsv41` (CUDA 13.0, sm_120), 200K context, DSpark drafting on (`--dspark auto --dspark-conf 0.65`).
All answers greedy (temperature 0), thinking off.

### Speed

| | Writing, mean of 3 answers (tok/s) | Writing, single code prompt | Reading a 2K / 8K prompt (tok/s) |
|---|---:|---:|---:|
| ISTA Coder (192) | 118.7 (155 / 117 / 84) | - | 1,024 / 1,653 |
| **CoderPlus240** | **132.0** (170 / 113 / 113) | **166.9** | 982 / 1,638 |
| IQ2_XXS 2.00 bpw unpruned, exact (does not fit the card) | 27.5 | 31.9 | 274 / 876 |

The writing figures are the server's own timings for three 256-token answers (code, explanation, email), the shape of
Project Maya's `--bench`. The 25% more experts cost no speed: they are on the card too.

### Answers to 10 mixed prompts

The same 10 prompts, one greedy answer each, compared side by side, with the unpruned IQ2_XXS build (all 384 experts)
as the reference. 400 tokens at most per answer.

| Prompt | ISTA Coder | CoderPlus240 | Unpruned IQ2_XXS |
|---|---|---|---|
| Translate a sentence into French and Spanish | **broken**: invents "réunionión", then argues with itself until the token limit | Spanish correct; the French starts right and slips into Spanish halfway | both correct |
| Order five people by height | correct | correct | correct |
| When do two trains meet | right method; reached the token limit before the final time | right method; small slip in the last step ("11:33:33" for 11:33:20) | correct (11:33:20) |
| Find the bug in a "second largest" function | correct fix | correct fix | correct fix |
| LRU cache class | working | working, with input checks | working |
| Extract a JSON object from text | correct | correct (identical) | correct |
| Why printing money causes inflation | good | good | good |
| TCP vs UDP for a junior developer | good | good | good |
| Short story | good | good | good |
| Polite but firm email | good | good | good |

Both pruned models also drop some braces in LaTeX (`\text hours`); the unpruned build does not.

### Agent tasks

Three coding-agent tasks from the KannaCode coding agent's model benchmark (each rebuilt
from a real merged commit of a Python repository and scored by its pytest checks), run through the KannaCode agent
against the model's OpenAI-compatible endpoint. Wall time includes every model call, tool run and test run.

| Task | ISTA Coder | **CoderPlus240** |
|---|---|---|
| version-source-of-truth (1 check) | pass, 350 s | **pass, 255 s** |
| surface-search-command (2 checks) | pass, 861 s | **pass, 736 s** |
| version-flag-and-status (2 checks) | pass, 396 s | **pass, 362 s** |

All 5 checks pass on both. CoderPlus240 finished each task faster.

### Quality reference (ISTA-DASLab's measurements, not ours)

Against the original model, KL divergence (lower is closer), from ISTA-DASLab's and Strata's published numbers:

| | Overall | Code | Agent | Documents | English | Multilingual |
|---|---:|---:|---:|---:|---:|---:|
| ISTA Coder (192) | 0.397 | 0.230 | 0.199 | 0.302 | 0.476 | 0.777 |
| ISTA IQ2_XXS 2.00 bpw (384) | 0.171 | 0.104 | 0.108 | 0.202 | 0.303 | 0.138 |

CoderPlus240's KL has not been measured. Its experts are a superset of the Coder's and a subset of the unpruned build's.

---

## Running it

The shards are in ISTA-DASLab's older **four-shard layout**, which Strata's DeepSeek engine reads. ISTA re-split their sets
on 2026-10-27 into a six-shard layout for llama.cpp b11870 and newer (different metadata keys, one attention tensor stored
in another shape, the Engram table as shard 6). llama.cpp has not been tried with these files and most likely needs that
newer layout.

With Strata's DeepSeek engine (`strata-dsv41`; upstream Strata removed DeepSeek support in 0.2.14, so a build that has it):
pack the four shards with its `tools/dsv41/pack_gguf.py` the way it packs ISTA's IQ2_XXS set, naming these shards with
`--shards`, the Engram file with `--engram`, the unsloth metadata shard with `--engram-meta`, the drafter with
`--dspark-gguf` (with `--dspark-noise-token 128799 --dspark-markov-rank 256`) and the vision projector with `--mmproj`;
then copy
`magma/expert-profile-240.bin` into the pack as `expert-profile.bin`. The engine reads the expert count from the files.

---

## Limitations

- **The same count in every layer.** The coverage table shows layers differ a lot, so spending the same budget unevenly
  (more experts where a layer is more sensitive) should do better. GGUF stores one expert count per model, so that needs
  an engine and pack format that allow a count per layer.
- **Usage-based selection.** ISTA's 192 come from an optimisation of the whole model's output (RCO). The 48 added here
  come from routing counts alone, which is simpler.
- **Not retrained.** The router was trained with all 384 experts; it is used as is, restricted to the 240.
- **Strata's engine only, so far.** Not tested on llama.cpp or other runtimes.

## License

MIT, as the base model ([DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)) and
ISTA-DASLab's GSQ-RCO releases, whose weights these are.

## Credits

- **DeepSeek** for DeepSeek-V4.1-Flash-1017.
- **ISTA-DASLab** (Institute of Science and Technology Austria) for the GSQ quantization, the RCO expert selection of the
  Coder and the GGUF files this is cut from:
  [GSQ](https://arxiv.org/abs/2604.18556), [RCO](https://arxiv.org/abs/2605.00649).
- **Strata** and **Project Maya** for the DeepSeek engine, its hot-expert profiles and benchmark shape.
