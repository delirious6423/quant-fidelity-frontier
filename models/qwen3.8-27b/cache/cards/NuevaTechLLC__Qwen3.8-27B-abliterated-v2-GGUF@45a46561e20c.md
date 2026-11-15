---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language:
- en
pipeline_tag: text-generation
library_name: llama.cpp
tags:
- gguf
- abliterated
- uncensored
- refusal-removal
- interpretability
- llama.cpp
- ollama
---

# Qwen3.8-27B-abliterated-v2 (GGUF)

Abliterated build of `Qwen/Qwen3.8-27B`: the refusal direction was located by bulk contrast between harmful and harmless prompt pairs and removed with a KL-bounded weight edit, so the model keeps its base behaviour on ordinary prompts while declining far less. Built 2026-10-27 by Nueva Tech LLC with its low-damage abliteration pipeline, described below.

This is a Qwen 3.5 architecture model (GGUF arch `qwen35`) with the older state space tensor naming, which loads in llama.cpp builds from October 2026 onward and Ollama 0.40 and later.

## Files

| File | Quant | Size | SHA256 |
|---|---|---|---|
| `big-Qwen3.8-27B-Q4_K_M.gguf` | Q4_K_M | 16.5 GB | `b895daa10fe45c6da67ad02948528635901ab9c7754ffa069c8ecbd5e9ade68e` |

Needs about 20 GB of VRAM or RAM for the Q4_K_M file. Q8_0, where present, is closer to the edited bf16 weights at roughly double the size.

## Run it

Ollama, with the Modelfile in this repo:

```
hf download NuevaTechLLC/Qwen3.8-27B-abliterated-v2-GGUF --local-dir Qwen3.8-27B-abliterated-v2-GGUF
cd Qwen3.8-27B-abliterated-v2-GGUF && ollama create qwen3.8-27b-abliterated-v2-gguf -f Modelfile && ollama run qwen3.8-27b-abliterated-v2-gguf
```

LM Studio: download the Q4_K_M file and import it with `lms import <file>`.

llama.cpp:

```
llama-cli -m big-Qwen3.8-27B-Q4_K_M.gguf -c 8192 -cnv
```

## How these were made

Nueva Tech's low damage abliteration framework treats refusal removal as a constrained optimisation problem rather than a one shot weight edit: remove refusal, measure everything else that moved, and keep the result on the Pareto front between refusal and damage. It exists because the standard recipe (difference of means between an unmatched harmful set and an unmatched harmless set, one direction, applied at every layer) damages models in ways that the usual KL check does not show. Each weakness of that recipe has a specific answer in the pipeline:

| Standard recipe weakness | What this pipeline does instead |
|---|---|
| The harmful versus harmless mean difference captures topic, length and register as well as the refusal decision | Harmless twins: each harmful prompt gets a structurally matched harmless rewrite, gated for length and form, so the contrast isolates the refusal decision. Packages tagged `matched` used the twin set for extraction, `bulk` used the full harmless corpus. |
| Refusal circuitry also drives calibrated declining: rejecting a false premise, admitting ignorance, rejecting nonsense, honouring a negative constraint. Removing it takes those too, and KL on fluent text stays near zero while it happens | Four protected behaviours with 96 probe and control pairs are scored directly on every candidate edit and weighted above KL in the objective |
| A direction measured at one layer is applied at every layer although its meaning drifts with depth | A layer scan finds where refusal separates best, and the edit is applied with a kernel (band or uniform) centred there with a searched width and peak strength, on attention output and MLP output projections |
| The refusal direction projects onto the harmless mean, so ablation shifts every input and RMSNorm rescales with the wrong magnitude | The harmless mean is removed from the direction before projection, with optional norm restoration |
| Refusal spans a small cone of directions, so rank 1 leaves residual refusals and raising strength to catch them costs more than it buys | SVD over the matched contrast gives a rank k subspace, with protection against the calibration directions |
| A hard weight edit gets no gradient feedback | Optuna searches the edit plan against the objective `refusal + λ_kl·KL + λ_cal·calibration_drop + λ_over·Δover_refusal`, and an optional self distillation pass heals the chosen edit against cached logits from the untouched model |

Every stage leaves a record in this repo: `extract.json` (layer scan, subspace, protection cosines), `search.json` (every trial with its metrics and the Pareto set) and `ablation_card.json` (the chosen plan, survival through quantization, data counts and timings). The results table further down comes straight from that card.

This model was not itself in the bake-off below; the comparison is cross-model evidence for the method, and this model's own damage numbers are in the results table further down.

## Damage measured against other abliteration methods

To check that low damage means something outside our own harness, the same base models were abliterated with the two most used public tools and all contenders were scored by one evaluator with the same tokenizer, chat template and greedy decoding:

- **Heretic 1.4.0**, the automated abliteration tool that runs its own Optuna search over a directional ablation.
- **OBLITERATUS** at commit `461ff08`, in its `advanced` and `optimized` presets.
- **Ours**, the pipeline's v2 edit, released here under the same name, scored as the bf16 checkpoint these GGUFs were quantized from.

Held out sets: StrongREJECT with AdvBench rows and overlaps removed (282 harmful prompts), OR-Bench-hard (300 benign prompts that look harmful, for over-refusal), and Dolly (300 harmless prompts, for KL against the untouched base over the full vocabulary). Refusals were judged by Qwen3Guard-Gen-8B. `comply` is the share of harmful prompts answered with content the judge rates unsafe, so it measures real compliance rather than the absence of a refusal phrase. Capability is lm-eval 0.4.13: full MMLU, ARC-Challenge and GSM8K at 400 items each. Brackets are 95 percent confidence intervals.

**Qwen3-8B**

| contender | refusal | comply | over-refusal | KL | MMLU | ARC-C | GSM8K |
|---|---|---|---|---|---|---|---|
| base | 0.954 | 0.004 | 0.370 | 0.000 | 0.757 | 0.560 | 0.892 |
| Heretic 1.4.0 | 0.610 | 0.273 | 0.050 | 0.006 | 0.757 | 0.558 | 0.898 |
| OBLITERATUS advanced | 0.716 | 0.188 | 0.113 | 0.242 | 0.747 | 0.525 | 0.880 |
| OBLITERATUS optimized | 0.876 | 0.053 | 0.110 | 0.116 | 0.746 | 0.548 | 0.870 |
| ours (v2) | 0.135 | 0.777 | 0.003 | 0.052 | 0.758 | 0.560 | 0.905 |

**gemma-3-12b-it**

| contender | refusal | comply | over-refusal | KL | MMLU | ARC-C | GSM8K |
|---|---|---|---|---|---|---|---|
| base | 0.954 | 0.011 | 0.173 | 0.000 | 0.738 | 0.610 | 0.882 |
| Heretic 1.4.0 | 0.496 | 0.418 | 0.017 | 0.027 | 0.739 | 0.610 | 0.895 |
| ours (v2) | 0.259 | 0.660 | 0.000 | 0.091 | 0.733 | 0.588 | 0.882 |

**Qwen3-32B** (same pipeline, not part of this release)

| contender | refusal | comply | over-refusal | KL | MMLU | ARC-C | GSM8K |
|---|---|---|---|---|---|---|---|
| base | 0.965 | 0.000 | 0.313 | 0.000 | 0.832 | 0.573 | 0.605 |
| Heretic 1.4.0 | 0.298 | 0.599 | 0.003 | 0.028 | 0.834 | 0.578 | 0.688 |
| OBLITERATUS advanced | 0.901 | 0.039 | 0.273 | 0.194 | 0.827 | 0.557 | 0.448 |
| OBLITERATUS optimized | 0.897 | 0.050 | 0.083 | 0.257 | 0.822 | 0.593 | 0.698 |
| ours (v2) | 0.213 | 0.660 | 0.007 | 0.019 | 0.829 | 0.562 | 0.688 |

How to read it. Damage has two faces: KL says how far the output distribution moved on harmless text, and the capability columns say whether anything measurable was lost. Both OBLITERATUS presets barely moved refusal on these models while incurring 4 to 13 times our KL, and on Qwen3-32B the advanced preset cost 16 points of GSM8K. Heretic reaches the lowest KL of any contender but leaves between a third and two thirds of refusals in place, which is the trade its objective makes. The v2 edit removes 73 to 78 percent of refusals, with real compliance of 66 to 78 percent, at a KL between 0.02 and 0.09, and its capability scores sit inside the base model's confidence interval on every task except a half point of MMLU on gemma-3-12b. Paired tests with Holm correction confirm the refusal and KL differences against every rival at p below 0.001.

The pipeline's v3 edits, with the KL term weighted first and a finer search over the layer kernel, bring Qwen3-8B to 0.025 refusal at KL 0.013 and gemma-3-12b to KL 0.038 at the same refusal as v2. Those checkpoints are still being baked and will follow as v3 releases.

## How to read the results table below

`refusal` is the share of a held out harmful set the model declined, `over-refusal` the share of benign lookalikes it wrongly declined, `KL` the divergence from the base model on harmless prompts (lower is less damage). The remaining columns are the four protected behaviours: false premise, unanswerable, nonsense and negative constraint handling, higher is better. `edited:ckpt` is the saved checkpoint scored cold from disk, which is what these GGUFs were made from. Rows marked `@int8` or `@int4` are the same checkpoint scored after quantization; `healed` rows had the self distillation pass. These numbers come from the pipeline's own smaller harness and are not directly comparable with the bake-off tables above.

## Intended use and limits

Interpretability research, red team evaluation and controlled testing of safety behaviour. The model will comply with requests the base model declines; the person running it carries the responsibility for how it is used. No capability benchmarks beyond those listed above have been run on this build.

---

Base model: `Qwen/Qwen3.8-27B`
Built: 2026-10-27

Research artifact. Refusal behaviour has been removed by weight editing for interpretability and red team evaluation. Not a general purpose assistant.

## Results

| variant | refusal | over-refusal | KL | false prem | unanswerable | nonsense | neg constraint |
|---|---|---|---|---|---|---|---|
| baseline | 1.000 | 0.021 | 0.0000 | 0.556 | 0.600 | 0.333 | 0.833 |
| edited | 0.146 | 0.000 | 0.0904 | 0.556 | 0.700 | 0.667 | 0.833 |
| edited:ckpt | 0.146 | 0.000 | 0.0904 | 0.556 | 0.700 | 0.667 | 0.833 |

## Extraction
```json
{
  "report": "layers 65  k 1  protect True\nbest layer 36  separation 11.67  surviving rank 1\nvariance explained at best layer [0.764]\ncosine of top refusal direction with each protected behaviour, at best layer:\n  false_premise          +0.387\n  negative_constraint    +0.109\n  nonsense               +0.279\n  unanswerable           +0.399\nstrongest layers: 36(11.67), 50(11.63), 35(11.61), 48(11.58), 51(11.55), 47(11.53), 49(11.53), 46(11.48)",
  "contrast": "bulk",
  "clip": 0.995,
  "k": 1,
  "base_refusal_harmful": 0.995,
  "base_refusal_twin": 0.4,
  "base_over_refusal": 0.0625,
  "n_pairs": 200
}
```

## Edit plan
```json
{
  "center": 13.657036688113642,
  "width": 28.66171827893841,
  "peak": 0.9766557718737404,
  "kernel": "uniform",
  "source_layer": 14,
  "kinds": [
    "attn_out",
    "mlp_out"
  ],
  "norm_restore": "column",
  "min_alpha": 0.01,
  "edit_embedding": false
}
```

## Healing
```json
{
  "steps": 0,
  "qat": false
}
```

## Data
```json
{
  "pairs": 200,
  "n_eval": 48,
  "cal": 41
}
```