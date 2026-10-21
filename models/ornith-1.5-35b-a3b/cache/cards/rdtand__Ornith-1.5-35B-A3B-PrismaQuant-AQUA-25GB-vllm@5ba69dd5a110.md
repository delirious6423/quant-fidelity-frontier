---
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: transformers
tags:
  - prismaquant
  - aqua
  - mixed-precision
  - nvfp4
  - fp8
  - compressed-tensors
  - vllm
  - moe
---

# Ornith-1.5-35B-A3B PrismaQuant AQUA (4.75 bpp)

A mixed-precision quantization of
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
that assigns each Linear its own numeric format instead of applying one format
to the whole model. The artifact is a stock `compressed-tensors` checkpoint.
Vanilla vLLM serves it with no forked runtime and no custom kernels.

The model is 24.68 GB, down from 71.90 GB in BF16 — a 2.91x reduction — and
costs 2.53% perplexity against the BF16 source on an identical harness.

## Results

Both rows below were measured on the served artifact through vLLM, not on a
local emulation of it. The BF16 row is the source model measured through the
same harness in the same session.

| Metric | This artifact | BF16 source | Delta |
|---|---|---|---|
| WikiText-2 perplexity | 11.7067 | 11.4181 | +2.53% |
| Mean NLL | 2.4602 | 2.4352 | +0.0250 nats |
| Worst 512-token chunk NLL | 2.9293 | 2.9078 | +0.0215 nats |
| Size | 24.68 GB | 71.90 GB | 2.91x smaller |
| Bits per quantizable parameter | 4.749 | 16 | — |

Perplexity uses the WikiText-2 test split, 8176 tokens at sequence length 512.

KL divergence against the BF16 source, on WikiText-2 at 8 sequences x 512
tokens, scoring all 4088 positions:

| Statistic | Value |
|---|---|
| Mean KL | 0.0881 |
| Mean KL over confident positions | 0.0534 (2146 positions) |
| p99 KL | 0.8924 |
| Max KL | 5.7459 |

These KL values are only comparable to values from the same evaluator at
the same depth on the same text: WikiText-2-raw train, corpus SHA-256
`fb23ad9643a34514eec5cb85ec2a6f49d1a33e6a3d5077dff5a403e1d18f5047`, 8
sequences at length 512, 52.5% of positions teacher-confident. A KL from a
different corpus is a different measurement, not a better or worse one.

Read the KL numbers with the truncation caveat in
[What these numbers do not cover](#what-these-numbers-do-not-cover).

## Format allocation

PrismaQuant assigned formats to 511 serving units. The allocation is the
product of the method, so it is worth reading directly:

| Unit role | BF16 | FP8 | NVFP4 |
|---|---|---|---|
| Routed packed experts (80) | — | 2 | 78 |
| Dense attention and MLP (300) | 176 | 122 | 2 |
| Shared experts (120) | 5 | 115 | — |
| Multi-token prediction (11) | 11 | — | — |

The shape of this table is the result, not an input. The allocator put almost
all routed-expert mass in NVFP4 and kept the dense path at 8 and 16 bits,
because that is where the measured cost of the bits landed. A uniform-format
method cannot express this assignment.

## Serve the model

```bash
vllm serve rdtand/Ornith-1.5-35B-A3B-PrismaQuant-AQUA-25GB-vllm \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 32768
```

The base model supports a 262144-token context. Serving the full context needs
KV cache memory well beyond the weights, so set `--max-model-len` to what your
hardware holds.

The checkpoint loads in both eager and CUDA-graph mode. The build validates both.

Every number on this card was measured on vLLM
`0.26.1rc1.dev693+g7f7a32cfe`, with PyTorch 2.13.0+cu130 and Transformers
5.15.0, on an NVIDIA GB10 (Blackwell, `sm_121`). Any vLLM build that reads
`compressed-tensors` NVFP4 and FP8 checkpoints serves this artifact.

## What a uniform format costs

The claim that per-Linear allocation beats a single format is testable, so it
was tested on this model rather than asserted.

Every body unit was forced to NVFP4 and re-scored through the evaluator the
allocator itself uses to pick a shipping point, on the same held-out
calibration split as every other row below, with a calibration-hash check
refusing the comparison if the arms had drifted apart. That
build lands at **4.500 bpp** — and it is the *only* uniform option near this
budget, because the served menu jumps from NVFP4 at ~4.5 bpp straight to FP8 at
~8 bpp. No uniform format reaches 4.75 bpp at all.

| Build | bpp | Held-out calibration KL |
|---|---|---|
| Uniform NVFP4 (every body unit) | 4.500 | 0.1378 - 0.1405 |
| PrismaQuant allocation | 4.600 | 0.0721 |
| **PrismaQuant allocation (shipped)** | **4.749** | **0.0554** |

**These three rows are a different measurement from the KL in
[Results](#results) and must not be read against it.** They come from the
allocator's selection evaluator: a full-sequence mean over the held-out
calibration split at sequence length 1024, scored on the rendered model. The
Results table reports a top-1024 confident-position KL over WikiText-2 at
sequence length 512, scored on the served artifact. The two land at similar
magnitudes by coincidence. Compare rows within this table only.

**Spending 5.5% more bytes buys a 2.5x reduction in KL.** In absolute terms
that is about 1.3 GB on a 24.68 GB artifact.

The reason is visible in the allocation. Routed experts hold 95.8% of the
quantizable parameters and stay in NVFP4 at every budget. The dense attention
and MLP path is only 4.2% of parameters but carries most of the divergence, and
the 0.25 bpp between the uniform build and the shipped one is spent almost
entirely on lifting that path out of 4-bit: 190 of 190 body dense units sit in
NVFP4 at 4.500 bpp, and 2 of 190 at 4.749.

Two caveats belong with these numbers. The uniform arm was scored in a separate
process from the frontier, so it carries cross-session variation; a replicate of
two near-identical assignments measured 2.0% apart, against a 2.5x effect. And
the allocator's own 4.500 bpp solution is itself uniform NVFP4 on 389 of 390
body units, which is not a failure of the method but the budget leaving it no
freedom - the comparison above is therefore a rate-distortion statement, not a
head-to-head between two different allocations at one budget.

## How the allocation is chosen

PrismaQuant separates two questions that most quantization methods answer
together:

- **How do you round one Linear best, given a fixed format?** GPTQ, activation
  ordering, and joint scale optimization answer this. They run underneath
  whatever format is chosen.
- **How many bits should each Linear get, and in which format?** This is what
  PrismaQuant allocates, and it is where the quality comes from.

The allocator prices every `(unit, format)` pair with a KL-adjoint Fisher cost
multiplied by the weight perturbation that the production render actually
produces, so the price reflects the bytes that ship rather than an idealized
rounding. AQUA adds an activation-aware term to that price. The term matters
because NVFP4 quantizes activations as well as weights, and a weight-only cost
cannot see that axis at all: NVFP4 and NVFP4A16 render weights bit-identically.

A multiple-choice knapsack solver then picks the assignment that minimizes
predicted KL under a byte budget. Candidate assignments are rendered and scored
on real held-out KL before one is selected.

The calibration set for the cost model is disjoint from the selection set, and
both are disjoint from the WikiText-2 text used for the numbers above. The build
verifies that disjointness rather than assuming it.

## What these numbers do not cover

This section is part of the result. Read it before comparing this artifact to
another one.

**KL is measured over the top 1024 logits, not the full vocabulary.** The
measurement collects reference probabilities through vLLM's `prompt_logprobs`
path, which returns a truncated distribution. Across the scored positions those
1024 logits hold 98.55% of the reference probability mass on average, but only
60.63% at the worst position. Mass outside the top 1024 contributes nothing to
the reported KL, so treat these values as a lower bound on full-vocabulary KL,
and do not compare them against KL numbers computed over a full vocabulary.

**The serving route is not attested.** The build records a route histogram for
every unit, and every one of the 511 units comes back as "no declared lane"
rather than as a confirmed native kernel route. No serving profile in this lane
publishes a machine-readable route status, so the build cannot state which
kernel vLLM selects for these bytes. The recorded count of units on a fallback
route is zero, and that zero is vacuous: it means nothing was checked, not that
nothing fell back. Treat kernel routing as unverified for this artifact.

**Some expert renders fell back to round-to-nearest.** Within the 80
packed-expert units rendered at NVFP4, 1836 of 20480 individual expert tensors
(8.96%) shipped a round-to-nearest render instead of a GPTQ-compensated one.
Every one of those was a held-out revert: GPTQ ran, its own held-out check found
the compensation made that expert worse, and the render reverted. The rate
concentrates in `down_proj`, reaching 36.3% at layer 23. The guard behaved
correctly, and the affected experts carry a plainer render than the rest.

**No downstream task suite was run.** These are perplexity and KL numbers. GSM8K,
IFEval, MMLU, and tool-calling fidelity were not measured on this artifact. A
model that holds perplexity can still shift behavior at a decision point, so
validate the model on your own task before deploying it.

**Bits per parameter counts quantizable parameters only.** The 4.749 figure
excludes `lm_head`, profile-pinned Linears, and the multi-token prediction
sidecar, following the convention used across PrismaQuant artifacts. Compare it
only against numbers computed the same way.

## Reproducibility

| Field | Value |
|---|---|
| Source snapshot | `e4dfb35a93d4b6822a811a7676f3488514abe7e2` |
| PrismaQuant commit | `03604c75f84f98893cfd2ac67c75db19e2f5b273` |
| Assignment hash | `7707b3b87f5f3677` |
| Layer config SHA-256 | `8fb3ea86f602181a4434e43222220464bf9dd02bf2ff7835331544cc88aa50d8` |
| Artifact SHA-256 | `4d54c5c638528e7537d80ce52b77f56dea92a627bbcc5bbd1a3b144d878daad1` |
| Render levers | GPTQ with fixed damping 1.0, static activation ordering, joint scale optimization |

The checkpoint ships in 24 shards of at most 1.001 GiB each.

## Citation

Built with [PrismaQuant](https://github.com/RobTand/prismaquant) by Robert Tand.
Contact: robert.tand@icloud.com

```bibtex
@software{tand_prismaquant,
  author = {Tand, Robert},
  title  = {PrismaQuant: production-faithful mixed-precision LLM quantization},
  url    = {https://github.com/RobTand/prismaquant}
}
```
