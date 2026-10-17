---
license: apache-2.0
base_model:
  - prism-ml/Ternary-Bonsai-2-27B-gguf
  - Qwen/Qwen3.8-27B
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - gsq
  - rco
  - ternary
---

# Bonsai 2 27B — full GSQ and RCO

An experimental refinement of [Prism's Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
using full-model ternary GSQ, task-loss RCO, and a final discrete allocation polish.
The RCO gradient searches alone selected the original weights; the separately
documented polishing step produced the changed allocation released here. The final file is
**6,496,332,448 bytes (6.496 GB)**.
Its dominant format is Prism **PTQ1_0**, with group size 128; standard TQ1_0 is a
different format. RCO can allocate Q4_0 to selected complete matrices while
keeping the GGUF below 6,500,000,000 bytes.

This is an independent experiment, not an official Prism, Qwen, or ISTA-DASLab release.
It adapts the methods from [GSQ](https://github.com/IST-DASLab/GSQ) and
[RCO](https://github.com/IST-DASLab/RCO); it does not reproduce the exact recipe or
benchmark protocol of [the reference GSQ/RCO model](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF).

## Measured results

All GGUF comparisons use the same pinned native runtime, examples, prompts and
scoring. Accuracy columns are percentages; lower perplexity is better.
WikiText-2 perplexity is **5.84% lower** than original Bonsai.
All four paired accuracy intervals include zero. The measured accuracy changes are small and mixed; this experiment does not establish a statistically clear accuracy improvement.

| Model | WikiText-2 PPL | ARC-C | ARC-E | HellaSwag | Winogrande |
|---|---:|---:|---:|---:|---:|
| Original Bonsai PTQ1_0 | 8.6690 | 61.01 | 79.76 | 77.65 | 73.80 |
| Bonsai GSQ + RCO | 8.1627 | 60.67 | 80.13 | 77.80 | 74.11 |
| Original Qwen3.8 BF16 | 6.5137 | 59.30 | 74.92 | 82.15 | 75.14 |

Relative to the original Bonsai quant, the perplexity change is -0.5063
and the unweighted mean of the four accuracy changes is +0.13 percentage points.
Read individual tasks and uncertainty alongside this small aggregate.
These measurements do not establish universal improvement, reasoning quality,
or equivalence to the BF16 base model.

| Task | Accuracy change (pp) | Paired bootstrap 95% interval |
|---|---:|---:|
| arc_challenge | -0.34 | [-1.79, +1.11] |
| arc_easy | +0.38 | [-0.51, +1.26] |
| hellaswag | +0.15 | [-0.55, +0.85] |
| winogrande | +0.32 | [-1.10, +1.74] |

WikiText-2 uses the raw test split, context 2048 and the native perplexity tool.
ARC-C uses 1,172 examples, ARC-E 2,376, HellaSwag a fixed random 2,000 validation examples,
and Winogrande 1,267 validation examples. This is a custom zero-shot continuation
likelihood harness without a chat template or generated reasoning. ARC and
HellaSwag use character-normalized likelihood; Winogrande scores the suffix after
placing each candidate in context. Raw and normalized scores, paired 5,000-resample
bootstrap results, and dataset revisions are included. Do not compare these
numbers directly to unrelated harnesses or thinking-mode leaderboard results.

## Training and allocation

GSQ optimized all **402/402 original ternary matrices**, including all 64
transformer blocks and every row of the embedding and output head. Corresponding
Q4 variants also received GSQ optimization. This was a full coverage run; no
unprocessed transformer layers were substituted for completed GSQ.

The teacher was original Qwen3.8-27B BF16. Sampled Bonsai F16 weights were already
exactly ternary, so reconstructing those values against themselves would not
recover lost base precision. Ternary optimization starts from Bonsai's codes;
Q4 starts from rotated round-to-nearest Qwen weights. Original Bonsai non-ternary
normalization/recurrent parameters and rotation metadata are retained.

Each block uses 1,024 disjoint FineWeb-Edu documents of 2,048 tokens, 64 held-out
validation documents, 10 epochs, effective batch 32, FP32 Gumbel logits and scales,
BF16 compute, Lion, temperature 2→0.05 and logit multiplier 100→500. All eligible
attention and MLP matrices train jointly within each block; hard chosen outputs
propagate to subsequent blocks. Fused Triton kernels implement the Gumbel
quantizers, gradients, dequantization, Hadamard rotations and optimizer updates.
The embedding and head optimize every row in 2,048-row chunks for 256 steps each.
The best validation checkpoint may retain an initialization when training does
not improve it. Full coverage describes optimization performed, not a guarantee
that every selected code changes.

RCO runs 768 end-to-end cross-entropy allocation updates with upstream tangent projection,
retraction and optimizer-vector transport, Gumbel straight-through sampling,
and exact multiple-choice knapsack rounding under a byte budget. It uses a
separate 128-document calibration split and 64-document selection split. The
initial search's strongest four checkpoints are reranked on all selection documents.
Benchmark examples do not select the allocation.
The initial 256-update search selected the original model on the full
selection split. A further 512-update refinement reused its
allocation logits with a +1 preference for the original option, Adam LR0.025,
and temperature0.8→0.05. Each refinement update averages two Gumbel assignments
with two calibration windows each. It covers all four512-token windows of all
128RCO training documents; six strong checkpoints are reranked on64selection
documents. Initial results and the full refinement protocol are retained.
The 768 gradient updates still selected the original allocation. A separately
documented discrete coordinate polish then measured all 804 single-matrix
alternatives on 16 RCO calibration documents. It tested an exact-budget additive
knapsack proposal and up to 128 greedy proposals on 32 different RCO calibration
documents, accepting only measured loss reductions. Original, predefined greedy
snapshots, the final greedy allocation and the additive proposal were compared
on the 64 selection documents. This extra finite-difference search is not counted
as additional RCO gradient updates. `polish-protocol.json` and `polish-final.json`
record its fixed document windows, decisions and final selection. Benchmark
examples were excluded throughout.

Selected matrix counts: **10 GSQ PTQ1_0, 370 original PTQ1_0,
22 GSQ Q4_0**. Original candidates remain available to RCO as a guard
against harmful reconstruction changes. See `tensor-allocation.json` for exact
types, bytes, parameters and the GGUF SHA-256.

The run used one 96 GB NVIDIA RTX PRO 6000 Blackwell, with a 20-hour wall-clock cap.
Native/Torch parity, packing round trips, fused-kernel gradients, RCO memory and
gradient checks, and native scoring comparisons are recorded in `reports/`.
Reconstruction errors are diagnostics, not accuracy benchmarks.

## Runtime

Use [Prism's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) at
`9a9394a895b96003ca842a6041cb28ac49a108f7` or a compatible runtime with PTQ1_0
and the model's rotation metadata. Standard llama.cpp compatibility is not assumed.

```bash
git clone https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp
git checkout 9a9394a895b96003ca842a6041cb28ac49a108f7
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli
build/bin/llama-cli -m /path/to/bonsai-2-27b-gsq-rco-PTQ1_0.gguf -ngl 999 -c 4096
```

## Reproduction and attribution

Owned implementation sources are in `source/`, exact corpus revisions and split
document hashes in `data/`, and raw scores/training history in `reports/`.
Read `REPRODUCE.md` before running them; machine-specific paths and the expired
run deadline must be configured. This inference release does not include every
rejected intermediate candidate or optimizer checkpoint.

Source Bonsai revision: `6ed5e12bf84b7a63069882c91dd9e9218647d17b`.
Qwen teacher revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
GSQ revision: `03fc16484c369e3127225615d5e03e8d3a6043e3`.
RCO revision: `9a1e09c07d468109cbe60a1b87d5036034a79d10`.
Model licensing follows the upstream Apache 2.0 license and notice. Preserve
`LICENSE`, `NOTICE.txt`, and upstream code licenses under `licenses/`.
