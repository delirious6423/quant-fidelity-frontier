---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language: [en, ko]
tags: [awq, w4a16, compressed-tensors, quantized, vllm]
pipeline_tag: text-generation
base_model_relation: quantized
library_name: transformers
---

# Qwen3.8-27B-W4A16-AWQ

**18.59 GB.** An AWQ counterpart to our [GPTQ build](https://huggingface.co/ThakiCloud/Qwen3.8-27B-W4A16-GPTQ)
of the same model at the same size, published mainly so the two can be compared under identical
conditions. If you are picking one to deploy, read the last section — for Korean workloads we would take
the GPTQ build.

## ⛔ Score this with `flexible-extract`, not `strict-match`

This is the single most important thing on the card. GSM8K, full 1,319-item set, greedy, one serving
session, identical config across arms:

| | en strict | en flexible | ko strict | ko flexible |
|---|---|---|---|---|
| bf16 | 0.5542 | 0.5550 | 0.6149 | 0.6315 |
| GPTQ sibling | 0.5133 | 0.5641 | 0.5686 | 0.6505 |
| **this build (AWQ)** | **0.4147** | 0.5299 | 0.5921 | 0.6156 |

Against bf16 this build is **−13.95pp on strict (7.24σ)** and **−2.51pp on flexible (1.29σ)**. Those two
numbers describe the same outputs. The gap is not capability — it is that this build often answers
correctly without wrapping the answer in the `#### N` form that strict-match requires.

We checked this rather than assuming it. Four blind judges read 40 disputed cases from an earlier run of
the same family: **35 of 40 were correct answers rejected on formatting**. AWQ shows the effect more
strongly than GPTQ does (−9.86pp against the sibling on strict, versus −3.42pp on flexible).

If you benchmark this checkpoint with strict-match and conclude it is broken, you have measured the
extractor, not the model.

## What it is

AWQ W4A16 via `llmcompressor`, group size 128, symmetric int4. Same base model, same target size, same
ignore set as our GPTQ build — the vision tower, `lm_head` and embeddings stay at original precision.
This is a hybrid VL model (64 layers: 16 full-attention, 48 linear-attention/GDN), which is why 4-bit
weights land at 18.59 GB rather than the ~14 GB the naive arithmetic suggests.

## Where it actually differs from the GPTQ build

Comparing the two 4-bit builds directly, and reading each gap against the **rebuild noise floor** we
measured for this pipeline (a second build of an identical recipe lands this far away by itself):

| metric | AWQ − GPTQ | floor | multiple | attributable? |
|---|---|---|---|---|
| en strict | −9.86pp | 3.56pp | 2.77x | yes — but this is the formatting axis above |
| en flexible | −3.42pp | 2.27pp | 1.51x | **no** |
| ko strict | +2.35pp | 2.13pp | 1.10x | **no** |
| **ko flexible** | **−3.49pp** | 1.37pp | **2.55x** | **yes** |

So: on English capability the two are not distinguishable. On **Korean** flexible-extract the AWQ build is
genuinely behind by more than twice the floor. That is not a formatting artifact, and it is the one place
we would let it decide a deployment.

**Recommendation: English or mixed workloads, either build. Korean-heavy workloads, take the GPTQ one.**

⚠️ The floor above was measured on our **GPTQ** pipeline. We have not measured the rebuild spread of the
AWQ pipeline, so "attributable" here is weaker than it looks. Floor data:
[`ThakiCloud/quantization-rebuild-noise-floor`](https://huggingface.co/datasets/ThakiCloud/quantization-rebuild-noise-floor).

## Usage

```bash
vllm serve ThakiCloud/Qwen3.8-27B-W4A16-AWQ \
  --max-model-len 131072 --max-num-seqs 256 \
  --kv-cache-dtype fp8 --mamba-cache-mode align
```

`--mamba-cache-mode align` is required (hybrid SSM). Native context is 262,144. KV costs 64 KiB/token on
this model; `--kv-cache-dtype fp8` gives 1.95x the KV pool at no measured throughput or quality cost.

W4A16 runs through Marlin dequantization, so it is the slow-but-small option: on a single B200 our NVFP4
build reaches 3.52x this family's ceiling throughput but needs 30.14 GB. On a 32 GB card that build does
not leave room for KV cache and this one does.

## Scope

One run, one task family (GSM8K en/ko, full sets), one hardware generation. Not measured: code,
long-context retrieval, MMLU-class breadth, or the rebuild spread of this AWQ pipeline. Two of the
comparisons above sit at 1.77σ and 1.86σ — borderline on a single run, and we would repeat before
treating them as settled.
