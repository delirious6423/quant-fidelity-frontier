---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- compressed-tensors
- qwen3_8
- int8
- autoround
---

# Qwen3.8-27B INT8 AutoRound

This is an unofficial W8A16 INT8 quantized version of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It was created using [AutoRound](https://github.com/intel/auto-round).

## Available versions

There are four branches. They vary along two axes: the quantization granularity, and whether the hybrid linear attention projections are quantized at all.

| Branch | Granularity | linear_attn | Size |
|---|---|---|---|
| `main` | per-channel | `in_proj_qkv`, `in_proj_z`, `out_proj` are INT8 | 28.8 GiB |
| `linear-attn-bf16` | per-channel | all of linear_attn stays BF16 | 33.9 GiB |
| `main-gs128` | group_size 128 | `in_proj_qkv`, `in_proj_z`, `out_proj` are INT8 | 29.3 GiB |
| `linear-attn-bf16-gs128` | group_size 128 | all of linear_attn stays BF16 | 34.3 GiB |

All four fit on 2x 24GB with tensor parallelism.

If you want the closest match to the base model, take `linear-attn-bf16-gs128`. If you want the smallest file, take `main`. The two `gs128` branches score better on mean, median and every reported percentile below p99.9, at a cost of about half a gibibyte, so they are the better default unless you are short on disk or already have the per-channel weights in place.

The per-channel branches are kept rather than replaced. They are what the earlier Qwen3.6-27B quantizations used, so they give a like-for-like reference for anyone comparing across model versions, and `linear-attn-bf16` has the mildest worst case of the four by a wide margin.

## Quantization details

Some layers are left unquantized, either because quantizing them costs more than it saves or because there is no calibration signal reaching them. The layer selection is identical across all four branches. Only the granularity and the linear attention treatment change.

| Branch | INT8 Linear | BF16 Linear |
|---|---|---|
| `main`, `main-gs128` | 407 | 210 |
| `linear-attn-bf16`, `linear-attn-bf16-gs128` | 263 | 354 |

The INT8 set covers `q_proj` / `k_proj` / `v_proj` / `o_proj` of self_attn, `gate_proj` / `up_proj` / `down_proj` of mlp, `in_proj_qkv` / `in_proj_z` / `out_proj` of linear_attn, and the MTP layers. The BF16 set covers `in_proj_a` / `in_proj_b` of linear_attn, `embed_tokens`, `lm_head` and the whole vision tower.

Notes on the choices.

* `in_proj_a` and `in_proj_b` are not a judgement call. Their output dimension is 48, which is not divisible by 32, so AutoRound skips them regardless of what the layer config asks for. Since `in_proj_b` and `in_proj_a` are fused into a single `in_proj_ba` by vLLM, both ends of that pair land on BF16 and the fused module stays consistent.
* The vision tower is excluded because the calibration set is text only, so no calibration input ever reaches it.
* `embed_tokens` and `lm_head` are excluded by the usual convention. The vocabulary is 248,320 wide, so a per-channel scale would cover 5,120 elements at a time.
* Calibration: `iters=250`, `nsamples=1024`, `seqlen=2048`, `batch_size=4`, `gradient_accumulate_steps=2`, dataset `NeelNanda/pile-10k` (256) plus `codeparrot/github-code-clean` (768). Identical for all four branches.
* Every layer in the INT8 set has an input dimension that is a multiple of 128, so `group_size=128` applies cleanly and no layer falls back to a coarser scale.

On why there are two granularities. The per-channel setting was carried over from the earlier Qwen3.6-27B recipe without being reexamined. Revisiting it on this model showed that `group_size=128` is better on mean KLD, median, p90, p95, p99 and top-1 agreement, for roughly half a gibibyte of extra scale storage. Whether a difference of that size is noticeable in use is a separate question, and this card does not claim that it is.

A note for anyone reproducing this. The shared-dict aliasing bug in AutoRound is still present in 0.15, it has only moved from `compressors/utils.py` to `compressors/layer_config/resolver.py`. The regex expansion assigns the same dict object to every matched layer, and the later shape check that forces `bits=16` on layers whose weights are not divisible by 32 mutates that shared dict, silently dragging every sibling layer to 16 bit with it. On this architecture a single `linear_attn` keyword is enough to trigger it, because the forced 16 bit on `in_proj_a` and `in_proj_b` propagates to `in_proj_qkv`, `in_proj_z` and `out_proj` and quietly erases the W8 request. Change

```layer_config[match] = val```

to

```layer_config[match] = copy.deepcopy(val)```

or specify each layer by its full suffix rather than by a shared keyword.

## Evaluation Results

### 1. KL Divergence (KLD)

The output distribution of each quantized model was compared directly to that of the base model across 122,640 scored next-token positions.

# **KL divergence cannot be compared across models, datasets, or evaluation methods, and is only comparable if all of them are the same. You canNOT compare these KLD values to others' KLD reports!**

# **If other quantized models report a very low / high KLD compared to mine in their READMEs, it is typically due to differences in the evaluation methodology.**

| Metric @ctx 512 | main | linear-attn-bf16 | main-gs128 | linear-attn-bf16-gs128 |
|---|---|---|---|---|
| Mean KLD | 0.00124 ± 0.00002 | 0.00110 ± 0.00001 | 0.00101 ± 0.00002 | 0.000956 ± 0.000021 |
| Median KLD | 0.000701 | 0.000628 | 0.000553 | 0.000507 |
| P90 / P95 / P99 KLD | 0.00231 / 0.00340 / 0.00825 | 0.00215 / 0.00306 / 0.00719 | 0.00201 / 0.00273 / 0.00666 | 0.00195 / 0.00256 / 0.00610 |
| P99.9 / Max KLD | 0.0335 / 1.843 | 0.0250 / 0.882 | 0.0258 / 1.500 | 0.0217 / 1.471 |
| Top-1 Agreement | 98.40% | 98.47% | 98.52% | 98.59% |
| PPL (Base to Quantized) | 7.9112 to 7.9101 | 7.9112 to 7.9132 | 7.9112 to 7.9114 | 7.9112 to 7.9110 |
| ln(PPL Ratio) | -0.000140 | +0.000247 | +0.000020 | -0.000027 |

Evaluation set: WikiText-2 (test), 240 sequences x 512 tokens = 122,640 scored next-token positions. Values are in nats, and one shared reference pass is used for all four columns, so they are directly comparable.

Because the scored positions are shared, the branches can be compared by a paired difference rather than through their standard errors alone. Taking `main` as the baseline, the paired mean difference is -0.000149 for `linear-attn-bf16` (t = -6.9), -0.000233 for `main-gs128` (t = -9.1) and -0.000289 for `linear-attn-bf16-gs128` (t = -11.9).

One row does not follow the pattern. The single worst position is far milder on `linear-attn-bf16` (0.882) than on any other branch (1.47 to 1.84), even though that branch is beaten everywhere else. Leaving linear attention in BF16 removes the layers where the largest single-position errors originate, and the two granularities barely differ on that point. If your concern is the worst case rather than the average, that is the branch to look at.

### 2. How to read these numbers

KLD is not an absolute property of a checkpoint. It is a property of a checkpoint measured on a corpus, and the corpus dominates the result. Text that the base model already predicts confidently leaves little probability mass for quantization to move, so the divergence is small. Text with genuine uncertainty leaves a lot, so it is larger. The same checkpoint can land several times apart on two evaluation sets, and the ordering of two checkpoints can change with it.

Two consequences are worth stating plainly.

* A mean KLD taken from one model card cannot be placed next to a mean KLD from another unless the corpus, the scored position count, the teacher checkpoint and the precision path all match. Totals in the low thousands drawn from a single document are the most sensitive case, because the positions are correlated and a predictable document pulls the mean down as context accumulates.
* Mean KLD and top-1 agreement move together within one measurement setup. Across two different setups they do not, so a checkpoint can show both a worse KLD and a better top-1 than another purely because it was measured on easier text. A pair of numbers only orders checkpoints that were measured the same way.

The protocol used here is stated in full so that it can be reproduced or disputed.

* Teacher: `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, loaded in BF16 through the Hugging Face path with `use_cache=False`.
* Corpus: WikiText-2 raw test, every row concatenated, tokenized once, first 122,880 tokens reshaped to 240 sequences of 512. The first 511 positions of each sequence are scored, giving 122,640.
* Metric: `KL(P_teacher || Q_candidate)` over the full 248,320 token vocabulary, in nats, with FP32 `log_softmax`. No sampling and no top-k truncation.
* Teacher log-probabilities are cached as FP16 and renormalized in FP32 before the divergence is taken, which removes the systematic offset the FP16 round trip would otherwise add.
* All four branches are scored against one shared reference pass, so the columns differ only by the checkpoint.

The reference pass was rebuilt from scratch partway through this work and `main` was rerun against it, reproducing every reported statistic exactly, so the measurement is deterministic end to end.

If you want another checkpoint added to the table above, open a discussion and I will run it through this same script and post the result, including if it comes out ahead of these.

### 3. Benchmarks

Not run.

## Variants

The two `linear-attn-bf16` branches are not separate AutoRound runs. Each takes the corresponding quantized checkpoint and swaps the three quantized linear attention projections back to the original BF16 weights. The rounding of the remaining layers was optimized under the assumption that linear attention is INT8, so those layers end up very slightly mismatched, but the substitution only makes the linear attention exact, which moves the block output error down rather than up.

## How to Use

Tested with the existing vLLM image built for Qwen3.6-27B, which resolves this model natively since Qwen3.8-27B reports the same `qwen3_5` model type and `Qwen3_5ForConditionalGeneration` architecture. No image rebuild was needed, and the `gs128` branches load through the same path as the per-channel ones.

The following configurations were verified to start and answer a request on 2x RTX 3090 with `--tensor-parallel-size 2` and `--gpu-memory-utilization 0.97`.

| Branch | Configuration | Result | Weights per rank | KV cache |
|---|---|---|---|---|
| main | `--language-model-only` | starts | 14.13 GiB | 240,688 tokens |
| main | vision enabled | starts | 14.57 GiB | 217,168 tokens |
| main | MTP speculative decoding | starts | 14.36 GiB | 204,800 tokens |
| linear-attn-bf16 | `--language-model-only` | starts | 16.65 GiB | 157,584 tokens |
| linear-attn-bf16 | vision enabled | starts | 17.10 GiB | 134,064 tokens |
| linear-attn-bf16 | MTP speculative decoding | starts | 16.89 GiB | 127,200 tokens |
| main-gs128 | `--language-model-only` | starts | 14.28 GiB | 231,280 tokens |
| main-gs128 | vision enabled | starts | 14.72 GiB | 208,544 tokens |
| main-gs128 | MTP speculative decoding | starts | 14.51 GiB | 201,600 tokens |
| linear-attn-bf16-gs128 | `--language-model-only` | starts | 16.77 GiB | 150,528 tokens |
| linear-attn-bf16-gs128 | vision enabled | starts | 17.22 GiB | 127,792 tokens |
| linear-attn-bf16-gs128 | MTP speculative decoding | starts | 17.01 GiB | 125,600 tokens |

Measured with `--max-model-len auto`, which resolves to the model's full 262,144 context, and `--max-num-seqs 1`. The MTP rows use `{"method": "mtp", "num_speculative_tokens": 3}`.

Two things are worth knowing before you size a deployment around these.

* The pool is smaller than the context window on every configuration here, so a single request cannot reach 262,144 tokens on 2x 24GB. The `main` branch with the language model only comes closest at 240,688. If you need the full window, you need more memory rather than a different branch.
* Setting `--max-model-len` to a small value does not leave the rest of the pool for other requests, it shrinks the pool. At `--max-model-len 8192` the same MTP configurations reported roughly half these figures. If you measure this yourself and get a much lower number, check that setting first.

## Acknowledgements

- [Alibaba / Qwen team](https://huggingface.co/Qwen) for the base [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) model
- [Intel AutoRound](https://github.com/intel/auto-round) team for the quantization framework
- [vLLM project](https://github.com/vllm-project/vllm) for the inference engine and Qwen3_5 MTP support
