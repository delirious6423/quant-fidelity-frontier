---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- deepseek
- deepseek-v4.1
- nvfp4
- modelopt
- vllm
- blackwell
- quantized
- atomic-chat
---

# DeepSeek-V4.1-Flash-NVFP4

[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) with its routed experts
re-encoded as NVFP4 by NVIDIA's DeepSeek-V4 ModelOpt recipe, calibrated, and measured against the original on
the same engine and GPUs. Produced by one run of
[`foundry-nvfp4.sh`](https://huggingface.co/AtomicChat/DeepSeek-V4.1-Flash-NVFP4-nvidia/blob/main/reproduce/foundry-nvfp4.sh) on
4×B200, 2026-09-10. Every log, the raw top-512 log-probability dumps of all ten measurement runs, the calibration
amax and the aggregation script are in
[AtomicChat/DeepSeek-V4.1-Flash-NVFP4-metrics](https://huggingface.co/datasets/AtomicChat/DeepSeek-V4.1-Flash-NVFP4-metrics).

## What this is, and what it is not

V4.1-Flash already ships its routed experts in MXFP4. NVFP4 keeps every E2M1 nibble and rewrites the block scales
(E4M3 per 16 plus one fp32 per tensor, instead of E8M0 per 32). ModelOpt's `--cast_mxfp4_to_nvfp4` does this in
closed form; all 16,986,931,200 expert blocks had scales inside the exactly representable window. We then checked
the result the direct way: 12 experts × 3 projections (0.42 G parameters) dequantized from the source and from this
checkpoint and compared element by element: **identical, worst difference 0.0**. The packed bytes differ in 11 % of
positions, every one of them a −0 nibble that the export normalized to +0. Script and report: `logs/cast-check.py`
and `logs/cast-check.txt` in the metrics dataset.

Engram tables, attention, shared experts, the LM head and the MTP/DSpark draft stay in their original FP8 / BF16.
The checkpoint is **17.00 GB (15.83 GiB) larger** than the original: 527.29 GB against 510.30 GB, because the expert
scales double. Nothing here is smaller. What NVFP4 buys is the W4A4 tensor-core path on Blackwell; what it costs
is measured below. Speed was not measured and is not claimed.

Calibration, per NVIDIA's recipe: `cnn_dailymail` + `nvidia/Nemotron-Post-Training-Dataset-v2`, 64 samples of
512 tokens each, through DeepSeek's reference `inference/model.py` under
[NVIDIA Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) at commit `079078de` with two one-line patches
for V4.1 (tokenizer into the Transformer, FP8 block size read from the scale shape). 15,246 of 15,360 routed experts
(99.3 %) received calibration tokens; 342 expert projections took the exporter's fallback scale, the maximum over
all calibrated experts of the same projection across layers. The only thing the calibration changes is
`input_scale`, one fp32 per expert projection, which places the E4M3 window for the FP4 activations on Blackwell's
NVFP4 MoE kernels.

## Measured against the original

**Reference:** the original MXFP4 checkpoint served by the same vLLM build (`dsv41-feat` at `e47aa780`,
[PR #56214](https://github.com/vllm-project/vllm/pull/56214)) on the same 4×B200, tensor parallel 4. The original
takes the FlashInfer TRT-LLM MXFP4×MXFP8 MoE path; this checkpoint takes the FlashInfer TRT-LLM NVFP4 path. The
difference measured below is the difference between those two execution paths as a whole, not an isolated cost of
FP4 activations.

**Protocol:** three corpora from [AtomicChat/calib-corpora](https://huggingface.co/datasets/AtomicChat/calib-corpora),
hashes as used in `logs/corpora-sha256.txt`: `neutral` (30 languages), `code`, and `agentic`, which is agentic
dialogue rendered in Muse Glimmer's markup (`<|start|>`, `atem:function_calls`), not DeepSeek's DSML; V4.1's
tokenizer scores it as plain text with unfamiliar control strings, which is fine for comparing builds on identical
input and is not "the model's own markup". The corpora are disjoint from every calib-corpora build by construction;
overlap with the CNN/DailyMail and Nemotron samples the recipe calibrates on was not checked. Windows of 4096 tokens
starting with BOS, the second half of each window scored, 24 windows per corpus, 49,152 scored positions each, raw
token ids in, no chat template, no speculative decoding. Teacher-forced next-token distributions only: this says
nothing about tool calls or long free trajectories.

**KL is a lower bound.** vLLM returns top-512 log probabilities. The reference's top-512 ids that every compared run
also ranked form the set S (median mass 1.00000, first percentile above 0.993); p and q are exact on S and
everything else is one bucket on both sides. Coarsening onto a common partition can only lower the KL, so the
number is at or below the true divergence, and no upper bound is claimed. Intervals are 95 % bootstrap over windows
of 2048 positions and, in the paired table, also over each build's runs, resampled independently with the
reference fixed; with three runs per build that is coarse, and it is the interval to quote.

**Runs:** three of this checkpoint, three of the uncalibrated cast (`flat`: same bits, `input_scale = 1.0`, which
is what the community NVFP4 versions carry), two repeats of the original plus one with batch size 1. Each run is a
fresh engine start.

Reference perplexity: 2.9685 (neutral), 1.8919 (code), 1.3861 (agentic).

| corpus | build | runs | KL lower bound, mean of runs (run SD) | top-1 agreement (run SD) | ppl | Δ ppl |
| --- | --- | --- | --- | --- | --- | --- |
| neutral | **this checkpoint** | 3 | 0.03548 (0.00020) | 94.10 % (0.05) | 2.9938 | +0.85 % |
| | uncalibrated cast | 3 | 0.03459 (0.00040) | 94.29 % (0.10) | 2.9901 | +0.73 % |
| | original, repeated | 2 | 0.01596 (0.00013) | 96.13 % (0.08) | 2.9689 | +0.01 % |
| | original, batch 1 | 1 | 0.01676 | 95.90 % | 2.9645 | −0.13 % |
| code | **this checkpoint** | 3 | 0.02008 (0.00031) | 96.65 % (0.09) | 1.8991 | +0.38 % |
| | uncalibrated cast | 3 | 0.01942 (0.00060) | 96.77 % (0.06) | 1.8995 | +0.40 % |
| | original, repeated | 2 | 0.01022 (0.00017) | 97.69 % (0.00) | 1.8907 | −0.06 % |
| | original, batch 1 | 1 | 0.01108 | 97.55 % | 1.8922 | +0.02 % |
| agentic | **this checkpoint** | 3 | 0.00888 (0.00011) | 98.34 % (0.05) | 1.3873 | +0.09 % |
| | uncalibrated cast | 3 | 0.00856 (0.00019) | 98.37 % (0.04) | 1.3875 | +0.10 % |
| | original, repeated | 2 | 0.00537 (0.00020) | 98.67 % (0.05) | 1.3856 | −0.04 % |
| | original, batch 1 | 1 | 0.00585 | 98.65 % | 1.3863 | +0.02 % |

Paired differences, A − B, 95 % bootstrap over windows and runs (windows-only intervals, which are narrower, are
in `logs/aggregate.txt`):

| corpus | this checkpoint − cast, KL | this checkpoint − cast, top-1 | cast − original, KL | cast − original, top-1 |
| --- | --- | --- | --- | --- |
| neutral | +0.0009 [+0.0001, +0.0019] | −0.19 pt [−0.41, +0.03] | +0.0186 [+0.0159, +0.0217] | −1.84 pt [−2.11, −1.59] |
| code | +0.0007 [−0.0002, +0.0015] | −0.12 pt [−0.29, +0.04] | +0.0092 [+0.0061, +0.0125] | −0.92 pt [−1.29, −0.58] |
| agentic | +0.0003 [−0.0001, +0.0008] | −0.02 pt [−0.13, +0.09] | +0.0032 [+0.0025, +0.0040] | −0.30 pt [−0.40, −0.20] |

Three readings:

1. **The calibration showed no convincing advantage in this experiment.** The point estimates lean the other way:
   the calibrated checkpoint sits 0.0009 KL and 0.19 points of top-1 behind the uncalibrated cast on `neutral`, but
   once run-to-run variation is in the interval only the neutral KL difference stays clear of zero, and barely;
   top-1 on every corpus and KL on `code` and `agentic` do not, and code perplexity is marginally *lower* for the
   calibrated build (1.8991 against 1.8995). The defensible statement: no measurable benefit from the calibration on
   this model, and a slight lean toward the plain cast that three runs cannot confirm. A mechanism consistent with
   the lean, and predicted before measuring, is that DeepSeek trained V4.1 with clamped SwiGLU activations (expert
   inputs bounded at 189 and 150), a flat E4M3 window already covers that range, and a per-expert ceiling set from a
   few hundred calibration tokens can only clip what it did not see. That remains a hypothesis: no clipping was
   observed, and a changed scale also moves the rounding of every value, so nothing here shows `input_scale = 1.0`
   to be optimal. We publish the calibrated build because it is the recipe; the cast is the same bits with
   `input_scale = 1.0`.
2. **The NVFP4 path costs 1.8 points of top-1 agreement and +0.7 to +0.9 % perplexity on general text**, about half
   that on code, a tenth on agentic dialogue, relative to the original on its native path. This is separated from
   the original's own run-to-run spread on every corpus, with either interval.
3. **Two runs of the original diverge from each other by 0.016 KL and disagree on 4 % of top-1 tokens**, yet a
   build's mean divergence is stable to 0.0002–0.0006 across runs. One batch-size-1 run of the original diverged
   from the batch-4 reference about as much as the batch-4 repeats did, which argues against batching as the source
   without settling it. Non-deterministic MoE kernels flipping near-tied experts is the working hypothesis, not an
   established cause.

## Running it

Needs Blackwell (B200/B300/GB200/GB300, or RTX PRO 6000 via the CUTLASS backends) and a vLLM that knows
`deepseek_v41`: as of 2026-09-10 that is the branch above, built from source. The loader keys on
`moe_quant_algo: NVFP4` in `config.json`. Measured with `tensor_parallel_size=4`, `max_model_len=4097`, no
speculative decoding; the DSpark draft experts are left in MXFP4 and that combination has not been exercised.

## Reproduce

[reproduce/runbook-nvfp4.md](https://huggingface.co/AtomicChat/DeepSeek-V4.1-Flash-NVFP4-nvidia/blob/main/reproduce/runbook-nvfp4.md) and [reproduce/foundry-nvfp4.sh](https://huggingface.co/AtomicChat/DeepSeek-V4.1-Flash-NVFP4-nvidia/blob/main/reproduce/foundry-nvfp4.sh), copied from our pipeline repository at commit `fb559ea` (the toolbox is self contained: `source` it on the box). Together with the metrics dataset:
the runbook, the toolbox, the estimator (`nvfp4_kld.py`), the paired comparison (`nvfp4_compare`), the
aggregation over windows and runs (`logs/aggregate.py`) and the cast check (`logs/cast-check.py`), the latter two
in the metrics dataset. The upstream reference code, encoding and license files are DeepSeek's, copied unchanged.
