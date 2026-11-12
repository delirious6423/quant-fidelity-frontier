---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
- en
- zh
- ko
tags:
- deepseek
- deepseek-v4.1
- gguf
- vector-quantization
- vq
- gptq
- gptvq
- hessian
- error-feedback
- imatrix
- calibration
- mixture-of-experts
- engram
- mtp
- speculative-decoding
- vision
- 1m-context
- dgx-spark
- gb10
- ds4
- ds4-dfm-rs
---

# DeepSeek-V4.1-Flash-1017 hVQ GGUF

**The [iVQ release](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-iVQ-GGUF) of DeepSeek's October 17 V4.1 Flash with the codes of its routed experts reassigned by error feedback through the input correlation: the same codebooks, layout, size and runtime.**

The lineage is VQ → iVQ → hVQ. VQ fits its codebooks to the weights alone. iVQ weighs
every input channel of every expert by how strongly real inputs drive it: the diagonal
of the input second moment `H = Σ x xᵀ`. hVQ also uses the off-diagonal part, GPTQ /
GPTVQ style: the 8-weight vectors of a row are assigned in order, and the error each one
leaves is fed forward into the targets of the vectors not yet assigned, so that errors
on correlated input channels cancel in the expert's output instead of adding up.

Only the code indices and the row gains change. The codebooks are iVQ's bytes, and every
byte outside the 40 routed-expert blobs is the VQ base's, so the comparison with iVQ
measures the off-diagonal term and its error feedback alone, and the files drop into a
setup that already serves VQ or iVQ.

> **Status, 2026-11-11:** converted, verified and uploaded. The base has VQ's exact tensor
> inventory and byte layout (1,000 tensors, 113,556,639,424 bytes; 8,359,513,152 bytes
> outside the expert blobs compared byte for byte) and passes the ds4-dfm-rs DeepSeek
> V4.1 loader check (`bind: slots=1000 bound=1000 required-missing=0`).
> **Measured against the original FP4/FP8 model:** Mean per-layer median routed-output error 0.2876 → 0.1885 (-34.5%); generated-position KL 0.2899 → 0.2408 (-17.0%). Generated-position NLL 1.3466 → 1.3589 (+0.9%).
> **On DGX Spark (GB10), the released GGUF through ds4-dfm-rs:** generated-position KL 0.296 → 0.247 (-16.7%), same top-1 as the original 81.9% → 84.0%, NLL 1.348 → 1.360 (+0.012, inside its 95% interval). `ds4 --serving-check` passes; chat, thinking, tools, vision and all four API surfaces answer ([below](#on-dgx-spark)).

> **Experimental low-bit release.** The routed experts remain at about 1.6 bits
> per weight. The measurements below describe teacher-forced fidelity. Sampled
> generation was checked on a small seeded set only; no task benchmark has been run.
>
> **Problems or odd output?** Post the prompt, sampling settings and runtime commit
> in the [Community tab](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-hVQ-GGUF/discussions).

## At a glance

| | |
|---|---|
| Source | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) at `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Starting point | [Baekpica/DeepSeek-V4.1-Flash-1017-iVQ-GGUF](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-iVQ-GGUF) (its codebooks), patched into the [VQ base](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF) (YoungAi `vq8sh14-q4k-mtpnative`) |
| What changes | In the 40 routed-expert VQ blobs: the code indices and the F16 row gains |
| What does not change | The codebooks (iVQ's, byte for byte), the bit allocation (layers 0–13 13-bit K=8192, layers 14–39 12-bit K=4096), file layout and size. The q4_K skeleton, the MTP draft towers, the vision sidecar and the Engram tables are byte-identical to the VQ release |
| Setting | `t = 1`, damping `0`, chosen once on four layers and applied to all 40 ([below](#how-t-was-chosen)) |
| Calibration | The iVQ set: 3,095 sequences, 3,653,893 tokens, same teacher stream |
| Fidelity (reference forward, vs the original) | Mean per-layer median routed-output error 0.2876 → 0.1885 (-34.5%); generated-position KL 0.2899 → 0.2408 (-17.0%). Generated-position NLL 1.3466 → 1.3589 (+0.9%). |
| On DGX Spark (released GGUF, vs the original) | Generated-position KL 0.296 → 0.247 (-16.7%); same top-1 81.9% → 84.0%; serving checklist passes |
| Speed on DGX Spark | Prefill 1,118–1,519 tok/s, decode 28–35 tok/s with automatic MTP (2K–64K full-prefill sweep, [below](#speed)), with [ds4-dfm-rs PR #147](https://github.com/Baekpica/ds4-dfm-rs/pull/147) |
| Size | Base GGUF 113,556,639,424 bytes (40 parts), the same as VQ and iVQ. Routed experts about 1.6 bits per weight |
| Context | 1,048,576 tokens, from the model metadata |
| Runtime | [ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs) on DGX Spark (CUDA), the same as VQ and iVQ. YoungAi `ds4-server` reads the same layout |

## What the feedback changes

A routed expert's weight matrix is cut into 8-weight vectors along each row. Each row
is divided by its gain, and every vector is replaced by one of K shared codewords. The
output error of an expert is the error of `W·x`. For one row with weights `w` and
quantized weights `q` it is

$$\textstyle (w - q)\,H\,(w - q)^\top,\qquad H = \sum_{\text{tokens}} x\,x^\top$$

iVQ keeps the diagonal of `H` (its column weights `D`), which lets every vector be
assigned on its own. hVQ uses

$$\textstyle H_t = D^{1/2}\,\big((1 - t)\,I + t\,R\big)\,D^{1/2}$$

with the same `D` and with `R` the correlation matrix of the expert's input. With
`H_t = V Vᵀ` (`V` upper triangular) the row error splits into one term per 8-column
block, and block `B`'s term involves the blocks before it only. So the blocks are
assigned in order: block `B`'s target is its weights plus what the errors of the
earlier blocks leave on it through `V`, and its codeword is the nearest one under the
block's own 8×8 metric. This is GPTQ's update written with the factor of `H` instead of
the factor of its inverse. The row gain is then refit by least squares under `H_t`, and
the pass runs once more with the new gains, as in iVQ.

- `t = 0` is iVQ exactly: no feedback, the same column-weighted distance, the same
  gains. On layer 14, the converter at `t = 0`, a rerun of the iVQ converter and
  the published iVQ blob match byte for byte. Synthetic tests cover both bit widths.
- `t = 1` is full GPTQ feedback on the correlation. At about 1.6 bits per weight the
  error of a block is a large fraction of the block itself, and the estimate of `R`
  for an expert with few routed tokens is noisy, so `t` is chosen on held-out tokens.
- `R` comes from sums of `x xᵀ / mean(x²)` over the tokens routed to the expert: every
  token counts the same. Raw sums would describe the handful of massive tokens (BOS
  and a few others) that carry 94–100% of the activation energy from layer 10 on.
- Before forming the correlation, each expert's second moment is shrunk toward the
  layer's mean per token (gate and up) or toward the identity at its own mean diagonal
  (down), with a pseudo-count of 4096 tokens.

**Weight-space and fitted quadratic error** (mean over the layers in each group;
each cell is iVQ → hVQ). The `H_t` columns use the calibration-derived metric above.
The held-out routed-output errors in the next section are measured directly.

| MoE layers | Plain weight error, iVQ → hVQ | Activation-weighted (D) error | H_t error, gate/up | H_t error, down |
|---|---:|---:|---:|---:|
| 0–13 (13-bit) | 0.148 → 0.193 (+30.0%) | 0.131 → 0.190 (+44.8%) | 0.095 → 0.039 (-59.2%) | 0.116 → 0.091 (-20.9%) |
| 14–39 (12-bit) | 0.173 → 0.204 (+17.7%) | 0.121 → 0.151 (+24.6%) | 0.125 → 0.066 (-47.6%) | 0.141 → 0.119 (-15.8%) |
| All 40 | 0.165 → 0.200 (+21.5%) | 0.125 → 0.165 (+32.0%) | 0.115 → 0.056 (-51.0%) | 0.132 → 0.109 (-17.4%) |

## Measured against the original

All numbers come from the reference forward in `tools/vq_calib/ds41_ref.py` (DeepSeek's
`inference/model.py` on the prefill path with its FP8/FP4 kernels emulated), on Runpod
1× H200. The original is the FP4/FP8 checkpoint; VQ, iVQ and hVQ replace only the routed
experts with their dequantized blobs. VQ and iVQ were measured again here, on the same
GPU and in the same runs as hVQ, so their numbers differ slightly from the ones on their
cards (8× A100). The held-out set is the iVQ card's: 26 cases, 58,803 tokens.

| vs the original | Floor | VQ | iVQ | hVQ | hVQ vs iVQ |
|---|---:|---:|---:|---:|---:|
| Routed-expert error per layer, per token (median, mean of 40 layers) | — | 0.309 | 0.288 | 0.189 | -34.5% |
| Routed-expert set overlap with the original (mean of layers 1–39) | — | 75.2% | 76.5% | 82.2% | +5.7 pt |
| Final hidden state cosine | — | 0.778 | 0.798 | 0.869 | +0.071 |
| KL, all positions | 0.037 | 0.370 | 0.352 | 0.273 | -22.3% |
| KL, generated positions | 0.012 | 0.293 | 0.290 | 0.241 | -17.0% |
| KL, prompt positions | 0.111 | 0.601 | 0.538 | 0.371 | -31.1% |
| Same top-1, all positions | 95.6% | 82.42% | 82.41% | 84.67% | +2.26 pt |
| Same top-1, generated positions | 96.3% | 82.39% | 82.22% | 84.39% | +2.17 pt |
| NLL, all positions (original 1.260) | 1.259 | 1.236 | 1.252 | 1.387 | +0.14 |
| NLL, generated positions (original 1.235) | 1.235 | 1.334 | 1.347 | 1.359 | +0.01 |

**Numerical floor:** The same original model run twice, the held-out cases batched differently. This
measures the reference forward's numerical variation under a batching change;
small differences should be read alongside it.

**Generated positions:** Raw-text cases, and the assistant turns of chat cases.

**Per layer.** With the original's own inputs and routes, the routed output
`Σ w_k E_k(x)` of every held-out token is computed with the original experts and with
each conversion's, and its relative error `‖Δy‖² / ‖y‖²` taken per token:

Each group row averages the per-layer medians and energy-weighted ratios; it is
not a single median pooled across layers.

| MoE layers | Per token (median): VQ → iVQ → hVQ | hVQ vs iVQ | Layers hVQ < iVQ | Energy-weighted: VQ → iVQ → hVQ | hVQ vs iVQ |
|---|---:|---:|---:|---:|---:|
| 0–13 (13-bit) | 0.239 → 0.215 → 0.115 | -46.4% | 14 of 14 | 0.220 → 0.164 → 0.074 | -55.0% |
| 14–39 (12-bit) | 0.347 → 0.327 → 0.228 | -30.2% | 26 of 26 | 0.345 → 0.075 → 0.014 | -81.7% |
| All 40 | 0.309 → 0.288 → 0.189 | -34.5% | 40 of 40 | 0.301 → 0.106 → 0.035 | -67.3% |

**By kind of text:**

| Held-out text | Tokens | KL: VQ → iVQ → hVQ | hVQ vs iVQ | Same top-1: VQ → iVQ → hVQ | NLL: original → VQ → iVQ → hVQ |
|---|---:|---:|---:|---:|---:|
| WikiText-2 | 4,096 | 0.497 → 0.482 → 0.419 | -13% | 78.6% → 78.5% → 80.4% | 1.12 → 1.43 → 1.41 → 1.42 |
| Web en / ko / zh / ja / es | 10,346 | 0.377 → 0.403 → 0.350 | -13% | 75.3% → 74.4% → 76.8% | 1.90 → 2.05 → 2.11 → 2.11 |
| Math web | 1,476 | 0.217 → 0.204 → 0.138 | -32% | 92.5% → 93.8% → 95.4% | 0.09 → 0.29 → 0.28 → 0.22 |
| Code files (Python, C) | 4,094 | 0.109 → 0.103 → 0.087 | -15% | 96.7% → 96.4% → 97.0% | 0.10 → 0.17 → 0.16 → 0.15 |
| Chat ko / zh / ja | 6,047 | 0.264 → 0.237 → 0.134 | -44% | 84.4% → 85.1% → 89.2% | 1.11 → 1.10 → 1.09 → 1.15 |
| Reasoning (think) | 6,142 | 0.157 → 0.139 → 0.084 | -40% | 88.2% → 88.4% → 91.6% | 0.87 → 0.81 → 0.83 → 0.88 |
| Instruction following | 1,517 | 0.323 → 0.333 → 0.326 | -2% | 79.8% → 80.5% → 81.9% | 0.98 → 0.99 → 1.01 → 1.08 |
| Tool calls / agents (DSML) | 2,659 | 1.330 → 1.015 → 0.563 | -45% | 69.8% → 71.3% → 76.5% | 2.37 → 1.26 → 1.42 → 2.61 |
| SWE agentless | 1,923 | 1.059 → 1.024 → 0.800 | -22% | 71.5% → 72.2% → 75.0% | 2.25 → 1.37 → 1.36 → 2.47 |
| Competitive programming (think) | 4,095 | 0.397 → 0.391 → 0.195 | -50% | 79.7% → 80.5% → 85.7% | 1.21 → 1.03 → 1.01 → 1.22 |
| Long 8K (web, finance) | 16,382 | 0.292 → 0.290 → 0.275 | -5% | 84.7% → 84.3% → 84.8% | 1.23 → 1.29 → 1.29 → 1.35 |

The direct routed-output error falls in all 40 layers under both summaries. The mean
per-layer median falls from 0.2876 to 0.1885 (34.5%); the mean energy-weighted ratio
falls from 0.1061 to 0.0347 (67.3%). The median reduction is largest in layers 0–2
(79.1%, 61.6% and 60.6%) and smallest in layers 26, 30 and 27 (18.7%, 18.9% and
19.6%). The large energy-weighted gain gives more influence to the massive routed
outputs; the median describes a typical token within each layer. The error tails
also fall, although an average 2.2% of tokens per layer have higher error than iVQ.
Plain weight error rises 21.5% and the diagonal-weighted error rises 32.0%, while
the directly measured routed outputs become closer to the original.

The final prediction distribution also moves closer to the original: generated-position
KL falls 17.0%, top-1 agreement rises 2.17 percentage points, final hidden-state cosine
rises from 0.7985 to 0.8692, and mean route-set overlap rises from 76.55% to 82.23%.
All-position KL is lower in 18 of 19 domains; long finance rises from 0.1606 to
0.1659. Generated-position KL is lower in 17 of the 18 domains with generated
targets; instruction-following chat rises slightly, from 0.3263 to 0.3280.
The long-finance case has no generated targets. The original's batching floor is
0.0368 KL over all positions and 0.0120 over generated positions; this is a
measurement of numerical variation, not a statistical confidence interval.

NLL on the actual held-out text worsens: 1.3466 → 1.3589 at generated positions
(+0.9%), and 1.2517 → 1.3869 over all positions (+10.8%). The largest all-position
NLL increases are in tool/agent and SWE cases, while raw math and code improve.
KL measures agreement with the original distribution; NLL scores the particular
target tokens, so the two need not move together. These results support improved
fidelity to the original, with a measured NLL tradeoff. They are teacher-forced
measurements of routed-expert substitutions: the non-expert body is the original
checkpoint, so the GGUF's q4_K body and MTP are not evaluated by these scores.
The same held-out set also selected the pilot setting. No quality threshold was
used to accept or reject the build.

The complete [evaluation record](EVALUATION.md) includes all 40 layers, all 19
domains at both all and generated positions, error tails, and conversion time and
memory for each layer.

## On DGX Spark

The tables above replace the routed experts inside the original body. These
run the released files on one DGX Spark (GB10, CUDA 13.4, clocks 300–2,200
MHz) with ds4-dfm-rs `main` at `502019da`: the q4_K body and the engine's
arithmetic are included. The engine's teacher-forced score entry wrote every
position's logits for the same 26 cases; the original's logits are its
reference final states times `head.weight`.

| vs the original, generated positions (44,157) | Floor | VQ | iVQ | hVQ | hVQ − iVQ (95% interval over cases) |
|---|---:|---:|---:|---:|---:|
| KL | 0.012 | 0.302 | 0.296 | 0.247 | -0.050 (-0.062 to -0.038) |
| Same top-1 as the original | 96.3% | 81.96% | 81.87% | 84.02% | +2.1 pt (+1.5 to +3.0) |
| Top-1 is the text's token (original 71.72%) | 71.71% | 68.95% | 68.80% | 69.47% | +0.7 pt (+0.2 to +1.2) |
| NLL (original 1.235) | 1.235 | 1.340 | 1.348 | 1.360 | +0.012 (-0.003 to +0.031) |
| Entropy (original 0.816) | 0.818 | 1.155 | 1.077 | 0.889 | -0.188 (-0.229 to -0.143) |

Over all 58,777 positions KL is 0.379 → 0.362 → 0.277.

- **The released file keeps the gain.** The full GGUF adds 0.004–0.010 KL to
  each of the three against the reference substitution. Generated-position KL
  is lower under hVQ in all 18 domains with generated targets, all-position
  KL in 25 of the 26 cases.
- **Why NLL does not follow KL.** hVQ is nearly as sharp as the original; VQ
  and iVQ are flatter. Where the original's top-1 is the text's token (72% of
  generated positions) the NLL is 0.154 for the original, 0.350 for iVQ and
  0.290 for hVQ. Where it is not, hVQ follows the original (4.07 against
  3.98) and the flatter iVQ hedges (3.88). hVQ is the closer copy of the
  original, including where the original is wrong.

**Serving.** `ds4 --serving-check` passes on this artifact: prefix reuse,
continue and disk KV each reproduce 16 of 16 tokens, as for iVQ. Behind
`ds4-server` the model is resident 153 s after launch with 10.5 GiB of memory
left available. One request per surface answered as expected: chat, thinking,
SSE streaming, a tool call and its follow-up, a base64 image, prefix reuse,
`/v1/messages`, `/v1/responses`, `/v1/completions` and a 24K-token prompt
with one planted fact.

A seeded set at temperature 1.0, the same seeds for both artifacts:

| 44 requests per artifact | iVQ | hVQ |
|---|---:|---:|
| Thinking arithmetic (en, ko): exact answer | 24 / 24 | 24 / 24 |
| Thinking closed and finished with `stop` | 24 / 24 | 24 / 24 |
| Tool call valid (name, city, unit) | 12 / 12 | 12 / 12 |
| Chat ko / en / zh / ja, code, translation: `stop` | 8 / 8 | 8 / 8 |

It shows no gross failure, not a ranking. Tables by kind of text and the method are
in the [evaluation record](EVALUATION.md#on-dgx-spark-gb10).

### Speed

![DeepSeek V4.1 Flash 1017 hVQ prefill and decode on one DGX Spark, 2K to 64K, before and after the November 12 prefill rounds](assets/ds41-hvq-spark-64k-2026-11-12.png)

[ds4-dfm-rs PR #147](https://github.com/Baekpica/ds4-dfm-rs/pull/147) makes prefill of this artifact 52%
faster over the sweep, from 18% at 2K to 65% at 64K. VQ and iVQ share its tensor shapes and kernels and take the same
paths. The Engram prefetch buffer no longer keeps the size of the first block a server
process saw, index scores run on tensor cores, and routed experts run on FP8 tensor
cores with E4M3 activations. Each run is a fresh `ds4-server` on one DGX Spark at
2,184–2,190 MHz. Every frontier prefills its whole prompt and generates up to 128
greedy tokens with automatic MTP.

| 2K–64K sweep, 2026-11-12, fresh runs: 3 before, 3 after | Prefill before | Prefill after | Decode before | Decode after |
|---|---:|---:|---:|---:|
| Median of the runs' means over 32 frontiers | 955.5 | **1,448.3** | 32.4 | 32.1 |
| 2K frontier (2,149 tokens) | 949.7 | **1,118.1** | 34.8 | 31.6 |
| 8K frontier (8,449 tokens) | 1,028.0 | **1,409.4** | 31.7 | 33.6 |
| 16K frontier (16,546 tokens) | 1,028.8 | **1,486.9** | 36.0 | 33.7 |
| 32K frontier (32,749 tokens) | 932.9 | **1,445.9** | 32.7 | 30.5 |
| 64K frontier (66,608 tokens) | 876.6 | **1,442.8** | 34.7 | 31.2 |

Before is [`main` 2f7a19e5](https://github.com/Baekpica/ds4-dfm-rs/commit/2f7a19e5), after is
[`2ad7267a`](https://github.com/Baekpica/ds4-dfm-rs/commit/2ad7267a567722eb049e899449792b159366aa50); cells are tok/s, per-frontier cells medians.
The first two changes leave every logit bit-identical. The FP8 experts change
arithmetic: on the 26 held-out cases above, generated-position KL to the original is
0.2467 before and 0.2464 after, NLL 1.3603 and 1.3605, and the same top-1 as the
original over all positions 84.38% for both.
The logits move 1.3 times as far as they do between 1,024- and 2,048-token prefill
chunks. Greedy continuations differ from their first tokens on, so decode differs by
prompt; over the 32 frontiers it does not move beyond its noise. `DS4_DS41_VQ_FP8=0`
restores the BF16 expert kernels. The one pass first published here (958–1,157 tok/s)
was taken on a server that had not lost its Engram prefetch; a freshly booted server,
as in the before column, had. [Rounds and gates](https://github.com/Baekpica/ds4-dfm-rs/blob/2eb910ee5a3068cb8d3c2f922008d6e26597b44a/docs/deepseek41-optimization-2026-11-12.md) ·
[sweep record](https://github.com/Baekpica/ds4-dfm-rs/blob/2eb910ee5a3068cb8d3c2f922008d6e26597b44a/docs/benchmarks/ds41-hvq-spark-64k-2026-11-12/README.md).

## How t was chosen

Chosen once on layers 3, 11, 19 and 27 by the mean ratio to iVQ of the median routed-output error over 58,803 held-out tokens. The same setting is used on all 40 layers.

Pilot executed in the previous pod; raw sweep JSONL and probe files were lost. The table and precision summaries below are recovered verbatim from the archived session output, not rerun on this pod.

| t | damping | shrink (tokens) | Layer 3 (13-bit) | Layer 11 (13-bit) | Layer 19 (12-bit) | Layer 27 (12-bit) | Mean ratio to iVQ | Target energy, last quarter of the blocks (gate/up) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VQ (weights only) | | | 0.2231 | 0.2612 | 0.2897 | 0.3683 | 1.091 | |
| iVQ (t = 0) | | | 0.1936 | 0.2466 | 0.2589 | 0.3561 | 1.000 | 1.00 |
| 0.25 | 0 | 256 | 0.1094 | 0.1899 | 0.1688 | 0.3032 | 0.7097 | 1.02 |
| 0.5 | 0 | 256 | 0.0922 | 0.1746 | 0.1496 | 0.2908 | 0.6448 | 1.05 |
| 0.75 | 0 | 256 | 0.0831 | 0.1666 | 0.1389 | 0.2853 | 0.6107 | 1.09 |
| 0.9 | 0 | 256 | 0.0799 | 0.1649 | 0.1352 | 0.2857 | 0.6014 | 1.13 |
| 1 | 0 | 256 | 0.0794 | 0.1653 | 0.1345 | 0.2883 | 0.6024 | 1.19 |
| 0.75 | 0.01 | 256 | 0.0834 | 0.1670 | 0.1391 | 0.2854 | 0.6116 | 1.09 |
| 1 | 0.01 | 256 | 0.0794 | 0.1652 | 0.1345 | 0.2878 | 0.6020 | 1.18 |
| 0.75 | 0.1 | 256 | 0.0860 | 0.1687 | 0.1419 | 0.2858 | 0.6198 | 1.08 |
| 0.75 | 0 | 4096 | 0.0837 | 0.1670 | 0.1397 | 0.2854 | 0.6127 | 1.09 |
| **1** | **0** | **4096** | **0.0793** | **0.1647** | **0.1345** | **0.2863** | **0.6003** | 1.17 |


## Lineage and prior work

**All three stages have prior work at their core. What is new lies in the setting, the combination and the evidence, and one result runs against an existing paper.** The claims were checked in the papers themselves.

**1. Lineage**

- **VQ** (weights as codebook entries over small groups). Bringing VQ/PQ to network weights is commonly credited to Gong et al. (2014). The gain–shape split has an LLM precedent in PVQ (2024); AQLM (2024) is closer in format, a per-row scale times a learned codebook over groups of 8 weights.
- **iVQ** (diagonally weighted k-means). Its objective, a diagonal-Hessian-weighted distortion with weighted-mean centroids, has the form of Choi et al.'s Hessian-weighted k-means (ICLR 2017). k-means weighted by input activations for output reconstruction came first in Stock et al. (ICLR 2020). In LLMs SqueezeLLM does the same in one dimension with the Fisher diagonal. "imatrix + codebooks over groups of 8" has a precedent in llama.cpp's i-quants.
- **hVQ** (block-wise error propagation). The assignment has the skeleton of QuIP#'s BlockLDLQ: g columns rounded together by a vector quantizer, earlier columns' error fed back through a block LDL of `H`. Learned codebooks with d-column propagation originate in GPTVQ. "GPTQ written with the factor of `H`" is the equivalence the QuIP line already established.

**2. Element by element**

| Element | Verdict | Nearest prior work |
|---|---|---|
| 8-column block propagation + VQ | existing | QuIP# BlockLDLQ, GPTVQ |
| `H_t = D + t·(off-diagonal)` | existing form, opposite conclusion | DASH-Q; Schäfer and Strimmer's shrinkage |
| Token-normalized statistics | known problem, different remedy | P-GPTQ |
| Per-expert statistics + shrink to the layer mean | known problem, the same remedy not found | MoEQuant |
| 500B-class MoE, 1.6 bpw, one GPU for 2 hours, drop-in | this combination not found | QMoE (scalar), GPTVQ (dense) |

- GPTQ propagation on a very large MoE with one GPU: QMoE (1.6T parameters, 0.8 bpw, under a day), scalar and ternary.
- Per-expert sample imbalance and gate-weighted Hessians: MoEQuant (ICML 2025).
- A 2026 survey records that an expert's `XᵀX` is singular when its routed tokens are fewer than the input dimension, so the rounder runs only on the damped Hessian.
- BOS and the first tokens bias the Hessian of deep down-projections; excluding those positions is MiniCPM4's P-GPTQ. An implementation report found the opposite, that removing BOS made results worse. hVQ's "keep every token, at equal weight" is a third option between the two.

**3. The most meaningful point: the opposite conclusion to DASH-Q**

- The shrinkage family `H(ρ) = D + ρ·O` was defined for quantization by DASH-Q (EuroMLSys '26), whose final choice is ρ = 0, the diagonal alone. The form is Schäfer and Strimmer's correlation shrinkage (2005).
- Its noise analysis is one layer of Llama-2-7B with 8–2,048 samples under scalar quantization. It sweeps ρ only on a Hessian-stability metric and never quantizes at an intermediate ρ.
- hVQ swept t from 0 to 1: the error improved up to t = 0.9 and stayed flat within 0.2% from there to 1 (3.65M tokens, VQ, rare experts shrunk). That off-diagonals pay off even at 1.6 bpw once calibration data is ample and shrinkage is applied is a regime result with no direct counterpart in the prior work found.

**4. What a paper would still need**

- No direct baselines: GPTVQ and QuIP# (BlockLDLQ + E8) on the same model.
- No ablations: token normalization against raw sums and excluded positions; the shrink target (layer mean against plain damping).
- One model, no independent test set, no generation benchmark.
- The fair positioning today: GPTVQ / BlockLDLQ configured for a shared codebook, MoE and an extreme low bit rate, plus a regime result.

The search was limited: "not found" does not mean "does not exist".

Sources: [GPTVQ](https://arxiv.org/abs/2402.15319) · [QuIP#](https://arxiv.org/abs/2402.04396) · [QuIP](https://arxiv.org/abs/2307.13304) · [GPTQ](https://arxiv.org/abs/2210.17323) · [DASH-Q](https://arxiv.org/abs/2604.13806) · [Schäfer and Strimmer 2005](https://strimmerlab.github.io/publications/journals/shrinkcov2005.pdf) · [QMoE](https://arxiv.org/abs/2310.16795) · [MoEQuant](https://arxiv.org/abs/2505.03804) · [MiniCPM4 (P-GPTQ)](https://arxiv.org/abs/2506.07900) · [Choi et al. 2017](https://arxiv.org/abs/1612.01543) · [Stock et al. 2020](https://arxiv.org/abs/1907.05686) · [SqueezeLLM](https://arxiv.org/abs/2306.07629) · [VPTQ](https://arxiv.org/abs/2409.17066) · [AQLM](https://arxiv.org/abs/2401.06118) · [PVQ](https://arxiv.org/abs/2410.16926) · [Gong et al. 2014](https://arxiv.org/abs/1412.6115) · [2026 survey (Jokar)](https://arxiv.org/abs/2608.25188) · [llama.cpp i-quants](https://github.com/ggml-org/llama.cpp/pull/4861) · [BOS report (llm-compressor #3365)](https://github.com/vllm-project/llm-compressor/pull/3365)

## Calibration data

This is the iVQ release's frozen calibration set, rebuilt with the same seed and
pinned sources: 3,095 sequences, 3,653,893 tokens. Both token IDs and sequence
boundaries match its content SHA-256. Chats use the model's own
`encoding/encoding.py`, including thinking mode and V4.1 DSML tool calls.

The mix and the source-by-source counts are unchanged from the
[iVQ calibration table](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-iVQ-GGUF#calibration-data):
web, math, code, instruction following, multilingual chat, reasoning, DSML tools,
SWE, competitive programming and long finance. No domain exceeds 12.4%; 19.6% of
tokens are in sequences of at least 4,096 tokens. The longest sequence has 31,721
tokens. WikiText is held out, and calibration rows sharing non-boilerplate
64-token windows with the held-out set were removed.

The 26-case held-out set is separate from calibration. Unlike iVQ, this release
uses it to choose t, damping and shrink on four layers as well as to report the
final measurements; it is not an independent second test set.

## Files

The file names and layout are the VQ and iVQ releases', so a directory or script set
up for one of them serves this one once its path is changed. Do not mix files between
the directories. The part names are the same, and `SHA256SUMS` catches a mix.

| Path | Contents | Size |
|---|---|---:|
| `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf.part01-of-40` … `part40-of-40` | hVQ base GGUF, including the MTP drafter, as 40 byte-range parts (24 × 2,838,915,986 B, then 16 × 2,838,915,985 B) | 113,556,639,424 B |
| `DeepSeek-V4.1-Flash.vision.gguf` | Vision sidecar for `--vision-gguf` (identical to the VQ release) | 971,066,016 B |
| `deepseek-engram/model-00047-of-00048.safetensors` | 1017 Engram shard (official file, unmodified) | 101,535,150,936 B |
| `deepseek-engram/model-00048-of-00048.safetensors` | 1017 Engram shard (official file, unmodified) | 101,537,926,640 B |
| `SHA256SUMS` | SHA-256 of the assembled base, every part, the vision sidecar and the two Engram shards | |
| [`RECIPE.md`](RECIPE.md) | What changes against iVQ, statistics, the choice of t, commands, pinned inputs | |
| [`EVALUATION.md`](EVALUATION.md) | Full per-layer, per-domain and conversion-cost tables | |
| [`SOURCE.md`](SOURCE.md) | Provenance and licenses | |
| [`tools/`](tools) | The hVQ tools and tests (ds4-dfm-rs `tools/vq_calib/`, branch `feat/ds41-hvq`) | |
| `LICENSE`, `LICENSE-ds4-dfm-rs`, `LICENSE-DeepSeek` | YoungAi tools and format (MIT, with the GGML notice), ds4-dfm-rs calibration tools (MIT), DeepSeek weights (MIT) | |

SHA-256 of the assembled base (all parts are in `SHA256SUMS`):

```
d000e79199acd3e443fa6485e4927a0d5f207fa860d0b48f05bbc3dd514f52df  DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
```

The download is about 318 GB. Plan for about 114 GB more while the parts are
assembled, or use the append-and-delete loop below. If you already have the VQ or iVQ
release, only the 40 parts are new. The vision sidecar and the Engram shards are the
same files.

## Quick start on DGX Spark

```sh
set -e

# 1. Download and verify (parts, vision sidecar, Engram, checksums)
hf download Baekpica/DeepSeek-V4.1-Flash-1017-hVQ-GGUF --local-dir ds41-1017-hvq
cd ds41-1017-hvq
sha256sum -c --ignore-missing SHA256SUMS

# 2. Assemble the base. This loop needs only one extra part of free disk.
B=DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
test ! -e "$B"     # start from the 40 downloaded parts and no assembled file
for i in $(seq -w 1 40); do
    cat "$B.part$i-of-40" >> "$B"
    rm "$B.part$i-of-40"
done
sha256sum -c --ignore-missing SHA256SUMS      # the assembled .gguf must report OK

# 3. Build ds4-dfm-rs and serve
git clone https://github.com/Baekpica/ds4-dfm-rs.git && (cd ds4-dfm-rs && make cuda-spark)
./ds4-dfm-rs/ds4-server --cuda -m "$B" \
    --engram-dir deepseek-engram \
    --vision-gguf DeepSeek-V4.1-Flash.vision.gguf \
    --mem-budget-mb 110000 --host 0.0.0.0 --port 8000
```

If you already have the VQ or iVQ release or the official 1017 checkpoint, add
`--exclude "deepseek-engram/*" --exclude "*.vision.gguf"` to the download. Then
point `--engram-dir` and `--vision-gguf` at the files you have. They are
byte-identical.

The server provides OpenAI-style `/v1/chat/completions`, `/v1/completions`
and `/v1/responses`, and Anthropic-style `/v1/messages`. The embedded drafter
is controlled with `--mtp-mode off|auto|on`. DeepSeek recommends
`temperature` 1.0 and `top_p` 1.0 for 1017. Send them explicitly in each
request, because sampling on this path follows the request. See the
[ds4-dfm-rs DeepSeek V4.1 guide](https://github.com/Baekpica/ds4-dfm-rs/blob/main/docs/deepseek41-flash.md)
for the full flag set and serving limits. Standard llama.cpp, Ollama and
Transformers loaders do not read this format.

## Use the 1017 Engram, not the V4.1 one

The same caveat as the VQ release applies. 1017 changed the two Engram shards
without changing their names or sizes. An engine pointed at the older
`deepseek-ai/DeepSeek-V4.1-Flash` tables loads them without error and runs a
mismatched model. Use the copies in `deepseek-engram/` (SHA-256 `76a05106…` and
`9dc51433…`), or the same files from the 1017 checkpoint. Do not mount YoungAi's
V4.1 domain sidecars (zchain / GR-RB) either. They were solved against other
weights, and any GR sidecar fitted to the VQ or iVQ indices does not fit these.

## Limits

- The same held-out set selected the hyperparameters on four layers and measured the final build. No second independent held-out set was measured.
- The pilot survives as printed session tables, not raw sweep JSONL or precision-probe tensors.
- Real-layer t=0 identity was verified on layer 14; synthetic tests cover both 12- and 13-bit layouts.
- The measurements are teacher-forced: a reference forward on one H200 and the engine
  on one DGX Spark. Sampled generation was checked on 44 seeded requests. No task
  benchmark and no long sampled generation has been run.
- No GPTVQ, QuIP# or AQLM conversion of this model was run for comparison.
- The statistics are the original model's: a quantized layer's input is the original's
  output of the layers before it, not the quantized model's. Feedback therefore
  corrects each layer on its own.
- The MTP draft towers are the VQ release's, unchanged.
- YoungAi reports that this weight recipe under-weights stop decisions (`</think>`,
  EOS), so long thinking segments can fail to close. Its server therefore defaults
  to non-thinking mode. Here all 24 short seeded thinking requests closed and
  stopped; long thinking segments were not tested.
- The custom VQ format needs ds4-dfm-rs or YoungAi's engine.

## Credits and sources

- **[YoungAi](https://github.com/fodelf/YoungAi)** by Wenzhou Wu (吴文周,
  [fodelf](https://github.com/fodelf) /
  [wenzhouwu](https://huggingface.co/wenzhouwu)), MIT. YoungAi supplies the
  `vq8sh14-q4k-mtpnative` recipe and the DQVL v3 payload format, and its tools
  built the VQ base this release patches: the q4_K skeleton, the MTP towers and
  the GGUF layout. Its notice is kept in [`LICENSE`](LICENSE).
- **Prior work** ([lineage](#lineage-and-prior-work)): **GPTQ** (Frantar et al.),
  **QuIP** and **QuIP#** (Chee et al.; Tseng et al.: LDLQ, BlockLDLQ) and **GPTVQ**
  (van Baalen et al.) for error feedback and its use with vector codes; Choi et
  al., Stock et al., **SqueezeLLM** and **VPTQ** for the weighted k-means behind
  iVQ; **AQLM** for per-row scales with learned 8-weight codebooks; **DASH-Q** and
  Schäfer and Strimmer for the shrinkage family; **QMoE** and **MoEQuant** for MoE
  quantization; MiniCPM4's **P-GPTQ** for token bias in the statistics.
- **[llama.cpp / GGML](https://github.com/ggml-org/llama.cpp)**: the importance-matrix
  idea iVQ carried over to vector quantization, the GGUF container and the q4_K layout.
- **[ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs)**: the runtime and the
  calibration tools (`tools/vq_calib/`), MIT ([`LICENSE-ds4-dfm-rs`](LICENSE-ds4-dfm-rs)).
  Its DeepSeek V4.1 support was ported by [JLFN](https://github.com/JLFN)
  ([#80](https://github.com/Baekpica/ds4-dfm-rs/pull/80), vision in
  [#83](https://github.com/Baekpica/ds4-dfm-rs/pull/83)).
- **[antirez/ds4](https://github.com/antirez/ds4)** (DwarfStar) by Salvatore
  Sanfilippo, the DeepSeek V4 engine YoungAi started from.
- **[DeepSeek-AI](https://huggingface.co/deepseek-ai)**: the model, its Engram
  tables and the reference inference code, MIT
  ([`LICENSE-DeepSeek`](LICENSE-DeepSeek)).
- **Calibration text**: NVIDIA Nemotron post-training datasets, FineWeb-Edu,
  FineWeb-2 and FineMath (Hugging Face), github-code-clean (codeparrot),
  Chinese-DeepSeek-R1-Distill-data-110k and COIG-CQIA. Only statistics derived
  from them are in the weights.

## License

MIT. The weights derive from DeepSeek-V4.1-Flash-1017 (MIT,
[`LICENSE-DeepSeek`](LICENSE-DeepSeek)). The VQ format, recipe and base tools are
YoungAi's (MIT, [`LICENSE`](LICENSE), which includes the GGML notice). The
calibration tools are ds4-dfm-rs's (MIT, [`LICENSE-ds4-dfm-rs`](LICENSE-ds4-dfm-rs)).
