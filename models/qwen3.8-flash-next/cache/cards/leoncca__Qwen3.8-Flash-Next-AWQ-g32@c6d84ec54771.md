---
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- qwen4-exp
- awq
- w4a16
- e4m3
- revolta
---

# Qwen3.8 Flash Next AWQ g32 + QSA FP8 E4M3 KV

**2026-10-17 update: 137.07 GB (127.66 GiB) across 85 model shards.**

AWQ-format derivative of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
All routed-expert projections have been requantized from the same official BF16
mother with damped GPTQ, using real captured activations. Layer smoothing s1 and
per-expert s2 were searched again on calibration data. Non-routed compensation
carriers were recomputed in FP32 and stored in FP16. PLE uses the fixed official
FP8 weights. Existing QSA FP8 E4M3 K/V scales are preserved; they were not
recalibrated for these new weights.

On a fixed 66-record text subset, mean top-20-plus-tail KL to the BF16 reference
decreases from 0.0305 to 0.0207 (about 32%), and top-1 agreement rises from
96.32% to 97.06%. These are bounded fidelity checks, with the approximation and
runtime limitations below; they are not general capability benchmark scores.

## Recommended runtime on V100

On NVIDIA V100 (SM70) we recommend [ReVolta](https://github.com/Leonccaa/ReVolta), our open-source
vLLM-based engine for 4 × V100 32 GB. It contains the SM70 support this
checkpoint needs (AWQ g32 MoE kernels, QSA with calibrated FP8 E4M3 KV, the FP8
MTP draft and multimodal input). We serve the Uncensored variant of this
checkpoint with it, and a pre-release build of it ran the validation below.
Its README has the launch command and the supported settings.
On Ampere or newer GPUs, see **Stock vLLM** under Loading requirements.

## Checkpoint

- Mother revision: `f5d08274bafd880402bd16f5e3e6c514136ec06c`
- Official FP8 PLE revision: `bcd9f01ddc9cff2316eb84281bebcd5b058bddce`
- Format: standard asymmetric AWQ W4A16, group size 32, integer zero point, GEMM packing
- Quantized: `gate_proj`, `up_proj`, `down_proj` of all 512 routed experts in all 48 layers (24,576 experts)
- Optimizer: damped GPTQ, gate/up damping 0.3 and down damping 1.0
- Expert scales: **FP16**, changed from the previous release's BF16 scales
- s1: searched again per layer; 384 non-routed compensation carriers recomputed in FP32 and stored FP16
- Indexed tensors: 222,771, including 24 target FP32 scalar K/V scales and 31 default MTP tensors
- Shards: 48 expert layer files, three retained base files, 33 PLE files and one K/V-scale file
- Merged index SHA-256: `3e1468b43ca1cffb7bd942b495768cf5ab3967534775bd44b8b910ab48765381`
- Target K/V-scale shard SHA-256: `c89b08bacde7d869342190b2f95046a83af37080241ae00ccae3a3d6888852fd`

The compensation carriers are router, shared expert (gate/up/down),
shared-expert gate, HC norm, and both HC down-projection carriers. HC norm uses
`(1+w)/s1-1`; shared down does not receive input smoothing and is converted only
for storage. Other non-routed tensors are retained from the source checkpoint;
no new PLE conversion is performed. No default MTP K/V scale entries are added
to the root index. The optional `mtp-fp8/` directory is separate and unchanged.

Identical index hashes can describe the same tensor names and shard layout in
different checkpoints; they do **not** prove identical weights. Use `SHA256SUMS`
for complete file identity. Public provenance and results are in
`MODEL_PROVENANCE.json`, `VALIDATION.json`, `EXPERT_COVERAGE.json` and
`PUBLIC_RELEASE_AUDIT.json`.

## Why AWQ g32

W4A16 provides a weight-only INT4 path with 16-bit activation/compute precision.
The native g32 layout matches width-640 routed experts under TP4:
`640 / 4 = 160`, which is divisible by 32 but not 128.

AWQ describes the serialized layout here; the new expert weights were optimized
with damped GPTQ. They do not require a GPTQ checkpoint loader. Bits, group size,
asymmetric zero and GEMM packing are unchanged. Loaders must accept FP16 expert
scales and compensation carriers, and preserve other mixed source dtypes.

## Calibration

Weight calibration uses real activations captured from the previous release
served in vLLM, on the `public80-v1mix20` calibration corpus, across all 48
layers (169,614 calibration rows per layer). Layer s1 and per-expert s2 are
searched again using calibration activation second moments. Previous s1 is
recovered only to map captured inputs back to the mother-model space; it is
not reused as the new smoothing. No heldout data selects these scales or damping.

The Hessian and damped GPTQ use the captured previous-checkpoint top-10 routing.
Scales are rounded to FP16 before code selection, integer zero points are
stored, and codes follow `clip(round(w / scale + zero), 0, 15)` on the current
GPTQ residual with feedback. Standard AWQ GEMM packing is used for all experts.

All experts are quantized, with no routing bypass. Natural sample counts are
published by layer and expert in `EXPERT_COVERAGE.json`. Only the seven
zero-natural-coverage layer/expert pairs use 2,048 equally spaced rows from
their layer as a proxy. Every expert with natural coverage uses its natural
samples plus damping; a low positive count does not trigger proxy replacement.
Captured route counts do not describe the new checkpoint's end-to-end routing.
The calibration fit/select self-check conditions on scales selected from full
calibration data and is not an independent heldout evaluation.

The preserved target K/V scales were previously calibrated from 4,130,597
processed prompt tokens. The 24 finite, positive FP32 scalars cover K/V at all
12 QSA layers. They are KV-storage scales, separate from AWQ weight scales and
PLE scales. Reusing them does not mean new scale calibration.

## Loading requirements

The loader must support:

- Qwen4Exp per-expert asymmetric AWQ W4A16 g32 in GEMM layout, including FP16 expert scales and compensation carriers;
- the official FP8 PLE tensors and the complete merged tensor index;
- the 24 target QSA K/V scale entries through `model.safetensors.index.json`;
- full Qwen4Exp multimodal handling, including MRoPE and video-token-pruning initialization, for images and videos.

Tested on four 32 GB NVIDIA V100 GPUs (TP4, FP16 compute, FP8 E4M3 KV storage)
with a pre-release build of [ReVolta](https://github.com/Leonccaa/ReVolta). Generic AWQ support alone is
insufficient for this architecture. No stock-runtime compatibility or default
launch configuration is implied. Image/video regression has not been run on
this new official checkpoint.

**Stock vLLM.** Upstream vLLM (checked in v0.31.0 and v0.32.0) runs the Qwen4Exp QSA
layers only in BF16, which needs compute capability 8.0 or newer. `config.json`
therefore declares `bfloat16`, so the default `--dtype auto` selects BF16 on Ampere
or newer GPUs and no extra flag is needed. A user has reported that the Uncensored
variant (same architecture and format) runs on stock vLLM v0.31.0 with BF16 compute;
we have not run either on stock vLLM ourselves. If you see
`NotImplementedError: Qwen4Exp QSA currently requires BF16`, the run is using FP16,
for example from an older copy of `config.json` or an explicit `--dtype float16`.
Stock vLLM cannot run this model on pre-Ampere GPUs such as V100 or T4; there, vLLM
falls back to FP16 automatically because the GPU has no BF16. In BF16 mode the FP16
expert scales are rounded to BF16 on load (at most 0.4% relative per scale); the GPU
fidelity figures below were measured with FP16 compute, not BF16.

The bundled config keeps native 262,144-token context and default RoPE. The
comparison uses matched positional settings; it does not change the bundled
defaults or establish extended-context capability.

## Validation

### BF16 fidelity comparison — 2026-10-17

The same stratified 66-record subset provides 39,101 scored tokens. BF16
reference distributions come from a transformers CPU full-vocabulary forward
over the original official text model, expanding BF16 weights exactly to FP32
for the main computation. The quantized arms were served in the configuration
listed under Loading requirements. The comparison therefore includes runtime
arithmetic and KV effects as well as weight quantization.

KL is `KL(quantized arm || BF16)`, in nats per scored token. Means are weighted
by valid token count. Top-1 agreement compares captured rank 1 with the BF16
FP32 argmax before logprob storage rounding.

| Subset | Records / tokens | Previous KL approx. | New KL approx. | Previous top-1 | New top-1 |
|---|---:|---:|---:|---:|---:|
| All | 66 / 39,101 | 0.0305 | 0.0207 | 96.32% | 97.06% |
| Instructions (English / Chinese / code) | 40 / 9,383 | 0.0427 | 0.0278 | 93.78% | 95.56% |
| Dialogue incl. math | 50 / 12,752 | 0.0372 | 0.0236 | 94.71% | 96.17% |
| Documents / code | 16 / 26,349 | 0.0273 | 0.0194 | 97.11% | 97.49% |

Instructions are English, Chinese and code instructions. Dialogue also includes
math reasoning; Instructions is a subset of Dialogue. All = Dialogue + Documents.
All 15 reported groups (11 leaf categories and four overlapping summaries)
have lower coarse-grained KL and higher top-1 agreement. The overall
coarse-grained KL decrease is 32.04%.

**Approximation limit:** only the quantized arm's top 20 probabilities and total
tail mass are captured. Merging the tail into one bin gives an exact
coarse-grained KL and a lower bound before numerical uncertainty. It is also
a point approximation to full KL if the unknown arm tail follows the BF16
conditional tail distribution. Different arms can have different top-20 bins.
Tail differences are not measured, so a lower coarse-grained value does not
prove a lower full-vocabulary KL. Accounting for FP16 reference storage rounding
and an explicit serving-logprob tolerance of ±1e-6, conservative full forward-KL
bounds for All are [0.030235, 0.327424]
previous and [0.020465, 0.317644] new. These are numerical/unknown-tail bounds,
not statistical confidence intervals. Exact group results are in `VALIDATION.json`.

This is a fixed stratified subset, not the full corpus or a broad benchmark.
The reference is text-only; these KL results do not evaluate vision or MTP.
CPU FP32 reference arithmetic differs from GPU FP16 serving; it cannot isolate
quantization error from kernel, tensor-parallel reduction or KV effects.

### Serving check — 2026-10-17

This official checkpoint loaded and started on the V100 configuration above
without error, and served all 66 KL collection requests without error. It has
**not** undergone a separate full serving regression covering concurrency,
image/video inputs, long-context needle retrieval or optional FP8 MTP integration.
A full serving regression was performed on the Uncensored variant using the
same quantization approach; those results do not establish acceptance of this
official checkpoint.

Previously published GSM8K, HumanEval, MBPP and IFEval results describe the
**previous weights**. They have not been rerun on this release and are not
claimed for it. The historical card remains available at repository revision
`16e5ccf6c2a7395e12428748bff0e19e51798ca1`.

## Notes and limitations

- The same-mother BF16 comparison is bounded to the text subset and approximation above; no universal BF16-to-AWQ capability-loss figure is claimed.
- The reported loading and KL collection checks do not certify every production, agent, long-output or multimodal workload.
- Preserve the bundled K/V scales; they are reused historical calibration, not newly fitted scales for these weights.
- This derivative uses the included Qwen Community License 1.0.

## Integrity

Run `sha256sum -c SHA256SUMS` after downloading. The checksum list covers every
file in this root release package except itself: all indexed shards,
tokenizer/processor assets, config, license, this card and public metadata.
It excludes the separately preserved `mtp-fp8/` directory, which has its own
checksum manifest.

## Optional FP8 MTP

The existing optional draft in [`mtp-fp8/`](mtp-fp8/README.md) is **unchanged**
by this release. It stores only MTP routed experts in FP8 E4M3 with explicit
block scales; other draft parameters retain source precision. Select that
directory explicitly as the speculative draft with `modelopt_mixed` in a
compatible Qwen4Exp MTP runtime. It retains its own K/V scales and documentation.
A full unfiltered repository download also includes its 4.88 GiB tensor payload.

This draft has not been separately regression-tested with the new official
target. The BF16 distribution comparison covers the target model only, not the
draft. The draft's earlier bounded checks remain documented in its directory.
