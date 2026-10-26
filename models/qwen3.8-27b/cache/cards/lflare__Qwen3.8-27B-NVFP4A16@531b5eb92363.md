---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - qwen
  - qwen3_5
  - multimodal
  - image-text-to-text
  - mtp
  - speculative-decoding
  - compressed-tensors
  - llm-compressor
  - nvfp4
  - w4a16
---

# Qwen3.8-27B-NVFP4A16

**NVFP4A16 W4A16, attnbf16** — 4-bit FP4 weights with FP8 E4M3 per-group scales and **bf16 activations**, applied to the **MLPs only**. The entire `self_attn` path, the `linear_attn` (SSM) block, the vision tower, `lm_head` and the MTP head all stay in bf16.

A quantisation of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) — a dense, multimodal
Qwen3.5-family model (`qwen3_5`) with hybrid linear + full attention, a vision tower,
and a multi-token-prediction head.

| | |
|---|---|
| Scheme | NVFP4A16 (W4A16) |
| Size | **28.8 GiB** (6 shards) vs 51.7 GiB bf16 |
| Quantised modules | **192** (the 64x3 MLP projections) |
| Kept in bf16 | `self_attn`, `linear_attn`, `model.visual`, `lm_head`, `mtp` |
| Recipe | [`recipe.yaml`](recipe.yaml) (`nvfp4a16_attnbf16.yaml`) |

## Why only the MLPs

`Qwen3.8-27B` sets `attn_output_gate: true`, which fuses the attention **output gate**
into `q_proj` — making it `[12288, 5120]` instead of `[6144, 5120]`. Half of that tensor
is a multiplicative per-head gate on what attention writes into the residual stream, and
quantisation error on a multiplicative gate behaves worse than on an additive projection.

Only 16 of the 64 layers are `full_attention` (the rest are `linear_attn`), so the whole
`self_attn` block is a small share of the parameters — holding it in bf16 costs little
while removing the riskiest quantisation target. The MLPs supply nearly all of the
compression.

The `linear_attn` (Mamba/SSM) block is kept in bf16 for the same reason it is in every
build in this line: quantising the recurrent projections measurably degrades this
architecture.

## Measurements

KLD is per-token, 8 samples from `neuralmagic/calibration` (LLM split) at `max_seq=1024`.
PPL is `wikitext-2-raw` test, non-overlapping 2048-token chunks. Baseline for both is
**bf16 `Qwen/Qwen3.8-27B` itself**, so what is measured is quantisation loss and nothing else.

**Two KLD columns, because one number cannot be fair to both checkpoints.** The official
release is DeepSeek-format FP8, which has to be dequantised to plain bf16 before it will
run at all on Blackwell — and a dequantised checkpoint runs with **bf16 activations**. Our
checkpoint, loaded normally, still applies its scheme's **dynamic FP8 activation quant**.
Comparing those two directly measures W8A8 against W8A16 and flatters whichever model was
dequantised. So both are reported both ways.

| Model | Modules quantised | KLD (weight-only) | KLD (as deployed) | PPL | ΔPPL |
|---|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` (bf16) | — | 0 | 0 | 6.9416 | — |
| [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (official) | 407 | 0.0523 | 0.1001 | 6.9539 | +0.18 % |
| **This model** | 192 | **0.1267** | **0.1267** | **7.1497** | **+3.00 %** |

*Weight-only* = weights round-tripped through the quantised format, activations bf16.
*As deployed* = the same weights plus dynamic per-token FP8 activation quantisation on
exactly the modules each checkpoint actually converts. This build is **A16** — it does not quantise activations at all, so its two columns are the same measurement.

**This build has lower divergence from the bf16 parent than the official release on both
measures** — 142% higher weight-only and 27% higher as deployed — while quantising
192 modules to the official's 407.

On PPL the two are not meaningfully separable; both sit within noise of the bf16 parent.
A negative ΔPPL does **not** mean this build beats its own parent, only that the loss is
below measurement noise.

### What does *not* explain the difference

The intuitive story is that scale layout matters — the official uses 128x128 **block**
scales, `llm-compressor` emits one scale per **output channel**. We tested that directly,
simulating the FP8 E4M3 round-trip on real weight tensors from this model
(raw numbers: [`block_scale_experiment.json`](block_scale_experiment.json)):

| scheme | #scales | rel. Frobenius err | max abs err |
|---|---|---|---|
| per-tensor | 1 | 0.02648 | 0.02065 |
| per-output-channel (this build) | 5120 | 0.02643 | 0.02065 |
| block 128x128 (official) | 5440 | 0.02643 | **0.00426** |
| *control:* INT8 per-tensor | 1 | 0.0750 | — |
| *control:* INT8 per-output-channel | 17408 | **0.0101** | — |
| *control:* INT8 block 128x128 | 5440 | 0.0144 | — |

All FP8 variants land within 0.1 % of each other — **including per-tensor, with a single
scale for the entire matrix**. The reason is that FP8 is a *floating-point* format: every
element carries its own exponent, so the scale only has to bring values into range rather
than resolve dynamic range the way a fixed-point scale must. The INT8 control shows the
method is sensitive enough to detect layout effects where they exist (a 7.4x spread) — and
that in *that* regime per-output-channel actually beats 128x128 block, having more scales.

Block scaling does buy a ~5x lower **worst-case** weight error; that is an outlier-tail
property, not an average one.

We also checked whether the vendor used a smarter rounding than round-to-nearest, by
comparing their stored weights against an RTN round-trip of the same parent tensors
(raw numbers: [`rounding_experiment.json`](rounding_experiment.json)):
their reconstruction error matches RTN to within 0.3 %. They used plain round-to-nearest.

So the gap is not scale layout and not rounding — it is **which modules get quantised**.
This build leaves the attention path, the SSM block and the MTP head in bf16; the official
quantises all of them.

So the reasons to prefer this build are **architectural, not average-case accuracy**:

* the SSM and attention paths are untouched bf16, which is what you want if you care about
  long-context or multi-turn behaviour and are unwilling to accept quantised recurrent
  projections;
* it is `compressed-tensors` format, so it **loads in `transformers` on Blackwell (SM120)**.
  The official checkpoint is DeepSeek-format FP8 (`quant_method: fp8` + `weight_scale_inv`,
  block-128), which does not — `transformers`' `finegrained_fp8` integration has no SM120
  recipe, and it has to be hand-dequantised even to be evaluated. (That dequantisation is
  how the official row above was measured, on this same harness, so the comparison is
  apples-to-apples.)

Both FP8 rows measure **weight** precision loss only; neither models dynamic activation
quantisation, so the caveat applies equally to both and does not favour either.

## MTP / speculative decoding

All **15 `mtp.*` tensors are present** in bf16, re-grafted from the bf16 parent after
quantisation (llm-compressor does not save modules `transformers` never instantiates).

The 8 MTP Linear modules are listed in `quantization_config.ignore`. This matters:
without it, vLLM's compressed-tensors loader treats the bf16 head as a quantised target,
finds no scales, mis-loads it, and **every draft is rejected — 0 % acceptance at every
position** while the head still appears to load correctly in the logs.

Measured on vLLM with `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'`
over greedy decoding of mixed chat/code/reasoning prompts:

| | draft tokens | accepted | acceptance |
|---|---|---|---|
| This model | 576 | 484 | **84.0%** |

## Smoke tests

Text generation, vision, and MTP key presence all pass:

* MTP tensors present: `True`
* Text: `True`
* Vision: `True` — the tower correctly described the test image
  (The user wants a single short sentence describing the image. I need to identify the key subject and setting. T...)

## Inference

### vLLM (with MTP speculative decoding)

```bash
vllm serve lflare/Qwen3.8-27B-NVFP4A16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --quantization compressed-tensors \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
```

`--max-num-seqs 32` is not optional on this architecture: the default (1024) exceeds the
available Mamba cache blocks and vLLM aborts CUDA-graph capture. FP4 weights are unpacked at compute time. Native FP4 GEMM requires Blackwell (SM100+); on older GPUs vLLM dequantises at runtime.

### transformers

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

repo = "lflare/Qwen3.8-27B-NVFP4A16"
proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    repo, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
```

## Provenance

Built with `llm-compressor` 0.13.0 + `compressed-tensors` 0.18.0 on `transformers` 5.14.1,
torch 2.13.0+cu130, on an RTX PRO 6000 Blackwell (SM120). Plain quantisation — no
abliteration, no fine-tuning.

The recipe and every measurement JSON ship in this repo, alongside the weights.
