---
language:
- en
license: apache-2.0
library_name: mlx
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
tags:
- mlx
- mxfp4
- gptq
- qwen3.5
- qwen3.8
---

# Qwen3.8 27B calibrated GPTQ MXFP4 for MLX

This is a native-format MXFP4 conversion of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). It is not a
round-to-nearest repack: activation-calibrated sequential GPTQ solved every
eligible linear tensor with native E8M0 block-scale search.

The intended runtime is MLX-LM on Apple Silicon or the MLX CUDA backend. The
checkpoint uses ordinary MLX quantized tensors and does not require a custom
model loader.

## Quantization

| Property | Value |
| --- | --- |
| Weight format | MXFP4 E2M1 |
| Scale format | E8M0 |
| Block size | 32 weights |
| Solver | Sequential GPTQ with native-scale search |
| Quantized tensors | 496 linear tensors |
| Layer coverage | 64/64 transformer layers |
| Calibration windows | 256 × 2,048 tokens |
| Damping | 0.01 |
| Clipping | MSE |
| Seed | 17 |

Sensitive tensors remain BF16 where required by the runtime layout. In
particular, this is the ordinary target checkpoint: its native Qwen3.5-family
MTP tensors are not packaged here.

## Use

```bash
mlx_lm.generate \
  --model RESMP-DEV/Qwen3.8-27B-MLX-GPTQ-MXFP4 \
  --prompt "Explain why calibrated quantization can outperform naive RTN."
```

The Apple Silicon smoke used the current MLX/MLX-LM stack. A separate CUDA 13
smoke used MLX 0.32.0 and MLX-LM 0.31.3 on one RTX 3090 Ti:

- generation completed successfully;
- peak device memory was 18.685 GB;
- a two-question reasoning prefill peaked at 20.863 GB;
- the model fit one 24 GB card without tensor parallelism.

For OpenAI-compatible serving, keep concurrency bounded on 24 GB GPUs:

```bash
mlx_lm.server \
  --model RESMP-DEV/Qwen3.8-27B-MLX-GPTQ-MXFP4 \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --decode-concurrency 1 --prompt-concurrency 1
```

When the server returns both fields, score `message.content` separately from
`message.reasoning`. Concatenating them changes exact-match and code-evaluation
semantics.

## Provenance and integrity

- Base model: `Qwen/Qwen3.8-27B`
- Base revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Calibration corpus: `RESMP-DEV/ptq-calibration-corpus`
- Corpus revision: `ffa2e74b9361d573167373140f10dfd30c69a327`
- Calibration token SHA-256:
  `d7dd281b1f1756a6492361e1dfb0c65fed259f8a3d65eabbcb51809b50bd35d7`
- Calibration engine commit:
  `7131c407af8b8991fb5a8148a49cf2f323c7afd3`
- Calibration manifest SHA-256:
  `430087dedf944b584fed8424a899b9c5666bba6b13838ee84bbfdde9c513b258`

All 496 calibrated tensors were injected during packing; there were zero RTN
fallbacks. Per-file SHA-256 values are recorded in `GPTQ_RECEIPT.md`.

## Validation and limitations

Validation completed for source revision, calibration-manifest matching, tensor
packing, MLX loading, greedy generation, CUDA loading, and generation smoke.
The calibration reconstruction has median relative error `0.1165285` and range
`0.11245`–`0.146238`.

## Local DFlash2 diagnostic

A three-repeat, 256-token diagnostic on one Apple M4 Max with 128 GB unified
memory compared this checkpoint with `z-lab/Qwen3.8-27B-DFlash2` and the
community 4-bit control. The runtime was `dflash-mlx` commit `6080323`, MLX
0.32.4, and MLX-LM 0.32.2. DFlash2 used explicit affine W4 draft weights so
local target paths could not silently fall back to BF16.

| Target | Baseline tok/s | DFlash2 tok/s | Speedup | Acceptance | DFlash peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibrated MXFP4 | 26.385 | 53.156 | 2.029× | 72.266% | 19.688 GB |
| Community 4-bit | 28.048 | 69.410 | 2.475× | 72.266% | 16.775 GB |

With DFlash2 left in BF16, the calibrated MXFP4 target reached 49.580 tok/s
and 22.293 GB peak memory. Explicit W4 therefore improved this diagnostic
while reducing peak memory by about 2.61 GB.

For each target, all 256 generated token IDs matched that target's stock-MLX
baseline. Against the pinned BF16 source on the same prompt, the calibrated
MXFP4 checkpoint agreed on 44.141% of 256 generated IDs and first diverged at
position 97; the community control agreed on 1.953% and first diverged at
position 3. This is one diagnostic prompt, not a general quality benchmark.

No downstream quality claim is made from calibration error. A matched held-out
comparison against the pinned BF16 source and an RTN control remains pending.
Users should run their own task-level evaluations before deployment.
