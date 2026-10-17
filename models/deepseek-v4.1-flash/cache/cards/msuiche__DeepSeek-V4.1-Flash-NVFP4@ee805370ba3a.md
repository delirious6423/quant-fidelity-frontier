---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
library_name: Model Optimizer
tags:
- nvfp4
- quantized
- deepseek
- modelopt
---

# DeepSeek-V4.1-Flash-NVFP4 (community, weight-only)

NVFP4 quantization of [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
(revision `fb2764a5cf32`). Unofficial community artifact, not affiliated with DeepSeek or NVIDIA.

**510 GB -> 415 GB (19% smaller).** The first NVFP4 quant of this checkpoint.

## What is quantized

| Component | Source | Here | Notes |
|---|---|---|---|
| Routed experts (`*.ffn.experts.*`) | MXFP4 (UE8M0 scales) | **unchanged** | already 4-bit in the source checkpoint; passed through byte-identical |
| Attention projections (`*.attn.wq_a/wq_b/wkv/wo_a/wo_b`, indexer `wq_b`) | FP8 E4M3, UE8M0 32x32 | **NVFP4 W4** | per-16 E4M3 block scales + FP32 global |
| Shared experts (`*.ffn.shared_experts.*`) | FP8 E4M3 | **NVFP4 W4** | |
| Engram hash tables (`layers.1/14.engram.embed`) | FP8 E4M3, per-row 1x32 | **NVFP4 W4** | see risk note below |
| Engram projections (`engram.wkv`), MTP/drafter dense (`mtp.*.attn.*`, `mtp.*.shared_experts.*`, `mtp.0.main_proj`) | FP8 E4M3 | **NVFP4 W4** | drafter kept consistent with main model |
| Token embeddings, lm_head, vision tower, norms, router gates, hyper-connection params | BF16/F32 | **unchanged** | |

Layout per converted tensor mirrors `nvidia/DeepSeek-V4-Pro-0813-NVFP4` exactly:
`X.weight` (U8, two E2M1 per byte, low nibble first), `X.weight_scale` (F8_E4M3, per-16),
`X.weight_scale_2` (F32 scalar). No `input_scale` (weights only; activations stay dynamic).
The `quantization_config` keeps the source's `fp8`/`ue8m0`/`expert_dtype: fp4` fields and adds
`quant_algo: MIXED_PRECISION` plus a `quantized_layers` map naming every NVFP4 module
(the convention the vLLM DeepSeek-V4 NVFP4 path reads).

## Method

- Weight-only PTQ with NVIDIA Model Optimizer 0.46.1 (`NVFP4QTensor.quantize`), streaming
  shard-by-shard converter; no model instance, no activations.
- Global scale per tensor chosen by an MSE sweep over multipliers {0.8, 0.9, 1.0, 1.1, 1.25}
  around the standard `amax/(448*6)` default.
- No activation calibration set was used: there is currently **no runtime that can execute
  this model** (see below), so activation-aware calibration was impossible. Weight-only
  max/MSE calibration needs no data.
- Format verified bit-exact: modelopt's NVFP4 dequant of `nvidia/DeepSeek-V4-Pro-0813-NVFP4`
  expert bytes reproduces the source MXFP4 weights exactly; this repo emits the same layout.

## Measured quantization error (per-tensor, vs the FP8 source weights)

| Tensor class | rel. L2 error |
|---|---|
| MTP/drafter dense | 9.4% (n=25) |
| attention indexer wq_b | 9.4% (n=8) |
| attention projections | 9.4% (n=200) |
| engram hash tables | 9.4% (n=2) |
| engram wkv | 9.5% (n=2) |
| shared experts | 9.4% (n=120) |

The engram tables are the largest component converted (203 GB FP8 -> 111 GB).
Row-level rel. L2 error is 9.4%. An error-propagation simulation through the real
engram gate (sampled table rows + real `wkv`/gate weights, synthetic hidden states) gives output
cosine similarity > 0.9999 and contribution-weighted output error < 1% for hidden-state RMS >= 10
(0.7% at RMS 10, 0.07% at RMS 100). At hidden RMS ~1 the contribution ratio is large and output
error reaches ~5%. End-to-end measurement (below) confirms the simulation: 32K needle-in-a-haystack
retrieval, the most engram-sensitive workload tested, shows no engram-specific damage.

## End-to-end measured (2026-09-16)

`vllm/vllm-openai:deepseekv41-flash-0909` executes `deepseek_v41` natively but has **no
dense-NVFP4 path**: its `deepseek_v4_1/quant_config.py` forces FP8/UE8M0 for all LinearBase
layers (experts can be NVFP4 via `moe_quant_algo`), and it never reads this repo's
`quantized_layers` map. The repo therefore still cannot be served as-is; see Runtime support.

Quality was instead measured through a **dequantized-FP8 twin**: every NVFP4 tensor dequantized
and re-encoded as FP8 E4M3 + UE8M0 block scales (the source repo's exact layout), served on
8xH200 with the stock image. The FP8 re-encode adds ~2.6% rel error on top of the NVFP4 error,
so twin scores are a **pessimistic bound** on true NVFP4 quality. Identical 0-shot lm-eval
harness for all checkpoints (MMLU 10/subject; zero-shot n=200; NIAH 32K n=10):

| task | FP8 source | this repo (twin) |
|---|---|---|
| MMLU (acc) | 85.1 | 81.8 |
| ARC-c (acc_norm) | 59.5 | 61.5 |
| ARC-e (acc_norm) | 83.5 | 83.5 |
| HellaSwag (acc_norm) | 71.5 | 68.5 |
| BoolQ (acc) | 87.5 | 87.0 |
| PIQA (acc_norm) | 82.5 | 81.5 |
| WinoGrande (acc) | 68.0 | 69.5 |
| NIAH 32K (EM) | 0.600 | 0.700 |

Weight-only NVFP4 costs ~3.3pt MMLU (true value smaller — twin is a lower bound), zero-shot is
a wash, and needle retrieval is intact. The MMLU loss lives in the dense attention/indexer
tensors, not the engram tables. A v2 that leaves attention at FP8 (~+7 GB) should recover most
of it; activation-aware PTQ for the dense tensors is the follow-up, now that a runtime exists
to collect activations.

**If the engram tables turn out to be quality-critical**, you can reconstruct an FP8-engram
variant: download `model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors`
from the [source repo](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/fb2764a5cf32),
swap them in, and remove the `layers.1.engram`/`layers.14.engram` entries from
`quantization_config.quantized_layers`.

## Runtime support (updated 2026-09-16) — read before downloading

- **vLLM** `vllm/vllm-openai:deepseekv41-flash-0909` serves `deepseek_v41` (including the FP8
  source checkpoint) but **cannot load this repo**: dense-NVFP4 is unsupported for this model
  type — LinearBase layers are hard-wired to FP8/UE8M0, only MoE experts have an NVFP4 branch
  (`moe_quant_algo: NVFP4`, the convention NVIDIA's V4-Pro port uses). The per-layer
  `quantized_layers` map in this repo's config is not consumed by that loader. Serving this
  repo as-is needs a loader patch (route `quantized_layers` entries to the ModelOpt NVFP4
  linear method); the engram tables additionally need an NVFP4 embedding path.
- **transformers**: `deepseek_v41` is not in 5.17.0 (latest PyPI) nor in `main`. The source
  checkpoint ships DeepSeek's native `inference/` runtime instead, which reads the *source*
  FP8/MXFP4 format, not this repo's NVFP4 dense tensors.
- **SGLang**: no `deepseek_v41` model file.

End-to-end quality is nevertheless measured — see the section above (dequant-FP8 twin on
stock vLLM, 8xH200).

## Reproduction

Converter + validation harness: see the model card discussion / commit history. Method:
`nvfp4_shard.py` (streaming converter), `local_validate.py` (range-fetch error harness),
ModelOpt 0.46.1. Source sha `fb2764a5cf32`. Conversion took ~47 min on a 16-core CPU container (repo upload ~37 min more); no GPU used.

## License

Inherited from the source model (mit).
