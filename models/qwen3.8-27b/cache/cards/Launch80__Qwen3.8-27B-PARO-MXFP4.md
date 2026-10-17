---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
- paroquant
- mxfp4
- w4a8
- rocm
- rdna4
- vllm
- quantized
---

# Qwen3.8-27B-PARO-MXFP4

Qwen3.8-27B with **ParoQuant rotations on OCP MXFP4 weights**, built for AMD RDNA4 (Radeon AI PRO
R9700, gfx1201) and served through a W4A8 fp8-WMMA path in a patched vLLM.

- **Weights:** e2m1 (4-bit) elements with one e8m0 shared scale per 32 along K, i.e. OCP MXFP4,
  4.25 bits/weight. Stored per projection as `weight [N, K/2]` uint8 (two e2m1 per byte, even
  index in the low nibble) and `weight_scale [N, K/32]` uint8, the same packing AMD's Quark MXFP4
  releases use.
- **Rotations:** the learned pairwise Givens rotations (`pairs`, `theta`, 8 layers of 64 disjoint
  pairs per 128-channel group) and pre-inverted `channel_scales` from
  [z-lab/Qwen3.8-27B-PARO](https://huggingface.co/z-lab/Qwen3.8-27B-PARO) (ParoQuant, ICLR'26).
  Rotations trained from scratch on a consumer-RAM calibration budget come out as identity past
  layer ~10, so this checkpoint reuses z-lab's trained ones unchanged.
- **How it was made:** bf16 base weights x z-lab's channel scales, rotated with z-lab's rotations,
  quantized one-shot to MXFP4 with the OCP shared exponent, then a stage-2 fine-tune of the weights
  and per-block exponent bias under the MXFP4 grid (rotations frozen; 256 samples x 2 epochs).
  Inference identity: `y = ((x * channel_scales) R^T) dequant(Q)^T`.
- **Config:** `quantization_config.quant_method = "paroquant_mxfp4"`, `format = "mxfp4"`,
  `group_size = 128` (rotation group), `krot = 8`. The `lm_head` and embeddings are bf16, as in
  the source checkpoints. Tokenizer, chat template and generation config are Qwen's.

## Loading

This is **not** loadable by stock transformers or stock vLLM: the `paroquant_mxfp4` quant method
is a vLLM quantization plugin that fuses the rotation with a per-token fp8 activation quant and
runs the MXFP4 x fp8 WMMA GEMM. It ships in
[radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4) (`paroquant/`,
`PAROQUANT.md`), which builds the kernels in-container on ROCm for gfx1201:

```bash
./setup-paroquant.sh
MODEL_DIR=Qwen3.8-27B-PARO-MXFP4 MODE=prod SPEC=7 ./paroquant/run_paroquant.sh
```

Other hardware would need the rotation prologue and GEMM ported; the format itself is plain OCP
MXFP4 plus the rotation tensors, so a dequantize-and-rotate reference is a few lines
(`paroquant/test_mxfp4_loader.py` has one).

## Results (2 x R9700, TP=2, fp8 KV, DFlash2-FP8 drafter, SPEC=7)

| | int4 ParoQuant (z-lab) | **this checkpoint** | AMD MXFP4 (Quark) |
|---|---|---|---|
| GSM8K 500q, greedy, served path | 97.4-98.0% | **97.4-97.6%** | 97.8% |
| decode step @ctx 25 / 8k / 32k | 24.19 / 25.74 / 26.70 ms | **24.53 / 25.89 / 26.80 ms** | 28.6 ms @25 |
| prefill sweep, 2k / 8k / 16k / 32k / 64k (BetterBench PP t/s) | 3782 / 3700 / 3725 / 3621 / 3450 | **4770 / 4827 / 4649 / 4495 / 4273** | -- |
| BetterBench combined decode, single pass | 226 t/s | 203 t/s | 186 t/s |
| in-serve kernel check vs exact dequant (real inputs) | -- | rel 0.0012-0.0021 | -- |

The MXFP4 grid removes both per-group FMAs from the int4 GEMM's inner loop (+25% GEMM throughput
at prefill); decode traffic is identical to int4 g128 (4.25 bits/weight both). Perplexity was not
the gate; task accuracy and the served-path kernel check were.

## Credits

- Qwen team for Qwen3.8-27B (Apache-2.0).
- z-lab for ParoQuant and the trained rotations
  ([paper](https://arxiv.org/abs/2511.10645), [checkpoint](https://huggingface.co/z-lab/Qwen3.8-27B-PARO)).
- AMD for the OCP MXFP4 packing conventions this follows.
