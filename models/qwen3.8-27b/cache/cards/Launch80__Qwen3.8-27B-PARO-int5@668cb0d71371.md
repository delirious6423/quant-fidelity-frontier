---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
- paroquant
- int5
- w5a8
- rocm
- rdna4
- vllm
- quantized
---

# Qwen3.8-27B-PARO-int5

Qwen3.8-27B with **ParoQuant rotations on uniform asymmetric int5 weights**, built for AMD RDNA4
(Radeon AI PRO R9700, gfx1201) and served through a W5A8 fp8-WMMA path in a patched vLLM. It is the
fidelity-first sibling of [Qwen3.8-27B-PARO-MXFP4](https://huggingface.co/Launch80/Qwen3.8-27B-PARO-MXFP4):
**4.2x lower KL divergence** for +2.6 ms/step and 3 GB more on disk.

- **Weights:** uniform asymmetric int5, group size 128 along K, fp16 group scales and zero points
  (5.25 bits/weight effective). Stored as `"int5-bitplane"`: `qweight`/`qzeros` hold the low four
  bits in the int4 AWQ packing, and `qweight_hi`/`qzeros_hi` are `[K, N/32]` int32 fifth-bit planes.
- **Rotations:** the learned pairwise Givens rotations (`pairs`, `theta`, 8 layers of 64 disjoint
  pairs per 128-channel group) and pre-inverted `channel_scales` from
  [z-lab/Qwen3.8-27B-PARO](https://huggingface.co/z-lab/Qwen3.8-27B-PARO) (ParoQuant, ICLR'26),
  reused unchanged — rotations trained from scratch on a consumer-RAM calibration budget come out
  as identity past layer ~10.
- **How it was made:** bf16 base weights x z-lab's channel scales, rotated with z-lab's rotations,
  quantized one-shot to int5 g128 RTN, then a stage-2 fine-tune of the weights and group scales
  under the int5 grid (rotations frozen). Inference identity:
  `y = ((x * channel_scales) R^T) dequant(Q)^T`.
- **Config:** `quantization_config.quant_method = "paroquant"`, `bits = 5`, `format = "int5-bitplane"`,
  `group_size = 128`, `krot = 8`. `lm_head` and embeddings are bf16, as in the source checkpoints.
  Tokenizer, chat template and generation config are Qwen's.

## Why five bits

The int4 kernel feeds the fp8 WMMA the signed code `c - 8`, which is exact in e4m3. With 5-bit codes,
`c - 16` spans -16..15 and **every integer in that range is also exact in e4m3** — so the GEMM algebra,
the zero-point fold, the per-token activation quant, the rotation-stream producers and the split-K /
A-tiled bands all carry over unchanged. Only weight staging differs: low nibbles keep the existing word
layout and the fifth bit rides in a byte-per-(slot, lane) plane in the same fragment order. int6 does
**not** have this property (codes to ±32 are not exact in e4m3) and would need an int8-WMMA rewrite.

At decode the 5-bit kernel costs 1.20-1.26x the 4-bit kernel at M<=8 — exactly the 1.235x byte ratio,
so it is purely bandwidth-bound with no unpack penalty.

## Loading

This is **not** loadable by stock transformers or stock vLLM: `paroquant` is a vLLM quantization plugin
that fuses the rotation with the activation quant and runs the int5 x fp8 WMMA GEMM. It ships in
[radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4) (`paroquant/`, `PAROQUANT.md`),
which builds the kernels in-container on ROCm for gfx1201:

```bash
./setup-paroquant.sh
MODEL_DIR=Qwen3.8-27B-PARO-int5 MODE=prod SPEC=7 \
  RADIANCE_PQ_I8=1 RADIANCE_PQ_PG=1 RADIANCE_PQ_ZPE=1 \
  ./paroquant/run_paroquant.sh
```

Those three flags select the configuration all the numbers below were measured with: int8 per-group
activations (`I8` + `PG`) and the zero-point epilogue (`ZPE`). Other hardware would need the rotation
prologue and GEMM ported; the format itself is plain int5 plus the rotation tensors, so a
dequantize-and-rotate reference is a few lines.

## Results (2 x R9700, TP=2, fp8 KV, DFlash2-FP8 drafter, SPEC=7)

KL is `KL(bf16 || candidate)` over wikitext, 96 x 500-char chunks, top-256, with the bf16 reference
**served on the same stack** — a reference collected on a different image charges quantization ~0.018
nats that belong to the serving stack's own numerics.

| | int4 ParoQuant | PARO-MXFP4 | **this checkpoint** |
|---|---|---|---|
| bits/weight | 4.25 | 4.25 | **5.25** |
| KL top-5 / top-256 | 0.0195 / 0.0285 | 0.0296 / 0.0419 | **0.0070 / 0.0100** |
| top-1 agreement | 91.5% | 90.3% | **95.21%** |
| GSM8K 500q, greedy, served path | 97.4-98.0% | 97.4-97.6% | **97.40%** |
| decode step @ctx 25 / 8k / 32k | 23.5 ms | 23.3 ms | **25.90 / 27.46 / 28.19 ms** |
| prefill 2k / 8k / 32k / 64k (PP t/s) | 3808 / 3646 / 3495 / 3349 | 4770 / 4827 / 4495 / 4273 | **3941 / 3790 / 3616 / 3436** |
| KV cache tokens | 854k | 862k | **760k** |
| on disk | 18 GB | 18 GB | **21 GB** |

Where the fidelity comes from, cumulatively: fp16 group scales instead of e8m0 shared exponents
(-32% KL), the fifth bit (0.0285 -> 0.0155), the stage-2 fine-tune (-> 0.0126), and int8 per-group
activations instead of e4m3 per-token (-> 0.0097). The last one matters more than it looks: per-*token*
int8 was the wrong granularity, not the wrong format, and per-group int8 lands **below** the
weights-only RTN number.

## Choosing between this and PARO-MXFP4

Task accuracy does not separate them — GSM8K is 97.4-97.8% for every variant, inside noise at 500
questions. The separation is distributional fidelity: at 0.0100 vs 0.0419 nats this checkpoint stays
markedly closer to the bf16 base's full output distribution, which is what matters for
logprob-sensitive work, draft-model acceptance, and long agentic chains where small per-token
divergences compound. If you want maximum prefill throughput and KV headroom instead, take PARO-MXFP4.
