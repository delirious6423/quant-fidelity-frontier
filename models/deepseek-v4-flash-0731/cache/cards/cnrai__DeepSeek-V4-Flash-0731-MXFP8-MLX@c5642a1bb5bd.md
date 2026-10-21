---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
pipeline_tag: text-generation
library_name: transformers
tags:
- deepseek_v4
- conversational
- mxfp8
- 8-bit
---

# cnrai/DeepSeek-V4-Flash-0731-MXFP8-MLX

MXFP8-quantized DeepSeek-V4-Flash for Apple Silicon, optimized for PAVE Core, OpenPAVE and vllm-mlx

## Model Details

| Feature | Details |
| --- | --- |
| **Base model** | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| **Quantization** | MXFP8 (8-bit floating point with block-wise scaling) |
| **Original format** | FP8 (E4M3 weights, E8M0 block scales) |
| **Architecture** | Hybrid attention + pooling (DeepSeek V4 MoE, 43 layers) |
| **Parameters** | ~274 GB on disk |
| **Designed for** | Mac Studio M3 Ultra 512GB + PAVE Core |

## Why MXFP8?

The original DeepSeek-V4-Flash-0731 checkpoint uses FP8 (E4M3) weights with E8M0 block scales. MLX's Metal kernel backend doesn't natively read HuggingFace F8_E4M3/E8M0 dtypes from safetensors. This checkpoint converts them to MXFP8 format, which MLX loads natively via `mx.quantized_matmul(mode='mxfp8')`.

### FP8 to MXFP8 Conversion

The conversion is performed directly from the **original `deepseek-ai/DeepSeek-V4-Flash-0731`** checkpoint — not from a re-quantized derivative. This is critical: any intermediate re-quantization (e.g. MXFP4 to MXFP8 via a third party) compounds quantization noise and degrades quality. By starting from the source, we guarantee no double-quantization.

#### Source checkpoint layout

The original checkpoint stores two different quantization formats, despite being labeled "FP8":

| Layer type | Source format | Bits per weight | Layout |
|---|---|---|---|
| Attention (q/k/v/o projections) | FP8 E4M3 | 8-bit | Weight tensor + E8M0 block-scale tensor |
| Shared MLP (gate/up/down) | FP8 E4M3 | 8-bit | Weight tensor + E8M0 block-scale tensor |
| MoE experts (switch_mlp, 256 experts) | **MXFP4** E2M1 | **4-bit** | 2 nibbles packed per byte + E8M0 block-scale |

#### Conversion pipeline (per shard)

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Load original weights via PyTorch                │
│  (reads HF F8_E4M3/E8M0 and MXFP4-packed I8 dtypes)       │
├─────────────────────────────────────────────────────────┤
│  STEP 2: Dequantize to bf16                               │
│                                                           │
│  FP8 E4M3 path:                                           │
│    bf16 = E4M3_value × 2^(block_scale_byte - 127)         │
│    (lossless: bf16 mantissa=7 bits > E4M3 mantissa=3)     │
│                                                           │
│  MXFP4 E2M1 path:                                         │
│    bf16 = mx.dequantize(mxfp4_packed, block_scales)       │
│    (lossless: bf16 mantissa=7 bits > E2M1 mantissa=1)     │
├─────────────────────────────────────────────────────────┤
│  STEP 3: Re-quantize bf16 → MLX MXFP8                     │
│    mx.quantize(bf16_weights, bits=8, mode='mxfp8')        │
│    (MLX computes optimal E4M3 + E8M0 block scales)        │
└─────────────────────────────────────────────────────────┘
```

#### Why quality is retained

1. **No double quantization.** We convert from the original source checkpoint, not from an intermediate MXFP4 derivative. The rejected path (MXFP4 → bf16 → MXFP8) would compound two rounding errors; our path (source → bf16 → MXFP8) introduces only one.

2. **bf16 intermediate is lossless.** Both FP8 E4M3 (3 mantissa bits) and MXFP4 E2M1 (1 mantissa bit) fit exactly into bf16's 7 mantissa bits. The dequantization to bf16 introduces zero additional rounding error.

3. **Expert weights gain precision.** MoE experts go from 4-bit (MXFP4 E2M1, 1 mantissa bit) to 8-bit (MXFP8 E4M3, 3 mantissa bits). This is a strict precision upgrade — 3× more mantissa bits reduces quantization error by ~8× per weight. Since V4-Flash has 256 small experts (2048 intermediate each), the extra precision directly improves MoE routing and output quality.

4. **Block scales are recomputed optimally.** The original E8M0 block scales are designed for the source format's dynamic range. MLX's `mx.quantize` recomputes block scales that are optimal for MXFP8's E4M3 mantissa range, rather than reusing scales designed for a different format.

#### What we explicitly rejected

- **inferencerlabs MXFP4→MXFP8 path**: double quantization. Their MXFP4 checkpoint was already quantized from the original FP8; re-quantizing MXFP4→bf16→MXFP8 compounds rounding errors. User-confirmed quality degradation (garbled thinking, broken tool calls, repetition loops).
- **Vontra MXFP4 checkpoint**: standard MLX layout but still 4-bit experts. Correct outputs on simple prompts but degrades on complex reasoning due to MXFP4 precision loss on V4's 256 tiny experts.

Output: 45 safetensors shards, ~274 GB total. Expert weights doubled from 4-bit to 8-bit as intended.

## Performance

**Mac Studio M3 Ultra (512GB unified memory):**

| Metric | Value |
|---|---|
| Decode throughput | **up to 26 tok/s** (without MTP speculative decoding) |
| Prefill throughput | ~106 tok/s |
| Model load size | ~274 GB |
| Quantization overhead | ~14% slower vs MXFP4 (31 tok/s) due to 2x bandwidth |

### Benchmark conditions

- Single-user, TP=1, `--max-num-seqs 1`
- `--cache-memory-mb 8192` prefix cache
- `--ssd-cache-dir` for SSD-tier KV cache
- Temperature=0, max_tokens=256
- 5 diverse prompts (code, creative, reasoning, translation, math)
- Measured via server-side `Chat completion: N tokens in Xs` log lines

## PAVE Core Integration

This model is designed to work perfectly with **PAVE Core** — a highly customized fork of vllm-mlx optimized for DeepSeek-V4-Flash and GLM-5.2 on Apple Silicon.

### Features

- **System KV cache** — snapshots the 12.8K-token system prefix (system prompt + tool definitions), saving ~90% prefill compute on every request
- **Block-aligned prefix cache** — block-aligned (128 tokens) prefix matching that respects V4's hybrid PoolingCache/SSM architecture
- **Deep-copy snapshot** — pre-decode KV snapshots with deep-copied arrays prevent in-place mutation corruption across requests
- **SSD-tiered KV cache** — large cache entries spill to SSD for cross-session reuse
- **MXFP8/MXFP4 hybrid quantization** — attention/shared layers in MXFP8, MoE experts in MXFP8
- **Streaming tool calls** — DSML tool-call parser with streaming partial-marker buffering
- **Reasoning content** — V4 reasoning parser with `reasoning_effort` support and server-side reasoning preservation
- **CJK ban mask** — prevents CJK token generation in non-CJK contexts
- **Auto-compact** — server-side context compaction with KV cache splicing for long conversations
- **GPU keepwarm** — bistable residency management for Apple Metal (prevents cold-start decode penalties)
- **TB5 pipeline parallelism** — PP=2 prefill offload to a second Mac over Thunderbolt 5 (80 Gb/s)

### OpenPAVE

Built specifically for OpenPAVE — the AI coding agent.