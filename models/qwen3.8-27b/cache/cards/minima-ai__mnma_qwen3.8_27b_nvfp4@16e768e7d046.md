---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
library_name: vllm
tags:
  - nvfp4
  - w4a4
  - quantized
  - compressed-tensors
  - gated-deltanet
  - fp8-kv-cache
  - qwen3.8
---

# mnma Qwen3.8-27B NVFP4 (Minima)

**A fully W4A4 build of Qwen3.8-27B by post-training quantization
alone: every one of the 496 backbone linear layers — all 48 Gated
DeltaNet (GDN) layers and their gate projections included — quantized
to NVFP4, plus calibrated FP8 KV-cache scales. No quantization-aware
training, no distillation — calibration only.**

Early public 4-bit builds of this model kept the GDN block at 8/16-bit
on the assumption that recurrent-state layers are fragile under
quantization. Our accompanying paper shows the opposite — the
architecture's own gating and delta-rule correction make the recurrent
half the *easy* half to quantize — and this checkpoint is the proof:
it matches BF16 within seed noise on every task we measured while
being the smallest and fastest-prefill recipe in the cohort the paper
compares.

## What exactly is quantized

| Block | Tensors | Precision |
|---|---|---|
| GDN (48 layers) | `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` | NVFP4 W4A4 (group 16) |
| Attention (16 layers) | `q/k/v/o_proj` | NVFP4 W4A4 |
| MLP (64 layers) | `gate/up/down_proj` | NVFP4 W4A4 |
| KV cache (16 attention layers) | `k_scale`/`v_scale` | calibrated static FP8 scales (32 tensors) |
| Embeddings, `lm_head`, `conv1d`, norms, `A_log`/`dt_bias` | — | BF16 |

Built with [llm-compressor](https://github.com/vllm-project/llm-compressor)
(`NVFP4` scheme + `kv_cache_scheme` fp8/tensor/static), calibrated on a
frozen 128-sample × 32K-token set. Text-only `Qwen3_5ForCausalLM`
extraction (no vision tower).

**Global scales are harmonized.** vLLM fuses `in_proj_qkv+z` and
`in_proj_b+a` into single NVFP4 GEMMs with one global scale each, while
llm-compressor calibrates them per module (up to 2.8× apart). Serving
an unharmonized checkpoint silently mis-scales the GDN decay/write
gates. This checkpoint has the fused groups rewritten to shared global
scales with the ratio folded into the per-block scales, so it serves
correctly out of the box.

## Serving

Requires vLLM ≥ 0.27 on hardware with native NVFP4 (SM120 / Blackwell):

```bash
vllm serve minima-ai/mnma_qwen3.8_27b_nvfp4 --kv-cache-dtype fp8
```

`--kv-cache-dtype fp8` picks up the calibrated k/v scales shipped in
the checkpoint.

## Evaluation

Measured under one serving regime (vLLM 0.27.1, TP=1, one 96 GB
RTX PRO 6000, FP8 KV cache), against the BF16 model served identically:

| | BF16 | **this model** |
|---|---|---|
| WikiText-2 PPL @4K / @32K | 6.95 / 10.35 | 7.68 / 10.50 |
| MMLU-Pro | 80.4 | 79.7 |
| GSM8K | 95.5 | 95.5 |
| AIME'25 (pass@1, 4 seeds) | 86.7 | 86.7 |
| GPQA-Diamond (pass@1, 4 seeds) | 86.5 | 85.1 |
| LiveCodeBench v6 | 79.0 | 78.5 |
| RULER NIAH single/multikey @32K, @64K | 100 ×4 | 100 ×4 |
| Weights in VRAM | 50.13 GiB | **17.53 GiB** |
| Decode tok/s (1K in / 1K out, concurrency 32) | 621 | **1,154** |
| TTFT, 32K-token prefill | 6.90 s | **4.03 s** |

No task score is CI-separated from BF16. Task rows were measured on
the scale-free twin of this checkpoint (identical weights; a KV-cache
ablation showed KV precision moves no task score for either model);
PPL, RULER, and throughput were measured on this checkpoint directly.
The perplexity gap (+0.73 @4K, +0.15 @32K) *shrinks* with context —
quantization error does not accumulate in the recurrent state; see the
paper for the mechanism study.

## Notes

- Long-context perplexity of the *base* model rises inside long
  requests (BF16 shows the same 6.95 → 10.35 pattern); compare models
  only under matched serving paths.
- Quantized with the thinking-capable chat template intact; evaluation
  used `enable_thinking=false` for MMLU-Pro/GSM8K and thinking-on
  sampling for AIME/GPQA/LCB.

Paper: [Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for
the Recurrent Half of a Hybrid 27B LLM](https://arxiv.org/abs/2609.04098)
(arXiv:2609.04098).
