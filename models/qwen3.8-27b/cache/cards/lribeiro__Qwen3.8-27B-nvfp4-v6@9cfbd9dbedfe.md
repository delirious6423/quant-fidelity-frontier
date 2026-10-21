---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - compressed-tensors
  - fp4
  - gptq
  - nvfp4
  - quantized
  - vllm
  - w4a4
  - mixed-precision
  - blackwell
  - sm120
  - qwen3.8
language:
  - multilingual
---

# Qwen3.8-27B-NVFP4-GPTQ-v6

NVFP4 (W4A4) + FP8 (W8A8) mixed-precision, GPTQ-quantized variant of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [llm-compressor](https://github.com/vllm-project/llm-compressor) and
saved in `compressed-tensors` format for direct use with vLLM.

This is experiment **v6** in a 38-run quantization sweep. The first 16 MLP layers use
NVFP4 4-bit weights + 4-bit activations with GPTQ Hessian error correction; the remaining
48 MLP layers and all attention layers use FP8 E4M3 (W8A8). v6 is the Pareto-optimal
config in the NVFP4 W4A4 line — the best fidelity achievable with native FP4 tensor cores
before KLD degrades below the Acceptable tier.

**The best NVFP4 W4A4 config, but not the best model overall.** v17+ (all-FP8 W8A8)
achieves lower KLD at comparable speed and strictly dominates the entire NVFP4 W4A4
line. v6 is the recommended choice only when 4-bit NVFP4 tensor-core prefill throughput
is required and KLD must stay within the Acceptable band.

| Topic  |  Notes |
|---|---|
| **Checkpoint size** | 26.30 GB (14 safetensors shards) |
| **BF16 base** | 55.6 GB → **2.11× smaller** |
| **Hardware requirement** | NVIDIA Blackwell (SM120), FP4 tensor cores |
| **Engine** | vLLM 0.26.1rc0 (infernal-invocation fork), FlashInfer backend |

---

## Model Details

**Qwen3.8-27B** (`qwen3_5_text`):

- **Type:** Causal language model (hybrid linear + full attention)
- **Parameters:** 27B
- **Layers:** 64 — 48 linear-attention (GatedDeltaNet) + 16 full-attention (every 4th layer: 0, 4, 8, …, 60)
- **Hidden size:** 5,120
- **Intermediate size:** 17,408
- **Attention heads:** 24 × 256 dim
- **Vocabulary:** 248,320
- **Activation:** SiLU
- **Context:** 8,192 tokens (benchmark configuration)

Each linear-attention layer has `in_proj_qkv`, `in_proj_z`, `out_proj` plus recurrent state
paths `in_proj_a`/`in_proj_b`. Each full-attention layer has standard `q/k/v/o_proj`. All
layers share `gate_proj`/`up_proj`/`down_proj` MLP blocks.

---

## Quantization Configuration

### Precision Assignment

| Module group | Format | Weights | Activations | Layers |
|---|---|---|---|---|
| MLP (gate/up/down) | **NVFP4 W4A4** | 4-bit float, sym, group_size=16, E4M3 scales, `tensor_group`, `memoryless_minmax` | 4-bit float, sym, group_size=16, E4M3 scales, dynamic=local, `static_minmax` | **0–15** |
| MLP (gate/up/down) | **FP8 W8A8** (E4M3) | 8-bit float, sym, `channel`, `memoryless_minmax` | 8-bit float, sym, `token`, dynamic | **16–63** |
| self_attn (Q/K/V/O) | FP8 W8A8 (E4M3) | same | same | all 16 full-attn layers |
| linear_attn (qkv/z/out) | FP8 W8A8 (E4M3) | same | same | all 48 lin-attn layers |
| lm_head | FP8 W8A8 (E4M3) | same | same | — |
| KV cache | FP8 E4M3, `tensor`, static | — | — | — |

**GPTQ:** Hessian-based weight correction, sequential targets `Qwen3_5DecoderLayer`,
`actorder=static`, default `dampening_frac` (no override).

### Modules Kept in BF16

| Pattern | Reason |
|---|---|
| `visual.*` | Vision tower — small, quantization-sensitive |
| `linear_attn.norm` | Numerically fragile normalization path |
| `linear_attn.in_proj_a` | Non-power-of-64 dims — CUTLASS constraint |
| `linear_attn.in_proj_b` | Non-power-of-64 dims — CUTLASS constraint |
| `mtp.*` | Multi-token-prediction head — kept BF16 (MTP speculative decoding not used in this config) |

### Calibration

- **Source:** `malaiwah/qwen38-27b-fidelity-suite-v3` tokens (181 contexts × 2048 tokens)
- **Samples:** 181
- **Max sequence length:** 2048
- **Format:** Decoded token IDs from suite, not raw text

---

## Usage

```bash
# vLLM serve (requires Blackwell SM120 + FlashInfer)
vllm serve /models/Qwen3.8-27B-nvfp4-gptq-v6 \
  --max-model-len 8192 \
  --max-num-seqs 512 \
  --quantization compressed-tensors
```

> **`--max-num-seqs` matters on this architecture.** 48 of 64 layers use linear attention,
> and vLLM allocates one Mamba-style cache block per decode sequence. The default
> `max_num_seqs=1024` can exceed available blocks and fail during CUDA graph capture with
> `max_num_seqs (1024) exceeds available Mamba cache blocks`. Lower to 512 or raise
> `--gpu-memory-utilization`. This is a property of the base model, not of quantization.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/models/Qwen3.8-27B-nvfp4-gptq-v6",
          quantization="compressed-tensors",
          max_model_len=8192)
out = llm.generate(
    ["Explain 4-bit quantization in two sentences."],
    SamplingParams(temperature=0.7, max_tokens=256),
)
print(out[0].outputs[0].text)
```

---

## Evaluation

### Distribution Fidelity (KLD)

Hidden-state capture after final RMSNorm → shared BF16 LM-head replay → full-vocabulary
`KL(BF16_ref ‖ candidate)`, two-pass log-sum-exp normalization. Body-only comparison
(candidate LM head unused). Suite: 136 contexts × 2048 tokens, 278,392 scored positions
over 248,320-token vocabulary.

| Metric | Value |
|---|---:|
| **Token mean KLD** | **0.015972** |
| Token median KLD | 0.002991 |
| p95 KLD | 0.062037 |
| p99 KLD | 0.245719 |
| p99.9 KLD | 0.894317 |
| Max KLD | 4.341776 |
| Mean JSD (bits) | 0.005468 |
| **Top-1 agreement** | **95.64%** |
| Validation tier | Acceptable (KLD < 0.031, Top-1 > 94%) — matches EXL3 K4 band |

### KLD by Domain

| Stratum | Contexts | Mean KLD |
|---|---:|---:|
| scientific | 39 | 0.00602 |
| multilingual | 7 | 0.00616 |
| encyclopedic | 13 | 0.01175 |
| code | 36 | 0.01465 |
| literary | 41 | 0.02962 |

Literary text is the dominant KLD contributor — 4.9× the scientific stratum. 4-bit
quantization disproportionately affects creative/prose generation.

### Throughput

Single NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB), vLLM 0.26.1rc0, FlashInfer backend.
Duration-based sustained decode, 30s cells, temperature 0.0, `ignore_eos=true`.

| Metric | Value |
|---|---:|
| **Prefill (2k context)** | **7,756 tok/s** |
| TTFT (2k) | 0.263 s |
| Decode C1 (0 ctx) | 52.3 tok/s |
| Decode C1 (2k ctx) | 52.0 tok/s |
| Decode C4 (0 ctx) | 189.1 tok/s |
| Decode C4 (2k ctx) | 183.1 tok/s |

### Composite Score

$$\text{score} = \frac{\text{prefill}_{2k} \times \text{decode}_{c1}}{\text{KLD} \times 10^6} = \frac{7756 \times 52.3}{0.01597 \times 10^6} = \mathbf{25.42}$$

---

## Context Within Experiment Series

v6 is the second data point in the NVFP4 W4A4 mixed-precision sweep (Phase 1), testing
how many early MLP layers can use 4-bit NVFP4 before KLD degrades unacceptably. KLD
scales roughly linearly with NVFP4 layer count — each additional 8-layer block adds
~0.004–0.005 KLD.

| ID | NVFP4 MLP layers | FP8 MLP layers | KLD | Top-1 | Prefill | Decode | Score |
|---|---|---|---:|---:|---:|---:|---:|
| **v6** | **0–15** | **16–63** | **0.01597** | **95.64%** | **7,756** | **52.3** | **25.42** |
| v7 | 0–23 | 24–63 | 0.01975 | 95.16% | 8,020 | 54.1 | 21.97 |
| v4 | 0–31 | 32–63 | 0.02596 | 94.48% | 8,488 | 56.5 | 18.48 |
| v2 | 0–47 | 48–63 | 0.06575 | 92.17% | 9,243 | 60.5 | 8.51 |
| v1 | 0–55 | 56–63 | 0.07553 | 91.59% | 9,660 | 63.6 | 8.13 |

v6 is the Pareto-optimal NVFP4 W4A4 config — the only point in the sweep that stays
within the Acceptable KLD tier (< 0.031) while still using native FP4 tensor cores for
prefill. Later phases (v17+: all-FP8 W8A8) superseded this entire line — FP8 achieves
KLD 0.0123 at comparable speed, strictly dominating NVFP4 W4A4 for this model.

---

## Provenance & Integrity

| Artifact | SHA-256 |
|---|---|
| Model index | `3cb54aa0c6ccc887610d28c9766de8f9218b9142f6e0977a7efb22ab15e3a7c2` |
| Config | `fa744eae724e6b7519428f0cfd396988efe89fa12223e54f1f5e1800c7348e4b` |
| Suite tokens | `3f9d17f1b55f64872ad3ac19c8711654e09ba70b7ca14b0851525088fe735691` |
| Capture manifest | `2d9a8f99c2102cc305cc64653c8f2d1728da413cebd84a5027d543987be3569a` |
| Shared LM head | `25a30fd5f826da0abc4efc4cc71def9f02bcb8085f7175eee284d221dee4cfff` |

14 model shards individually hashed in `reports/report-nvfp4-gptq-v6.json` →
`candidate_identity.shard_sha256`. Full fidelity report, benchmark JSON, and hidden-state
captures archived in the experiment workspace.

---

## Hardware & Runtime

| Field | Value |
|---|---|
| GPU | 4× NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 96 GiB GDDR7 each) |
| Driver | 595.58.03 |
| Benchmark GPU count | 1 |
| VRAM used | 86,648 / 97,887 MB (88.5%) |
| KV cache budget | 1,888,640 tokens (1,135 blocks × 1,664) |
| GPU util (avg / max) | 88.9% / 100.0% |
| Power (avg / max) | 272 W / 298 W (limit 300 W) |
| Temp (avg / max) | 52.2 °C / 58.0 °C |

---

## Reproduction

```bash
# Inside Docker (vLLM infernal-invocation image, FlashInfer, SM120)
CUDA_VISIBLE_DEVICES=0 python /data/scripts/quantize_nvfp4_gptq_v6.py \
  --model /data/models/Qwen3.8-27B-bf16 \
  --output /data/models/Qwen3.8-27B-nvfp4-gptq-v6 \
  --calib-samples 181 \
  --calib-max-len 2048
```

Calibration tokens from `$SUITE_DIR/tokens` (default `/data/suite-v3/tokens`).
Script: `scripts/quantize_nvfp4_gptq_v6.py`. Log: `quant_nvfp4_gptq_v6.log`.

---

## Limitations

- **Not the best config.** v17 (all-FP8 W8A8) achieves 23% lower KLD (0.0123 vs 0.0160)
  at comparable speed and strictly dominates. Use v6 only when native FP4 tensor-core
  prefill is required and KLD must stay within the Acceptable band.
- **Blackwell-only.** NVFP4 W4A4 requires SM120 FP4 tensor cores via FlashInfer CUTLASS.
  No fallback path for non-Blackwell GPUs.
- **NVFP4 prefill only for MLP.** FlashInfer native FP4 MMA is used for MLP GEMMs only;
  attention layers run FP8 MMA. No native FP4 path for attention.
- **Body-only KLD.** Fidelity measured against a shared BF16 LM head; end-to-end
  generation quality may differ slightly.
- **Literary domain degradation.** 4.9× higher KLD on literary text vs scientific —
  creative/prose generation is disproportionately affected by 4-bit quantization.
- **Quantization is lossy.** Validate on your own workload before production use.
