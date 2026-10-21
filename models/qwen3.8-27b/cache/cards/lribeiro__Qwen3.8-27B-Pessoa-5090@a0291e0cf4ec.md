---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - compressed-tensors
  - autoround
  - int4
  - int5
  - int6
  - mixed-precision
  - weight-only
  - w4a16
  - w5a16
  - w6a16
  - quantized
  - vllm
  - blackwell
  - sm120
  - rtx-5090
  - qwen3.8
  - vision
  - image-text-to-text
  - vlm
  - mtp
language:
  - multilingual
---

# Qwen3.8-27B-Pessoa-5090 (AR11)

Mixed-precision **AutoRound** INT quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
saved in `compressed-tensors` (`pack-quantized`) format for direct use with vLLM.

This is a **three-tier weight-only** checkpoint tuned for the RTX 5090 / Blackwell
SM120 memory envelope:

- **MLP layers 0–31** — **INT4 W4A16**, group size 64, symmetric (heaviest compression
  on the early, less error-sensitive blocks).
- **MLP layers 32–63** — **INT5 W4A16**, group size 64, symmetric (one extra bit on the
  late blocks where output distributions are more sensitive).
- **All attention projections** (self-attn Q/K/V/O + linear-attn qkv/z/out) —
  **INT6 W6A16**, group size 128, symmetric (attention kept higher-precision to preserve
  routing/recall fidelity).
- **Activations are not quantized** (weight-only): weights are dequantized to BF16 at the
  kernel boundary, so there is zero activation quantization error.
- **Visual encoder, MTP head, lm_head, embeddings, and linear-attention recurrent
  projections** (`in_proj_a`, `in_proj_b`, `norm`, `conv1d`) are kept in BF16.

The 32/32 INT4-vs-INT5 MLP split plus INT6 attention lands at **22.59 GB** — small enough
to fit comfortably on a single RTX 5090 (32 GB) with room for KV cache and MTP speculative
decoding — while holding KLD to **0.00397 nats** (top-1 agreement **97.64%**) on the
278,392-position fidelity suite.

| Topic | Notes |
|---|---|
| **Checkpoint size** | 22.59 GB (7 safetensors shards) |
| **BF16 base** | 55.6 GB → **2.46× smaller** |
| **Architecture** | `Qwen3_5ForConditionalGeneration` (VLM: vision encoder + language model + MTP) |
| **Quantization** | AutoRound 0.14.2, 200 iters, 181 calibration contexts |
| **Vision tower** | 27-layer ViT, 460.7M params, BF16 (unquantized) |
| **MTP head** | 1 layer, BF16 (unquantized, speculative decoding) |
| **Hardware target** | NVIDIA RTX 5090 / Blackwell SM120 (FP8/INT MMA) |
| **Engine** | vLLM, FlashInfer backend, MTP speculative decoding |

---

## Model Details

**Qwen3.8-27B** (`qwen3_5` / `Qwen3_5ForConditionalGeneration`):

- **Type:** Vision-language model (hybrid linear + full attention language model + vision encoder)
- **Parameters:** 27.79B total
- **Layers:** 64 — 48 linear-attention (GatedDeltaNet) + 16 full-attention (every 4th layer: 0, 4, 8, …, 60)
- **Hidden size:** 5,120
- **Intermediate size:** 17,408
- **Attention heads:** 24 × 256 dim (full-attn); 16 key × 128 dim / 128 value head (linear-attn)
- **Vocabulary:** 248,320
- **Activation:** SiLU (SwiGLU MLP, `output_gate_type=swish`)
- **Context:** 262,144 tokens (max_position_embeddings)
- **Vision encoder:** 27-layer ViT, hidden size 1,152, patch size 16, 16 heads
- **MTP:** 1 layer, `num_speculative_tokens=3`, shared embeddings

Each linear-attention layer has `in_proj_qkv`, `in_proj_z`, `out_proj` plus recurrent
state paths `in_proj_a` / `in_proj_b`, a `norm`, and a `conv1d` (kernel dim 4). Each
full-attention layer has standard `q_proj` / `k_proj` / `v_proj` / `o_proj`
(4 KV heads, partial rotary 0.25). All layers share `gate_proj` / `up_proj` / `down_proj`
MLP blocks. The vision tower (`model.visual.*`) processes image/video patches and injects
embeddings into the language model via image/video token boundaries.

---

## Quantization Configuration

### Precision Assignment

| Module group | Format | Weights | Activations | Layers |
|---|---|---|---|---|
| MLP (gate/up/down) — early | **INT4 W4A16**, g64, sym | 4-bit int, `group`, `memoryless_minmax` | BF16 (not quantized) | 0–31 |
| MLP (gate/up/down) — late | **INT5 W4A16**, g64, sym | 5-bit int, `group`, `memoryless_minmax` | BF16 (not quantized) | 32–63 |
| self_attn (Q/K/V/O) | **INT6 W6A16**, g128, sym | 6-bit int, `group`, `memoryless_minmax` | BF16 (not quantized) | all 16 full-attn layers |
| linear_attn (qkv/z/out) | **INT6 W6A16**, g128, sym | 6-bit int, `group`, `memoryless_minmax` | BF16 (not quantized) | all 48 lin-attn layers |

**Weight-only:** `input_activations: null`, `output_activations: null` for all groups.
Weights are dequantized to BF16 at the kernel boundary — no activation quantization error.
KV cache is left in BF16 (no `kv_cache_scheme`).

### Modules Kept in BF16

| Pattern | Reason |
|---|---|
| `re:.*visual.*` | Vision tower — small, quantization-sensitive |
| `re:.*mtp.*` | MTP head — kept BF16 for speculative decoding |
| `lm_head` | Output projection — kept BF16 for precision |
| `model.language_model.embed_tokens` | Token embeddings — kept BF16 |
| `model.language_model.layers.*.linear_attn.in_proj_a` / `in_proj_b` | Recurrent state path — non-power-of-64 dims |
| `model.language_model.layers.*.linear_attn.norm` | Numerically fragile normalization path |
| `model.language_model.layers.*.linear_attn.conv1d` | 1D conv weights — tiny |
| `model.language_model.layers.*.linear_attn.A_log` / `dt_bias` | Mamba-style decay/bias — tiny |
| `model.language_model.layers.*.self_attn.k_scale` / `v_scale` | Attention scale factors — tiny |
| `model.language_model.layers.*.input_layernorm` / `post_attention_layernorm` | RMSNorm — not quantized |

### Calibration

- **Tool:** [AutoRound](https://github.com/intel/auto-round) 0.14.2 (MLLM block-walker path)
- **Iters:** 200
- **Source:** `malaiwah/qwen38-27b-fidelity-suite-v3` tokens (181 contexts × 2048 tokens)
- **Samples:** 181, max sequence length 2048
- **Format:** Decoded token IDs from the fidelity suite, split 2/3 user / 1/3 assistant
- **Scheme base:** `W4A16`, overridden per-module by an explicit `layer_config` (exact
  module names — AutoRound 0.14.2 does not expand regex keys for the MLLM path)
- **Seed:** 0

### Build command

```bash
python scripts/quant_pessoa_autoround_v7.py \
  --model /models/Qwen3.8-27B \
  --output /models/Qwen3.8-27B-Pessoa-AR11-w6-late5-32 \
  --iters 200 --calib-samples 181 --calib-max-len 2048 \
  --lat-bits 6 --sat-bits 6 \
  --late-mlp-bits 5 --late-mlp-start 32 \
  --base-mtp-dir /models/Qwen3.8-27B
```

---

## Fidelity Evaluation (KLD)

Distribution-fidelity measured against the BF16 base on the
`qwen38-27b-fidelity-suite-v3` capture-replay harness: 136 contexts, 278,392 scored
positions, comparing the full next-token softmax distribution (KL divergence in nats,
float64 accumulation, two-pass) using the shared BF16 `lm_head`.

### Headline Metrics

| Metric | Value |
|---|---:|
| **Mean KLD** | **0.003971** nats |
| Median KLD | 0.001230 nats |
| p95 KLD | 0.01436 nats |
| p99 KLD | 0.05407 nats |
| p999 KLD | 0.1941 nats |
| Max KLD | 1.5037 nats |
| **Top-1 agreement** | **97.64%** |
| Mean JSD | 0.001408 bits |
| Contexts | 136 |
| Scored positions | 278,392 |
| Vocab size | 248,320 |

### Bootstrap CI (mean KLD, 10,000 resamples, 41 clusters)

| | Value |
|---|---:|
| Mean | 0.003971 |
| 95% CI low | 0.003176 |
| 95% CI high | 0.004916 |

### KLD by Stratum

| Stratum | Contexts | Mean KLD |
|---|---:|---:|
| Multilingual | 7 | 0.001888 |
| Scientific | 39 | 0.002028 |
| Encyclopedic | 13 | 0.003125 |
| Code | 36 | 0.003856 |
| Literary | 41 | 0.006545 |

As is typical for weight-only quantization of this family, the **literary** stratum
(Project Gutenberg prose) carries the highest mean KLD, while **scientific** and
**multilingual** contexts are the closest to the base distribution.

### Worst Contexts

| # | Stratum | Cluster | Mean KLD | Max KLD | Top-1 |
|---:|---|---|---:|---:|---:|
| 156 | literary | pg2701 (Moby-Dick) | 0.01769 | 1.5037 | 97.07% |
| 79 | literary | pg174 (Scrooge) | 0.01710 | 0.4504 | 95.55% |
| 51 | literary | pg98 (Tale of Two Cities) | 0.01597 | 0.3673 | 94.63% |
| 80 | literary | pg219 | 0.01593 | 0.7814 | 96.09% |
| 74 | literary | pg174 | 0.01376 | 0.7430 | 96.53% |

The worst individual token (max KLD 1.50 nat) occurs in a Moby-Dick passage; even there
the top-1 agreement remains above 97%.

### Comparator

- Vocab chunk: 24,832 · Device: CUDA · Accumulation: float64 · Two-pass: true

---

## AutoRound Ladder (Pessoa series)

AR11 sits in a sweep of AutoRound mixed-precision variants, all with INT6 attention
(lat + sat) and a progressively later INT4→INT5 MLP boundary:

| Variant | MLP early / late split | MLP late bits | Mean KLD | Top-1 | Size |
|---|---|---:|---:|---:|---:|
| AR6-INT4 | all INT4 | 4 | 0.009011 | 96.54% | — |
| AR7-w6 | no late split | — | 0.008491 | 96.58% | — |
| AR8-w6-late5 | ≥8 | 5 | 0.007774 | 96.95% | — |
| AR9-w6-late5-16 | ≥16 | 5 | 0.006509 | 97.18% | — |
| AR10-w6-late5-24 | ≥24 | 5 | 0.004798 | 97.49% | — |
| **AR11-w6-late5-32** | **≥32** | **5** | **0.003971** | **97.64%** | **22.59 GB** |
| AR12-w6-late5-48 | ≥48 | 5 | 0.003591 | 97.75% | — |

KLD falls monotonically as the INT4→INT5 boundary moves earlier (more layers at INT5).
AR11's 32/32 split is the knee of the curve: each step beyond it yields diminishing
fidelity gains for the added size, while each step before it costs noticeably more KLD.

---

## Usage

```bash
# vLLM serve (RTX 5090 / Blackwell SM120 + FlashInfer)
vllm serve /models/Qwen3.8-27B-Pessoa-AR11-w6-late5-32 \
  --served-model-name qwen38-27b-pessoa-5090 \
  --max-model-len 131072 \
  --max-num-seqs 32 \
  --quantization compressed-tensors \
  --trust-remote-code \
  --attention-backend flashinfer \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The endpoint is OpenAI-compatible. MTP speculative decoding is enabled
(`num_speculative_tokens=3`), providing a decode speedup when acceptance is high.

---

## Hardware & Runtime

| Field | Value |
|---|---|
| Target GPU | NVIDIA RTX 5090 (32 GB) / Blackwell SM120 |
| Checkpoint | 22.59 GB (fits 32 GB with KV cache + MTP headroom) |
| Engine | vLLM, FlashInfer backend |
| MTP speculative decoding | 3 speculative tokens |
| Max model len | 131,072 (recommended serve) |
| Vision | BF16, full image/video understanding retained |

---

## Limitations

- **No task-accuracy benchmarks yet.** This model has a KLD fidelity report but has not
  been run through the GSM8K / MMLU-Pro / GPQA-Diamond accuracy suite. KLD is a strong
  proxy for distribution closeness but is not a direct task-score measurement. Validate
  on your own workload before production use.
- **Quantization is lossy.** Weight-only INT4/INT5/INT6 eliminates activation error but
  retains weight rounding error. The literary stratum shows the highest divergence.
- **Blackwell-optimized.** The INT pack-quantized MMA paths are exercised on SM120
  (Blackwell). Validate on your hardware (Ada/Hopper SM89/SM90 also support the relevant
  integer paths).
- **AutoRound 0.14.2 MLLM path quirk.** Regex `layer_config` keys are not expanded for the
  MLLM block-walker; this checkpoint uses explicit per-module keys (400 entries) and a
  post-build config guard verifying exported group bits match the recipe.
- **RTX 5090 naming.** The "5090" in the model name denotes the target memory envelope
  (single 32 GB card). The checkpoint is hardware-agnostic compressed-tensors and runs on
  any vLLM-supported GPU with sufficient VRAM.

---

## Acknowledgements

Standing on the shoulders of giants. This work would not exist without the
[**rtx6kpro**](https://github.com/local-inference-lab/rtx6kpro) community and the
broader local-inference-lab Discord — the open protocol, tooling, and baseline
measurements that made systematic quantization comparison possible.

The fidelity suite (`malaiwah/qwen38-27b-fidelity-suite-v3`) and the KLD capture-replay
harness provided the distribution-level validation. AutoRound
([Intel](https://github.com/intel/auto-round)) produced the optimized rounding.
