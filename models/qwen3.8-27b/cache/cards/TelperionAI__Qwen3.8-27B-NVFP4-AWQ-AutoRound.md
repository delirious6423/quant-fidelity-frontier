---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - nvfp4
  - fp4
  - awq
  - autoround
  - llm-compressor
  - compressed-tensors
  - vllm
library_name: transformers
---

# Qwen3.8-27B-NVFP4-AWQ-AutoRound

Mixed-precision **NVFP4** quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B),
built with [llm-compressor](https://github.com/vllm-project/llm-compressor) using
**AWQ activation-aware scaling followed by AutoRound**.

**24.7 GB** — 6 GB smaller than FP8 and ~22% faster, while recovering **84%** of the
gap between a plain NVFP4+GPTQ build and FP8 on high-confidence token agreement.

## Recipe

| component | precision |
|---|---|
| `mlp.{gate,up,down}_proj`, layers 0–55 | **NVFP4** (4-bit, group-16, FP8-e4m3 scales → 4.5 effective bits) |
| `mlp.{gate,up,down}_proj`, layers 56–63 | FP8 e4m3 |
| `self_attn.{q,k,v,o}_proj` | FP8 e4m3 |
| `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (GDN) | FP8 e4m3 |
| `lm_head`, `embed_tokens`, norms, GDN state params, vision tower | **BF16** |

Two passes:

1. **AWQ** — per-input-channel scaling on `post_attention_layernorm → {gate_proj, up_proj}`
   and `up_proj → down_proj`. Gate and up share one input, so the reciprocal scale folds
   into the norm weights: **zero size and zero throughput cost.** The scales merge fully
   into weights, so unlike rotation methods (QuIP/SpinQuant) this still runs under
   tensor parallelism.
2. **AutoRound** — SignSGD-optimized rounding and clipping (200 iters) against a
   **block-wise** reconstruction loss, replacing GPTQ. Mean block loss fell ~18%.

Calibration: 1358 × 1024-token packed sequences (1.39M tokens) from a balanced Nemotron-v2 blend
(25% code, 25% math, 20% STEM, 20% chat, 10% multilingual).

`lm_head` and `embed_tokens` stay BF16, matching Qwen's own official FP8 release.

## Benchmarks

Measured against the BF16 base on 142,727 tokens of self-distilled thinking-mode
output plus 200 free greedy generations. vLLM 0.27.1, TP=2, 2×B300.

| checkpoint | size ↓ | top-1 ↑ | near-tie ↓ | moderate ↓ | confident ↓ | certain ↓ | divmed ↑ | tok/s ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.8-27B-FP8` *(8-bit ref)* | 30.9 GB | 96.15% | 22.70% | 3.48% | 1.45% | 0.08% | 47 | 8711 |
| **this model** | 24.7 GB | **93.38%** | **34.18%** | **8.67%** | **1.85%** | **0.17%** | **28** | 10590 |
| `RadixArk/Qwen3.8-27B-NVFP4` | **21.9 GB** | 90.23% | 43.80% | 14.49% | 3.29% | 0.70% | 11 | **11436** |
| `unsloth/Qwen3.8-27B-NVFP4` | 23.4 GB | 91.75% | 40.12% | 10.32% | 3.91% | 0.25% | 19 | 11069 |

Bold marks the best value in each column **among the FP4 checkpoints**; the FP8 row is a
reference at a different precision and size class, so it is excluded from the comparison.
All sizes are on-disk tensor bytes and **include the ~0.85 GB BF16 MTP head**, which every
checkpoint in this table ships. Subtract ~0.85 GB for a no-MTP comparison.

**Columns.** `top-1` is raw argmax agreement with BF16. The four bucket columns are
*disagreement* rates, split by how confident the base model was at that position
(top1−top2 logprob margin): `near-tie` <0.5, `moderate` 0.5–2, `confident` 2–5,
`certain` >5. **Only `confident` and `certain` are real damage** — a flip where the base
model was itself nearly tied is numerical noise. `divmed` is the median token index at
which free greedy generation first diverges from BF16 (higher is better).

**Perplexity is deliberately excluded.** On this comparison it is anti-correlated with
quality — the checkpoint with the best perplexity (`RadixArk`, −1.75%) has the worst
`certain`-bucket damage measured (0.70%, 4× this model's). Do not rank FP4 checkpoints
of this model by perplexity.

### What AutoRound contributes

Holding the recipe fixed and swapping only the rounding algorithm, `confident` damage
falls from **2.69% (GPTQ) → 1.85% (AutoRound)** — a paired McNemar z of 9.26 over
33,812 high-confidence positions. Against a plain NVFP4+GPTQ build without AWQ (3.97%),
the two passes together close **84%** of the distance to FP8.

The likely reason: GPTQ minimizes a per-layer weight-MSE proxy, while AutoRound optimizes
the block's actual output. Several methods we tested that sharpened the *local* proxy
(GPTQv2, error-guided layer allocation, a 3× larger AWQ grid) all made end-to-end quality
*worse* on this model.

## Usage

```python
from vllm import LLM
llm = LLM("TelperionAI/Qwen3.8-27B-NVFP4-AWQ-AutoRound", tensor_parallel_size=2)
```

Requires a Blackwell-class GPU for native NVFP4, and vLLM with `compressed-tensors`.

## Speculative decoding (MTP)

The model's **MTP (multi-token prediction) head is included**, in BF16, and works with
vLLM's `mtp` speculative decoding:

```python
from vllm import LLM
llm = LLM("TelperionAI/Qwen3.8-27B-NVFP4-AWQ-AutoRound", tensor_parallel_size=2,
          speculative_config={"method": "mtp", "num_speculative_tokens": 2})
```

`Qwen3_5ForConditionalGeneration` does not carry `mtp.*` in its state dict, so
llm-compressor never sees it and it is silently dropped, even though `config.json` still
declares `mtp_num_hidden_layers: 1`. It is grafted back in here from the base checkpoint
and excluded from quantization (`re:.*mtp.*` in
`quantization_config.ignore`; without that exclusion the quantization target regexes also
match `mtp.layers.0.mlp.*` and vLLM fails to load). Draft quality drives acceptance rate,
so it is kept at full precision rather than quantized.

Acceptance rate has not been measured; the head is verified to load and generate.

## Limitations

- **AutoRound ran at effective batch size 1.** The default (8) raised
  `this model has not been supported` on this architecture. Gradients are noisier than
  intended, so these numbers likely *understate* what the method can do here.
- **Calibration sequences are packed** to a uniform 1024 tokens (AutoRound concatenates
  batch elements), so they cross document boundaries.
- **Single evaluation corpus.** All numbers come from one self-distilled corpus.
- Vision tower untouched (BF16); evaluated as a text model.

## KV-cache quantization (calibrated scales included)

This checkpoint carries per-layer `k_scale` / `v_scale` in
`model-kv-scales.safetensors`, with `kv_cache_scheme` recorded under
`quantization_config` — the on-disk contract llm-compressor emits and vLLM reads
in `BaseKVCacheMethod`. With no scales present vLLM falls back to 1.0.

```
--kv-cache-dtype fp8      # 32 KiB/token, from 64 KiB at fp16
```

Qwen3.8-27B is a hybrid stack: only 16 of its 64 layers are `full_attention` and
carry a KV cache at all, so this is 32 scalars in a 4 KB file.

Scales were observed on **this quantized model**, at the point vLLM caches K/V
(post-RoPE, at the attention dispatch), over a 2353-document / 3,713,778-token
corpus, taking a running absmax per layer (`memoryless_minmax`,
llm-compressor's default KV observer); `k_scale = amax/448`.

Corpus size matters more than it looks, and only for V. absmax is a tail
statistic: it sits flat, then jumps when a rare document appears, so a small
calibration set looks converged when it is not. A 48-document pass
under-estimated V by up to 1.9x and a 231-document pass was still up to 1.5x
low; at this size **no tensor set a new maximum over the final 50 documents**.
Under-estimating clips exactly the outliers a cache most needs to keep. K is
converged by ~100 documents and needs none of this; V's range also grows ~13x
from the first attention layer to the last, which is why the scales are
per-layer rather than global.
