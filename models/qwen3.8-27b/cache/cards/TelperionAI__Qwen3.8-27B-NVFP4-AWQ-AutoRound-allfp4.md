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

# Qwen3.8-27B-NVFP4-AWQ-AutoRound-allfp4

Mixed-precision **NVFP4** quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B),
built with [llm-compressor](https://github.com/vllm-project/llm-compressor) using
**AWQ activation-aware scaling followed by AutoRound**.

**23.8 GB** — all 64 MLP layers at 4 bits, with no FP8 last-band protection, and
*statistically indistinguishable* from the 24.7 GB variant that keeps it. 7 GB smaller than
FP8 and ~25% faster.

## Recipe

| component | precision |
|---|---|
| `mlp.{gate,up,down}_proj`, **all 64 layers** | **NVFP4** (4-bit, group-16, FP8-e4m3 scales → 4.5 effective bits) |
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
| **this model (all-FP4)** | 23.8 GB | **93.05%** | **36.65%** | 8.70% | **2.00%** | **0.16%** | **22** | 10927 |
| sibling with FP8 MLP 56–63 | 24.7 GB | 93.38% | 34.18% | 8.67% | 1.85% | 0.17% | 28 | 10590 |
| `RadixArk/…-NVFP4-BF16-LMHead` *(identical precision map, RTN)* | 23.8 GB | 91.06% | 41.04% | 12.53% | 3.23% | 0.71% | 12 | **11380** |
| `unsloth/Qwen3.8-27B-NVFP4` | 23.4 GB | 91.75% | 40.12% | 10.32% | 3.91% | 0.25% | 19 | 11069 |

Bold marks the best value in each column **among the ~23.8 GB checkpoints**; FP8 and the
24.7 GB sibling are shown for reference.
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

### The FP8 last band is not needed

Most NVFP4 recipes for this model protect MLP layers 56–63 at FP8. This build removes that,
putting all 64 MLP layers at 4 bits. Against the otherwise-identical sibling that keeps it:
`confident` 1.85% → 2.00% (paired McNemar z = 1.88, **not significant**) and `certain`
0.17% → 0.16% (z = 0.20, **not significant**). The protection costs 0.87 GB and 3%
throughput and buys nothing measurable here.

It *is* load-bearing for round-to-nearest builds, which is presumably why it became
convention: among RTN checkpoints of this model, those protecting the band sit at 0.25%
`certain` and those that do not sit at 0.70%.

### What the algorithm contributes

`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` has a **byte-identical precision map** to this
model — same 23.8 GB, NVFP4 on all 64 MLP layers, FP8 attention and GDN, BF16 `lm_head` —
differing only in using round-to-nearest:

| | RTN | AWQ + AutoRound (this model) | z |
|---|---:|---:|---:|
| confident disagreement | 3.23% | **2.00%** | 12.72 |
| certain disagreement | 0.71% | **0.16%** | **15.18** |
| median greedy divergence | 12 tok | **22 tok** | — |

At identical size and precision, the algorithm is worth a **4.4×** reduction in the worst
damage bucket.

**AWQ** rescales per input channel; because `gate_proj` and `up_proj` share one input, the
reciprocal folds into `post_attention_layernorm` at zero size cost. **AutoRound** optimises
rounding and clipping with SignSGD against a block-wise reconstruction loss rather than
GPTQ's per-layer weight-MSE proxy.

## Usage

```python
from vllm import LLM
llm = LLM("TelperionAI/Qwen3.8-27B-NVFP4-AWQ-AutoRound-allfp4", tensor_parallel_size=2)
```

Requires a Blackwell-class GPU for native NVFP4, and vLLM with `compressed-tensors`.

## Speculative decoding (MTP)

The model's **MTP (multi-token prediction) head is included**, in BF16, and works with
vLLM's `mtp` speculative decoding:

```python
from vllm import LLM
llm = LLM("TelperionAI/Qwen3.8-27B-NVFP4-AWQ-AutoRound-allfp4", tensor_parallel_size=2,
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
`model-kv-scales.safetensors`, with `kv_cache_scheme` under
`quantization_config` — the on-disk contract llm-compressor emits and vLLM reads
in `BaseKVCacheMethod`. Without them vLLM falls back to a scale of 1.0.

```
--kv-cache-dtype fp8      # 32 KiB/token, from 64 KiB at fp16
```

**Provenance:** these scales were measured on
[Qwen3.8-27B-NVFP4-AWQ-AutoRound](https://huggingface.co/TelperionAI/Qwen3.8-27B-NVFP4-AWQ-AutoRound), a different quantization of
the same base model, over a 2,353-document / 3.7M-token corpus — the size at
which the post-RoPE V absmax stops moving. K/V range is a property of the model
rather than of the weight format (per-layer amax agrees to a median ratio of
1.0002 across our trellis / FP8 / INT4 / NVFP4 builds), so they transfer. They
were **not** measured on this checkpoint specifically; cross-build agreement is
excellent at the median but ranges 0.88–1.36 in the tails.
