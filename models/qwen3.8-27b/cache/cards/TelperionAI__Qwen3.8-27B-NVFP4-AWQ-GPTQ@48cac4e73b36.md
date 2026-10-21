---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - nvfp4
  - fp4
  - awq
  - gptq
  - llm-compressor
  - compressed-tensors
  - vllm
library_name: transformers
---

# Qwen3.8-27B-NVFP4-AWQ-GPTQ

Mixed-precision **NVFP4** quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B),
built with [llm-compressor](https://github.com/vllm-project/llm-compressor) using
**AWQ activation-aware scaling followed by GPTQ**, with an `imatrix_mse` observer.

**24.7 GB.** At the same size as a plain NVFP4 build, it cuts high-confidence damage by
roughly a third, and it is the most accurate NVFP4 checkpoint of this model we have measured.

## Recipe

| component | precision |
|---|---|
| `mlp.{gate,up,down}_proj`, layers 0–55 | **NVFP4** (4-bit, group-16, FP8-e4m3 scales → 4.5 effective bits) |
| `mlp.{gate,up,down}_proj`, layers 56–63 | FP8 e4m3 (dynamic) |
| `self_attn.{q,k,v,o}_proj` | FP8 e4m3 (dynamic) |
| `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (GDN) | FP8 e4m3 (dynamic) |
| `lm_head`, `embed_tokens`, all norms, GDN state params, vision tower | **BF16** |

Two passes, in order:

1. **AWQ** — per-input-channel scaling on `post_attention_layernorm → {gate_proj, up_proj}`
   and `up_proj → down_proj`. Gate and up share one input, so the reciprocal scale folds
   into the norm weights: **the accuracy gain costs zero bytes and zero throughput.**
   The scales merge into weights entirely, so unlike rotation-based methods (QuIP/SpinQuant)
   this checkpoint still runs under tensor parallelism.
2. **GPTQ** on every quantized module (`actorder="static"`, `dampening_frac=0.01`).

Calibration: 1024 sequences × 1024 tokens of a balanced Nemotron-v2 blend
(25% code, 25% math, 20% STEM, 20% chat, 10% multilingual).

`lm_head` and `embed_tokens` are left in BF16 — matching Qwen's own official FP8 release,
which does the same.

## Benchmarks

Measured against the BF16 base model on 142,727 tokens of self-distilled thinking-mode
output, plus 200 free greedy generations. vLLM 0.27.1, TP=2, 2×B300.

| checkpoint | size ↓ | top-1 ↑ | near-tie ↓ | moderate ↓ | confident ↓ | certain ↓ | divmed ↑ | tok/s ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.8-27B-FP8` *(8-bit ref)* | 30.9 GB | 96.15% | 22.70% | 3.48% | 1.45% | 0.08% | 47 | 8711 |
| **this model (NVFP4+AWQ)** | 24.7 GB | **93.44%** | **33.86%** | **7.74%** | **2.69%** | **0.19%** | **29** | 10680 |
| `RadixArk/Qwen3.8-27B-NVFP4` | **21.9 GB** | 90.23% | 43.80% | 14.49% | 3.29% | 0.70% | 11 | **11436** |
| `unsloth/Qwen3.8-27B-NVFP4` | 23.4 GB | 91.75% | 40.12% | 10.32% | 3.91% | 0.25% | 19 | 11069 |

Bold marks the best value in each column **among the FP4 checkpoints**; the FP8 row is a
reference at a different precision and size class, so it is excluded from the comparison.
All sizes are on-disk tensor bytes and **include the ~0.85 GB BF16 MTP head**, which every
checkpoint in this table ships. Subtract ~0.85 GB for a no-MTP comparison.

**Columns.** `top-1` is raw argmax agreement with BF16. The four bucket columns are
*disagreement* rates, split by how confident the base model was at that position
(top1−top2 logprob margin): `near-tie` <0.5, `moderate` 0.5–2, `confident` 2–5,
`certain` >5. **Only `confident` and `certain` represent real damage** — a flip where
the base model itself was nearly tied is numerical noise, not a quality loss.
`divmed` is the median token index at which free greedy generation first diverges
from BF16 (higher is better).

**Perplexity is deliberately excluded.** On this comparison it is anti-correlated with
quality — the checkpoint with the best perplexity (`RadixArk`, −1.75%) has the worst
`certain`-bucket damage of any arm measured (0.70%, 3.7× this model's). Do not rank
FP4 checkpoints of this model by perplexity.

In an internal ablation, removing the AWQ pass and keeping everything else identical
raises `confident` damage from 2.69% to 3.97% — so AWQ closes **about half** of the
gap to FP8, at **no size or speed cost**.

## Usage

```python
from vllm import LLM
llm = LLM("selimaktas/Qwen3.8-27B-NVFP4-AWQ-GPTQ", tensor_parallel_size=2)
```

Requires a Blackwell-class GPU for native NVFP4, and vLLM with `compressed-tensors`.

## Speculative decoding (MTP)

The model's **MTP (multi-token prediction) head is included**, in BF16, and works with
vLLM's `mtp` speculative decoding:

```python
from vllm import LLM
llm = LLM("TelperionAI/Qwen3.8-27B-NVFP4-AWQ-GPTQ", tensor_parallel_size=2,
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
- **Single evaluation corpus.** All numbers come from one self-distilled corpus. The
  margins over the public NVFP4 checkpoints are large and statistically solid, but the
  comparison has not been replicated on a second distribution.
- Vision tower is untouched (BF16); this was evaluated as a text model.

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
