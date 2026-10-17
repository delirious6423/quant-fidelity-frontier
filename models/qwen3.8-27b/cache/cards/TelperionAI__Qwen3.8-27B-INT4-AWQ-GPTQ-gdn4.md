---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - int4
  - w4a16
  - awq
  - gptq
  - llm-compressor
  - compressed-tensors
  - vllm
library_name: transformers
---

# Qwen3.8-27B-INT4-AWQ-GPTQ (GDN-4bit)

Mixed-precision **INT4 (W4A16)** quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B),
built with [llm-compressor](https://github.com/vllm-project/llm-compressor) using
**AWQ activation-aware scaling followed by GPTQ**.

**22.6 GB, and measurably more faithful to the BF16 model than the official FP8 release**
on high-confidence token agreement — at 8.3 GB less. Runs on anything Turing or newer
(no FP8/FP4 hardware required).

This is the size-optimised variant. A 25.1 GB sibling,
[`Qwen3.8-27B-INT4-AWQ-GPTQ`](https://huggingface.co/TelperionAI/Qwen3.8-27B-INT4-AWQ-GPTQ),
keeps the GDN projections at 8-bit and is more accurate again (0.93% vs 1.15% confident).

## Recipe

| component | precision |
|---|---|
| `mlp.{gate,up,down}_proj`, layers 0–55 | **INT4, group-32, asymmetric** (~4.6 effective bits) |
| `mlp.{gate,up,down}_proj`, layers 56–63 | INT8 W8A16 |
| `self_attn.{q,k,v,o}_proj` | INT8 W8A16 |
| `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (GDN) | **INT4, group-32, asymmetric** |
| `lm_head`, `embed_tokens`, norms, GDN state params, vision tower | BF16 |

Two passes:

1. **AWQ** — per-input-channel scaling on `post_attention_layernorm → {gate_proj, up_proj}`
   and `up_proj → down_proj`. Gate and up share one input, so the reciprocal scale folds
   into the norm weights: zero size and zero throughput cost.
2. **GPTQ** — Hessian-based error compensation, `actorder="static"`, `dampening_frac=0.01`.

Calibration: 924 sequences × 1024 tokens of a balanced Nemotron-v2 blend
(25% code, 25% math, 20% STEM, 20% chat, 10% multilingual).

`lm_head` and `embed_tokens` stay BF16, matching Qwen's own official FP8 release.

## Benchmarks

Against the BF16 base on 142,727 tokens of self-distilled thinking-mode output plus
200 free greedy generations. vLLM 0.27.1, TP=2, 2×B300.

| checkpoint | size ↓ | top-1 ↑ | near-tie ↓ | moderate ↓ | confident ↓ | certain ↓ | divmed ↑ | tok/s ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.8-27B-FP8` *(8-bit ref)* | 30.9 GB | 96.15% | 22.70% | 3.48% | 1.45% | 0.08% | 47 | 8711 |
| **this model (GDN 4-bit)** | 22.6 GB | **95.88%** | **25.26%** | **3.60%** | **1.15%** | 0.13% | **51** | 4716 |
| our 25.1 GB sibling (GDN 8-bit) | 25.1 GB | 96.30% | 22.29% | 3.52% | 0.93% | 0.09% | 48 | 4617 |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` | **21.0 GB** | 94.35% | 33.74% | 5.59% | 1.35% | **0.12%** | 29 | 4787 |

Bold marks the best value in each column **among the ~21–23 GB checkpoints**; the FP8 row and
our own 25.1 GB sibling sit in different size classes and are shown for reference only.

**Columns.** `top-1` is raw argmax agreement with BF16. The four bucket columns are
*disagreement* rates, split by how confident the base model was at that position
(top1−top2 logprob margin): `near-tie` <0.5, `moderate` 0.5–2, `confident` 2–5,
`certain` >5. **Only `confident` and `certain` are real damage** — a flip where the base
model was itself nearly tied is numerical noise. `divmed` is the median token index at
which free greedy generation first diverges from BF16 (higher is better).

**Perplexity is deliberately excluded.** On this model it is anti-correlated with quality —
the FP4 checkpoint with the best perplexity we measured also had the worst `certain`-bucket
damage. Do not rank quantizations of this model by perplexity.

### What the GDN precision costs

Moving the GDN projections from 8-bit to 4-bit is what takes this build from 25.1 GB to 22.6 GB.
It costs `confident` **0.93% → 1.15%** and `certain` **0.09% → 0.13%** — real but modest.
Note the two are not a pure bit-width comparison: at 4-bit the GDN projections also move
into GPTQ's error compensation, where at 8-bit they were plain round-to-nearest, so the
true cost of 4-bit GDN in isolation is likely a little larger than shown.

At matched size against `cyankiwi` (21.0 GB), this build is still ahead on `confident`
(1.15% vs 1.35%) and level on `certain` — so the AWQ+GPTQ recipe, not the extra bits,
accounts for most of the margin.

### Why INT4 rather than NVFP4

At ~4.6 effective bits, INT4 group-32 **asymmetric** gives 16 uniformly-spaced levels plus
a per-group zero point. NVFP4 gives 8 non-uniform FP4 levels at 4.5 bits with no zero point.
On this model the zero point matters — MLP weight distributions are skewed — and the same
recipe in NVFP4 lands at 1.85% confident versus 0.93% here.

NVFP4's advantage is **hardware**, not numerics: on Blackwell it decodes at 10590 tok/s
against 4617 here, because native NVFP4 tensor-core paths beat Marlin INT4. Pick NVFP4 if
you are throughput-bound on Blackwell; pick this if you want fidelity, or you are on
Ampere/Ada where FP4 and FP8 do not exist.

## Usage

```python
from vllm import LLM
llm = LLM("TelperionAI/Qwen3.8-27B-INT4-AWQ-GPTQ-gdn4", tensor_parallel_size=2)
```

W4A16 needs only compute capability ≥ 7.5 (Turing), so this runs on 3090/4090/A100/H100
as well as Blackwell. Throughput above was measured on B300 and is not representative of
Ampere, where Marlin INT4 is comparatively better tuned.

## Speculative decoding (MTP)

The model's **MTP (multi-token prediction) head is included**, in BF16, and works with
vLLM's `mtp` speculative decoding:

```python
from vllm import LLM
llm = LLM("TelperionAI/Qwen3.8-27B-INT4-AWQ-GPTQ-gdn4", tensor_parallel_size=2,
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

- **22.6 GB is still tight on a single 24 GB card** once KV cache is accounted for.
  Quantizing `lm_head` would free a further ~1.3 GB but was not measured here.
- **Throughput on Blackwell is ~2.3× below NVFP4.** This checkpoint trades speed for fidelity.
- **Single evaluation corpus.** All numbers come from one self-distilled corpus. The margins
  over FP8 and cyankiwi are statistically solid but have not been replicated on a second
  distribution, nor on downstream task benchmarks.
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
[Qwen3.8-27B-INT4-AWQ-GPTQ](https://huggingface.co/TelperionAI/Qwen3.8-27B-INT4-AWQ-GPTQ), a different quantization of
the same base model, over a 2,353-document / 3.7M-token corpus — the size at
which the post-RoPE V absmax stops moving. K/V range is a property of the model
rather than of the weight format (per-layer amax agrees to a median ratio of
1.0002 across our trellis / FP8 / INT4 / NVFP4 builds), so they transfer. They
were **not** measured on this checkpoint specifically; cross-build agreement is
excellent at the median but ranges 0.88–1.36 in the tails.
