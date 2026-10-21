---
license: other
license_name: glm-5.3
license_link: https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE
base_model:
- zai-org/GLM-5.3
base_model_relation: quantized
pipeline_tag: text-generation
library_name: transformers
tags:
- compressed-tensors
- int4
- int8
- w4a16
- w8a16
- moe
- glm
- dgx-spark
- gb10
- vllm
---

# GLM-5.3 Int4-Int8Mix (743B) — the quant that fits 4x NVIDIA DGX Spark

> ### ✅ Live — all 282 shards uploaded, fully open (2026-08-29)
> The complete model is here: 282 safetensors shards + config + tokenizer, ~378 GB, **public and ungated**. Everything below is verified on our own hardware. Recipe + launchers + benchmarks: **[github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark)**.

An **Int4-Int8 mixed-precision** quantization of the full **[zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)** (743B total / ~40B active MoE, `glm5_next`, 78 layers, 1M context), in the **`compressed-tensors`** format for **vLLM**. Produced with the QuantTrio-style recipe (data-free, mixed W4A16 / W8A16), it is, as far as we can tell, the **first quant of the big GLM-5.3 that actually serves on four DGX Spark (GB10) nodes** with real KV-cache headroom.

## Why this one fits when the NVFP4s do not

The existing full-GLM-5.3 NVFP4 quants are all ~465 GB because they keep the entire attention stack at bf16. Across 4 Sparks that is ~116 GB/rank, leaving only ~12 GB/rank, which is not enough for the KV pool plus framework, so they OOM in practice.

This quant takes the attention to **Int8** instead of bf16. That single change drops the footprint to **~378 GB** (~94.5 GB/rank at TP4), leaving **~33 GB/rank** for a real KV pool. It is the difference between "loads on paper" and "serves under real traffic."

| | weights | GB/rank (TP4) | KV headroom/rank | serves on 4 Sparks? |
|---|---|---|---|---|
| NVFP4 (attention bf16) | ~465 GB | ~116 | ~12 GB | no (OOM) |
| **This — Int4-Int8Mix** | **~378 GB** | **~94.5** | **~33 GB** | **yes** |

## The recipe (data-free, mixed precision)

Static, symmetric, weight-only, `compressed-tensors` / `pack-quantized`, group size 128. No calibration data. Per-layer map:

- **MoE experts (layers 3-77):** W4A16, group 128 — the bulk of the weights, where the size savings come from.
- **Dense + attention (layers 1-77):** W8A16, group 128 — `self_attn.*`, `mlp.gate_up/gate/up/down`, `shared_experts.*`.
- **MTP block (layer 78):** W8A16, channelwise.
- **Kept at full precision (this is what protects accuracy):** layer 0 entirely, every `mlp.gate` (the MoE router), `self_attn.indexer` / `indexers_proj` (the DSA sparse-attention selector), the MTP `eh_proj`/`enorm`/`hnorm`, and `shared_head.norm` / `shared_head.head` (the LM head).

The trick in one line: **Int4 only the bulky experts, Int8 the dense and attention, and leave the routing, indexer, and head alone.**

## Serving (vLLM, 4x DGX Spark GB10, TP4)

```bash
vllm serve <path-to-this-model> \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --served-model-name glm-5.3 \
  --max-model-len 131072   # raise per your KV pool
```

- **⚠️ GB10 / sm121 needs the sm12x kernel overlays** (the sparse-MLA + fp8-KV path is not in stock vLLM for this arch). Get them from the sibling recipe repo: **[tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s)**. On standard CUDA (H100/A100 etc.), `--quantization compressed-tensors` is all you need.
- `kv_cache_scheme: None` in the config — KV precision is a **serve-time** choice (`--kv-cache-dtype`), not baked into the weights.
- **Speculative decode:** pairs with **[incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2)** (a 4.9 GB DFlash2 drafter) at ~zero extra KV pool — the flagship stack. Native MTP (layer 78) is preserved as a fallback.
- **Long context:** add **Decode Context Parallel** (`--decode-context-parallel-size 4`) to split the KV across all four nodes for a ~600K single-request context (the tradeoff is a small cross-node cost per decode step).

## Verification (what's proven, and what isn't)

Produced by a shard-streaming RTN quantizer (reads one BF16 shard at a time, quantizes with `compressed_tensors`' own `calculate_qparams` / `quantize` / `pack_to_int32`, writes a 1:1 output shard; peak RAM ~10 GiB, runtime **28.2 min**). NOT `llmcompressor.oneshot` — accelerate disk-offload would need ~1.4 TB of scratch this fleet doesn't have.

**Proven:**
- **Structural (fail-closed):** 59,585 source tensors → 58,992 quantized modules → 177,569 output tensors (exact). Group counts **57,600 / 616 / 776** match the pre-quant dry-run exactly (75 layers × 256 experts × 3 = 57,600 ✓). Sacred modules (layer 0, `mlp.gate`, indexer, `lm_head`, MTP norms) all plain BF16, zero packed leaks. 282 shards, none missing.
- **Layout matches QuantTrio byte-for-byte** (verified via HTTP range-reads of their published safetensors headers): expert `down_proj.weight_packed` I32 `[6144,256]`, `weight_scale` BF16 `[6144,16]`, MTP scale BF16 `[6144,1]` — all identical.
- **Numerical round-trip:** int8 group/128 ≈ 0.70% rel error, int4 ≈ 12% (normal for 16 levels), full range used.
- **Serves live:** TP4 across 4× DGX Spark (GB10/sm121), vLLM 0.23.1rc1, weights 95.53 GiB/rank, fp8_ds_mla KV, 202,944-token KV pool @ 200K ctx, health 200. Passed the bat-and-ball coherence trap ($0.05 with correct algebra) — a mis-quantized router or indexer cannot do that.

**Performance (measured — TP4, 200K ctx, `fp8_ds_mla` KV, MTP k=4, thinking off, all four ranks clocking correctly):**

| concurrency | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| **aggregate tok/s** | 12.12 | 21.71 | 28.30 | 33.11 | **46.03** |

Single-stream is ~12 tok/s; **46 is the peak at 6 concurrent streams**. For reference, the GLM-5.2 QuantTrio recipe on the same four nodes reports 32.5 mean / 36 peak.

**NOT yet done (honest):**
- **No 69-scenario eval.** Coherence is proven; quality-vs-fp8 parity is NOT measured. Do not assume parity.
- **DFlash2 speculative decode is not yet enabled** (native MTP is). It is one flag away (`fp8_e4m3` KV, method `dflash`, `num_speculative_tokens` 7) and is the next speed step.

## Credits (what's whose)

We stand on a lot of other people's work here, so to be precise about who did what:

- **Base model:** [zai-org / Z.ai](https://huggingface.co/zai-org/GLM-5.3) — GLM-5.3 (743B / ~40B active MoE). Everything starts here.
- **Quantization format + tooling:** the **`compressed-tensors`** format and **`llm-compressor`** are by **[Neural Magic](https://github.com/neuralmagic) (now Red Hat)**. Mixed-precision weight quantization (some layers 4-bit, some 8-bit) is a general, well-established technique, not any one group's invention.
- **The GLM recipe (the layer map):** curated by **[QuantTrio](https://huggingface.co/QuantTrio)** (e.g. GLM-5.2-Int4-Int8Mix) — specifically the choice of which layers stay Int4 vs Int8 vs full-precision so GLM MoE routing and accuracy hold up. We reproduced that map for GLM-5.3 and verified its coverage before quantizing.
- **Speculative drafter:** **[IncoAI](https://huggingface.co/incoai)** — the DFlash2 drafter used in the serving stack.
- **This artifact** — the first Int4-Int8Mix of the big GLM-5.3, and the first one that serves on **4x NVIDIA DGX Spark** with the DFlash2 + DCP4 stack: **[@tonyd2wild](https://github.com/tonyd2wild)**.

## License

Inherits the [GLM-5.3 license](https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE) from the base model. Quantization changes weights only, not the license terms.
