---
license: other
license_name: glm-5.3-license
license_link: https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE
base_model: zai-org/GLM-5.3
base_model_relation: quantized
quantized_by: bot-lab-21
pipeline_tag: text-generation
library_name: vllm
tags: [glm, glm-5.3, moe, compressed-tensors, int4, int8, gptq, pollard, vllm, tensor-parallel, dgx-spark, gb10, mtp, speculative-decoding]
---

# GLM-5.3-Int4-Int8Mix-Pollard-GPTQ — int4/int8 mixed-precision GLM-5.3 (compressed-tensors), Pollard-method GPTQ cook

A production-serving quantization of **[zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)** (744B-parameter MoE, ~40B
active; 78 layers, 256 routed experts + 1 shared, top-8; MLA attention with DeepSeek-style sparse indexer; 1 MTP layer).
Cooked from the **native BF16 checkpoint** (not from a pre-quantized body) with **full-Hessian GPTQ error feedback** and
**measured per-tensor sensitivity** (the [Pollard Weights](https://github.com/WestWaters/pollard-weights) *method* — measure sensitivity, allocate under a byte budget, Hessian error feedback — with a vendored, extended `pollard_gptq` core), in-domain calibration, and every routed expert calibrated
on its own routed tokens. Format is plain `compressed-tensors` `pack-quantized` — it loads in **stock vLLM ≥ 0.28** with no
custom code.

This is the model we serve in production on an 8 × DGX Spark (GB10) cluster at **1M context with NVFP4 KV cache**. Numbers below
are measured on that deployment; if you run 8 × 128 GB-class GPUs or another TP8 unified-memory cluster you should see the same.

## What is in the box

| item | precision | notes |
|---|---|---|
| routed experts (75 MoE layers × 256) | **int4**, g128, symmetric, bf16 scales | GPTQ with per-expert routed Hessians |
| attention (`q_a`, `q_b`, `kv_a`, `kv_b`, `o`), all 78 layers | **int4**, g128 | "fast" allocation — measured to cost ~0.05 ppl vs int8 while saving ~6.5 GB of weight reads per decoded token |
| shared expert, dense MLP (layers 1–2), `lm_head` | **int4**, g128 | same |
| layer 0, routers (`mlp.gate`), DSA indexers, norms, embeddings | **bf16** | passthrough |
| MTP layer (layer 78) | int4 experts / int4 attention, GPTQ | present in the checkpoint; usable as the speculative draft with `method: mtp` |

Total: **396 GB** on disk (240 shards). `allocation.json` lists the per-tensor bit choice; `config.json` carries the
`compressed-tensors` groups and ignore rules.

## Measured quality (same battery, same fleet; "v2" = the int4/int8 round-to-nearest production quant it replaced)

| gate | v2 (RTN, attention int8) | **v4-fast (this repo)** |
|---|---|---|
| perplexity, fixed held-out texts (lower is better) | 4.838 | **4.82** |
| HumanEval+ pass@1 base / plus | 0.976 / 0.927 | **0.963 / 0.945** |
| MBPP+ pass@1 base / plus | 0.963 / 0.825 | **0.971 / 0.828** |
| needle-in-haystack, 672K planted in 900K | clean | **clean** |
| mixed-workload throughput, single stream / 4 streams (tok/s, TP8 GB10, MTP k=5) | 34.5 / 79 | **40.0 / 85.2** |
| speculative acceptance, model-author MTP layer, same-day replay of real traffic, k=5 (see *Speculative draft*) | RTN int8 draft 1.50 | **in-checkpoint GPTQ int4 layer 78: 1.59 · `mtp-draft-gptq-int8/`: 1.84** |

Reading: swapping round-to-nearest for Hessian error feedback at identical bytes improved next-token prediction and code pass
rates; taking attention and the shared expert from int8 to int4 (−20 % weight bytes per token) cost 0.05 ppl and was a wash on
code. The speed gain is the byte saving: GB10 decode is memory-bandwidth-bound and, in the int8 build, attention read more bytes
per token than the experts did.

## Recipe (short)

- **Source:** BF16 GLM-5.3, 1.5 TB. Never start from a pre-quantized body.
- **Calibration:** 384 rows × 2048 tokens (786K tokens) of real assistant traffic — agentic tool-call sessions, chat, code,
  long documents — rendered through the chat template. Held-out evaluation texts excluded by content hash.
- **Quantizer:** GPTQ (full damped Hessian, Cholesky, column-wise error propagation), symmetric int4/int8 g128 with bf16
  scales, packed to `compressed-tensors` `pack-quantized`. Routed experts each get a Hessian from the tokens the router actually
  sent them (experts with fewer than 512 calibration tokens fall back to RTN — only a handful in layers 3–5).
- **Allocation:** measured, not assumed. Every "candidate" tensor group (attention, shared experts, dense MLP, `lm_head`, MTP)
  was emitted at both int4 and int8; the Hessian-weighted reconstruction error ranked them, and the final choice was gated on the
  end-to-end battery above. This build (our internal lane "v4-fast") = all candidates at int4.
- **Cook:** band-parallel across 10 single-GPU nodes (each node owns a contiguous band of layers and resumes from the exact
  residual stream at its band start), ~2 h wall for the quantization proper. Method write-up and Pollard-related findings:
  see the linked notes in the discussion tab.

## Serving

Stock vLLM (≥ 0.28), tensor-parallel 8, MLA sparse attention backend, NVFP4 or fp8 KV cache. The MTP layer is in the checkpoint:

```bash
vllm serve <this-repo> --tensor-parallel-size 8 --trust-remote-code \
  --max-model-len 900000 --kv-cache-dtype fp8 \
  --speculative-config '{"model":"<this-repo>","method":"mtp","num_speculative_tokens":3}'
```

Per-node memory at TP8: ~50 GB weights + draft, leaving ~45 GB of KV on a 121 GB GB10 (≈ 900K–1M context at 41–57 KB/token).
On our fleet we run a fork with GB10-specific kernels; the numbers in the table were taken there. Stock vLLM 0.28 serves this
checkpoint correctly (same outputs on our correctness probe) but slower until its GB10 paths catch up.

## Speculative draft (`mtp-draft-gptq-int8/`)

The checkpoint's own layer 78 works as the MTP draft (point `speculative-config.model` at this repo). The subfolder
`mtp-draft-gptq-int8/` is a **better** draft: the same model-author MTP layer quantized with GPTQ at **int8 attention / int4
experts** (the allocation the production draft used, Hessian-rounded), packaged standalone (layer 78 + embeddings + head + final
norm, 13 GB, `compressed-tensors`). The same files are also published standalone as
[bot-lab-21/GLM-5.3-Int4-Int8Mix-Pollard-GPTQ-MTP-Draft](https://huggingface.co/bot-lab-21/GLM-5.3-Int4-Int8Mix-Pollard-GPTQ-MTP-Draft)
so `speculative-config.model` can be an HF id. Measured on the same day, same body, same replay of real traffic, k = 5 adaptive:

| layer-78 draft | accepted draft tokens / step |
|---|---|
| round-to-nearest int8 attention (our previous production draft) | 1.50 |
| this checkpoint's layer 78 (GPTQ, int4 attention) | 1.59 |
| **`mtp-draft-gptq-int8/` (GPTQ, int8 attention)** | **1.84** |

Reading: Hessian rounding on the draft layer alone is worth +22 % acceptance at identical bits, and a draft quantized like its
target beats an unquantized bf16 copy of the same weights (1.64). This probe drifts ~0.3 day to day on a fixed config, so
compare within a session only.

```bash
# download the repo, then:
vllm serve <repo-dir> --tensor-parallel-size 8 --trust-remote-code --max-model-len 900000 --kv-cache-dtype fp8 \
  --speculative-config '{"model":"bot-lab-21/GLM-5.3-Int4-Int8Mix-Pollard-GPTQ-MTP-Draft","method":"mtp","quantization":"compressed-tensors","num_speculative_tokens":3}'
# (or "model":"<repo-dir>/mtp-draft-gptq-int8" from the downloaded repo)
```

## Files

`model-00001-of-00240.safetensors … model-00240-of-00240.safetensors`, `model.safetensors.index.json`, `config.json`,
`generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` (GLM-5.3 text template),
`allocation.json` (per-tensor bits); `mtp-draft-gptq-int8/` (standalone MTP draft, 4 shards + config + tokenizer).

## License / provenance

Derivative of zai-org/GLM-5.3, distributed under the GLM-5.3 License (original copyright © 2026 Z.AI). Quantization and
evaluation by bot-lab-21. Quantizer core derived from Pollard Weights' `pollard_gptq` (Apache-2.0) with symmetric packing and
per-expert routed Hessians added. Calibration text is private traffic and is not distributed. Produced with the help of an AI
assistant operating the cluster; all numbers were measured on the deployed model.
