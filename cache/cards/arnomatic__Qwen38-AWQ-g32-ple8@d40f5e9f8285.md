---
license: apache-2.0
language:
- en
- de
tags:
- qwen3.8
- awq
- moe
- strix-halo
- gfx1151
- vllm
- expert-parallel
base_model:
- Qwen/Qwen3.8-Flash-Next
- leoncca/Qwen3.8-Flash-Next-AWQ-g32
---
# Qwen38-AWQ-g32-ple8 — Qwen3.8-Flash-Next AWQ-g32 (asym) + W8A16 PLE, tuned for 2× AMD Strix Halo

**What this is:** a quality-corrected quantization of `Qwen3.8-Flash-Next`
for two-node serving on 2× Strix Halo (gfx1151, 124 GiB UMA each):

- **Experts:** AWQ **g32 asymmetric** (`moe_wna16` GEMM layout), from
  [leoncca/Qwen3.8-Flash-Next-AWQ-g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-AWQ-g32).
  Asym expert quantization fixed systematic word-form degradation we
  measured with the symmetric (RTN/g128) production build — reasoning
  became ~3× shorter and targeted, "behaves like a normal cloud model".
- **PLE N-gram table:** W8A16 g32, **baked from the official FP8 PLE
  checkpoint** (`Qwen/Qwen3.8-Flash-Next-FP8`) using the flush convention
  (LSB + offset 128, g32 sym, amax/127). Correlation 0.9996 against the
  reference W8A16 table; max numeric diff 2.29e-04 (int8 noise).
  320,001,536 rows (16 heads × ~20M prime rows, head_dim 160, one PLE
  layer, `ple_layer_ids: [2]`), 128 shards → baked to 12 shards.
- **Everything else** (dense layers, MTP heads, KV scales) untouched from
  the leoncca AWQ base.

**Size:** 10 AWQ shards + 12 PLE shards, ~131 GB on disk.
**Runtime:** ~68 GiB weights per rank (GTT-measured; everything stays packed — the W8A16 N-gram table is dequantized only at lookup) + 8 GiB KV.

## Loading requirements (2-node)

- **TP2 + Expert Parallel** — mandatory, never TP1 (see below).
- `VLLM_PLE_W8A16_EMBED=1` — required for the W8A16-baked PLE table.
- vLLM with Qwen3.8-Flash-Next AMD support + **4 patch files**
  (`container/`, see `container/BUILD.md`).

```bash
export HEAD_IP=<head-ip> WORKER_IP=<worker-ip>
./container/start_ray_2node.sh   # Ray on the Thunderbolt link
./container/serve.sh             # vLLM TP2+EP, ctx 262144, MTP nst=2, port 8012
```

## Performance (2× Strix Halo, TP2+EP, MTP nst=2, batched tokens 2048)

Cold-cache benchmarks (`scripts/bench_vllm.py`-class streaming, unique
seed per run, 09.09. build):

| Prompt tokens | TTFT | Prefill | Generation |
|---|---|---|---|
| 113 | 0.89 s | — | **16.64 tok/s** |
| 5,971 | 12.1 s | **495 tok/s** | 16.53 tok/s |
| 96,993 | 198 s | **489 tok/s** | 16.20 tok/s |

Earlier long-output measurement (5.2k prompt / 2048 output, nst=2):
**12.7–13.4 tok/s** generation. Generation stays essentially flat from
short to ~97k context (−3 %); prefix-cached repeats re-read a ~97k prompt
in seconds (10× vs. uncached — see the sibling PLE8 report's §18 for the
statistical caveats). Load ~212 s, context 262144, KV pool 8 GiB
(~537k tokens).

⚠️ `--max-num-batched-tokens 2048` is **verified** on this build. With
1024 the engine deterministically faults (`gfxhub page fault, TCP client,
PERMISSION_FAULTS 0x3`) at exactly `num_computed_tokens = 103,360`
(= 101 × 1024) on ~104k-token prompts — chunk sizes must be re-tested per
build; the PLE8-era "2048 is fatal" finding does **not** apply here.

## Why TP2 + EP, never TP1

`moe_intermediate_size = 640` splits to 320 columns at TP2 and
`320 % 128 != 0` → vLLM rejects without `--enable-expert-parallel`
(experts are not split along the intermediate dim under EP).
TP1 fills a single node's unified memory (~140 GiB full model, packed,
single node + KV +
activations) and **freezes the machine** — hard reboot territory.

## Sampling

`generation_config.json` carries a single unified sampling block —
`temperature 1.0, top_p 0.95, top_k 20` (verified against the checkpoint;
no per-mode variants, no `presence_penalty`). vLLM picks these up
automatically — no flags needed.

## Reproducibility

- `container/bake_awq_ple8.py` — bakes the W8A16 PLE table from the FP8
  source (what created this checkpoint).
- `SHA256SUMS` — integrity of all files.
- Checkpoint geometry/audit: `MODEL_PROVENANCE.json`,
  `PUBLIC_RELEASE_AUDIT.json`, `EXPERT_COVERAGE.json`, `VALIDATION.json`.

## Credits & lineage

- Base quantization: [leoncca/Qwen3.8-Flash-Next-AWQ-g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-AWQ-g32) (AWQ g32 asym)
- PLE source: [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) (official)
- Reference W8A16 PLE table: aixiaoma/Qwen3.8-Flash-Next-W4A16 lineage
- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- License: Apache-2.0 (following the base model)
