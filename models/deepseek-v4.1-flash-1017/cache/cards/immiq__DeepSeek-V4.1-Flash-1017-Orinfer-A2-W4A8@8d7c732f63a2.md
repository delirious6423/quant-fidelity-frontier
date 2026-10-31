---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
language:
- zh
- en
tags:
- orinfer
- ascend
- ascend910b
- w4a8
- int4
- moe
- safetensors
inference: false
---

# DeepSeek-V4.1-Flash-1017 — Orinfer A2 W4A8

Native **orinfer** model data for **Altus A2 / Ascend 910B**, complete 40-layer text inference with TP8.
The original signed INT4 expert codes, per-channel scales, assist values and mixed-precision parameters are preserved. This release changes physical storage layouts without requantizing those weights.

## Runtime and download

Requires orinfer [92743c42674fd0f0ab20668c41a7cf6898805a95](https://github.com/iMMIQ/orinfer/tree/92743c42674fd0f0ab20668c41a7cf6898805a95) or a compatible engine, and the matching **separate execution package**:

```
ae00352aadb6f647a0969527286862080d354fc34d1a28af45d8d5c2ecd95b4b
```

This repository contains weights, configuration, tokenizer/chat assets, physical tensor index, checksums and model attribution. Operators, execution plans and compiled libraries are distributed separately. Obtain the matching A2 execution cache from your orinfer deployment/build; the general CUDA execution-package installer is not used for this native A2 package. See [native model package documentation](https://github.com/iMMIQ/orinfer/blob/e16afb99d7e65d233304a975ad882087b69cb504/docs/model-packages.md).

```bash
hf download immiq/DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8 --local-dir ./DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8

# Set this to the directory containing the separately supplied digest directory.
export ORINFER_EXECUTION_CACHE=/path/to/a2-execution-cache
orinfer serve ./DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8 \
  --model deepseek-v41-flash-1017-w4a8 --mtp-drafts 0 --drain-timeout-ms 120000
```

For fully resident host tables, use the shown 120-second drain budget and a longer external termination grace period. Model readiness is HTTP 200 from `/health`. The service CLI configures CANN termination-signal ownership; direct Rust API callers must set `ASCEND_COREDUMP_SIGNAL=none` before process startup.

These safetensors contain orinfer physical layouts. They are loaded through the matching native A2 provider. The original model configuration is retained; this package exports the tested **text path** and does not include a validated image-inference deployment.

## Physical layout

- **178 safetensors shards**, **5,505 physical tensors**, **510,089,196,120 container bytes** (about **475.06 GiB**).
- Files range from **1.23 to 3.75 GiB**. GPU files group one rank and four consecutive layers; shared parameters and CPU embedding have separate shards. Engram files group layer, hash head and contiguous half-head row ranges.
- Expert W4 banks use transposed-NZ packed bytes `[E,K/64,N/16,16,32]`. The ACL logical view is `[E,K,N]`; scales, assist values and output-channel order are unchanged. All dynamic row counts read one resident W4 bank.
- Engram records contain original group-32 INT8 values immediately followed by FP32 scale bytes, without extra row padding. The index `row_records` extension fixes `i8-f32-g32-v1` and ordered global row segments.
- Online loading reads prepared physical data directly. No full-bank transpose or requantization is performed at startup.
- Runtime KV uses group-64 INT8 with FP16 scales. KV and other sequence state remain request-private. The execution package declares up to 262,144 context tokens; this release's HTTP checks cover 32K plus a 4K continuation and prefix reuse.

`release-manifest.json` contains file and tensor SHA256 values. `SHA256SUMS` supports file verification:

```bash
sha256sum -c SHA256SUMS
```

## Validation and performance

Numerical checks cover lossless nibble/address mapping, FP32 scale-byte preservation, real up/down W4 BF16 outputs, changed INT8 inputs including boundary values, changed/empty/full routing groups, output tails and actual Graph replay. Host record checks cover shard boundaries, reverse/repeated queries, resident lifetime and cache reuse. Full native requests cover 2K, 2049 and B32 fixed-history scenarios. Default high-performance HCCL is used; cross-run bitwise equality is not a requirement of that communication mode.

Observed on the project A2 TP8 machine with MTP off. Mini service settings: 32 active-request limit, 128 front-end batch-token budget, 8192 MiB prefix cache, `ORINFER_A2_HOST_TABLE_MIB=212000`, `ORINFER_A2_HOST_ROW_CACHE_MIB=2048`, and `HCCL_OP_EXPANSION_MODE=AIV`; native packets have 2048-row capacity. CPU affinity is machine-specific.

| Native warm full-model scenario | Baseline | This layout |
| --- | ---: | ---: |
| 2K prefill, input tokens/s | 7,986 | 8,831 |
| 2049 prefill, input tokens/s | 6,143 | 6,547 |
| Single-stream decode, tokens/s | 51.3 | about 53 |
| B32 aggregate decode, tokens/s | 263.9 | 265.1 |

Native warm timings exclude load, admission/reset and Graph preparation. A single fixed **240-second Mini-Weka** replay completed **27 profiling requests vs 20** and decoded **5,056 tokens vs 4,085**; prefill progress was effectively unchanged. These are fixed-budget progress observations, not an aggregate TPS or statistical repeatability claim. Same-format full-resident service startup was **248 seconds vs 258 seconds** in one sequential comparison with cache/clock settings preserved. Combined ACLNN workspaces use **160 MiB less per rank**.

Full official-reference task/NLL quality acceptance was deferred in this release cycle; layout/numerical and request-path validation are the scope reported here. No one-hour replay was run.

## Attribution

Model weights retain the upstream MIT license in `LICENSE`. Original checkpoint attribution and documentation are preserved in `SOURCE-README.md`. The orinfer engine is a separate LGPL-3.0-or-later project; engine/operators are not included in this weight repository.
