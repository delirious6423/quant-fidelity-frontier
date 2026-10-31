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

Use [orinfer v0.3.0](https://github.com/iMMIQ/orinfer/releases/tag/v0.3.0) for Linux aarch64 and the matching separate A2 execution package:

```
c809979d55ac308b2e874d6b06dbac97b2154ce50ae43fc8790605647452b43c
```

This repository contains model data and attribution. Operators, plans and native libraries are distributed in the separate execution archive. Version 0.3.0 supports native A2 installation with the Rust CLI, without Python or an NPU context:

```bash
gh release download v0.3.0 -R iMMIQ/orinfer \
  -p 'orinfer-v0.3.0-linux-aarch64.tar.xz' \
  -p 'orinfer-exec-deepseek-v41-flash-1017-w4a8-ascend910b-v0.3.0.tar.xz' \
  -p SHA256SUMS
sha256sum --ignore-missing -c SHA256SUMS
tar -xf orinfer-v0.3.0-linux-aarch64.tar.xz
export PATH="$PWD/orinfer-v0.3.0-linux-aarch64/bin:$PATH"
orinfer package install orinfer-exec-deepseek-v41-flash-1017-w4a8-ascend910b-v0.3.0.tar.xz
hf download immiq/DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8 --local-dir ./DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8

# Run once for full content verification; normal loading performs no asset SHA256 scans.
orinfer validate-model ./DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8
source /usr/local/Ascend/cann/set_env.sh
export HCCL_OP_EXPANSION_MODE=AIV HCCL_NPU_SOCKET_PORT_RANGE=auto
orinfer serve ./DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8 \
  --model deepseek-v41-flash-1017-w4a8 --mtp-drafts 0 --drain-timeout-ms 120000
```

For the fully resident host-table configuration used in the reported tests, set `ORINFER_A2_HOST_TABLE_MIB=212000` and `ORINFER_A2_HOST_ROW_CACHE_MIB=2048` before starting the service. The table budget requires about 211 GiB of host memory in addition to other process memory. Use the shown 120-second drain budget and a longer external termination grace period. Model readiness is HTTP 200 from `/health`. The CLI configures CANN termination-signal ownership; direct Rust API callers must set `ASCEND_COREDUMP_SIGNAL=none` before startup.

These safetensors contain orinfer physical layouts for the tested text path. See the [native model package documentation](https://github.com/iMMIQ/orinfer/blob/v0.3.0/docs/model-packages.md).

## Physical layout

- **178 safetensors shards**, **5,505 physical tensors**, **510,089,196,120 container bytes** (about **475.06 GiB**).
- Files range from **1.23 to 3.75 GiB**. GPU files group one rank and four consecutive layers; shared parameters and CPU embedding have separate shards. Engram files group layer, hash head and contiguous half-head row ranges.
- Expert W4 banks use transposed-NZ packed bytes `[E,K/64,N/16,16,32]`. The ACL logical view is `[E,K,N]`; scales, assist values and output-channel order are unchanged. All dynamic row counts read one resident W4 bank.
- Engram records contain original group-32 INT8 values immediately followed by FP32 scale bytes, without extra row padding. The index `row_records` extension fixes `i8-f32-g32-v1` and ordered global row segments.
- Online loading reads prepared physical data directly. No full-bank transpose or requantization is performed at startup.
- Runtime KV uses group-64 INT8 with FP16 scales. KV and other sequence state remain request-private. The execution package declares up to 262,144 context tokens. Later bounded stress checks exercised lengths through 262,080 input tokens, B32 admission, overload recovery and clean shutdown; latency SLO acceptance is workload-specific.

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
