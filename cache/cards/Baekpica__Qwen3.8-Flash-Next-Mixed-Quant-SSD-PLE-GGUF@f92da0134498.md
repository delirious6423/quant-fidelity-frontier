---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/f5d08274bafd880402bd16f5e3e6c514136ec06c/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - gguf
  - mixed-quant
  - qwen4exp
  - qwen3.8-flash-next
  - dgx-spark
  - ds4
  - ssd-offload
---

# Qwen3.8-Flash-Next Mixed-Quant SSD-PLE GGUF

> **Two accelerator-resident precision variants are published with a shared BF16 SSD-PLE sidecar; an [official FP8 PLE sidecar](#choose-a-ple-sidecar-bf16-or-fp8) is also available.** [`MQ-Q6-SSD-PLE-BF16`](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF/tree/main/MQ-Q6-SSD-PLE-BF16) is the original Q5/Q6 recipe; [`MQ-Q5-SSD-PLE-BF16`](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF/tree/main/MQ-Q5-SSD-PLE-BF16) lowers the `Q5_K` and `Q6_K` targets by one tier, freeing **13.4082 GiB** of resident-memory headroom while retaining the optimized `Q5_0` down
>
> **Native still-image input is also verified end to end** across OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages.
>
> Identical image-bearing prompts can reuse live recurrent checkpoints and disk-KV records through decoded-pixel cache identity; different pixels remain isolated.
>
> **The Q5 artifact also completed an exact 262,144-token direct prefill at 277.06 tok/s**, with all 248,320 final logits finite and the independently recomputed argmax matching the recorded result. An instrumented incremental sweep measured 277.49 tok/s token-weighted prefill and the SSD-PLE latency distribution reported below.
>
> **[`ds4-dfm-rs@ccd2d39`](https://github.com/Baekpica/ds4-dfm-rs/commit/ccd2d394d6be720b58c71e8ee408d308e0717a33) also adds a one-bank 524,288-token Qwen YaRN path.** Its staged DGX Spark run used the same-layout Uncensored sibling Q5+Sidecar artifact and completed an HTTP 200 request with 524,240 prompt tokens plus one generated token at 215.4 prefill tok/s, with zero memory-census faults. This verifies the shared 512K runtime path, not this base artifact's long-context quality, multi-bank serving, MTP decode, or sustained throughput.
>
> **Update 2026-09-07:** three more prefill rounds on [`ds4-dfm-rs@6e036c4`](https://github.com/Baekpica/ds4-dfm-rs/commit/6e036c4) ([`docs/qwen38-prefill-2026-09-07.md`](https://github.com/Baekpica/ds4-dfm-rs/blob/1eb3773/docs/qwen38-prefill-2026-09-07.md)): the weighted SwiGLU is quantized straight into the fused expert-down's operands, the fused QSA attention runs four barriers per tile with conflict-free partial stores, and the SSD-PLE gather leases 16,384-row tiles under one lock. Same-hour cold single-shot `ds4-bench` medians of three against `main` `ef37468` on this Q5+Sidecar artifact: **1,429.9 → 1,504.6 tok/s at 8,192 tokens (+5.2%)**, **1,557.2 → 1,648.4 tok/s at 65,536 tokens (+5.9%)**, **1,439.9 → 1,513.0 tok/s at 196,608 tokens (+5.1%, one run each)**. On the production server shape (two banks, 196,608 context, `--mtp-draft 2`, three fresh workers per binary) the 8,259-token repeated-passage prompt went 1,481.7 → **1,559.9 tok/s** (+5.3%), the 7,937-token cold-PLE markdown prompt 1,555.9 → **1,652.6 tok/s** (+6.2%) and the 8,036-token x-prompt 1,677.7 → **1,761.5 tok/s** (+5.0%), with the 175-token greedy continuation and its MTP acceptance byte-identical before and after (30.3 / 30.4 tok/s). The 2K–64K incremental sweep (MTP draft 2, 2 GiB PLE cache) now measures **1,248.8 tok/s mean prefill** and **28.4 tok/s mean decode** (was 1,235.9 / 28.4). Every round is bit-identical to the kernels it replaces (8,192-token frontier logits byte-identical). See the 2026-09-07 campaign section below.
>
> **Update 2026-09-07 (image inputs):** [PR #15](https://github.com/Baekpica/ds4-dfm-rs/pull/15) reduces TTFT by **4.9–21.3%** across the tested photo, screenshot, document, and four-image inputs (three-run medians); the small-image case is unchanged. It also fixes a pre-existing vision LayerNorm race. See [image latency in agent workflows](#image-latency-in-agent-workflows-2026-09-07) for exact workloads, correctness gates, and cached-follow-up results.

> **Update 2026-09-14 — MTP runtime optimizations:** on this Q5 artifact with FP8 PLE, reduced draft vocabulary improved decode **3.04% at 8K and 3.87% at 64K**. Independently, cache-only MTP prefix preparation improved **64K prefill 2.55%**. Three fresh A/B pairs per frontier, 128 generated tokens, one DGX Spark at 300–2200 MHz; all frontier logits and generated IDs matched exactly. These cold-frontier measurements are separate from the older 2K–64K sweep below. [Runtime PR #40](https://github.com/Baekpica/ds4-dfm-rs/pull/40) · [Measurements, output checks and memory limits](https://github.com/Baekpica/ds4-dfm-rs/blob/b31e3db12c8b504103e9c95ed12a5131908b3f9e/docs/qwen38-perf-2026-09-14.md).

> **Update 2026-09-14 — QSA PV follow-up, no new default path:** [`ds4-dfm-rs@0f06cf8`](https://github.com/Baekpica/ds4-dfm-rs/commit/0f06cf8e798491d98c14a2abdc318708c115c0fe) measured fused-QSA PV 6×2 (bit-identical, +0.07% / +0.08% 8K/64K prefill), two-row selected-K/V reuse (full-model logit drift, no prefill gain), and HC mix_down/inject sharing (already tensor-core, ~1.2% of GPU time). None met the 2% bar. Published MTP draft/prefix numbers above are unchanged. [Runtime PR #41](https://github.com/Baekpica/ds4-dfm-rs/pull/41) · [Rejected-candidate measurements](https://github.com/Baekpica/ds4-dfm-rs/blob/0f06cf8e798491d98c14a2abdc318708c115c0fe/docs/qwen38-perf-2026-09-14.md).

> **Update 2026-09-08 — FP8 PLE, measured on this Q5 artifact:** the same 2K–64K card sweep with only the PLE sidecar changed measured **1,245.1 → 1,323.1 tok/s prefill (+6.3%)**, and **28.59 → 28.93 tok/s decode (+1.2%)**. Median of three run means per format, interleaved fresh processes on one DGX Spark; same main GGUF, binary, 2 GiB PLE cache, 16 workers and MTP draft 2. Automatic MTP quenching occurred in 1/3 BF16 and 0/3 FP8 runs; all samples are retained. Decode differences include generation and MTP policy effects. PLE files shrink from 95.37 to 47.68 GiB. [Protocol, raw CSV and limits](https://github.com/Baekpica/ds4-dfm-rs/blob/407509a06c366e372499817397918d6690bedcc4/docs/qwen38-ple-fp8.md).

![BF16 and FP8 PLE paired 2K–64K throughput on Qwen3.8 Flash Next base Q5](assets/qwen38-ds4-dfm-rs-2k-64k-throughput.png)

*Measured on this artifact. Curves are per-frontier medians; bands are observed min–max over three runs. One warm session per process, 2,048-token incremental prefill and 128 greedy tokens per frontier; prefill includes MTP prefix maintenance.*

## Support my work

I work on making large language models practical on hardware they were never really designed to fit on — through mixed quantization, inference optimization, custom kernels, and serving experiments.

While much of the development happens on local hardware, calibration, profiling, and large-scale validation often require expensive on-demand GPUs.

Contributions help pay for that compute, storage, and testing infrastructure, so I can keep experimenting and publishing the results openly.

<a href="https://www.buymeacoffee.com/baekpica" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" style="height: 60px !important;width: 217px !important;"></a> <a href="https://github.com/sponsors/Baekpica" target="_blank"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor Baekpica on GitHub" style="height: 60px !important;width: 217px !important;"></a>

This is a memory-hierarchy-aware mixed-precision conversion of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), pinned to revision [`f5d08274bafd880402bd16f5e3e6c514136ec06c`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/f5d08274bafd880402bd16f5e3e6c514136ec06c).

The design moves the 51.2B-parameter predictive latent embedding (PLE) table out of the accelerator-resident GGUF weight set and stores it as BF16 SSD-backed sparse lookup memory. Nearly all of the released residency budget is then reassigned to the 128.8B-parameter compute backbone. Compatibility with upstream llama.cpp, vLLM, SGLang, or other GGUF runtimes is **not** implied; a dedicated ds4 SSD-PLE loader is required.

These measurements are reproducible with [ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs), a Rust-host continuation of DwarfStar for explicit, hardware-validated large-model inference on NVIDIA DGX Spark. It preserves the antirez → Entrpi → Baekpica lineage and integrates a growing set of model families with minimal abstraction while retaining the native CUDA/MMQ kernels.

## Choose a PLE sidecar: BF16 or FP8

Both sidecar variants are available in this repository. Select one with
`hf download`; the commands below download only PLE files.

| Sidecar | Directory | Embedding files | Scale |
|---|---|---:|---|
| BF16 | `MQ-Q6-SSD-PLE-BF16/ple/` | 95.3682 GiB (4 files) | none |
| FP8 E4M3FN | [`PLE-FP8/`](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF/tree/main/PLE-FP8) | 47.6841 GiB (4 files) | one shared BF16 scalar (2 bytes) |

**BF16 sidecar:**

```bash
qwen_repo=Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF
qwen_root=./Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF

hf download "$qwen_repo" \
  --include 'MQ-Q6-SSD-PLE-BF16/ple/*' \
  --local-dir "$qwen_root"
(cd "$qwen_root/MQ-Q6-SSD-PLE-BF16/ple" && sha256sum -c SHA256SUMS)
```

**FP8 sidecar, including the required scale and manifest:**

```bash
qwen_repo=Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF
qwen_root=./Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF

hf download "$qwen_repo" \
  --include 'PLE-FP8/*' \
  --local-dir "$qwen_root"
(cd "$qwen_root/PLE-FP8" && sha256sum -c SHA256SUMS)
```

To download the Q5 **main GGUF only** before selecting a sidecar:

```bash
hf download "$qwen_repo" \
  --include 'MQ-Q5-SSD-PLE-BF16/*.gguf' \
  --include 'MQ-Q5-SSD-PLE-BF16/SHA256SUMS.main' \
  --local-dir "$qwen_root"
(cd "$qwen_root/MQ-Q5-SSD-PLE-BF16" && sha256sum -c SHA256SUMS.main)
```

The FP8 PLE is extracted byte for byte from
[`Qwen/Qwen3.8-Flash-Next-FP8@236dfdf`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13),
with all 128 embedding tensors and the original `weight_scale` preserved.
It uses 160-byte rows and one global BF16 scale; the scale is **not** a
128×128 block-scale array. See the [format and extraction details](./PLE-FP8/README.md).

FP8 runtime support is implemented in [ds4-dfm-rs PR #22](https://github.com/Baekpica/ds4-dfm-rs/pull/22) and
validated with the existing main GGUFs. Use a build containing this change;
the published `MQ-Q5-SSD-PLE-BF16` filenames and main GGUF metadata stay intact.

```bash
export DS4_QWEN_PLE_DIR="$(realpath "$qwen_root/PLE-FP8")"
# Run the existing ds4 / ds4-server / ds4-bench command with this environment.
```

The startup log must report `dtype=FP8_E4M3FN`. The runtime keeps FP8 pages in
the bounded cache and gathers the source-compatible scaled BF16 values for
the existing compute path. Unset the variable to use BF16. Use a separate
`--kv-disk-dir` for each main model and PLE format; cross-format snapshots are
rejected. See [usage, exact numerical contract and validation](https://github.com/Baekpica/ds4-dfm-rs/blob/407509a06c366e372499817397918d6690bedcc4/docs/qwen38-ple-fp8.md).
The paired Q5 measurements above establish runtime throughput on their
workload; they do not establish BF16-equivalent task quality or FP8 256K/512K
throughput.

## Variants and selective download

| Variant | Routed expert tiers | Main payload | Shared PLE | Runtime status |
|---|---|---:|---:|---|
| `MQ-Q6-SSD-PLE-BF16` | `Q5_K` / `Q6_K` / `Q5_0` | 90.9538 GiB | 95.3682 GiB | DGX Spark smoke passed |
| `MQ-Q5-SSD-PLE-BF16` | `Q4_K` / `Q5_K` / `Q5_0` | 77.5456 GiB | same files | DGX Spark 262K-config profile passed |

Download only the original Q6 variant, including its canonical PLE directory:

```bash
qwen_repo=Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF
qwen_root=./Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF
hf download "$qwen_repo" \
  --include 'MQ-Q6-SSD-PLE-BF16/*' \
  --local-dir "$qwen_root"
(cd "$qwen_root/MQ-Q6-SSD-PLE-BF16" && sha256sum -c SHA256SUMS)
```

Download only the lower-residency Q5 main GGUF plus the same canonical PLE files. The local link preserves the `ple/ple-manifest.json` path embedded in the GGUF without storing a second 95.37 GiB copy:

```bash
qwen_repo=Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF
qwen_root=./Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF
qwen_variant=MQ-Q5-SSD-PLE-BF16

hf download "$qwen_repo" \
  --include "$qwen_variant/*" \
  --local-dir "$qwen_root"
hf download "$qwen_repo" \
  --include 'MQ-Q6-SSD-PLE-BF16/ple/*' \
  --local-dir "$qwen_root"
test -e "$qwen_root/$qwen_variant/ple" || \
  ln -s ../MQ-Q6-SSD-PLE-BF16/ple "$qwen_root/$qwen_variant/ple"
(cd "$qwen_root/$qwen_variant" && sha256sum -c SHA256SUMS)
```

If the Q6 PLE directory is already present, only the first `hf download` and the link are needed; the additional transfer is about 77.56 GiB rather than downloading the PLE again.

The three Q5 main weight files occupy exactly **83,274,984,384 bytes / 77.5559 GiB**. A fresh local setup with the four shared PLE files occupies **185,675,770,816 weight-file bytes / 172.9240 GiB**; the PLE is not duplicated in this repository.

## Serve with ds4-dfm-rs

Build [ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs) with `make cuda-spark`, then start the weight owner from that checkout and keep it running:

```bash
MODEL=/path/to/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF/MQ-Q5-SSD-PLE-BF16/Qwen3.8-Flash-Next-MQ-Q5-SSD-PLE-BF16-00001-of-00003.gguf
MANIFEST=/tmp/qwen38-q5.weights.manifest

CUDA_VISIBLE_DEVICES=0 ./ds4_weight_server \
  --base "$MODEL" \
  --manifest "$MANIFEST" \
  --backend vmm \
  --scope base \
  --reserve-gb 32 \
  --no-repack-iq2-aligned \
  --no-repack-q2k-aligned \
  --repack-q8-aligned
```

After the owner reports `ready manifest=...`, start the Rust-host worker in another terminal. This is the 196,608-context, two-bank Q5+Sidecar configuration used for the measurements above:

```bash
MODEL=/path/to/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF/MQ-Q5-SSD-PLE-BF16/Qwen3.8-Flash-Next-MQ-Q5-SSD-PLE-BF16-00001-of-00003.gguf
MANIFEST=/tmp/qwen38-q5.weights.manifest
KV_DIR=./qwen38-q5-kv
mkdir -p "$KV_DIR"

CUDA_VISIBLE_DEVICES=0 \
DS4_CUDA_WEIGHT_IPC_MANIFEST="$MANIFEST" \
DS4_CUDA_WEIGHT_IPC_SCOPE=base \
DS4_MEMGOV=observe \
DS4_SESSION_GRAPH_FIT=0 \
DS4_SESSION_GRAPH_HEADROOM_MB=0 \
DS4_QWEN_BATCH=1 \
DS4_QWEN_PREFILL_CHUNK=8192 \
DS4_QWEN_PLE_CACHE_MB=512 \
DS4_QWEN_PLE_WORKERS=16 \
DS4_SERVER_CONTINUOUS=1 \
DS4_SERVER_COALESCE_MAX=2 \
DS4_SERVER_COALESCE_MAX_TOKENS=16384 \
DS4_SERVER_COALESCE_WAIT_MS=20 \
DS4_SERVER_WARM=1 \
DS4_SERVER_FORK=1 \
DS4_SERVER_FORK_PARTIAL=1 \
./ds4-server --cuda -m "$MODEL" -c 196608 -n 32768 \
  --host 0.0.0.0 --port 8002 --cors \
  --model-id Qwen3.8-Flash-Next-Mixed-Quant \
  --mem-floor-gb 2 --mtp-draft 2 --no-update-check \
  --kv-disk-dir "$KV_DIR" --kv-disk-space-mb 32768
```

The example binds to all interfaces; use `--host 127.0.0.1` unless remote access is intended. Keep the published `ple/` layout beside the first GGUF shard.

For the 512K-capable path, retain the same owner but use one worker bank with
`-c 524288 -n 256 --cont-width 1`, set `DS4_QWEN_PREFILL_CHUNK=1024` and
`DS4_SERVER_COALESCE_MAX=1`, and set `DS4_SERVER_WARM=0`,
`DS4_SERVER_FORK=0`, and `DS4_SERVER_FORK_PARTIAL=0`. The verified run did not
enable `--mtp-draft 2`. The runtime selects static YaRN factor 2 for this
context. `DS4_SESSION_GRAPH_FIT=0` remains an explicit fit-check override, not
a guarantee that the allocation fits. Direct completed-prompt evidence for
the base artifact itself remains the exact 262,144-token run above.

## Quantization targets by variant

| Model region | Q6 variant | Q5 variant | Residency policy |
|---|---:|---:|---|
| **51.2B-parameter PLE n-gram table (128 shards)** | BF16 SSD sidecar | same files | non-resident; bounded pinned page cache only |
| **Routed expert gate/up, interior layers 2–45** | `Q5_K` | `Q4_K` | accelerator-resident |
| **Routed expert gate/up, edge layers 0, 1, 46, 47** | `Q6_K` | `Q5_K` | accelerator-resident |
| **Routed expert down, main 512 columns** | `Q6_K` | `Q5_K` | accelerator-resident |
| **Routed expert down, 128-column tail** | `Q5_0` | unchanged `Q5_0` | accelerator-resident |
| **MTP routed experts and most always-active matrices** | primarily `Q8_0` | unchanged | accelerator-resident |
| **Hyper-connection, non-quantizable convolution/vision tensors** | BF16 | unchanged | accelerator-resident |
| **Norms, gates, recurrent/control state** | F32 where required | unchanged | accelerator-resident |
| **Integer PLE controls** | I64 | unchanged | accelerator-resident |

The original 128-column tail target was `Q5_K`, but standard GGUF `Q5_K` uses 256-value blocks and therefore cannot encode a tensor whose row width (`ne[0]`) is 128. The Q6 artifact realizes that tier as width-compatible `Q5_0`, and the Q5 variant deliberately retains all 48 `Q5_0` tail tensors so ds4 can continue using its current optimized tail accumulator and dispatch. Only the K-quant tiers are lowered: `Q5_K -> Q4_K` and `Q6_K -> Q5_K`. No Q5_0, Q8_0, BF16, F32, or I64 target changes between variants, and both are generated directly from the pinned source BF16 checkpoint rather than by requantizing the Q6 GGUF.

[`Baekpica/ds4@0951d03`](https://github.com/Baekpica/ds4/commit/0951d03695d3fbb6b43e1eec5387c1eca51f6535) recognizes the `MQ-Q5-SSD-PLE-BF16` label and validates its exact per-layer precision map. The existing `Q4_K` paired gate/up, routed `Q5_K` MMQ, and `Q5_0` tail paths are reused.

## Separate memory metrics

`Overall model BPW` is not the useful metric for this artifact because the shared PLE is intentionally non-resident. The audited resident-backbone census is:

| Metric | Q6 variant | Q5 variant |
|---|---:|---:|
| Main GGUF logical parameters | 128,799,735,699 | 128,799,735,699 |
| Main GGUF tensor count | 1,628 | 1,628 |
| **Resident backbone payload** | **97,660,877,400 bytes / 90.9538 GiB** | **83,263,928,920 bytes / 77.5456 GiB** |
| **Resident backbone effective BPW** | **6.0659** | **5.1717** |
| Actual main GGUF files | 97,671,932,960 bytes / 90.9641 GiB | **83,274,984,384 bytes / 77.5559 GiB** |
| Resident savings versus Q6 | — | **14,396,948,480 bytes / 13.4082 GiB** |
| Shared PLE parameters | 51,200,245,760 | same |
| **Shared PLE storage dtype** | **BF16 (16 bpw)** | same files |
| **PLE payload on SSD** | **102,400,491,520 bytes / 95.3679 GiB** | same files |
| PLE files including 4 KiB alignment | 102,400,786,432 bytes / 95.3682 GiB | not duplicated |

The published resident-PLE baseline spends 38,400,184,320 bytes / 35.7630 GiB on a Q5_1 PLE table. The original Q6 compute backbone costs 37,570,478,080 bytes / 34.9902 GiB more, so its resident payload is **829,706,240 bytes / 0.7727 GiB smaller** than that baseline. The lower-tier Q5 resident payload is **15,226,654,720 bytes / 14.1809 GiB smaller** than the same baseline. These are exact tensor payload calculations using GGUF block sizes; final file size, CUDA allocations, page cache, KV state, and runtime workspace are separate.

## Published artifact layout

```text
MQ-Q6-SSD-PLE-BF16/
├── Qwen3.8-Flash-Next-MQ-Q6-SSD-PLE-BF16-*.gguf
├── SHA256SUMS
├── quant-recipe.yaml
├── artifact-manifest.json
└── ple/
    ├── ple-bf16-00001-of-00004.bin
    ├── ple-bf16-00002-of-00004.bin
    ├── ple-bf16-00003-of-00004.bin
    ├── ple-bf16-00004-of-00004.bin
    └── ple-manifest.json

MQ-Q5-SSD-PLE-BF16/
├── Qwen3.8-Flash-Next-MQ-Q5-SSD-PLE-BF16-*.gguf
├── SHA256SUMS
├── SHA256SUMS.main
├── shared-ple-reference.json
├── quant-recipe.yaml
└── artifact-manifest.json
```

The Q5 directory intentionally does not duplicate the PLE files. After selective download, its local `ple` symlink points at the Q6 directory's canonical sidecar. The 128 logical checkpoint shards remain individually addressable through `ple-manifest.json`; four larger physical files reduce descriptor and Hub-transfer overhead. Every logical-shard start is 4 KiB aligned. The manifest records the source tensor, global row range, physical file and offset, row stride, per-head prime vocabulary sizes and offsets, SplitMix64-derived layer multipliers, total bytes, and checksums.

## SSD runtime contract

The full 95.37 GiB PLE sidecar must never become resident in the 128 GB unified-memory pool. The planned ds4 path uses exact CPU-side Qwen n-gram row derivation, asynchronous SSD prefetch overlapped with decoder layer 0, a bounded 512 MiB / 1 GiB / 2 GiB pinned mapped page cache, and a separate CUDA UVA gather kernel. `O_DIRECT` plus aligned asynchronous reads is preferred; a buffered fallback must evict sidecar pages with an explicit `POSIX_FADV_DONTNEED`-class policy.

Registering the whole file with CUDA, relying on unbounded `mmap`, or allowing Linux page cache to retain the full sidecar fails the residency objective. A successful load of the main GGUF alone is not proof that SSD-PLE works.

The current ds4 handoff implementation covers the correctness-critical core of
this contract: exact EOS-bounded CPU hashing, strict manifest validation,
bounded four-way page caching, asynchronous worker prefetch, `O_DIRECT` with a
page-cache-dropping fallback, mapped-cache UVA gathering, and cache-slot leases
held until the consuming CUDA stream has completed. Decoder-layer-0 overlap
is active; detailed SSD latency-distribution profiling remains target-hardware
integration work. Cache and worker-count measurements are reported below.

## Q6 variant DGX Spark ds4 serving verification

The results in this section apply to `MQ-Q6-SSD-PLE-BF16`, not yet to the new Q5 variant. On 2026-08-27, [`Baekpica/ds4@a06f06d`](https://github.com/Baekpica/ds4/commit/a06f06d)
on branch [`dfm`](https://github.com/Baekpica/ds4/tree/dfm) was verified on one
DGX Spark (NVIDIA GB10) with the server configured for a 262,144-token context.

- The worker imported a content-verified 91.05 GiB shared VMM weight cache and
  reserved a logical 14.33 GiB 262K session graph. QSA index/KV state was
  demand-mapped: the session census was 0.45 GiB before a request and 0.55 GiB
  after a 289-token cold prompt, with 91.65 GiB total device-live memory. These
  figures are for that short prompt; QSA pages grow as more context is used.
- The four PLE sidecars opened with `O_DIRECT` (`4/4`) behind a bounded 512 MiB
  cache. A bounded `mincore` check after inference found 0.00 MiB of the
  95.37 GiB sidecar set resident in Linux page cache; the sidecars were not
  present in the server's memory map.
- Chat Completions and Responses returned coherent deterministic output.
  Buffered and streaming Responses, a function call, and its
  `function_call_output` continuation completed successfully.
- Two simultaneous requests completed through the current serial lane with no
  request, graph-fit, memory-census, or memory-governor failures.
- Across three fresh eager-graph server processes, the same deterministic
  289-token synthetic cold prompt averaged 131.6 prefill tok/s, 2,254.9 ms
  TTFT, and 17.7 decode tok/s. Each run produced the same 74 output tokens.
  This is a short microbenchmark, not a sustained-throughput result.
- A 3,069-token marker-retrieval prompt returned the exact marker at 83.0
  prefill tok/s. This is the longest prompt used in this smoke test.

The host run used a 115 GiB hard memory cap and a 6 GiB admission floor; no
cgroup high/max/OOM event occurred. These observations establish functional
serving with the runtime configured for 262,144 tokens. They do **not** establish
a completed 262,144-token prompt run, full-model quality, sustained throughput,
or SSD latency percentiles.

## Q5 variant DGX Spark ds4 serving and prefill profile

On 2026-08-27, [`Baekpica/ds4@0951d03`](https://github.com/Baekpica/ds4/commit/0951d03695d3fbb6b43e1eec5387c1eca51f6535)
on branch [`dfm`](https://github.com/Baekpica/ds4/tree/dfm) was verified on one
DGX Spark (NVIDIA GB10) with `MQ-Q5-SSD-PLE-BF16` and the server configured for
a 262,144-token context. The measured configuration used
`DS4_QWEN_PREFILL_CHUNK=8192`, `DS4_QWEN_PLE_CACHE_MB=512`, `O_DIRECT` on 4/4
sidecars, a 115 GiB hard memory cap, and a 2 GiB admission floor. Prefix-cache
reuse was zero for every reported request.

| Workload | Measured requests | Mean prefill | Mean TTFT | Mean decode |
|---|---:|---:|---:|---:|
| 5,384-5,385 prompt tokens + 64 output tokens | 3 | 154.6 tok/s | 34.91 s | 18.7 tok/s |
| 21,037 prompt tokens + 64 output tokens | 2 | 149.1 tok/s | 141.28 s | 18.8 tok/s |

For the same 24,000-byte story workload family, three 256-token-chunk requests
(5,392-5,393 prompt tokens) averaged 139.7 prefill tok/s, 38.67 s TTFT, and
18.8 decode tok/s. The 8,192-token configuration improved mean prefill by
10.7% and reduced mean TTFT by 9.7%; decode differed by -0.5%. The request
prefix labels and resulting token counts differed slightly, so this is a live
serving A/B rather than a bit-exact kernel microbenchmark.

The 262K session graph reported a 24.85 GiB logical plan and 11,238.2 MiB of
measured committed memory. The 512 MiB-cache run peaked at 60,942,938,112 bytes
(56.76 GiB) in its guarded cgroup, with zero high/max/OOM events. A separate
21,037-token two-request check with a 2 GiB PLE cache averaged 150.0 prefill
tok/s, 140.43 s TTFT, and 18.7 decode tok/s: about +0.6%, -0.6%, and -0.5%
relative to 512 MiB. The 512 MiB default was retained because the larger cache
did not produce a material gain in this workload.

`/v1/models`, `/v1/stats`, Chat Completions, and text Completions all returned
successfully, and the generated text was coherent. This establishes functional
Q5 serving and preliminary prefill/TTFT/decode measurements with a 262K-configured
server. It does **not** establish a completed 262K prompt, full-model quality,
sustained multi-request throughput, or SSD latency percentiles.

### Optimized prefill rerun

Later on 2026-08-27, [`Baekpica/ds4@64f7e41`](https://github.com/Baekpica/ds4/commit/64f7e415f7ed92178139c5b606c4b7afd22f6569)
was tested on the same DGX Spark and Q5 artifact after Q5_0 down-tail tiling,
Gated DeltaNet state-column parallelization, and a compact bounded Q5_K routed
MMQ worklist were added. A fixed 8,192-token `ds4-bench` workload with no
decode measured 275.81, 293.20, and 293.55 prefill tok/s: **287.52 tok/s mean**
(275.81-293.55 range), versus 145.02 tok/s before those changes using the same
command and input (**+98.3%**).

A subsequent QSA reduction change in
[`Baekpica/ds4@4b5eb39`](https://github.com/Baekpica/ds4/commit/4b5eb39bd25fb0cee26fe29d5a1fefd225eb1aa9)
reused each selected value across the 12 query heads sharing a KV head. The
real-weight QSA check retained `1.2470e-07` relative RMS error and
`1.4544e-14` one-minus-cosine distance. The identical fixed 8,192-token direct
benchmark then measured 293.58, 321.00, and 285.32 prefill tok/s:
**299.97 tok/s mean** (285.32-321.00 range), **+4.3%** over the preceding
287.52 tok/s result and **+106.8%** over the initial 145.02 tok/s baseline.
This direct run used `--gen-tokens=0`, so API TTFT and decode throughput are
not applicable; the latest end-to-end values remain the `64f7e41` results
reported below.

The next PLE change in
[`Baekpica/ds4@11e1d219`](https://github.com/Baekpica/ds4/commit/11e1d2195f1f5e6cdaa8e3feeffe4e6e8ecd425f)
kept the shared 2,560-row key/value prefix on the paired Q8 kernel and sent the
remaining 7,680 key-only rows through the existing single-output kernel. On
the same fixed 8,192-token direct workload, a same-binary 3-by-3 A/B measured
302.46, 332.15, and 312.64 tok/s with the split enabled versus 287.06, 297.64,
and 286.06 tok/s with `DS4_CUDA_NO_Q8_PAIR_ASYM_SPLIT=1`: **315.75 versus
290.25 tok/s mean (+8.78%)**. The optimized mean is 5.3% above the previously
published 299.97 tok/s run, but the same-binary comparison is the controlled
result. A full-model 256-token on/off check matched all 248,320 logits exactly,
including the argmax and top-10 set.

The `64f7e41` build was also exercised through Chat Completions on a server
configured for a 262,144-token context, 8,192-token prefill chunks, two
independent Qwen banks, a 512 MiB PLE cache, and `O_DIRECT` on 4/4 sidecars:

| Prompt / output tokens | Prefill | TTFT | Decode |
|---:|---:|---:|---:|
| 8,774 / 64 | 317.8 tok/s | 27.637 s | 22.8 tok/s |
| 12,256 / 64 | 317.5 tok/s | 38.617 s | 22.7 tok/s |
| 8,095 / 64 | 296.0 tok/s | 27.360 s | 22.8 tok/s |
| **Mean (9,708 prompt tokens)** | **310.43 tok/s** | **31.204 s** | **22.77 tok/s** |

All three requests were cold (`cached_tokens=0`); token-weighted prefill was
311.30 tok/s. `/v1/models` advertised 262,144 tokens, all six requests in the
complete smoke sequence succeeded, and the server reported zero continuous
fallbacks and zero governor faults. Two simultaneous 3.8K prompts also
completed as `served=2 fallback=0`, establishing two-bank function but not
row-batched kernel throughput; their per-request decode rates were uneven and
are not included in the single-request mean above.

This 8,192-by-two-bank run required `DS4_MEMGOV=observe` because the internal
governor rejected cold materialization under enforcement despite the external
memory guard. The 109 GiB reclaim threshold and 115 GiB hard cgroup limit
remained active. Device-live memory stabilized at 97.88 GiB and observed system
available memory stayed at or above 15.46 GiB during the reported requests.
This remains a 262K-configured serving check, not a full-length 262K prompt run.

The `11e1d219` build was then checked through the same API shape with three
distinct cold prompts (`cached_tokens=0`):

| Prompt / output tokens | Prefill | TTFT | Decode |
|---:|---:|---:|---:|
| 7,121 / 64 | 360.3 tok/s | 19.799 s | 18.1 tok/s |
| 9,769 / 58 | 232.5 tok/s | 42.057 s | 18.0 tok/s |
| 5,660 / 64 | 258.3 tok/s | 21.954 s | 18.1 tok/s |
| **Mean (7,517 prompt tokens)** | **283.70 tok/s** | **27.937 s** | **18.07 tok/s** |

The server advertised a 262,144-token context and reported two banks. A
barrier-synchronized two-request smoke also completed, with 5/5 total requests
completed and zero failures. The 9,769-token request crossed the 8,192-token
chunk boundary and included second-chunk/bank warm-up, so these content- and
chunk-sensitive API values are reported as a serving check rather than a
controlled comparison with the earlier table. The external 115 GiB guard
remained active throughout.

On 2026-08-28, the QSA scoring path in
[`Baekpica/ds4@f9f14a7`](https://github.com/Baekpica/ds4/commit/f9f14a754ee916d1cbde3145ca1d87a13d732870)
replaced the fixed 24-query/2-KV shared-memory block reduction with a four-warp
register/shuffle reduction. A controlled cold fixed-8K 3-by-3 A/B measured
306.45, 309.58, and 334.18 tok/s before versus 369.77, 347.64, and 362.98 tok/s
after: **316.74 versus 360.13 tok/s mean (+13.70%)**. The real-weight arbitrary-
chunk and single-token checks retained `1.2470e-07` relative RMS error and
`1.4544e-14` one-minus-cosine distance.

The following Q5_0 expert-major tail change in
[`Baekpica/ds4@c733f5e`](https://github.com/Baekpica/ds4/commit/c733f5e5463a17db3be7f496e9c9fb9641f18eaa)
staged the existing 32 warp sums and let the first warp write contiguous output
columns. Its controlled cold fixed-8K 3-by-3 A/B measured 364.05, 359.54, and
370.08 tok/s before versus 384.28, 389.49, and 368.47 tok/s after:
**364.56 versus 380.75 tok/s mean (+4.44%)**. The full MMQ parity suite and the
real-weight MoE forward gate passed after this change.

The `c733f5e` build was then exercised through Chat Completions with three
distinct cold prompts (`cached_tokens=0`):

| Prompt / output tokens | Prefill | TTFT | Decode |
|---:|---:|---:|---:|
| 6,384 / 64 | 466.5 tok/s | 13.731 s | 17.9 tok/s |
| 9,807 / 64 | 474.6 tok/s | 20.687 s | 17.7 tok/s |
| 8,151 / 64 | 472.3 tok/s | 17.278 s | 17.2 tok/s |
| **Mean (8,114 prompt tokens)** | **471.13 tok/s** | **17.232 s** | **17.60 tok/s** |

Token-weighted prefill was 471.68 tok/s. This serving check used a 262,144-token
context, 8,192-token prefill chunks, two Qwen banks, the 2 GiB PLE cache and 16
PLE workers, `O_DIRECT` on 4/4 sidecars, `DS4_MEMGOV=observe`, and the external
109/115 GiB reclaim/hard guard. `/v1/models` advertised 262,144 tokens; all six
requests in the complete smoke sequence finished with zero failures, and a
barrier-synchronized pair completed as `served=2 fallback=0`. Final stats
reported 97.72 GiB device-live memory, 14.32 GiB system-available memory, and
zero governor/census faults. These content-sensitive API values are a serving
check, not a controlled comparison with earlier tables, a full-length 262K
prompt run, sustained-concurrency throughput, quality, or SSD-latency validation.

### Prefix reuse, recurrent disk KV, and embedded MTP

On 2026-08-28, [`Baekpica/ds4@c8bc5c9`](https://github.com/Baekpica/ds4/commit/c8bc5c9a87343603cef42ab0e64f914b1612aa05)
completed three Qwen runtime items that were previously listed as pending:

- Qwen recurrent session state is now serialized with the normal disk-KV
  payload. The payload includes PLE convolution state, all 36 Gated DeltaNet
  convolution/recurrent states, QSA KV/frontier state, and the live token
  checkpoint. Tagged-layout, truncation, and corruption checks fail closed.
  A real-Q5 cross-process API check restored all 1,152 cached prompt tokens
  after a server restart, computed zero prompt tokens, and returned the same
  deterministic continuation. Idle-bank stage/restore also passed.
- The opt-in two-bank runtime now retains a bounded 32-slot recurrent-state
  checkpoint pool and can fork either an exact frontier or the longest safe
  partial prefix. A divergent API branch reused 3,873 prompt tokens and
  computed only its 21-token suffix; the real-weight test also matched a cold
  oracle after a partial fork.
- The embedded Qwen MTP block now proposes one token and the target graph
  verifies `[normal, draft]` before accepting it. Rejected drafts roll the
  recurrent target state back, and a timing-based auto-quench returns to plain
  decode if speculation is slower. The real-Q5 gate produced the identical
  target token stream with multi-token accepts. This initial path was enabled
  with `--mtp-draft 2` for greedy scalar/session decode.

The combined guarded API check used a 196,608-token configured context,
`DS4_QWEN_BATCH=1`, two banks, 8,192-token prefill chunks, a bounded 512 MiB
PLE cache, and `O_DIRECT` on all four sidecars. Seven of seven requests
completed with zero failures; exact and partial forks were both observed, and
a synchronized three-request epoch reported `served=3 fallback=0`. Device-live
memory moved from 98.08 GiB after boot to 99.00 GiB during the run, observed
system-available memory remained at least 14.24 GiB, and the external 109/115
GiB reclaim/hard guard stayed active with zero governor or census faults. This
is a 196K-configured serving and feature check, not a full-length 196K or 262K
prompt run, sustained-concurrency throughput result, or proof of row-batched
Qwen kernels.

### Two-bank embedded-MTP follow-up

Later on 2026-08-28,
[`Baekpica/ds4@4226b60`](https://github.com/Baekpica/ds4/commit/4226b608f93000fb77321bd41f65270dc0a05012)
extended the same target-verified embedded MTP path to the opt-in continuous
two-bank loop. Each bank keeps independent draft state, verifies
`[current, draft]` in one two-row target call, rolls recurrent target state
back on rejection, and automatically quenches speculation when its measured
cost exceeds plain decode. Bank reset/copy, partial-prefix restore, disk-KV
restore, and graph retire/rebuild invalidate or rebuild the draft state.

The real-Q5 two-bank regression matched the existing serial target-verified
MTP output for both prompts. Both banks drafted (`6/6` cycles), accepted `5/4`
drafts, and then passed the existing disk-KV, partial-prefix, and graph
lifecycle gates.

A same-binary API A/B used four distinct cold 58-token prompts with a
128-token output budget:

| Decode mode | Mean prefill | Mean TTFT | Mean decode | Draft acceptance |
|---|---:|---:|---:|---:|
| Plain | 225.38 tok/s | 260.7 ms | 23.65 tok/s | n/a |
| Embedded MTP | 223.35 tok/s | 261.1 ms | 28.65 tok/s | 84.98% |

This is a **+21.1% mean decode-throughput** result with essentially unchanged
TTFT and prefill in that short workload. Device-live memory was 100.02 GiB
after the plain run and 101.00 GiB after the MTP run. One low-acceptance prompt
was slower than plain decode, so the gain is content-dependent and the
auto-quench remains part of the serving path. A synchronized two-request run
also completed, followed by Chat Completions, Responses streaming, and
Anthropic Messages checks. The final server counters were 10/10 requests
completed, 419 drafts, 362 accepts (86.40%), zero quench events, and zero
governor faults under the external 109/115 GiB reclaim/hard guard. This is a
two-bank functional and short-workload A/B, not a sustained-concurrency result
or a claim of true row-batched Qwen kernels.

### 262K API-surface and continuation follow-ups

On 2026-08-28, `Baekpica/ds4@63d4ce7` was checked with the Q5 artifact at a
262,144-token configured context, two Qwen banks, 8,192-token prefill chunks,
a 2 GiB PLE cache with 16 workers, partial-prefix reuse, and a 32 GiB disk-KV
budget. Chat Completions and Anthropic Messages ran simultaneously and returned
their native HTTP 200 response shapes. A high-reasoning Responses function
call and its `function_call_output` continuation also both returned HTTP 200;
the second turn reused 377 cached tokens and computed its 23-token suffix.
Final counters reported two serial Responses requests, zero serial refusals,
zero failed requests, and zero governor or census faults.

At that commit, the serial lane was required for this agent shape. Some live Responses
reasoning/tool frontiers deliberately route there; a cont-only launch with
`--no-serial` returns a typed 503 rather than silently recomputing or using the
wrong state. With two banks, boot device-live memory was 96.49 GiB with
15.46 GiB system-available. After the serial tool continuation allocated its
right-sized session graph, device-live memory was 106.22 GiB and
system-available memory was 5.55 GiB under the external 109/115 GiB guard.

On 2026-08-29,
[`Baekpica/ds4@0f66b89`](https://github.com/Baekpica/ds4/commit/0f66b899d689d49c648c579954e63254aab7e0f1)
removed that requirement when the continuation registry resolves an exact
bank-owned frontier. A guarded Q5 server used two banks, a 262,144-token
configured context, 8,192-token prefill chunks, the 2 GiB PLE cache with 32
workers, a fresh 16 GiB disk-KV directory, embedded MTP-2, and `--no-serial`.
The first streaming Responses request produced a function call after 42 hidden
reasoning tokens. The follow-up replayed that completed `function_call` without
the hidden reasoning and appended only its new `function_call_output`; it
returned HTTP 200 `response.completed` from the same bank with 395 cached
tokens plus a 27-token suffix. Final stats reported two continuous Responses
requests, zero serial requests, one continuous bank continuation, and zero
request, continuous-batch, refusal, census, or governor failures. Device-live
memory was 100.98 GiB and observed system-available memory was 9.01 GiB at the
final sample.

This supersedes the serial requirement only for a live bank-owned output-only
continuation whose generation and committed frontier still match. Buffered
first tool-generation requests remain serial because their model-visible
corrective retry has no row-local continuous equivalent; serial-owned,
expired, or unresolved continuation state is not silently accepted.

A three-bank 262K trial was not retained on the 128 GiB reference host. It
booted at 107.47 GiB device-live with 4.25 GiB system-available, but a small
serial tool continuation reduced system-available memory to 1.12 GiB, below
the configured 2 GiB floor. This establishes two banks, not three, as the
guarded 262K agent-serving configuration on this machine. These were API and
memory checks, not full-length 262K prompt or sustained-throughput results.

### Shared-expert Q8 artifact follow-up

On 2026-08-28,
[`Baekpica/ds4@02cac0b`](https://github.com/Baekpica/ds4/commit/02cac0bbca58b97fc924a11a9d24cd0c8140c25f)
completed three additional fixed-8K optimization rounds on the same DGX Spark
and Q5 artifact. Each row is a same-binary cold 3-by-3 A/B with the CPU helpers
pinned to cores `5-9,15-19`, 8,192-token prefill chunks, and the external
109/115 GiB reclaim/hard guard. The percentages are per-row comparisons and
are not additive.

| Round / change | Enabled runs | Disabled runs | Mean A/B | Disposition |
|---|---:|---:|---:|---|
| Restore the derived-artifact catalog for valid 5D vision tensors | 430.91, 433.12, 363.82 tok/s | 374.36, 350.16, 360.52 tok/s | 409.28 vs 361.68 tok/s (**+13.16%**) | Accepted |
| Admit the Qwen `2560 x 640` shared-expert pair to the existing aligned Q8 kernel | 469.30, 373.61, 421.78 tok/s | 372.77, 364.34, 422.40 tok/s | 421.56 vs 386.50 tok/s (**+9.07%**) | Accepted |
| Admit the `640 x 2560` shared-down matrix to the existing D2R kernel | 415.83, 405.10, 500.83 tok/s | 422.78, 376.81, 394.95 tok/s | 440.59 vs 398.18 tok/s (**+10.65%**) | Rejected and removed |

The third round changed the fixed-8K frontier by `5.38%` relative RMS and one
member of the top-10 set, so its apparent speedup was not retained. After its
removal, enabling or disabling only the accepted shared-pair path produced
byte-identical JSON for all 248,320 frontier logits, including the argmax and
top-10 set. The real-weight MoE, Gated DeltaNet, QSA, and two-bank recurrent
state/disk-KV/partial-fork/MTP lifecycle gates also passed. The wide run ranges
are reported rather than hidden; the Spark was thermally variable during these
measurements.

The accepted build was then served with a 262,144-token configured context,
two Qwen banks, 8,192-token prefill chunks, a 2 GiB bounded PLE cache, a fresh
32 GiB disk-KV directory, and embedded MTP-2. Three distinct streaming Chat
requests were cold (`cached_tokens=0`):

| Prompt / output tokens | Prefill | TTFT | Decode tail |
|---:|---:|---:|---:|
| 8,221 / 64 | 456.71 tok/s | 18.001 s | 23.34 tok/s |
| 8,941 / 64 | 453.58 tok/s | 19.712 s | 23.29 tok/s |
| 7,588 / 64 | 437.02 tok/s | 17.363 s | 24.13 tok/s |
| **Mean (8,250 prompt tokens)** | **449.10 tok/s** | **18.359 s** | **23.58 tok/s** |

Prefill is prompt tokens divided by measured streaming TTFT; decode tail is
the 63 tokens after the first token divided by the remaining stream time.
Token-weighted prefill was 449.38 tok/s. The three measured requests completed
with 88/98 MTP drafts accepted (89.80%) and no quench. A subsequent
barrier-synchronized pair returned two HTTP 200 responses and logged
`served=2 fallback=0`; the full gate completed 5/5 requests with zero request,
governor, or census failures. Device-live memory moved from 100.65 GiB at boot
to 102.34 GiB after the requests, while observed system-available memory was
8.21 GiB at the final sample. This remains a content-sensitive 262K-configured
serving check, not a full-length 262K prompt, sustained-concurrency, quality,
or SSD-latency-distribution result.

### Still-image input follow-up

On 2026-08-29,
[`Baekpica/ds4@cc771c0`](https://github.com/Baekpica/ds4/commit/cc771c046e97b59d5b10b7e145295384fa0bf608)
added native still-image input for the Qwen path. Chat Completions `image_url`,
Responses `input_image`, and Anthropic base64 image blocks share one ordered
request representation. The runtime applies the pinned
`Qwen3VLProcessor`/`Qwen2VLImageProcessorFast` geometry, executes the GGUF's
embedded 27-layer vision tower, replaces image-pad rows with projected image
features, and carries the reference three-axis M-RoPE positions through
prefill and decode.

The interface accepts base64 PNG/JPEG data URIs only: at most four images,
10 MiB decoded per image, and 20 MiB decoded per request. Network/file URLs,
video, and other image formats are rejected. In the original V1 commit, image
requests were deliberately cold because token-only cache keys cannot identify
pixel content; text-only exact, partial-prefix, and disk-KV reuse remained
enabled.

The Q5 artifact passed Chat Completions, Responses, and Anthropic Messages on
the guarded DGX Spark server configured for a 262,144-token context, two Qwen
banks, and 8,192-token prefill chunks. A synchronized Responses/Anthropic pair
completed as `served=2 fallback=0`. An 8,243-token Chat request placed its 64
projected image rows across the 8,192-token chunk boundary and completed at
489.4 prefill tok/s, 17.232 s TTFT, and 22.5 decode tok/s. A later text-only
repeat exercised a partial fork after both banks had served images. The
post-gate worker reported zero request, continuous-batch, census, or governor
failures under the external 109/115 GiB reclaim/hard guard.

[`Baekpica/ds4@9a55e03`](https://github.com/Baekpica/ds4/commit/9a55e03c5c5378af4f76d02d2dd0a476b1134a6e)
then corrected the vision-attention score broadcast after warp reduction. An
exact-source Responses API check read all nine visible labels from the
screenshot below: `ChatGPT`, `Claude`, `Perplexity`, `Grok`, `Cursor`,
`Hermex`, `Termius`, `Orca`, and `Grok Bot`.

| Source screenshot | Served result |
|---|---|
| <img src="assets/qwen38-still-image-source.png" alt="Source screenshot with nine app labels" width="320"> | <img src="assets/qwen38-still-image-response.png" alt="Qwen3.8 served response reading all nine app labels" width="720"> |

These are still-image API and integration checks, not a general vision
throughput or OCR benchmark, nor a cross-runtime hidden-state/logit comparison.
The first implementation used correctness-first per-query online attention;
the later tiled large-image follow-up is reported below.

On 2026-08-29,
[`Baekpica/ds4@f7830ba`](https://github.com/Baekpica/ds4/commit/f7830ba89129b7075637a96b28a6c473433289ac)
added image V2 cache identity. The server hashes decoded RGB pixels and source
geometry into an internal-only replay key; the marker is never tokenized or
returned by an API. A same-image continuation reused 90/110 prompt tokens,
while a same-geometry different-pixel request stayed cold. After a worker
restart, an image-bearing 1,890-token bank restored from disk and a 1,909-token
follow-up reused all 1,890 stored tokens. A separate same-image divergent
20,087-token request restored the 16,384-token recurrent checkpoint and
computed 3,703 prompt tokens. The final worker counters reported zero request,
continuous-batch, census, or governor failures under the external 109/115 GiB
reclaim/hard guard. Disk exact restore and live divergent partial reuse were
exercised separately; no cross-restart divergent-partial performance claim is
made.

### Image latency in agent workflows (2026-09-07)

[`ds4-dfm-rs@162de4d`](https://github.com/Baekpica/ds4-dfm-rs/commit/162de4dc9899f3df6ae9f72060859f9adc108af3)
([PR #15](https://github.com/Baekpica/ds4-dfm-rs/pull/15),
[protocol, fixtures, and raw samples](https://github.com/Baekpica/ds4-dfm-rs/blob/162de4dc9899f3df6ae9f72060859f9adc108af3/docs/qwen38-image-2026-09-07.md))
reuses K/V registers for two vision-attention queries per warp. It keeps
full attention, image resolution, patch counts, and accumulation order.
Small images keep the previous kernel.

Directly tested on this **MQ-Q5-SSD-PLE-BF16** artifact on one DGX Spark
GB10 (CUDA 13.3): context 262,144, two banks, 8,192-token prefill chunks,
2 GiB PLE cache / 16 workers, MTP draft 2. Median TTFT from three fresh
workers per variant, each with empty disk KV and the same six-image suite
in fixed order; every timed prompt had zero cached tokens. Temperature 0,
thinking disabled, one output token. PLE can warm within each suite.

| New image input | Original serving binary (`c3dfb36`) | Updated TTFT | Reduction |
|---|---:|---:|---:|
| Small dashboard, 256 × 256 | 0.275 s | 0.275 s | Within noise |
| Screenshot, 1024 × 768 | 1.824 s | 1.564 s | 14.3% |
| Invoice, 1536 × 1024 | 5.433 s | 4.399 s | 19.0% |
| Photo, 512 × 507 | 0.558 s | 0.531 s | 4.9% |
| Large screenshot, 1920 × 1080 | 8.635 s | 6.797 s | 21.3% |
| Four images, fixed order | 7.174 s | 5.829 s | 18.7% |

The change also fixes a pre-existing LayerNorm shared-memory race that
made repeated image forwards nondeterministic. With that fix on **both**
attention paths, all image embeddings and 248,320 frontier logits are
byte-identical on five full-model image cases, across paths and repeats;
Compute Sanitizer reports zero LayerNorm race hazards. Image outputs from
the previously racy encoder are not claimed to be identical.

Agent checks cover OCR, a photo, four-image tool arguments, tool-result
continuation, native Chat/Responses/Anthropic image schemas, two simultaneous
requests, and image KV restored after a worker restart. Generated text and
tool arguments match between the corrected legacy and new attention paths.
Immediate image follow-ups reused **800/828** tokens, tool follow-ups
**2,990/3,045**, and restart/disk reuse **824/846**. When the image is fully
covered by cached KV, the encoder is skipped; these cold-image speedups do
not apply to that follow-up. Same-size images with changed pixels stayed
isolated. This fixed suite is not a general vision-quality benchmark or a
claim of decode, sustained-concurrency, or Q6-artifact speedup.

### Tiled large-image vision attention

On 2026-08-29,
[`Baekpica/ds4@538fe62`](https://github.com/Baekpica/ds4/commit/538fe622e73e5735d59c8945946e2831305b1cd8)
changed vision attention from one warp rereading K/V for every query to an
8-query by 32-key shared-memory tile. It retains the same online-softmax order
and never allocates an N-by-N score matrix. A 67-row, two-image-segment test at
the model's actual 16 heads and 72-value head width was bit-exact against the
previous path. NCU measured 24.67 us for the tiled kernel versus 31.81 us for
the previous kernel on that check (-22.4%).

A same-binary 3-by-3 API A/B used a 4,408-patch document image, 1,159 prompt
tokens, 64 output tokens, and disabled live/disk prefix reuse:

| Vision attention | Mean prefill | Mean TTFT | Mean decode |
|---|---:|---:|---:|
| 8-query / 32-key tiled | 494.47 tok/s | 10.262 s | 13.77 tok/s |
| Previous per-query path | 494.50 tok/s | 14.143 s | 14.70 tok/s |

The controlled result is a 27.45% TTFT reduction for this image size. Decode
is reported for completeness but is not attributed to the vision kernel; the
64-token MTP tails were thermally variable. This is exact full attention and
still performs O(P^2) arithmetic. It is a shared-memory tiled online-softmax
kernel, not a tensor-core FlashAttention claim.

### Bounded SSD-PLE cache and worker tuning

On 2026-08-29,
[`Baekpica/ds4@1c9628a`](https://github.com/Baekpica/ds4/commit/1c9628ad727c4955f095fc1e8bb9ca9e1feb0213)
retained the existing same-chunk prefetch overlap with token embedding and
decoder layer 0, and changed the bounded page-read worker default from 16 to
32 after controlled tuning. Identical fixed 8,192-token direct runs used the
same binary and input with `--gen-tokens=0`, three runs per setting:

| Setting | Mean prefill |
|---|---:|
| 512 MiB cache, 16 workers | 252.17 tok/s |
| 1,024 MiB cache, 16 workers | 397.72 tok/s |
| 2,048 MiB cache, 16 workers | 450.26 tok/s |
| 2,048 MiB cache, 32 workers | 464.67 tok/s |
| 2,048 MiB cache, 64 workers | 466.88 tok/s |

The 32-worker default retained nearly all of the 64-worker result with less
host submission overhead. A cold 9,854-prompt / 64-output API check using the
2 GiB cache and 32-worker defaults measured 467.2 prefill tok/s, 21.111 s
TTFT, and 25.4 decode tok/s. A synchronized two-request check completed as
`served=2 fallback=0`; all four requests in the smoke sequence completed with
zero request, census, or governor failures. The external 109/115 GiB
reclaim/hard guard remained active and the minimum observed system-available
memory was 8.07 GiB. Direct runs do not measure TTFT or decode throughput, and
these measurements do not claim SSD latency percentiles or sustained
concurrency.

### State-safe two-bank decode row batching

On 2026-08-29,
[`Baekpica/ds4@7ccbfb9`](https://github.com/Baekpica/ds4/commit/7ccbfb99c931d72b09d42d7eea669e31d9b62b51)
replaced the two independent scalar calls for token embedding and all
Hyper-Connection transforms with one actual two-row operation whenever two
Qwen banks are decode-ready. At that commit, PLE, QSA, routed MoE, output
projection, and the remaining Gated DeltaNet stages retained bank-owned arithmetic. This is a
state-safe incremental row batch, not a claim that the complete Qwen graph is
one batched kernel. `DS4_QWEN_NO_ROW_BATCH=1` restores the scalar path.

The Hyper-Connection BF16 projections use a fixed native reduction for decode
so one-row and two-row execution are bit-identical across a two-bank-to-one-bank
transition. The same change superseded the earlier two-row MTP target verifier:
the target now verifies committed tokens sequentially with its ordinary
one-row arithmetic, advances the drafter from the first target hidden row
before evaluating an accepted draft, and no longer allocates a recurrent-state
rollback buffer.

The real-Q5 regression matched both 12-token outputs to their scalar oracles,
forced one request to finish after four tokens while the other continued to
12, retained MTP draft accepts, and passed disk-KV, partial-fork, and graph
retire/rebuild gates. The integrated Hyper-Connection test also matched the
two-row BF16 projections and residuals bit-for-bit against two scalar rows.

A same-process alternating 3-by-3 micro A/B used the two short test prompts
and generated 12 tokens per request on each pass:

| Decode carrier | Three elapsed times | Aggregate output rate |
|---|---:|---:|
| Two-row embedding/HC | 1.059, 1.057, 1.060 s | 22.67 tok/s |
| Scalar per-bank path | 1.150, 1.149, 1.151 s | 20.86 tok/s |

The controlled improvement was **8.65%** under the external 109/115 GiB
reclaim/hard guard. This is a short generation microbenchmark that isolates
the row-path switch; it is not an API prefill, TTFT, long-context decode, or
sustained-concurrency measurement.

### Paired two-bank GDN recurrent update

On 2026-08-30,
[`Baekpica/ds4@a34ef22`](https://github.com/Baekpica/ds4/commit/a34ef2263b466c752e22e7eabe4fb77ddf1ace2e)
placed the two independent Gated DeltaNet recurrent-state updates in one CUDA
grid. Each `(bank, value head)` still owns its original block, state, token
order, and arithmetic; only the duplicate kernel launch was removed. The
input projections are now covered by the later two-bank change below;
convolution, controls, gated norm, and output stay bank-owned.
`DS4_QWEN_NO_GDN_RECURRENT_BANK2=1` restores the two-launch path.

The production-shape CUDA gate used different inputs, controls, and initial
states for the two banks. Both outputs and both final recurrent states matched
two scalar launches bit-for-bit. The guarded real-Q5 regression also retained
both 12-token scalar-oracle streams and passed MTP, recurrent disk-KV,
partial-prefix fork, and graph retire/rebuild checks.

An alternating same-process 3-by-3 A/B isolated only this recurrent launch:

| GDN recurrent carrier | Three elapsed times | Aggregate output rate |
|---|---:|---:|
| Paired two-bank grid | 1.036, 1.031, 1.030 s | 23.24 tok/s |
| Two scalar launches | 1.053, 1.054, 1.051 s | 22.80 tok/s |

The measured improvement was **1.96%** under the external 109/115 GiB
reclaim/hard guard. This is a short two-request decode microbenchmark, not an
API prefill, TTFT, long-context decode, or sustained-concurrency result.

### Two-bank Q8 output projection

On 2026-08-30,
[`Baekpica/ds4@8e67e2d`](https://github.com/Baekpica/ds4/commit/8e67e2d)
sent the final Q8 output matrix for two decode-ready banks through the existing
two-row dense path instead of traversing the same weights once per bank. No new
kernel or persistent allocation was added. `DS4_QWEN_NO_OUTPUT_ROW_BATCH=1`
restores the two one-row calls for diagnosis.

At the model's 2,560-element output-head input width, the raw Q8 path matched
two one-row calls bit-for-bit across 514 checked outputs. The guarded
real-Q5 regression retained both scalar-oracle token streams and passed its
two-bank MTP, recurrent disk-KV, partial-prefix fork, and graph lifecycle gates.

An alternating same-process 3-by-3 A/B isolated only the output projection:

| Output projection | Three elapsed times | Aggregate output rate |
|---|---:|---:|
| Two-row Q8 path | 1.012, 1.012, 1.014 s | 23.70 tok/s |
| Two one-row calls | 1.039, 1.036, 1.034 s | 23.15 tok/s |

The measured improvement was **2.35%** under the external 109/115 GiB
reclaim/hard guard. This remains a short two-request decode microbenchmark;
it does not establish API prefill, TTFT, long-context decode, or sustained
concurrency performance.

### Two-bank SSD-PLE gather

On 2026-08-30,
[`Baekpica/ds4@64fd836`](https://github.com/Baekpica/ds4/commit/64fd836d129e0f37e8050b804dc5c7ba24710944)
combined the two decode-ready banks' 32 SSD-PLE row descriptors into one CUDA
gather and one bit-exact BF16-to-F32 promotion. Each bank still derives and
prefetches its own row IDs and retains independent key/value projections,
gating, convolution state, and residual arithmetic. No sidecar residency bound
or cache policy changed. `DS4_QWEN_NO_PLE_GATHER_BANK2=1` restores two gathers
for diagnosis.

The PLE primitive gate retained exact BF16 promotion and its existing
chunk/decode convolution-state checks. The guarded real-Q5 regression retained
both scalar-oracle token streams and passed two-bank MTP, recurrent disk-KV,
partial-prefix fork, and graph retire/rebuild checks.

An alternating same-process 3-by-3 A/B isolated only the gather change:

| SSD-PLE gather | Three elapsed times | Aggregate output rate |
|---|---:|---:|
| One two-bank gather | 1.001, 1.000, 1.002 s | 23.98 tok/s |
| Two one-bank gathers | 1.009, 1.006, 1.007 s | 23.82 tok/s |

The measured improvement was **0.64%** under the external 109/115 GiB
reclaim/hard guard. This is a short two-request decode microbenchmark, not an
SSD latency-percentile, API prefill/TTFT, long-context decode, or sustained
concurrency measurement.

### Paired two-bank Q5_0 expert-down tail

On 2026-08-30,
[`Baekpica/ds4@903d4d0`](https://github.com/Baekpica/ds4/commit/903d4d0e3cff9499ca9502c86f285cb0def31461)
placed the two decode-ready banks' assignment-major Q5_0 expert-down tails in
one two-dimensional CUDA grid. Each block still owns one bank, assignment,
and output tile and retains the scalar kernel's arithmetic. Router, main
routed MMQ, and shared-expert work remain independent. The expert-major
prefill fallback and the Q8_0 MTP tail are unchanged.
`DS4_QWEN_NO_Q5_TAIL_BANK2=1` restores two complete scalar MoE paths for
diagnostic A/B runs.

A production-shaped CUDA gate used different activation rows, expert IDs, and
initial output values for each bank; all 51,200 outputs per bank matched two
scalar launches bit-for-bit. The guarded real-Q5 regression retained both
target token streams and passed two-bank MTP, recurrent disk-KV,
partial-prefix fork, and graph retire/rebuild checks.

An alternating same-process 3-by-3 A/B isolated the paired tail path:

| Q5_0 tail carrier | Three elapsed times | Aggregate output rate |
|---|---:|---:|
| Paired two-bank grid | 0.976, 0.975, 0.973 s | 24.62 tok/s |
| Two scalar MoE paths | 0.992, 0.991, 0.991 s | 24.20 tok/s |

The measured improvement was **1.74%** under the external 109/115 GiB
reclaim/hard guard, with all three paired rounds faster. This remains a short
two-request decode microbenchmark; it is not an API prefill, TTFT,
long-context decode, or sustained-concurrency result.

### Two-bank QSA Q/gate projection

On 2026-08-30,
[`Baekpica/ds4@ef7a696`](https://github.com/Baekpica/ds4/commit/ef7a696ebd454861e6879418892848d5af05dc01)
sent the two decode-ready banks' QSA Q/gate projection through one two-row Q8
matmul in each of the 12 QSA layers. It reuses the carrier graph's contiguous
Hyper-Connection rows and the existing Q8 path; no new kernel or persistent
allocation was added. Index/KV history, RoPE, attention, and output remain
bank-owned. `DS4_QWEN_NO_QSA_QPROJ_BANK2=1` restores two independent
projections for diagnostic A/B runs.

The real layer-3 Q8 gate compared the two-row projection with two one-row calls
and matched all 98,304 output bytes exactly. The guarded real-Q5 regression
retained both target token streams and passed two-bank MTP, recurrent disk-KV,
partial-prefix fork, and graph retire/rebuild checks.

An alternating same-process 3-by-3 A/B isolated the QSA projection path:

| QSA Q/gate projection | Three elapsed times | Aggregate output rate |
|---|---:|---:|
| Two-row Q8 path | 0.955, 0.955, 0.954 s | 25.15 tok/s |
| Two one-row calls | 0.973, 0.974, 0.974 s | 24.65 tok/s |

The measured improvement was **2.01%** under the external 109/115 GiB
reclaim/hard guard, with all three paired rounds faster. This is a short
two-request decode microbenchmark; it is not an API prefill, TTFT,
long-context decode, or sustained-concurrency result.

### Two-bank Q5_K routed main

On 2026-08-30,
[`Baekpica/ds4@61af3f7`](https://github.com/Baekpica/ds4/commit/61af3f7fa26998f8daeb8617c80ecaeedaf28c9a)
kept each decode-ready bank's F32 router, top-k selection, Q4_K gate/up,
weighted SwiGLU, and packed activation construction independent, then joined
the two packed 10-row Q5_K expert-down main worklists in one existing bounded
routed-MMQ call. The second result is copied back to its bank before the paired
Q5_0 tail; shared-expert arithmetic remains bank-owned. No new CUDA kernel or
persistent allocation was added. `DS4_QWEN_NO_MOE_MAIN_BANK2=1` restores two
independent routed-main calls while retaining the paired tail.

The compact real-weight layer-2 gate used two distinct inputs and matched both
10-by-2,560 routed-main output tables bit-for-bit against two scalar calls. The
guarded real-Q5 regression retained both target token streams and passed
two-bank MTP, recurrent disk-KV, partial-prefix fork, and graph lifecycle
checks.

An alternating same-process 3-by-3 A/B isolated only the routed-main join:

| Q5_K routed main | Three elapsed times | Aggregate output rate |
|---|---:|---:|
| One two-bank worklist | 0.945, 0.944, 0.945 s | 25.40 tok/s |
| Two one-bank worklists | 0.963, 0.963, 0.965 s | 24.91 tok/s |

The measured improvement was **1.95%** under the external 109/115 GiB
reclaim/hard guard, with all three combined rounds faster. This remains a
short two-request decode microbenchmark; it is not an API prefill, TTFT,
long-context decode, or sustained-concurrency result.

### Two-bank router top-k

On 2026-08-30,
[`Baekpica/ds4@bd4d0be`](https://github.com/Baekpica/ds4/commit/bd4d0be)
kept the two F32 router projections bank-owned, staged their logits in two rows
of the existing workspace, and ran the existing router top-k kernel once over
both rows. The second bank's selected IDs and normalized weights are copied
back before its routed MoE work continues. No new CUDA kernel or persistent
allocation was added. `DS4_QWEN_NO_MOE_TOPK_BANK2=1` restores two independent
top-k launches while retaining the other accepted two-bank paths.

Four distinct 512-expert logits rows matched their one-row top-k calls
bit-for-bit. The guarded real-Q5 regression retained both target token streams
and passed two-bank MTP, recurrent disk-KV, partial-prefix fork, and graph
lifecycle checks.

An alternating same-process 3-by-3 A/B isolated only the top-k join:

| Router top-k | Three elapsed times | Aggregate output rate |
|---|---:|---:|
| One two-bank launch | 0.934, 0.931, 0.932 s | 25.74 tok/s |
| Two one-bank launches | 0.946, 0.944, 0.948 s | 25.37 tok/s |

The measured improvement was **1.48%** under the external 109/115 GiB
reclaim/hard guard, with all three combined rounds faster. This remains a
short two-request decode microbenchmark; it is not an API prefill, TTFT,
long-context decode, or sustained-concurrency result.

### Two-bank Gated DeltaNet input projections

On 2026-08-30,
[`Baekpica/ds4@320ee90`](https://github.com/Baekpica/ds4/commit/320ee90)
sent the two decode-ready banks' four Q8 Gated DeltaNet input projections
(`qkv`, `z`, `in_b`, and `in_a`) through the existing two-row dense path in
each of the 36 GDN layers. Bank 1's projected rows are copied back before its
independent stateful convolution and controls; the accepted paired recurrent
update is unchanged. No new CUDA kernel or persistent allocation was added.
`DS4_QWEN_NO_GDN_PROJ_BANK2=1` restores four independent projection calls
while retaining the paired recurrent update.

The compact real-weight layer-0 gate used two distinct inputs and matched both
complete 2,560-value GDN output rows bit-for-bit against two scalar calls. The
guarded real-Q5 regression retained both target token streams and passed
two-bank MTP, recurrent disk-KV, partial-prefix fork, and graph lifecycle
checks.

An alternating same-process 3-by-3 A/B isolated only the projection join:

| GDN input projections | Three elapsed times | Aggregate output rate |
|---|---:|---:|
| Four two-row Q8 calls | 0.878, 0.873, 0.873 s | 27.45 tok/s |
| Eight one-row Q8 calls | 0.935, 0.937, 0.936 s | 25.65 tok/s |

The measured improvement was **7.03%** under the external 109/115 GiB
reclaim/hard guard, with all three two-row rounds faster. This remains a short
two-request decode microbenchmark; it is not an API prefill, TTFT,
long-context decode, or sustained-concurrency result.

### Two-bank F32 router projection

On 2026-08-30,
[`Baekpica/ds4@1c3925b`](https://github.com/Baekpica/ds4/commit/1c3925b67b27adc4aedd472fcee3e62434496d3a)
sent both decode-ready banks' F32 router projections through one existing
two-row dense call with row-stable reduction order, then reused the accepted
combined top-k path. Bank-owned routing inputs and downstream expert work
remain independent. No new CUDA kernel or persistent allocation was added.
`DS4_QWEN_NO_MOE_ROUTER_BANK2=1` restores two independent router projections
while retaining the combined top-k path.

The compact real-weight gate matched both complete 512-value router-logit rows
and normalized top-k results bit-for-bit against two scalar calls. The guarded
real-Q5 regression retained both target token streams and passed two-bank MTP,
recurrent disk-KV, partial-prefix fork, and graph lifecycle checks.

An alternating same-process 3-by-3 A/B isolated only the router projection:

| F32 router projection | Three elapsed times | Aggregate output rate |
|---|---:|---:|
| One two-bank projection | 0.866, 0.867, 0.868 s | 27.68 tok/s |
| Two one-bank projections | 0.871, 0.870, 0.871 s | 27.57 tok/s |

The measured improvement was **0.40%** under the external 109/115 GiB
reclaim/hard guard, with all three combined rounds faster. This remains a
short two-request decode microbenchmark; it is not an API prefill, TTFT,
long-context decode, or sustained-concurrency result.

### Full-window prefill, narrow quality check, and SSD-PLE latency

A teacher-forced full-model comparison scored 2,048 tokens from each of two
fixed text fixtures on the higher-precision MQ-Q6 artifact and the target
MQ-Q5 artifact:

| Fixed slice | MQ-Q6 avg NLL / PPL | MQ-Q5 avg NLL / PPL | Q5-Q6 avg NLL |
|---|---:|---:|---:|
| Long-form essay | 2.184138 / 8.882991 | 2.140368 / 8.502567 | -0.043770 (-2.00%) |
| Repetitive structured security fixture | 0.068999 / 1.071435 | 0.081061 / 1.084437 | +0.012062 (+17.48%) |
| Equal-token aggregate (4,096 tokens) | 1.126569 / 3.085053 | 1.110714 / 3.036527 | -0.015854 (-1.41%) |

The two slices show no Q5 quality collapse in this narrow regression, but they
are not a representative evaluation suite and do not establish general Q5
superiority. The structured fixture is highly repetitive, so its low absolute
perplexity is useful only for the paired comparison.

The same Q5 artifact then completed an exact 262,144-token direct prefill with
`--gen-tokens=0`, an 8,192-token Qwen chunk, the bounded 2,048 MiB PLE cache,
and 32 PLE workers. It sustained **277.06 prefill tok/s**. The dumped final
output contained all 248,320 finite logits; the recorded and independently
recomputed argmax were both token 264. The run completed under the external
115 GiB memory guard with `DS4_MEMGOV=observe`; the lowest sampled
system-available memory was 7.8 GiB. This establishes full-windo

### 2026-09-02 prefill optimization campaign (ds4-dfm-rs)

Five incremental rounds on [`ds4-dfm-rs`](https://github.com/Baekpica/ds4-dfm-rs)
(`d9f96a8` → `230655d`), each measured against the previously adopted commit on
one NVIDIA DGX Spark, the canonical two-bank serving command
(`DS4_QWEN_BATCH=1`, 8,192-token prefill chunks, 512 MiB PLE cache with 16 page
workers, `--mtp-draft 2`, 196,608 configured context), three fresh workers with
fresh disk-KV directories per variant, `thinking` disabled, `max_tokens 1`, and
the API's `timings.prefill_tok_s`. The rounds were developed on the same-layout
[Uncensored sibling](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF)
`MQ-Q5-SSD-PLE-BF16` artifact and the final binary was then measured on this
artifact. Two prompts were used: a 31,000-character prompt made of one
7,750-character passage repeated four times (8,259 tokens; the last three copies
hit the PLE page cache, which reproduces the regime of the 2026-09-01 numbers
above) and a 23,150-character markdown/code document (7,937 tokens) whose
n-grams miss the PLE cache, so ~6.5 s of its prefill is SSD page reads.

| Round | Change | Commit | 8,259-token repeated prompt (median of 3, Uncensored sibling) | 7,937-token cold-PLE prompt (median of 3, Uncensored sibling) |
|---|---|---|---:|---:|
| baseline | `81e2cfa` | – | 556.2 tok/s | 417.6 tok/s |
| 1 | Q5_0/Q8_0 expert-down tail fused into the MMQ worklist kernel | `d9f96a8` | 635.0 (+14.2%) | 452.6 (+8.4%) |
| 2 | shared expert gate/up/down on the MMQ tier (K=640 down as one fused-tail expert) | `6a3ae08` | 740.5 (+16.6%) | 503.5 (+11.2%) |
| 3 | counting-sort expert id maps (bit-identical) + redundant non-finite passes dropped | `d575263` | 792.7 (+6.9%) | 520.0 (+3.3%) |
| 4 | Q5_K/Q8_0 gate/up on the compact worklist pair; PLE key/value and QSA k/v pairs on MMQ | `6e14b33` | 883.5 (+11.2%) | 591.7 (+13.8%) |
| 5 | hyper-connection rows converted to BF16 once | `230655d` | **910.7 (+3.3%)** | **597.1 (+0.6%)** |

**This artifact, same day, owner swapped to it:** unmodified `81e2cfa` measured
558.1 tok/s (554.6, 558.1, 559.5) and the final `230655d` **908.1 tok/s** (906.3,
908.1, 911.4) on the repeated prompt (**+62.7%**), and 584.4 tok/s (570.1, 584.4,
611.5) on the cold-PLE markdown prompt. The 64-token greedy continuation of the
repeated prompt is byte-identical between the two binaries on this artifact.
Per-round percentages are against the preceding round's rerun on the same day
and are not additive across days.

On the Uncensored sibling, rounds 1–3 and 5 leave the 64-token continuation
unchanged and round 4 changes it after the first ten tokens; on this artifact
the continuation is unchanged end to end, and disabling only round 4's pair
split on the final binary flips it again (at 771.4 tok/s), so that prompt sits
near a greedy tie under ~1e-3 numeric differences rather than tracking a
systematic drift. The new Q5_K/Q8_0 worklist pairs are bit-identical to the
generic routed matmul on the fixture; the PLE/QSA pairs move from Q8_0 DP4A
kernels to the Q8_1-activation MMQ tier used by every other projection.
A Linux-AIO rewrite of the PLE page workers was measured and rejected
(cold-PLE prompt 593 → 511 tok/s on the sibling); the raw NVMe sustains 53K–91K
random 4 KiB IOPS at 16–64 threads, so the remaining cold-PLE cost is a
pipeline limit, not a device limit.

### 2026-09-06 prefill campaign (ds4-dfm-rs)

Three more rounds on [`ds4-dfm-rs`](https://github.com/Baekpica/ds4-dfm-rs)
(`0510117` → `974d706`, [`docs/qwen38-prefill-2026-09-06.md`](https://github.com/Baekpica/ds4-dfm-rs/blob/2efbe15/docs/qwen38-prefill-2026-09-06.md)),
after the 2026-09-04 long-context rounds (PLE page cache, GDN, fused QSA,
block scorer, hyper-connection traffic). Same DGX Spark / GB10, this
`MQ-Q5-SSD-PLE-BF16` artifact, the resident VMM weight owner with aligned Q8
artifacts, and the production worker environment (8,192-token prefill chunks,
2 GiB PLE cache, 16 page workers). Each round was measured against the
preceding one as a cold single-shot `ds4-bench` prefill (one fresh worker per
run, three runs, medians; 32 greedy tokens after the prefill):

| Round | Change | Commit | cold 8,192-token prefill | cold 65,536-token prefill |
|---|---|---|---:|---:|
| baseline (`main`) | – | `0510117` | 1214.8 tok/s | 1382.7 tok/s |
| 1 | prompt opens with a 2,048-row chunk so the SSD-PLE reads of the full-size chunk behind it overlap the opening chunk's decoder layers | `7a5f872` | 1325.7 (+9.1%) | 1401.1 (+1.3%) |
| 2 | expert-down main read in place from the SwiGLU rows (no pack kernel); routed gate/up activation quantized once per token and scattered into the expert-sorted layout | `9696fd6` | 1348.1 (+1.7%) | 1422.4 (+1.5%) |
| 3 | routed expert sum, shared-expert sigmoid gate and residual add in one pass | `974d706` | **1362.1 (+1.0%)** | **1439.7 (+1.2%)** |

Cumulative: **1214.8 → 1362.1 tok/s (+12.1%)** on the cold 8K shape and
**1382.7 → 1439.7 tok/s (+4.1%)** at 64K; a cold 196,608-token prefill
went 1297.5 → 1325.6 tok/s. All three rounds are bit-identical to the
previous kernels on the fixtures (`tests/test_qwen4exp_moe`); the decode
after each prefill is unchanged (~24 tok/s without MTP in `ds4-bench`).

On the production server shape (the canonical two-bank command, 196,608
configured context, `--mtp-draft 2`, three fresh workers per binary,
`thinking` disabled, `max_tokens 1`, `timings.prefill_tok_s`), the same two
prompts as the 2026-09-02 table:

| binary | 8,259-token repeated prompt (median of 3) | 7,937-token cold-PLE markdown prompt (median of 3) | 8,036-token x-prompt (median of 3) |
|---|---:|---:|---:|
| `0510117` (before) | 1363.3 | 1406.1 | 1570.0 |
| `974d706` (after) | **1394.2 (+2.3%)** | **1467.5 (+4.4%)** | **1584.2 (+0.9%)** |

Decode on the 8,259-token chat prompt with 256 greedy tokens and MTP draft 2:
31.3 → 29.8 tok/s (52.7 → 53.7 ms per MTP step). The 256-token continuation
differs between the two binaries and so does its MTP acceptance (1.65 → 1.60
tokens per step), which is the whole decode difference: the new binary with
`DS4_QWEN_PREFILL_OPENING=0` reproduces the old text and the old decode rate
(31.4 tok/s, 1.65 tokens per step) exactly, so the opening chunk's prefill
GEMM shapes move a near-tie token, as a different `DS4_QWEN_PREFILL_CHUNK`
would; the kernels themselves are bit-identical.

Two attempts in the same campaign were measured and rejected: reading the
token-compact activation through a per-column map inside the MMQ tile
(gate/up kernel +22%, prefill −3%) and folding the weighted SwiGLU into the
up launch's store (bit-identical, but the scattered gate reads cost more than
the coalesced pass they replaced: cold 8K −0.7%).

### 2026-09-06 r4 (D2R, same day)

Three further rounds on `feature/qwen-prefill-opt-20260906-r4`
(`d9989bb` / `c00eacd` / `abdf25c`): dense D2R for K=2560 qkv/z/q, one
HC-mix Q8 emit per layer, and o_proj K=6144 onto D2R. Same-hour cold
`ds4-bench` medians of three on the aligned-Q8 owner (do not stack onto
the `0510117` → `974d706` series as one same-hour measurement):

| Round | Change | Commit | cold 8,192-token prefill | cold 65,536-token prefill |
|---|---|---|---:|---:|
| 4 | admit K=2560 onto dense D2R | `d9989bb` | 1391.1 (+2.8%) | 1504.6 (+5.1%) |
| 5 | HC mix emits Q8 once; D2R preq K%128 | `c00eacd` | 1406.0 (+0.8%) | 1518.4 (+0.9%) |
| 6 | D2R max K 4096 → 6144 (o_proj) | `abdf25c` | **1431.5 (+1.3%)** | **1554.8 (+2.5%)** |

The republished 2K–64K incremental sweep (MTP draft 2, 128 greedy tokens
per frontier, 2 GiB PLE cache) is **1,235.9 tok/s mean prefill** and
**28.4 tok/s mean decode**.

### 2026-09-07 prefill campaign (ds4-dfm-rs)

Three more rounds on [`ds4-dfm-rs`](https://github.com/Baekpica/ds4-dfm-rs)
(`ef37468` → `6e036c4`, [`docs/qwen38-prefill-2026-09-07.md`](https://github.com/Baekpica/ds4-dfm-rs/blob/1eb3773/docs/qwen38-prefill-2026-09-07.md)),
after the 2026-09-06 rounds. Same DGX Spark / GB10, this
`MQ-Q5-SSD-PLE-BF16` artifact, the resident VMM weight owner with aligned Q8
artifacts, and the production worker environment (8,192-token prefill chunks,
2 GiB PLE cache, 16 page workers). Each round was measured against its own
kill switch on the same binary in the same hour (off / on interleaved) as a
cold single-shot `ds4-bench` prefill (one fresh worker per run, three runs,
medians; 32 greedy tokens after the prefill):

| Round | Change | Commit | cold 8,192-token prefill (off → on) | cold 65,536-token prefill (off → on) |
|---|---|---|---:|---:|
| 1 | weighted SwiGLU quantized straight into the fused expert-down's Q8 operands (no F32 mid, two gathered quantize passes gone) | `1273c68` | 1432.9 → 1457.9 (+1.7%) | 1559.5 → 1581.5 (+1.4%) |
| 2 | fused QSA attention: four barriers per tile instead of six, XOR-swizzled partial stores (ncu-guided) | `30826d6` | 1455.0 → 1474.8 (+1.4%) | 1576.9 → 1597.6 (+1.3%) |
| 3 | SSD-PLE gather leases a whole tile under one store lock, 16,384-row tiles | `6e036c4` | 1482.9 → 1504.5 (+1.5%) | 1615.2 → 1644.3 (+1.8%) |

Cumulative, `ef37468` against `6e036c4` interleaved in the same hour:
**1,429.9 → 1,504.6 tok/s (+5.2%)** on the cold 8K shape, **1,557.2 →
1,648.4 tok/s (+5.9%)** at 64K, and 1,439.9 → 1,513.0 tok/s at 196,608 tokens (+5.1%, one run each). All three rounds
are bit-identical to the kernels they replace (fixtures in
`tests/test_qwen4exp_moe`, `tests/test_qwen4exp_qsa`,
`tests/test_qwen4exp_ple{,_cuda}`, and the 8,192-token frontier logits of
`ds4-bench --dump-frontier-logits-dir` byte-compared across rounds); decode
after each prefill is unchanged (~24 tok/s without MTP in `ds4-bench`).

On the production server shape (the canonical two-bank command, 196,608
configured context, `--mtp-draft 2`, three fresh workers per binary,
`thinking` disabled, `max_tokens 1`, `timings.prefill_tok_s`), the same
prompts as the 2026-09-02 and 2026-09-06 tables:

| binary | 8,259-token repeated prompt (median of 3) | 7,937-token cold-PLE markdown prompt (median of 3) | 8,036-token x-prompt (median of 3) |
|---|---:|---:|---:|
| `ef37468` (before) | 1481.7 | 1555.9 | 1677.7 |
| `6e036c4` (after) | **1559.9 (+5.3%)** | **1652.6 (+6.2%)** | **1761.5 (+5.0%)** |

Decode on the 8,259-token chat prompt with 256 requested greedy tokens and
MTP draft 2 (the continuation stops at the end token after 175 tokens):
30.3 → 30.4 tok/s, 1.63 tokens per step both, and the 175-token text is
byte-identical between the two binaries, as the kernels are.

Measured and rejected in the same campaign: `ld.global.nc.L1::no_allocate`
for the QSA key / value gathers (probe kernel 62.5 → 84.0 ms) and a value
tile parked in shared memory on its own (neutral; kept only as the structure
the barrier cut needs).

The 2K–64K incremental sweep (MTP draft 2, 128 greedy tokens per frontier,
2 GiB PLE cache, aligned-Q8 owner) on `6e036c4`: **1,248.8 tok/s
mean prefill** and **28.4 tok/s mean decode** (was 1,235.9 /
28.4 on `abdf25c`); 2K 1,050.1 / 64K 1,182.9 tok/s prefill.
