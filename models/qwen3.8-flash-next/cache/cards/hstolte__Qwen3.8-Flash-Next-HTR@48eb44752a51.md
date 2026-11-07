---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: text-generation
inference: false
tags:
- htr
- halo-trellis
- quantized
- strix-halo
- gfx1151
- custom-runtime
---

![Qwen3.8 Flash Next — Halo Trellis](media/hero.png)

# Qwen3.8-Flash-Next-HTR

**HTR release v0.2.2 — native serving, Octave KV and bounded PLE caching.** [Runtime release](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.2.2) · [Release history](docs/RELEASES.md).

HawgAuto's Halo Trellis quantization of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), built for **AMD Strix Halo / Radeon8060S** with the HTR-aware vLLM runtime. This is a serving update: **all checkpoint weights, tokenizer/configuration files and quantization are unchanged**.

## Recommended model — Htr5Core

**[Htr5Core](Htr5Core/) is the recommended inference model.** The quick start downloads this complete variant. **[Bf16Core](Bf16Core/)** is a complete research reference restoring the BF16 core; its GPU serving, performance and maximum context remain separately unqualified. [Variant details](docs/VARIANTS.md). Download one variant, not the whole repository; the root checkpoint remains a compatibility copy of Htr5Core.

## What v0.2.2 brings

- Ordinary **`vllm serve`** with native sampling, context, KV budget/format, concurrency and prefill controls. A special launcher is no longer required.
- Optional **`octave_k3v4`, `octave_k3v3`, `octave_k3v3_compact`** KV, native gfx1151 kernels, deterministic window compaction and the masked-unused-slot NaN fix.
- Accepted sampler/draft improvements, **wave32 GDN**, **bit-exact QSA preparation**, and the **exact HIP draft filter enabled by default**.
- Optional **bounded CPU-RAM PLE hot-row cache**, disabled by default; large PLE tables remain SSD-backed.
- Retained BF16, scaled8-bit and rotated4-bit KV options, MTP0/MTP3, packed HTR kernels, GPTQ8 hyperconnections and OpenAI-compatible tools/reasoning.
- Persistent per-configuration compiler-cache reuse. No new persistent unpacked16-bit weight shadow or mandatory downloaded seed.

## At a glance — Htr5Core

- **Parameters:** **125B with 6B activated, plus 51B n-gram embeddings and 4B MTP. Vision included.**
- **Platform:** Linux, AMD Strix Halo/gfx1151,128GB unified system memory and fast local SSD storage. Leave memory for model loading, runtime, KV, compilation and the OS; do not co-load another large shared-memory GPU workload.
- **Storage:**173.759GB complete checkpoint. Its102.4GB PLE tables stay on SSD; the optional hot-row cache consumes only its selected CPU-RAM budget.
- **Current qualification:**18 configurations (16 new-image plus2 inherited controls),324 requests,90 reconstructed SSE streams and five exact matched parity comparisons. The v0.2.2 matrix used one sequence, configured32768 context and2GiB KV. Native flags remain user-controlled; this tested geometry is not an artificial parser cap.
- **Context and concurrency history:** prior BF16 1M and quantized-KV2M/3M/4M recipes, and earlier concurrency results, remain available below on their original pinned images. They are not new v0.2.2 capacity or concurrency claims.
- **Quick start:** warm the native configuration once, keep `/workspace`, then restart with the same settings to reuse compiled programs. The observed v0.2.2 native fresh/seeded pair reached readiness in446.810/143.003s; one pair is not a universal startup guarantee. New shapes may compile.

[Complete v0.2.2 configuration/timing tables](https://github.com/HawgAuto/vllm-htr/blob/main/htr_runtime/releases/v0.2.2/QUALIFICATION.md) · [Machine-readable qualification](evidence/v0.2.2-release-qualification.json).

## Required runtime

HTR is a custom format, not GGUF. Use [vLLM HTR v0.2.2](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.2.2), not stock Transformers/vLLM or llama.cpp. Executable source: [`9a4917f41101926c8cb23d630018b8e5e4aec44c`](https://github.com/HawgAuto/vllm-htr/tree/9a4917f41101926c8cb23d630018b8e5e4aec44c). Public image alias: `ghcr.io/hawgauto/vllm-htr:v0.2.2`; use the exact digest below. No GitHub login, host compiler or separate launcher download is required.

The native path inherits checkpoint sampling defaults unless overridden: temperature0.7, top-p0.8, top-k20 for Htr5Core. The older maintainer-selected1.0/0.95 profile remains an optional native override, not a forced default. Thinking/template settings are unchanged. [Sampling and full flag reference](docs/RUNNING.md#sampling-for-existing-clients).

## Quick start and local cache reuse

### 1. Download Htr5Core and pull the runtime

```sh
VARIANT=Htr5Core
hf download hstolte/Qwen3.8-Flash-Next-HTR \
  --revision v0.2.2 --include "$VARIANT/*" --include "meta.yaml" \
  --local-dir ./Qwen3.8-Flash-Next-HTR

docker pull ghcr.io/hawgauto/vllm-htr@sha256:034898ebd182a240fa403ad5ea24bea3f647d933571e43420a27daf9af079d1c
```

Already have a verified Htr5Core checkpoint? Keep it; no weight redownload is needed. The original root checkpoint is Htr5Core too. Model and runtime versions are coordinated serving releases, not new quantizations.

### 2. Start the native runtime

Use absolute paths. The workspace must be writable by the image's UID1000. Replace `renderD128` with the AMD render node if needed. Run this in its own shell, or save it as a script:

```bash
#!/usr/bin/env bash
set -euo pipefail
IMAGE=ghcr.io/hawgauto/vllm-htr@sha256:034898ebd182a240fa403ad5ea24bea3f647d933571e43420a27daf9af079d1c
MODEL=${MODEL:-"$PWD/Qwen3.8-Flash-Next-HTR/Htr5Core"}
WORKSPACE=${WORKSPACE:-"$PWD/htr-cache/Htr5Core-v022"}
mkdir -p "$WORKSPACE"
exec docker run --rm --init --pull=never --name vllm-htr \
  --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add "$(stat -c %g /dev/kfd)" \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --pids-limit 8192 --shm-size 16g --ulimit core=0 \
  --security-opt no-new-privileges:true \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$MODEL,dst=/model,readonly" \
  --mount "type=bind,src=$WORKSPACE,dst=/workspace" \
  "$IMAGE" /model --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 --max-num-seqs 1 \
  --max-num-batched-tokens 2048 --kv-cache-memory-bytes 2147483648
```

This launches BF16 KV, MTP3, one sequence and a2GiB KV pool. Retain the workspace: first start compiles, subsequent starts reuse matching programs. This is compilation caching, not an expanded weight cache. There is no new `--build-seed` wrapper in the native path. Do not pass old launcher-only `--cpu-check`, `--mtp`, `--context` or `--kv-cache-gib` flags to `vllm serve`.

To turn MTP off, append `--speculative-config null`. Select KV with `--kv-cache-dtype`; use `--max-model-len`, `--kv-cache-memory-bytes`, `--max-num-seqs` and `--max-num-batched-tokens` for native budgets. You can save the configuration in YAML with `--config`. For optional16MiB PLE hot-row caching, add `-e HTR_PLE_ROW_CACHE_BYTES=16777216` before the image reference. Keep different profiles/variants in their own workspace; new request shapes can compile. [Complete guide and optional source build](docs/RUNNING.md).

### 3. Connect

After readiness, use `http://127.0.0.1:8000/v1` with model ID `qwen38-htr` (or discover it from `/v1/models`):

```sh
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-htr","messages":[{"role":"user","content":"Explain how speculative decoding works."}],"max_tokens":512}'
```

Thinking consumes output tokens. Supply OpenAI-compatible `tools` normally. The bind is loopback-only; add authentication and a reverse proxy before external exposure.

## Preserved v0.2.0 performance and concurrency measurements

The following rows remain tied to their original cached-process, greedy, fixed-length v0.2.0 qualification stack. They are not new v0.2.2 timings or sampled-chat promises. The old launcher admitted MTP0 1–32 and MTP3 1–4 streams; the current native configuration remains user-controlled, with new release qualification scoped above.

| Measured performance on Radeon 8060S | MTP0 | MTP3 |
|---|---:|---:|
| **Decode at 2K input** | **33.78 tok/s** | **43.47 tok/s** |
| **Decode at 8K input** | **33.81 tok/s** | **55.28 tok/s** |
| **Prefill at 2K input** | **1,020.06 tok/s** | **1,234.60 tok/s** |
| **Prefill at 8K input** | **1,227.13 tok/s** | **1,147.94 tok/s** |
| **Concurrent streams supported by the launcher** | **1–32** | **1–4** |
| **4-stream decode: all four streams active** | **68.90 tok/s total** | **52.42 tok/s total** |
| Mean per stream during that decode window | 17.22 tok/s | 13.10 tok/s |
| 4-stream decode: whole wave, including overlapping prefill stalls | 40.18 tok/s total | 26.40 tok/s total |
| 4-stream output / full request-wave wall time | 22.28 tok/s total | 18.80 tok/s total |

**How to read these speeds:** single-request rows use 2,048 / 8,192 input tokens and 128 output tokens. The four-stream rows use a mixed 512 / 2,048 / 8,192 / 8,192-token workload. “All four streams active” measures only simultaneous decoding; the whole-wave and full-wall rows show the effect of prompt processing and scheduling. These are cached-process, greedy fixed-length measurements from the v0.2.0 qualification stack, before its final self-check scratch-lifetime cleanup—not sampled-chat promises. [Exact settings, fresh/cached results and observed ranges](docs/SERVING-PERFORMANCE.md).

## Preserved v0.2.1 context and KV-accuracy profiles

Use the [pinned v0.2.1 guide](docs/RUNNING-v0.2.1.md) for the old profile-launcher commands; do not apply those aliases to the new native entrypoint. The tables below preserve their original v0.2.1 image/measurement scope, including the retained v0.2.0 BF16 baseline.

### Choose your context profile

| KV cache | MTP | Max context | YaRN | KV pool |
|---|---:|---:|---:|---:|
| BF16 KV ([retained v0.2.0 profile](docs/RUNNING-v0.2.0.md)) | 0 / 3 | **1,000,000** | ×4 | 27 GiB |
| Scaled 8-bit per-token/head KV (`int8_per_token_head`) | 0 | **2,000,000** | ×8 | 27.75 GiB |
| Scaled 8-bit per-token/head KV (`int8_per_token_head`) | 3 | **2,000,000** | ×8 | 27.75 GiB |
| Rotated 4-bit per-token/head KV (`int4_per_token_head`) | 0 | **4,000,000** | ×16 | 26.875 GiB |
| Rotated 4-bit per-token/head KV (`int4_per_token_head`) | 3 | **3,000,000** | ×16 | 22.375 GiB |

The four quantized-KV profiles use **one sequence** and **2,048-token prefill chunks**. They passed allocation, API readiness and generation; these are capacity-qualified profiles, not filled-window benchmarks. [Copyable launch commands and memory settings](docs/QUANTIZED-KV.md).

### KV accuracy retention versus BF16

| KV cache | MTP | BF16 prediction agreement | Prediction disagreement | Tie-aware agreement |
|---|---:|---:|---:|---:|
| Scaled 8-bit | 0 | **98.18%** (486/495) | **1.82%** | 99.80% |
| Rotated 4-bit | 0 | **98.18%** (486/495) | **1.82%** | 99.39% |
| Scaled 8-bit | 3 | **98.79%** (489/495) | **1.21%** | 99.80% |
| Rotated 4-bit | 3 | **98.38%** (487/495) | **1.62%** | 99.60% |

Measured on **495 teacher-forced prompt positions** against the same Htr5Core model with BF16 KV. Strict agreement compares complete top-choice sets; tie-aware agreement permits a shared tied top token. Disagreement is the measured prediction change, not a broad task-accuracy loss. All six functional fixtures passed in every compared cell. [Exact counts, logprob deltas and evidence](evidence/v0.2.1-release-qualification.json).

This is a serving update: **Htr5Core weights, tokenizer and weight quantization are unchanged**. Vision weights/processor assets remain included.

## Quantization recipe

The recommended Htr5Core model uses a mix of precisions rather than treating every component the same.

| Component | Htr5Core precision |
|---|---|
| Selected dense language-model weights / shared experts | HTR5 (5-bit) |
| Routed experts | HTR4 (4-bit) |
| Output head | HTR4 (4-bit) |
| Hyperconnections | GPTQ8 (8-bit integer) |
| PLE, vision tower and merger | Original BF16 |
| Norms, supporting tensors and integer controls | Retained component formats |
| Speculative draft | Included mixed-precision MTP draft |

## Method and lineage

HTR combines calibration-guided error correction with a trellis-based weight format designed for Strix Halo. The goal is to reduce weight storage while keeping the model close to its reference behavior.

Its lineage includes [exlr8](https://github.com/kindlingai/exlr8), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [QTIP](https://arxiv.org/abs/2406.11235). See the [methodology](https://github.com/HawgAuto/HTR/blob/main/docs/METHODOLOGY.md) and [full acknowledgments](https://github.com/HawgAuto/HTR/blob/main/docs/LINEAGE.md).

![The HTR quantization pipeline](media/method-map.png)

<a id="size-memory-and-1m-context"></a>

## Size, memory and context profiles

The **Htr5Core** checkpoint contains **37 weight files**: 23 target safetensors shards, 13 indexed HTR expert shards and one draft safetensors shard. An additional 26 files supply the configuration, tokenizer, processors, manifests and license. Shard consolidation is lossless, not requantization.

| Download component | Files | GB | GiB |
|---|---:|---:|---:|
| Target checkpoint (including PLE and vision) | 23 | 109.027 | 101.539 |
| Routed experts | 13 | 60.705 | 56.536 |
| MTP draft | 1 | 3.978 | 3.704 |
| **Complete checkpoint + supporting files** | **63** | **173.759** | **161.826** |

Exact checkpoint total: **173,759,296,105 bytes**; card, artwork and reports are additional. The largest shard is 4.997 GB. [File sizes and SHA-256](evidence/checkpoint-manifest.json) · [Sizing data](evidence/package-sizing.json)

**PLE is already included in these totals.** Its n-gram embedding tables account for 102.400 GB (95.368 GiB) of the target payload. The runtime reads requested PLE rows from SSD instead of placing the entire table on the GPU. Keep the model on a fast local SSD; download size is not resident GPU memory.

### Resident memory and context profiles — Htr5Core

The intended host has **128 GB shared system memory**. On the historical v0.2.0/default MTP3 profile, the engine logged **65.68 GiB model-load allocation**, **8 GiB BF16 KV**, and **0.19 GiB graph memory**. Those are separate allocations, not a total-RAM or co-loading allowance: the runtime, compilation, activations and filesystem cache also need headroom. Keep the large n-gram tables on a fast local SSD and avoid another large shared-memory GPU workload.

**Historical v0.2.1 quantized-KV profiles:** choose 2M, 3M or 4M from the [v0.2.1 profile table](#choose-your-context-profile). The cache budget and MTP mode must match.

**Retained BF16 1M context:** use `--context 1000000 --yarn-factor 4 --kv-cache-gib 27`. MTP0/MTP3, with either one or four slots, passed the public builder and cold replay on the v0.2.0 final image. The measured profiles reached approximately **99.2–104.1 GiB peak GPU shared-memory usage (GTT)** across build/replay and used swap during startup/compilation. The documented commands do not impose a Docker memory cap.

With four slots, the shared 27 GiB KV pool has an equal-share estimate of **281,684 tokens per slot for MTP0** or **255,030 for MTP3**; it is not four independent million-token histories. Warmups reached **32,768 input tokens**. Full-million prefill and long-position retrieval have not been measured. [Per-profile memory, capacity and startup results](docs/RUNNING-v0.2.0.md#measured-bf16-kv--yarn-4-profiles).

## Quantization quality

This matched **4-bit dense-layer comparison on WikiText** illustrates the choice of HTR. Only the selected dense weights change; both methods use the same reference model, whose experts are already quantized.

| Dense-layer format | PPL ↓ | KL ↓ | Top-1 agreement ↑ |
|---|---:|---:|---:|
| BF16 dense-layer reference | 4.295 | 0.0000 | 100.00% |
| HTR 4-bit | **4.308** | **0.0284** | **93.77%** |
| GPTQ 4-bit | 4.344 | 0.0520 | 91.68% |

**PPL** measures next-token prediction on the text; lower is better. **KL** measures divergence from the reference over its top 128 tokens plus a remaining-mass bucket; lower is better. It is not full-vocabulary KL. **Top-1 agreement** is how often the models choose the same next token—not task accuracy. Bold values mark the better quantized result.

HTR stays closer to the reference than GPTQ in this comparison. This illustrative HTR4 study is not a final-composite measurement of either shipped variant. These are component-level results, not a final score for the complete mixed-precision release. The [full quality report](docs/RECIPE-COMPARISONS.md) covers the other precision and calibration choices, including their tradeoffs; [source metrics](evidence/recipe-quality.json) retain the detailed measurements.

## Release verification — Htr5Core

The v0.2.2 integration matrix covers ordinary/MTP3 generation, all three Octave formats, retained INT8/INT4 KV, bounded PLE off/on and native fresh/seeded startup. Five matched pairs have exact tokens/text/finish and teacher-probe logprobs; legacy probes did not request logprobs. The EOS-enabled prose probes hit their128-token cap, so they prove bounded nonempty generation rather than natural termination. Tool-result turns stopped normally. [Qualification evidence](evidence/v0.2.2-release-qualification.json) · [Public-command delivery receipt](evidence/v0.2.2-delivery.json).

Single-sequence/configured32768/2GiB-KV execution does not establish new filled-window, concurrency, broad model-accuracy or image/video scores. Earlier qualification and benchmark evidence is retained with its original version/image/sampler labels; the same checkpoint does not make those measurements new-runtime results. FP8 KV and unrelated deferred research are not claimed.

[Historical v0.2.0 qualification](evidence/v0.2.0-release-qualification.json) · [Historical serving measurements](evidence/v0.2.0-serving-summary.json) · [Release history](docs/RELEASES.md) · [Vision and MTP inventory](evidence/vision-and-mtp.json).

## Download statistics

Full-repository downloads include [`meta.yaml`](meta.yaml) automatically. For selective downloads, include it alongside the checkpoint files so [Hugging Face's normal download counter](https://huggingface.co/docs/hub/models-download-stats) can record the request. No extra client telemetry is added; Hub counts are not a count of complete downloads or unique users.

## License and attribution

Source model: Qwen/Qwen3.8-Flash-Next, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Quantization/tooling: HawgAuto HTR; serving infrastructure: the separately pinned HawgAuto vLLM/HaloLoom stack.

The weights and derivative are governed by **Qwen Community License 1.0**, preserved verbatim in [LICENSE](LICENSE). This is not Apache-2.0 or the HTR code repository's MIT license. The license contains conditions for commercial Model-as-a-Service and AI Work Assistant use; review the complete text before deployment. Model authorship and capabilities belong to Qwen; HTR does not change those license obligations.
