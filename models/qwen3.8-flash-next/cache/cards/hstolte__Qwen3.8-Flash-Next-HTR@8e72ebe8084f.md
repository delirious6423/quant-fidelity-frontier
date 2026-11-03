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

**HTR release v0.2.0 — first serving update.** [Runtime release](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.2.0) · [Release history and update roadmap](docs/RELEASES.md).

HawgAuto's Halo Trellis quantization of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), built for **AMD Strix Halo / Radeon 8060S**.

## At a glance

- **Parameters:** **125B with 6B activated, plus 51B n-gram embeddings and 4B MTP. Vision included.**
- **Hardware and memory:** **AMD Strix Halo / Radeon 8060S with 128 GB unified memory**, Linux and a fast local SSD. The default MTP3 launch logs **65.68 GiB model-load allocation**, plus **8 GiB BF16 KV** and **0.19 GiB graph memory**; leave room for runtime, compilation and the OS.
- **Context:** **262,144 tokens by default; a 1M-token profile with YaRN ×4 and a 27 GiB BF16 KV pool.** The pool is shared across concurrent requests. The 1M profiles passed startup, seed building and cold replay, with prefill warmups through 32K. [Memory and context details](#size-memory-and-1m-context).
- **SSD-backed n-gram embeddings:** the **102.4 GB PLE tables stay on SSD**, not resident in GPU memory. The complete checkpoint download is **173.76 GB**.
- **Concurrency:** **MTP0 supports 1–32 streams; MTP3 supports 1–4 streams.** Four-stream measured rates are below; choose the mode for your latency and throughput needs.
- **Quick-start builder included:** run **`--build-seed` once per serving profile**, retain the workspace, then start normally to reuse its compile/autotune cache. The final-image 1M profiles measured **430–554 seconds for build + warmup** and **170–201 seconds to readiness on cold replay**. [Copyable quick start](#quick-start-and-local-cache-builder).

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

## What v0.2.0 brings

- **TAU v2 attention** for eligible sparse-attention prefills, with split-K fallback.
- **Partition4 MoE dispatch** and mode-specific reference reduction geometry.
- **Per-call dense-prefill dequantization**, without a persistent expanded BF16 weight copy; temporary MoE self-check scratch reduces ready-state memory use.
- **Configurable concurrency, context and BF16 KV**, plus a profile-aware local quick-start builder.
- **OpenAI-compatible text, reasoning and structured tool calling**, with MTP0 or three-token MTP3 decoding.

This is a serving update: checkpoint weights, tokenizer and quantization are unchanged. Vision weights and processor assets are included; image/video quality has not been benchmarked by these text-serving tests.

## Required runtime

HTR is a custom format, not GGUF. Use the **HTR-aware vLLM runtime**, not stock llama.cpp, Transformers or vLLM.

- [HTR format and quantization tools](https://github.com/HawgAuto/HTR)
- [vLLM HTR v0.2.0](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.2.0), executable source [`ad52c7df6d461d653199264f6d5bc216984aea3b`](https://github.com/HawgAuto/vllm-htr/tree/ad52c7df6d461d653199264f6d5bc216984aea3b)
- Public prebuilt image **`ghcr.io/hawgauto/vllm-htr:v0.2.0`**, pinned in the commands below. No GitHub login or host compiler is required; [building from source](docs/RUNNING.md#build-the-public-runtime-source) is optional.

Default serving uses **TAU v2 attention and probabilistic MTP3**, with thinking enabled, temperature **0.7**, top-p **0.8** and top-k **20**. These release sampling settings are distinct from the fixed-length greedy performance screen above.

## Quick start and local cache builder

### 1. Download the model and pull the runtime

```sh
hf download hstolte/Qwen3.8-Flash-Next-HTR \
  --revision v0.2.0 --local-dir ./Qwen3.8-Flash-Next-HTR

IMAGE=ghcr.io/hawgauto/vllm-htr@sha256:4c3bbfc5f42dad17c1a3bca2a911502aa0670aff1314e87e11bc900eaff491df
docker pull "$IMAGE"
```

Already have a complete v0.1.0 checkpoint? Keep it—this serving update needs **no weight redownload or requantization**.

### 2. Set up a reusable launch command

Use absolute paths. The image runs as UID 1000: make the workspace writable by that UID. On multi-GPU machines, replace `renderD128` with the AMD render node.

```sh
MODEL=/absolute/path/to/Qwen3.8-Flash-Next-HTR
WORKSPACE=/absolute/path/to/writable-htr-workspace
mkdir -p "$WORKSPACE"

htr() {
  docker run --rm --init --pull=never --name vllm-htr \
    --device /dev/kfd --device /dev/dri/renderD128 \
    --group-add "$(stat -c %g /dev/kfd)" \
    --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --pids-limit 8192 --shm-size 16g --ulimit core=0 \
    --security-opt no-new-privileges:true \
    -p 127.0.0.1:8000:8000 \
    --mount "type=bind,src=$MODEL,dst=/model,readonly" \
    --mount "type=bind,src=$WORKSPACE,dst=/workspace" \
    "$IMAGE" "$@"
}
```

### 3. Build once, then serve

```sh
# Starts the real server, warms representative shapes, writes SEED.json, and exits.
htr --build-seed && htr
```

The builder warms prompt lengths from **64 to 32,768 tokens** and, for multi-stream profiles, concurrent requests. Normal startup reuses the cache under `/workspace/cache/<profile>`. Keep the workspace across restarts. The cache is bound to the image/runtime and compile-relevant settings, including MTP mode, stream count, context, YaRN, batch budget and graph sizes. New request shapes can still compile; this is a compiler cache, not an expanded model-weight cache. No prebuilt seed archive needs to be downloaded.

For **four-stream serving**, use the same flags when building and starting:

```sh
htr --mtp 0 --max-num-seqs 4 --build-seed && \
  htr --mtp 0 --max-num-seqs 4
# Alternatively, use --mtp 3 on both commands for four-stream MTP3.
```

For a **1M-context, four-stream MTP3 profile**:

```sh
htr --mtp 3 --max-num-seqs 4 --context 1000000 --yarn-factor 4 \
  --kv-cache-gib 27 --max-num-batched-tokens 12288 --build-seed && \
  htr --mtp 3 --max-num-seqs 4 --context 1000000 --yarn-factor 4 \
  --kv-cache-gib 27 --max-num-batched-tokens 12288
```

Use one running server at a time; stop it before changing profiles. The [running guide](docs/RUNNING.md) includes no-device preflight, all four 1M profiles, measured builder/replay timings, and the complete flag reference.

### 4. Connect

After readiness, use **`http://127.0.0.1:8000/v1`** with model name **`qwen38-htr`**:

```sh
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-htr","messages":[{"role":"user","content":"Explain how speculative decoding works."}],"max_tokens":512}'
```

Thinking consumes completion tokens; allow enough output budget. Supply OpenAI-compatible `tools` normally. The bind is loopback-only; add authentication and a reverse proxy before exposing it externally.

## Concurrency and expected throughput

Set **`--max-num-seqs`** to admit concurrent requests: **1–32 with `--mtp 0`**, or **1–4 with `--mtp 3`**. The default is one stream. Additional clients queue when all slots are occupied. Slot count is not a separate full context allocation for every user; all slots share the configured KV budget.

On the cached-process four-stream mixed-prompt screen, MTP0 delivered **68.90 aggregate decode tok/s** while all four streams were decoding, or **17.22 tok/s per stream on average**. MTP3 delivered **52.42 aggregate**, averaging **13.10 per stream**. These per-stream figures are averages within that shared window, not an equal-share guarantee. Whole-wave decode was **40.18 / 26.40 tok/s** for MTP0 / MTP3; including complete prompt-processing and request wall time yielded **22.28 / 18.80 output tok/s**.

For this workload, **MTP3 is the faster single-request decode mode; MTP0 is the higher-throughput four-stream mode**. Longer prompts, newly compiled shapes, output length, speculation acceptance and SSD activity change the balance. See the [complete performance report](docs/SERVING-PERFORMANCE.md) for both four-stream workloads, fresh versus cached processes, per-run ranges and all preserved source rows. The 32-stream setting is a launcher capability, not a 32-stream throughput claim from this four-stream screen.

## Quantization recipe

The model uses a mix of precisions rather than treating every component the same.

| Component | Representation |
|---|---|
| Selected dense language-model weights | HTR5 (5-bit) |
| Routed experts | HTR4 (4-bit) |
| Output head | HTR4 (4-bit) |
| Hyperconnections | GPTQ8 (8-bit integer) |
| Vision tower and merger | Original BF16 |
| Other embeddings, norms and supporting tensors | Retained in their component-specific formats |
| Speculative draft | Included for MTP3 |

## Method and lineage

HTR combines calibration-guided error correction with a trellis-based weight format designed for Strix Halo. The goal is to reduce weight storage while keeping the model close to its reference behavior.

Its lineage includes [exlr8](https://github.com/kindlingai/exlr8), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [QTIP](https://arxiv.org/abs/2406.11235). See the [methodology](https://github.com/HawgAuto/HTR/blob/main/docs/METHODOLOGY.md) and [full acknowledgments](https://github.com/HawgAuto/HTR/blob/main/docs/LINEAGE.md).

![The HTR quantization pipeline](media/method-map.png)

## Size, memory and 1M context

The checkpoint contains **37 weight files**: 23 target safetensors shards, 13 indexed HTR expert shards and one draft safetensors shard. An additional 26 files supply the configuration, tokenizer, processors, manifests and license. Shard consolidation is lossless, not requantization.

| Download component | Files | GB | GiB |
|---|---:|---:|---:|
| Target checkpoint (including PLE and vision) | 23 | 109.027 | 101.539 |
| Routed experts | 13 | 60.705 | 56.536 |
| MTP draft | 1 | 3.978 | 3.704 |
| **Complete checkpoint + supporting files** | **63** | **173.759** | **161.826** |

Exact checkpoint total: **173,759,296,105 bytes**; card, artwork and reports are additional. The largest shard is 4.997 GB. [File sizes and SHA-256](evidence/checkpoint-manifest.json) · [Sizing data](evidence/package-sizing.json)

**PLE is already included in these totals.** Its n-gram embedding tables account for 102.400 GB (95.368 GiB) of the target payload. The runtime reads requested PLE rows from SSD instead of placing the entire table on the GPU. Keep the model on a fast local SSD; download size is not resident GPU memory.

### Resident memory and context profiles

The intended host has **128 GB shared system memory**. On the default MTP3 profile, the engine logged **65.68 GiB model-load allocation**, **8 GiB BF16 KV**, and **0.19 GiB graph memory**. Those are separate allocations, not a total-RAM or co-loading allowance: the runtime, compilation, activations and filesystem cache also need headroom. Keep the large n-gram tables on a fast local SSD and avoid another large shared-memory GPU workload.

**1M context:** use `--context 1000000 --yarn-factor 4 --kv-cache-gib 27`. MTP0/MTP3, with either one or four slots, passed the public builder and cold replay on the final image. The measured profiles reached approximately **99.2–104.1 GiB peak GPU shared-memory usage (GTT)** across build/replay and used swap during startup/compilation. The documented commands do not impose a Docker memory cap.

With four slots, the shared 27 GiB KV pool has an equal-share estimate of **281,684 tokens per slot for MTP0** or **255,030 for MTP3**; it is not four independent million-token histories. Warmups reached **32,768 input tokens**. Full-million prefill and long-position retrieval have not been measured. [Per-profile memory, capacity and startup results](docs/RUNNING.md#measured-bf16-kv--yarn-4-profiles).

## Quantization quality

This matched **4-bit dense-layer comparison on WikiText** illustrates the choice of HTR. Only the selected dense weights change; both methods use the same reference model, whose experts are already quantized.

| Dense-layer format | PPL ↓ | KL ↓ | Top-1 agreement ↑ |
|---|---:|---:|---:|
| BF16 dense-layer reference | 4.295 | 0.0000 | 100.00% |
| HTR 4-bit | **4.308** | **0.0284** | **93.77%** |
| GPTQ 4-bit | 4.344 | 0.0520 | 91.68% |

**PPL** measures next-token prediction on the text; lower is better. **KL** measures divergence from the reference over its top 128 tokens plus a remaining-mass bucket; lower is better. It is not full-vocabulary KL. **Top-1 agreement** is how often the models choose the same next token—not task accuracy. Bold values mark the better quantized result.

HTR stays closer to the reference than GPTQ in this comparison. This illustrative HTR4 study is not the HTR5 dense-trunk configuration shipped here. These are component-level results, not a final score for the complete mixed-precision release. The [full quality report](docs/RECIPE-COMPARISONS.md) covers the other precision and calibration choices, including their tradeoffs; [source metrics](evidence/recipe-quality.json) retain the detailed measurements.

## Release verification

The final image passed ordinary text, separated reasoning, MTP execution, structured tool calling and a tool-result round trip through the documented launch path. Its final memory-lifetime check matched 18/18 complete outputs, including 16 pairs with returned logprobs; all 1,488 observed MoE self-checks passed. All four 1M-profile builder/replay workflows passed. These are numerical and serving checks, not a broad model-quality or image/video score.

The current serving stack uses TAU v2 sparse attention with split-K fallback. Concurrent scheduling can change generated outputs. BF16 KV is the released path; FP8 KV and further short-prefill / PLE I/O tuning remain future work.

[Final-image qualification](evidence/v0.2.0-release-qualification.json) · [Current serving measurements](evidence/v0.2.0-serving-summary.json) · [Release history](docs/RELEASES.md) · [Vision and MTP inventory](evidence/vision-and-mtp.json)

## Download statistics

Full-repository downloads include [`meta.yaml`](meta.yaml) automatically. For selective downloads, include it alongside the checkpoint files so [Hugging Face's normal download counter](https://huggingface.co/docs/hub/models-download-stats) can record the request. No extra client telemetry is added; Hub counts are not a count of complete downloads or unique users.

## License and attribution

Source model: Qwen/Qwen3.8-Flash-Next, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Quantization/tooling: HawgAuto HTR; serving infrastructure: the separately pinned HawgAuto vLLM/HaloLoom stack.

The weights and derivative are governed by **Qwen Community License 1.0**, preserved verbatim in [LICENSE](LICENSE). This is not Apache-2.0 or the HTR code repository's MIT license. The license contains conditions for commercial Model-as-a-Service and AI Work Assistant use; review the complete text before deployment. Model authorship and capabilities belong to Qwen; HTR does not change those license obligations.
