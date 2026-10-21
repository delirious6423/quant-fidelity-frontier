---
library_name: tachyon
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - unsloth/Qwen3.8-27B-NVFP4
base_model_relation: quantized
tags:
  - tachyon
  - qwen3.8
  - nvfp4
  - fp8
  - w4a4
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
model-index:
  - name: Qwen3.8-27B-nvfp4-Tachyon
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: IFBench
          type: ifbench
        metrics:
          - type: accuracy
            value: 77.00
            name: Prompt-level strict (0-shot, rule)
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2025
          type: aime25
        metrics:
          - type: accuracy
            value: 96.67
            name: Accuracy (0-shot, rule)
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2026
          type: aime26
        metrics:
          - type: accuracy
            value: 96.67
            name: Accuracy (0-shot, rule)
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: GPQA-Diamond
          type: gpqa_diamond
        metrics:
          - type: accuracy
            value: 90.40
            name: Accuracy (0-shot, rule)
      - task:
          type: image-text-to-text
          name: Image Text to Text
        dataset:
          name: ERQA
          type: erqa
        metrics:
          - type: accuracy
            value: 66.25
            name: Accuracy (0-shot, rule)
      - task:
          type: image-text-to-text
          name: Image Text to Text
        dataset:
          name: RealWorldQA
          type: real_world_qa
        metrics:
          - type: accuracy
            value: 83.53
            name: Accuracy (0-shot, rule)
---

# Qwen3.8-27B NVFP4 for Tachyon

This model card is the version-controlled source for
[pinvou/Qwen3.8-27B-NVFP4-Tachyon](https://huggingface.co/pinvou/Qwen3.8-27B-NVFP4-Tachyon).

The repository contains a mixed NVFP4/FP8 representation of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It combines the official BF16 checkpoint
with the fixed packed Text weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) in the native
Tachyon `.tachyon` artifact format. The artifact is intended
only for Tachyon; it is not a Transformers checkpoint, Safetensors distribution, or GGUF file.

The artifact uses the Qwen3.5 Dense architecture. All 64 Text layers use NVFP4 MLP weights
and an NVFP4 output head; the token embedding, attention input/output projections, and GDN
Q/K/V/Z and output projections use row-scaled FP8. Control weights use BF16, with separate MTP, Vision and DFlash2 weights.

## Release 88251e38 (October 2026)

Engine package `tachyon-serve-gb10-88251e38.tar.gz` supersedes `7fca766a`:

- **Startup no longer fails on page-cache-heavy machines.** Runtime capacity now credits
  reclaimable host page cache (`MemAvailable` beyond `MemFree`) on unified-memory GB10
  systems. Devices that previously required `drop_caches` or a reboot before enabling the
  card start directly; the workaround is retired.
- **New discovery and monitoring endpoints.** `GET /get_server_info` returns
  `{"version","engine","model","max_model_len"}` without authentication, for client probes;
  `GET /metrics` renders the Prometheus text format (authenticated when `--api-key` is set).
  The startup log line now carries the build version.
- **Vision enabled in the card launch config.** The artifact has always shipped the Vision
  tower; the serving example and the store card now pass `--vision` so image/video prompts
  work out of the box.
- **Thinking-intensity control** (vLLM/SGLang-compatible, no app release needed):

| Field | Location | Values | Notes |
|---|---|---|---|
| `enable_thinking` | top level or `chat_template_kwargs.enable_thinking` | `true` / `false` | per-request override of the server default |
| `reasoning_effort` | top level or `chat_template_kwargs.reasoning_effort` | `none` / `low` / `medium` / `xhigh` | this artifact's chat template accepts exactly these; `high`/`minimal`/`max` return 400 — map the deepest tier to `xhigh` |
| `preserve_thinking` | top level or `chat_template_kwargs.preserve_thinking` | `true` / `false` | retain closed-turn reasoning in later prompts |

  Reasoning streams as `delta.reasoning_content` (streaming) and returns as
  `message.reasoning_content` (non-streaming), never inlined into `content`. Reasoning
  billing is `usage.completion_tokens_details.reasoning_tokens`. Reasoning shares the
  request `max_tokens` budget.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_nvfp4.tachyon` |
| Size | 22,240,278,272 bytes (20.71 GiB) |
| SHA-256 | `05c74c488b56a30097b25973888b31a16ddcfa1dec90556a2ae5e0c3201e50bb` |
| Container version | 3 |
| Architecture | `Qwen3_5ForCausalLM` |
| Public model name | `qwen3.8-27b` |
| Chat template | `qwen3_8.jinja`; override with `--chat-template FILE` |
| Template defaults | thinking off by default (`--thinking` to opt in); effort `xhigh` when enabled |
| Stored objects | 1,078 (1,072 tensors and 6 resources) |
| NVFP4 row tensors | 129 (all 64 Text-layer MLP parents and the output head) |
| Row-scaled FP8 tensors | 129 (128 FP8-row projections and the generated token embedding) |

The file contains Text, Vision, MTP, DFlash2, the optimized proposal head and frontend resources.
Vision and speculative weights are loaded only when selected at startup. Source-derived NVFP4 and
FP8 words are preserved without decode and requantization; only the official BF16 token embedding
is encoded locally as row-scaled FP8.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '05c74c488b56a30097b25973888b31a16ddcfa1dec90556a2ae5e0c3201e50bb' \
  'qwen3_8_27b_nvfp4.tachyon' | sha256sum --check
```

This release includes the complete DFlash2 companion weights from
`z-lab/Qwen3.8-27B-DFlash2` at revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. Select
`--spec dflash2 --draft-tokens 7 --lm-head-draft`; draft counts 1..15 are supported.
DFlash2 requires the runtime revision listed below. Existing performance and evaluation tables
retain their stated MTP configurations and revisions.

## Requirements

- a current Tachyon checkout, built from source; its artifact/component support must include
  revision `88251e3` or later;
- 64-bit Linux;
- a GPU supported by the selected Tachyon build: GB10 / DGX Spark (`sm_121a`, current
  project target) or RTX 5090 (`sm_120a`, explicit build selection);
- a CUDA toolkit capable of compiling the selected architecture.

The published artifact measurements below used RTX 5090 and CUDA 13.1. Engine support does not
establish GB10 validation for every component, context length, or weight profile in this card.
Use the GB10 run records (`docs/performance.md` in the selected checkout) for their measured
scope, and its build guide for toolchain requirements.

Already have the official v2 file? Upgrade it locally (see `docs/weight-conversion.md`)
without downloading the weights again.

Tachyon does not provide an install target or packaged binary. See the repository README for source-build dependencies.

## Download and run a CLI example

```bash
hf download pinvou/Qwen3.8-27B-NVFP4-Tachyon \
  qwen3_8_27b_nvfp4.tachyon \
  --local-dir models

./build/apps/tachyon models/qwen3_8_27b_nvfp4.tachyon \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec dflash2 --draft-tokens 7 \
  --lm-head-draft \
  --no-thinking
```

For images, videos, and structured chat history, see the CLI guide (`docs/cli.md`).

## Start a local server

```bash
./build/apps/tachyon-serve models/qwen3_8_27b_nvfp4.tachyon \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 262144 \
  --kv-capacity 262144 \
  --max-concurrency 8 \
  --kv-dtype fp8 \
  --device-state-slots 16 \
  --host-state-slots 16 \
  --host-kv-mib 8192 \
  --prefill-chunk 1024 \
  --spec dflash2 --draft-tokens 7 \
  --lm-head-draft \
  --no-thinking \
  --vision
```

Draft K=7 is the best default across mixed content; for structured, high-acceptance
serving (counting, code generation farms) restart with `--draft-tokens 13` — the
measured throughput profile reaches 405-461 tok/s aggregate at eight-way concurrency.

Each request has a 262,144-token logical ceiling (the model maximum). The shared 262,144-token
Device KV pool admits up to eight active requests when their combined completion reservations
fit; long-context requests automatically narrow the active batch while retaining the full pool
for a single deep-context request. Sixteen extra Device checkpoint slots, sixteen pinned Host
State slots, and 8 GiB of pinned Host KV retain reusable continuations under resource pressure.
Single-stream decode speed is identical to a smaller `--max-concurrency` setting; the wider
admission only adds concurrent capacity.

See the HTTP serving guide (`docs/serving.md`) for the API surface and the resource scheduling
reference (`docs/maintainer/resource-scheduling-and-context-cache.md`) for cache and admission
semantics.

## Supported use

The artifact supports:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages;
- MTP speculative decoding with draft windows from one to five;
- DFlash2 with draft windows from one to fifteen using the included
  DFlash2 companion weights (`--spec dflash2 --draft-tokens 7`, optionally `--lm-head-draft`);
- BF16, INT8, FP8, NVFP4, and K8V4 KV cache;
- CUDA Graph decode and compatible-prefix reuse;
- startup-bounded small-scale concurrent serving with true batched decode;
- the Tachyon CLI;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages serving.

## Performance


### GB10 / MegaCube serving (October 2026, DFlash2 K=7)

Measured on one GB10 (MegaCube) with Tachyon revision `7fca766`, CUDA 13.0, FP8 KV, thinking disabled, and the exact serve command above. Sampling matches the
public SGLang NEXTN baseline on the same hardware (temperature 1.0, top-p 0.95, top-k 20).

Single-request decode by content class (stochastic; greedy where higher):

| Content class | Acceptance | Draft K | tok/s |
|---|---|---|---|
| Structured counting | 90%+ | 12-13 | 55-99.8 (greedy) |
| Code generation | 40-60% | 7 | 45-57 |
| Technical writing | 45-55% | 7 | 30-36 |
| Simple explanations | 45% | 7 | 26-31 |
| Essays / freeform | 25-40% | 7 | 20-24 |

Concurrency, mixed content (aggregate, client-side, includes admission ramp):

| Concurrency | tok/s | SGLang NEXTN baseline (same hardware/sampling) |
|---|---|---|
| 1 | 33.4 geomean (greedy 36.5) | 23.9 |
| 4 | 70-89 | 65.2 |
| 8 | 109-110 | 94.2 |

Steady-state 8-way structured counting reaches 405-461 tok/s aggregate (95%+ acceptance,
server metric; ~51 tok/s per lane at 90% acceptance). Hot-turn TTFT on a 16.7K-token session
with prefix reuse: 223-227 ms. Speculative decoding never changes the target distribution;
greedy output differs from non-speculative decode only in near-tie floating-point orderings.

The tables below are historical RTX 5090 measurements retained for reference.

The MTP0 measurements below were collected at Tachyon revision
`f08597d`, and the MTP3 measurements at revision
`32c9881`.
Both campaigns used one NVIDIA GeForce RTX 5090, CUDA 13.1 compile/runtime, CUDA driver API 13.3,
stochastic sampling, INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and prefix reuse
disabled. MTP0 used no speculative backend and a 262,144-token context limit; MTP3 used a
131,072-token per-request context limit and three draft tokens.

### Concurrent MTP=3 corpus makespan

The fixed corpus contains three long-reasoning and twelve cross-scenario fixtures with five seeds
each, for 75 requests. Every concurrency point starts a fresh server and uses the same shuffle seed
and ordered HTTP send sequence. C=1 is the serial single-request corpus. Makespan includes prefill,
decode, workload transitions, and final drain.

| C | Requests | Decode tokens | Makespan | Requests/s | Corpus decode (tok/s) | Avg batch | MTP acceptance | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 752,160 | 4,670.27 s | 0.0161 | 161.1 | 1.00 | 60.8% | 1.00× |
| 2 | 75 | 739,951 | 2,510.78 s | 0.0299 | 294.7 | 1.98 | 59.2% | 1.86× |
| 4 | 75 | 713,384 | 1,647.74 s | 0.0455 | 432.9 | 3.29 | 58.0% | 2.83× |
| 8 | 75 | 723,602 | 2,164.90 s | 0.0346 | 334.2 | 2.36 | 57.6% | 2.16× |

All 300 requests completed without a request, CUDA, or out-of-memory failure. C=4 gives the
shortest complete-corpus makespan. C=8 is limited by memory pressure, which makes its
complete-corpus result slower than C=4. Sampling is stochastic, so the fixed prompts and seeds do
not imply token-identical continuations across concurrency-specific numerical routes; exact
decode-token totals are shown above.

### Long-context baseline (MTP disabled)

Each value is the arithmetic mean ± sample standard deviation over five fixed seeds after server
warm-up.

| Prompt tokens | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
|---:|---:|---:|---:|
| 7,680 | 8,340.4 ± 13.0 | 931.6 ± 1.6 | 71.2 ± 0.1 |
| 64,512 | 5,297.9 ± 259.2 | 12,281.1 ± 561.5 | 65.7 ± 0.8 |
| 130,048 | 3,544.7 ± 25.3 | 36,853.5 ± 259.4 | 59.6 ± 0.9 |
| 260,096 | 2,203.1 ± 13.4 | 118,354.8 ± 717.2 | 52.9 ± 2.3 |

### MTP=3 single-request long-reasoning decode

The C=1 point supplies five samples for each fixture. Values are arithmetic mean ± sample standard
deviation from server phase timings and speculative counters.

| AIME 2026 fixture | Completion tokens | Decode phase (tok/s) | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Problem 1 | 1,465.4 ± 417.3 | 195.2 ± 4.6 | 76.0% ± 2.4% | 3.28 ± 0.07 |
| Problem 15 | 65,414.4 ± 271.9 | 151.4 ± 2.0 | 56.2% ± 1.1% | 2.69 ± 0.03 |
| Problem 30 | 50,023.4 ± 14,839.1 | 167.5 ± 23.7 | 64.6% ± 14.9% | 2.94 ± 0.45 |

### MTP=3 single-request cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Decode phase (tok/s) | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|
| Code | 194.3 ± 6.1 | 76.4% ± 3.9% | 3.29 ± 0.12 |
| Story | 126.1 ± 10.9 | 37.4% ± 5.8% | 2.12 ± 0.17 |
| Translation | 192.3 ± 11.9 | 75.0% ± 6.5% | 3.25 ± 0.19 |
| Structured output | 219.8 ± 8.6 | 90.8% ± 5.1% | 3.72 ± 0.15 |

See `docs/performance/qwen3.8-27b.md` in the Tachyon checkout for metric definitions and the
exact reproduction command.

## Evaluation

The artifact was evaluated through Tachyon's OpenAI-compatible serving route with thinking enabled,
MTP=3, and INT8 group-64 KV. EvalScope 1.9.0 used 0-shot prompts, rule-based scoring, and one sample
per problem with temperature 1.0, top-p 0.95, top-k 20, presence penalty 0.0, and seed 42. The text
suite ran at a 252,928-token context limit; the multimodal suite ran with `--vision` at a
81,920-token limit.

| Benchmark | Tachyon NVFP4 | Correct / total | Official Qwen3.8-27B BF16 |
|---|---:|---:|---:|
| IFBench (prompt-level strict) | 77.00% | 231 / 300 | 79.5 |
| AIME 2025 | 96.67% | 29 / 30 | — |
| AIME 2026 | 96.67% | 29 / 30 | — |
| GPQA-Diamond | 90.40% | 179 / 198 | 89.2 |
| ERQA | 66.25% | 265 / 400 | 65.5 |
| RealWorldQA | 83.53% | 639 / 765 | 85.9 |

All 1,723 configured samples completed and were scored. IFBench additionally reports 80.50%
instruction-level strict, 80.33% prompt-level loose, and 83.50% instruction-level loose. These are
single-sample results, not pass@k.

The official Qwen3.8-27B BF16 figures come from the
[upstream model card](https://huggingface.co/Qwen/Qwen3.8-27B); its sampling settings and IFBench
metric level are not stated there, so the last column is not a same-protocol comparison. The NVFP4
deltas stay within ±2.5 points on the four overlapping benchmarks, and the upstream card reports no
AIME results.

## Limits

- Tachyon executes on one GPU and one CUDA device, with a startup-fixed capacity of 1–8 active
  requests per Engine.
- It does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  multi-GPU execution, CPU/GPU offload, or distributed serving.
- Context allocation is subject to GPU memory and the selected KV-cache type.
- Tachyon does not execute generated tool calls.

## Provenance

| Field | Value |
|---|---|
| Base repository | `Qwen/Qwen3.8-27B` |
| Base revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Base download source | `modelscope.cn/models/Qwen/Qwen3.8-27B` |
| Quantized source repository | `unsloth/Qwen3.8-27B-NVFP4` |
| Quantized source revision | `60e813d4dbbdc5d64cf3f5a8caf2897bedf03679` |
| Conversion recipe | `qwen3_8_27b_nvfp4` |
| Embedding encoder | `fp8_row_maxabs` |
| Minimum runtime revision | `7fca766aac205252b4e86fe0d84372107ed0fe61` |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

The artifact identity, summarized object inventory, and conversion provenance are published in
[`artifact-manifest.json`](https://huggingface.co/pinvou/Qwen3.8-27B-NVFP4-Tachyon/blob/main/artifact-manifest.json).
The exact storage contract is maintained in the Tachyon v3 container reference
(`docs/maintainer/artifact-container.md`).

## License

This Tachyon artifact is distributed under the Apache License 2.0. The
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) base repository and the
[quantized source repository](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) are also licensed
under Apache-2.0. Users remain responsible for complying with the license and applicable laws.
