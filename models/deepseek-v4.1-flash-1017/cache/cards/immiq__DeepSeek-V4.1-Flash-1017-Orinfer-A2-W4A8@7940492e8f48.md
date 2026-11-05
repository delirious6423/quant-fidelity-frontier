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
- ascend910b
- w4a8
- safetensors
- dspark
- vision
inference: false
---

# DeepSeek-V4.1-Flash-1017 — Orinfer A2 W4A8

Native model data for **Orinfer v0.3.1-fix1**, Linux aarch64, Ascend 910B TP8 and **CANN 9.2**. This version includes text, thinking, function tools, images, multiple images and DSpark parallel drafts. It uses Orinfer's native runtime and separate execution archive.

## Install

```bash
gh release download v0.3.1-fix1 -R iMMIQ/orinfer \
  -p 'orinfer-v0.3.1-fix1-linux-aarch64.tar.xz' \
  -p 'orinfer-exec-deepseek-v41-flash-1017-w4a8-ascend910b-v0.3.1-fix1.tar.xz' \
  -p SHA256SUMS
sha256sum --ignore-missing -c SHA256SUMS
tar -xf orinfer-v0.3.1-fix1-linux-aarch64.tar.xz
export PATH="$PWD/orinfer-v0.3.1-fix1-linux-aarch64/bin:$PATH"
orinfer package install orinfer-exec-deepseek-v41-flash-1017-w4a8-ascend910b-v0.3.1-fix1.tar.xz
hf download immiq/DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8 \
  --revision v0.3.1-fix1 --local-dir ./DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8

# Independent, explicit content verification. Normal loading does not scan payload hashes.
orinfer validate-model ./DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8

source /usr/local/Ascend/cann/set_env.sh
export ORINFER_A2_HOST_TABLE_MIB=212000
export ORINFER_VISION_CACHE_MIB=1024
export HCCL_OP_EXPANSION_MODE=AIV HCCL_NPU_SOCKET_PORT_RANGE=auto
unset HCCL_DETERMINISTIC ORINFER_A2_WORKER_CPUS RAYON_NUM_THREADS
mkdir -p ./run
orinfer serve ./DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8 \
  --model deepseek-v4.1 --listen 127.0.0.1:8000 \
  --mtp-drafts auto --max-speculative-tokens 127 \
  --prefix-cache-mib 8192 --kv-offload-mib 262144 \
  --request-body-mib 64 \
  --queue-timeout-ms 60000 --output-timeout-ms 180000 \
  --drain-timeout-ms 120000 --gpu-lock ./run/a2.lock
```

The matching execution-package digest is:

```text
d78cdcc6c916e7d62413bf7e4da5535846d306ebf79d3c8b7896ab3a77129e2a
```

Older installations should pin model revision `v0.3.0` with the v0.3.0 engine and execution archive. Use the complete matching set when upgrading. The model ID in clients must match `--model`.

`--mtp-drafts 0` uses ordinary decode. The execution package supports drafts through 16; `--mtp-drafts 16` enables adaptive selection over that range. The full automatic verification packet, including mixed prefill rows, stays within this package's 127-row communication domain. Independent prefill retains its large-row plans.

The profile above keeps the Engram host table fully resident, requiring about 206 GiB plus other host memory. That mode disables the duplicate file-row cache; `ORINFER_A2_HOST_ROW_CACHE_MIB` applies to file-backed mode. Leaving `ORINFER_A2_HOST_TABLE_MIB` unset selects bounded file-backed caching, which is not the mode used in these performance runs. Allow more than 120 seconds in the external termination grace period. Configure authentication before exposing the server outside the trusted host.

## Data and capabilities

- 189 prepared safetensors files include the original W4 expert representation, DSpark parameters and `vision.safetensors`. The v0.3.0 to v0.3.1 update preserves 170 existing shards byte for byte; eight final-layer shards change and eleven shards are added.
- Operators, execution plans and native libraries remain in the separate GitHub execution archive. They are not included in this model repository.
- Runtime KV uses group-64 INT8 with FP16 scales. KV and the other sequence states remain request-private. Prefix reuse and CPU KV offload use the shared runtime manager.
- Native vision uses the official encoder and aligner, with at most 1024 visual tokens per image. With the fix1 engine, image history is bounded by the model context and host-memory budget; the earlier 8192-feature history cap is removed. Encoding proceeds in bounded waves using the same execution package. Media identity is included in prefix-cache matching. `image_processing` can apply stricter request budgets; `--request-body-mib` and `--image-cache-mib` control HTTP body and preprocessing-cache limits.
- Streaming OpenAI-compatible chat supports native DSML function calls, schema constraints, thinking, image inputs and multiple images. See [deployment and OpenCode examples](https://github.com/iMMIQ/orinfer/blob/v0.3.1-fix1/docs/serving.md).
- `release-manifest.json` and `SHA256SUMS` describe file integrity. Payload verification runs through the explicit validation command.

## Measured deployment tuning — 2026-11-05

For a dedicated A2 with at least **96 CPU cores and 1 TB host RAM**, the practical recommendation from this sweep is **MTP auto, default preprocessing, a 1 GiB image-feature cache, and a fully resident Engram table**. The startup command above uses that profile. `auto` has an effective maximum of 5 drafts in this execution package. Keep preprocessing at its defaults (2 workers / 2048 MiB); no explicit thread counts or rank CPU lists are needed.

These performance tables were measured with the original **v0.3.1** engine and unchanged weights/execution package; they are not a new fix1 performance benchmark. The service was restricted to 96 logical CPUs with `MemoryMax=1000000000000`, using eight 64 GiB 910B3 devices, CANN 9.2 and the exact released binary/execution package. The benchmark CPU mask was `0-23,48-71,96-119,144-167`, reflecting this host's topology; it is not a portable CPU-number recommendation. Assign the service its actual CPUs through the deployment cpuset. Peak observed process RSS across these checks was approximately **219.0 GiB**; caches grow on demand.

All rows below used the same short workload order. Values are **warm seconds for the entire concurrent cohort**, including API completion; they are not kernel timings. The tool case used 16 concurrent fixed two-call requests and the median of two warm rounds. Other columns contain one warm repeat after a first pass. Thinking is the first planning response, capped at 128 thinking tokens for this bounded probe; a model can return the first call and wait for its result. Long-prefix reuse used four approximately 32K prompts. The image case reused 32 distinct 1024px images while changing the text before them to avoid a full token-prefix hit; the multi-image case used four requests with two images each.

| Profile | Tool C16 | Thinking C16 | Long-prefix C4 | Reused images C32 | Two images C4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Auto / 128 MiB vision / 16 GiB CPU tier | 6.01 | 17.02 | 2.98 | 9.43 | 1.60 |
| Auto / 1 GiB vision / 256 GiB CPU tier | 6.45 | 19.04 | 2.81 | 6.55 | 1.53 |
| Same larger caches / 8 preprocessors | 5.81 | 16.71 | 2.55 | 6.99 | 1.73 |
| Same larger caches / draft cap 16 | 9.03 | 21.13 | 4.49 | 6.73 | 1.64 |

Raising the draft cap to 16 made the measured text paths slower. Eight preprocessors had mixed results and no consistent overall benefit; the default is retained. Increasing the image-feature cache helped the repeated-image working set; it does not accelerate unique-image encoding. The CPU prefix budget changed alongside the image cache, but the micro workloads did not fill the CPU tier: **256 GiB is a capacity ceiling within the 1 TB deployment envelope, not a measured optimal capacity or an isolated TPS improvement**.

The selected auto profile was loaded again and checked at several client concurrencies. Each tool response generated 97 completion tokens. Warm values use the last two of three rounds:

| Client concurrency | Cohort completion, s | Aggregate end-to-end output tokens/s | First-delta p95, s |
| --- | ---: | ---: | ---: |
| 1 | 0.85 | 114.3 | 0.13 |
| 4 | 1.91 | 203.6 | 0.37 |
| 16 | 6.04 | 257.2 | 1.46 |
| 32 | 12.48 | 248.8 | 3.01 |

The tested service retained its default active-request ceiling of **32**. That ceiling does not wait for a full batch. Sixteen concurrent tool requests had a better per-request completion time; 32 concurrent tool requests provided similar aggregate throughput with longer individual latency. The warm thinking C32 first-response cohort took **29.42 seconds**, producing **6404 completion tokens** at **217.7 aggregate end-to-end output tokens/s**; first reasoning-delta p95 was **1.41 seconds**. These planning responses can differ in length and are not complete agent loops. Admission-limit 16 versus 32 under the same 32-request burst was not separately compared. Use the table to choose the desired load/latency operating point rather than interpreting more concurrency as a free per-user speedup.

The other tested defaults were batch-token budget 128, latency scheduling, 200 ms prefill budget, 400 ms soft TPOT target and 1024 MiB memory reserve. Device prefix caching stayed at 8192 MiB per device; CPU prefix caching is a global budget shared across TP ranks. Retain startup-probe allowance of 600 seconds and an external shutdown grace longer than 120 seconds. Output-channel timeout is a slow-reader deadline, not a complete inference deadline.

One final **240-second**, thinking-enabled, concurrency-16 Mini-Weka code-agent replay completed **16 warmup and 43 profiling requests**. The replay reached its wall limit and was incomplete. This private replay is a bounded progress diagnostic; no complete benchmark score or matched Mini speedup is claimed. The source, model weights and kernels were unchanged, and no one-hour replay was run.

See [the aggregate configuration report](benchmarks/a2-v0.3.1-config-20261105.json) for exact settings, workload scope and resource measurements. The fix1 update changes no model tensors or execution identities. Download `--revision v0.3.1-fix1` for the current card and validation report; the original `v0.3.1` model tag remains immutable. The known semantic-quality limitation below remains applicable.

## Validation and known limits

The fix1 engine is pinned to source revision `a666d5b6f1b2123b89c6879039a95b85dd517f82`. Its `make check` passed 625 Rust tests, 427 Python tests and 15 fixture-coverage checks; 91 hardware/external-fixture tests were conditionally ignored. The execution-package digest and all model tensor files are unchanged. The preceding native vision and operator checks remain applicable. Actual fix1 acceptance used the native A2 TP8 service with CANN 9.2; see the DSH section below.

A fixed two-tool greedy request passed 74 of 75 responses in the final short checks. One response used `b=10` instead of the requested `b=5` in the second call; the function name, argument types and protocol remained valid. Earlier ordinary-decode runs also exhibited this error pattern. Attribution to model quality, numerical execution or request state is not established. This is not a claim of complete task-quality acceptance.

In an earlier small same-package HTTP comparison with a different CPU-placement/logging setup, warm complete B16 tool requests took about 7.66 seconds versus 7.91 seconds before the controller convergence. B4 had variability, and first-cohort calibration remained slower. These measurements include request completion and do not represent a general decode TPS or a guaranteed latency SLO. A preceding 240-second Mini replay was truncated and is not a complete benchmark score. No one-hour replay was run.

## DeepSeek Harness (DSH) — fix1 acceptance

Fix1 accepts `chat_template_kwargs` from the DSH Chat Completions adapter, including thinking mode, effort, thinking-token budget and history preservation. Explicit request settings override deployment defaults; non-null top-level fields take precedence over matching nested aliases. Template rendering and reasoning-stream parsing use the same effective thinking mode. `/health.chat_capabilities.chat_template_kwargs` reports support.

Use Node.js and `npx @deepseek-ai/dsh@0.2.1-rc.5 --profile headless --help`, then merge the [DSH configuration example](https://github.com/iMMIQ/orinfer/blob/v0.3.1-fix1/examples/dsh-cordis.patch.yml) into the headless profile. Set the service URL, matching model ID and `ORINFER_API_KEY` environment variable. The example uses the OpenAI Chat Completions provider with `input: [text, image]` and dynamically maps DSH thinking settings. See the [installation and thinking guide](https://github.com/iMMIQ/orinfer/blob/v0.3.1-fix1/docs/deepseek-harness.md).

The actual DSH CLI, through a byte-preserving recording proxy, completed five short tasks against native A2 TP8 inference: plain text, two file-tool loops in a resumed session, thinking arithmetic with a nonempty reasoning stream, and two image-tool reads with the correct red/blue ordering. All **8 API requests returned HTTP 200**, and **4 tool calls completed**. The server ended ready with zero active/queued requests and exited cleanly. This is bounded integration acceptance, not a complete agent-quality benchmark. See the [aggregate validation record](benchmarks/a2-v0.3.1-fix1-dsh-20261105.json).

Upgrade the engine to fix1; existing matching weights and the digest-pinned execution package can be reused. The v0.3.1 and v0.3.0 release tags remain available. Frontend image caching is bounded and supports UUID references with resend after eviction; the single-image encoding budget remains 1024 visual tokens.

## Attribution

Model weights retain the upstream MIT license in `LICENSE`; the original documentation is in `SOURCE-README.md`. Orinfer is a separate LGPL-3.0-or-later project. See the [GitHub release](https://github.com/iMMIQ/orinfer/releases/tag/v0.3.1-fix1) for engine, execution assets and source.
