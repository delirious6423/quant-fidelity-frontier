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

Native model data for **Orinfer v0.3.5**, Linux aarch64, Ascend 910B TP8 and **CANN 9.2**, with **glibc 2.34 or later**. This version includes text, thinking, function tools, images, multiple images and DSpark parallel drafts. It uses Orinfer's native runtime and separate execution archive.

## Install

```bash
gh release download v0.3.5 -R iMMIQ/orinfer \
  -p 'orinfer-v0.3.5-linux-aarch64.tar.xz' \
  -p 'orinfer-exec-deepseek-v41-flash-1017-w4a8-ascend910b-v0.3.5.tar.xz' \
  -p SHA256SUMS
sha256sum --ignore-missing -c SHA256SUMS
tar -xf orinfer-v0.3.5-linux-aarch64.tar.xz
export PATH="$PWD/orinfer-v0.3.5-linux-aarch64/bin:$PATH"
orinfer package install orinfer-exec-deepseek-v41-flash-1017-w4a8-ascend910b-v0.3.5.tar.xz
hf download immiq/DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8 \
  --revision v0.3.5 --local-dir ./DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8

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
  --mtp-drafts auto --max-speculative-tokens 0 --graph full \
  --prefix-cache-mib 8192 --kv-offload-mib 262144 \
  --request-body-mib 64 \
  --queue-timeout-ms 60000 --output-timeout-ms 180000 \
  --drain-timeout-ms 120000 --gpu-lock ./run/a2.lock
```

The matching execution-package digest is:

```text
2d030ec319ce288154bb92ad7809f3b0bfb088b9a105fd8a979fee50da5f0929
```

Older installations should pin model revision `v0.3.0` with the v0.3.0 engine and execution archive. Use the complete matching set when upgrading. The model ID in clients must match `--model`.

`--mtp-drafts auto` adaptively selects up to 5 drafts. The execution package declares 192 decoder verification rows, including one anchor per owner, so 32 owners can use up to 5 drafts each when profitable. `--max-speculative-tokens 0` uses the package budget. Mixed prefill rows retain the shared packet capacity and all row counts reuse the existing dynamic operators. Explicit larger draft limits remain subject to package capacity and measured costs; `--mtp-drafts 0` selects ordinary decode.

The profile above keeps the Engram host table fully resident, requiring about 206 GiB plus other host memory. That mode disables the duplicate file-row cache; `ORINFER_A2_HOST_ROW_CACHE_MIB` applies to file-backed mode. Leaving `ORINFER_A2_HOST_TABLE_MIB` unset selects bounded file-backed caching, which is not the mode used in these performance runs. Allow more than 120 seconds in the external termination grace period. Configure authentication before exposing the server outside the trusted host.

## Version 0.3.5

Upgrade the engine and the matching execution archive from [the v0.3.5 release](https://github.com/iMMIQ/orinfer/releases/tag/v0.3.5), then obtain this model's `v0.3.5` metadata. The A2 execution package digest is `2d030ec319ce288154bb92ad7809f3b0bfb088b9a105fd8a979fee50da5f0929`. Weight tensors, tokenizer, chat template and generation-config bytes are unchanged; an existing complete model installation needs only the small metadata files, not another weight download or conversion. Operators and execution plans remain separate from the weights.

The engine adds boundary-correct token decoding and the `/v1/chat/completions/render` interface for Estuary, overlaps accepted-output processing with the next device step, reuses long-prompt encodings and improves packet compilation/readback. DSpark cost forecasts now use the same actual row admissions as verification, including atomic final windows, checkpoint and media boundaries. The earlier experimental source-prefix Graph and cost-family changes are not included. Shared fragment Graph remains opt-in.

Recommended production parameters remain `--mtp-drafts auto --max-speculative-tokens 0 --graph full`; retain the CPU/memory/cache budgets in the deployment example above. The default is temperature=1/top_p=1 and coupled DSpark. The package includes device grammar/probability-report sampling plus an optional `ORINFER_A2_DSPARK_SAMPLING=rejection` experimental path; enabling it is an explicit choice, not a change to the default deployment.

The pre-release source `a6d24e669e351fd9a75fda0aebf12480bfc55047` was tested with T=1/top_p=1, DSpark auto/max5 and **explicit rejection sampling**. In the fixed mixed long-context short, mean decoder-request latency improved 11.73% and cold-request latency improved 7.52%; pure C16 T1 latency was 2.21% higher. A 240-second Mini completed 47 formal requests with no returned errors. Over 42 strictly matched requests with equal bodies, token lengths and cached-token counts, cumulative end-to-end latency improved 5.98%, decode duration improved 7.20%, and TTFT was 2.04% higher. Across all 44 common equal-length requests, including two with different cache-read counts, end-to-end latency improved 5.27%. A separate 41-request subset versus the frozen 0.3.3 T0 reference showed current T1 end-to-end latency 10.35% lower. These subsets differ; do not combine their percentages. Per-request sums overlap and are not device wall time or throughput. The wall-limited replay includes warmup and cutoff censoring; it is not a complete agent-task benchmark, production stress certification or a default-coupled speedup claim.

The final archived `0e293fda1462d90f3b6758c3fd0b71f98a6e6024` engine passed its glibc 2.34 / CANN 9.2.0 TP8 smoke: default/greedy/mixed sampling, prefix repeat, render token count, strict JSON, function calls, token probability reports, two images and thinking SSE, followed by normal shutdown and release of eight devices. Both engine and embedded adapter require at most glibc 2.34. Full Rust/CPU checks and all 10 execution-package archive/ABI/install checks pass. The nine CUDA SM87 execution-package digests are reused from 0.3.4; no Orin GPU revalidation or long stress replay was performed on this A2 release host. Details are in [the aggregate acceptance report](benchmarks/a2-v0.3.5-acceptance-20261109.json).

## Version 0.3.4

DSv4.1 now defaults to `temperature=1`, `top_p=1`, `top_k=0` (no truncation), matching the DeepInfer serving defaults. Clients can explicitly request greedy decoding with `temperature=0`. The matching execution package samples target tokens on the NPU and couples compatible DSpark Markov drafts to the same request/position/token random identity. Grammar constraints, penalties and probability reporting retain their complete sampling paths. The runtime provides opt-in shared FFN Graph reuse for large prefill packets (`ORINFER_A2_SHARED_GRAPH=1`); the recommended deployment retains the default setting. The execution package also includes fused expert collectives. Request-private state and cache identity retain their existing contracts.

Upgrade the engine, execution archive and this model's `v0.3.4` metadata together. Model weight tensors, tokenizer, chat template and generation-config bytes are unchanged; no weight download or conversion is required for an existing complete installation. The A2 archive contains operators and execution plans separately from the weights, and preserves the compact RoPE generation format. Existing engine/execution/model revisions remain available for rollback.

The accepted sampling implementation previously completed a 600-second mixed-load schedule over concurrency 1/8/16/32, with 2,664 requests, 142,159 completion tokens, no request errors, no detected output/repetition issues and no truncated responses. Including drain, the measured wall time was 661.94 seconds and aggregate completed output was 214.76 tok/s; this is a mixed workload, not decode-only throughput. Requests included arithmetic, generated code, tool history and 2K/16K/100K retrieval plus 50K agent review. These are recorded checks, not a guarantee against all long-context loops or a full task-quality benchmark. The measurements precede the repository cleanup and dependency update; they were reused rather than rerun for this release.

A separate paired C16 100K-prefix/256-output sampling probe measured a temperature-1 per-request decode median of 21.22 tok/s versus 19.09–19.19 previously, and draft acceptance of 38.79% versus 30.13–33.38%. A 240-second Mini replay with the same initialization sequence showed cumulative end-to-end latency −12.76%, decode time −15.56% and TTFT +0.17% over 27 identical completed requests. Time-limited replay has cutoff censoring; these figures are neither a full benchmark score nor a universal speedup. Full records and scope are in [the v0.3.4 aggregate report](benchmarks/a2-v0.3.4-acceptance-20261108.json).

The final archived v0.3.4 binary (`e0304c9e1d9489830fb0f7f3c346933fdf5eab52`) passed bounded real-model startup and requests in the glibc 2.34 / CANN 9.2.0 TP8 container: default and greedy sampling, mixed requests, prefix repeat, function calls and two images, followed by normal shutdown and release of all eight devices. This release also passed the full CPU/Rust checks and execution-package installation checks. Rust is pinned to 1.99.1 and all 443 dependency release records satisfy the seven-day minimum age. CUDA SM87 execution-package content/digests are reused from v0.3.3; Orin GPU validation was not repeated on this A2 host.

## Historical version 0.3.3

The engine overlaps CPU resident-table preparation with bounded device-weight loading, merges adjacent reads, reuses staging buffers and keeps large tensors aligned to 2 MiB inside shared arenas. CPU table pages use fallible zero initialization and bounded retirement. Full CPU tables, weights, vision resources and warmup still complete before API readiness. Normal loading does not scan asset payload hashes.

The compact A2 execution archive is approximately 7.1 MB, compared with 101.8 MB in v0.3.2. Installation reconstructs the original RoPE bytes exactly and verifies them; shared execution plans preserve the original numeric and launch contracts. Use the v0.3.3 engine to install this archive. The new engine also accepts the v0.3.2 model metadata and its original execution archive. Model tensor, tokenizer and chat-template bytes are unchanged; only metadata, documentation and the aggregate benchmark report are updated in this model revision. Operators and execution plans remain outside this weight repository.

With full resident Engram tables, API readiness was measured at 86.59 seconds for the release pressure configuration; the preceding paired loader test measured 219.82 to 82.75 seconds. These measurements use the current filesystem cache state, not a cold-disk claim. The paired hot C32, 2K/256-token decode probe measured about 838 total tok/s with DSpark auto (previous loader 832); it is a separate workload from the agent replay. Large-weight alignment padding adds about 593 MiB per device and is charged to the memory budget.

The final glibc 2.34 / CANN 9.2.0 / TP8 build completed a 600-second C16 Mini-Weka thinking pressure window: 16 warmup plus 152 profiling requests, 0 returned errors, and no observed ACL/worker failure. Cold replay warmup took 189.49 seconds. Backend output counters measured 43.34 tok/s across the full window and 63.50 tok/s after warmup. This timed diagnostic intentionally stops unfinished requests and is not a complete Mini task score or a production-duration guarantee. It uses 16 GiB total CPU KV offload for comparability; the 256 GiB deployment setting above is a capacity ceiling, not a separately measured throughput gain. Per-minute counters, latency distributions, paired requests and native phase aggregates are summarized in [the v0.3.3 pressure report](benchmarks/a2-v0.3.3-pressure-20261106.json).

The archived engine source revision is `f978afbc95dcbd7f0f0dab6537b744ab0b69a869`. Its full `make check` passed; the archived executable and embedded adapter require glibc 2.34 or later. After pressure cancellation and drain, a two-image request returned the expected codes and all eight devices were released on normal shutdown. CUDA SM87 execution packages retain the previously released bytes and digests; this release did not repeat Orin GPU validation. Older model tags remain available for rollback.

## Historical version 0.3.2

This release retains the CANN 9.2 per-device Graph callback registry isolation fix for ACL 507047 and the embedded adapter compatibility fix for glibc 2.34. It adds stage-scoped Graph reuse, bounded DSpark cost extrapolation, fused mHC normalization/Sinkhorn, and the verification capacity above. Model tensors, tokenizer and chat-template bytes are unchanged from the preceding model revision; only this documentation, publication metadata and execution-package reference change. Operators and execution plans remain in the separate GitHub archive. Pin model revision `v0.3.2` together with the matching engine and execution archive. Older model tags remain available.

A 600-second C16 Mini-Weka thinking replay completed 147 profiling requests plus 16 cold warmup requests, with no returned request errors or ACL/worker failure. Warmup took 191.56 seconds. Backend output counters measured about 42.3 tok/s across the full window and 62.3 tok/s after warmup. These are timed replay diagnostics, not a completed task-quality score. Profiling TTFT P95 was 73.4 seconds; workload-dependent long tails remain. The comparison configuration, cache-budget differences and scope are recorded in [the aggregate replay report](benchmarks/a2-v0.3.2-pressure-20261106.json). No one-hour replay was run.

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

See [the aggregate configuration report](benchmarks/a2-v0.3.1-config-20261105.json) for exact settings, workload scope and resource measurements. The fix1 update changes no model tensors or execution identities. Download `--revision v0.3.1-fix1` for the historical fix1 card and validation report; the original `v0.3.1` model tag remains immutable. The known semantic-quality limitation below remains applicable.

## Validation and known limits

The v0.3.2 engine is pinned to source revision `4c68aff3a45b0324cda397a71a25164ffa549fdd`. The release runs `make check`, validates the archived engine and matching execution package, and checks glibc 2.34 compatibility of the executable and embedded registry adapter. Runtime/Graph acceptance uses native A2 TP8 and CANN 9.2. The model tensor bytes and preceding native vision kernels are unchanged. The DSH acceptance below records the earlier fix1 integration, retained by this release; it is not a newly repeated DSH task suite.

A fixed two-tool greedy request passed 74 of 75 responses in the final short checks. One response used `b=10` instead of the requested `b=5` in the second call; the function name, argument types and protocol remained valid. Earlier ordinary-decode runs also exhibited this error pattern. Attribution to model quality, numerical execution or request state is not established. This is not a claim of complete task-quality acceptance.

In an earlier small same-package HTTP comparison with a different CPU-placement/logging setup, warm complete B16 tool requests took about 7.66 seconds versus 7.91 seconds before the controller convergence. B4 had variability, and first-cohort calibration remained slower. These measurements include request completion and do not represent a general decode TPS or a guaranteed latency SLO. A preceding 240-second Mini replay was truncated and is not a complete benchmark score. No one-hour replay was run.

## DeepSeek Harness (DSH) — fix1 acceptance

Fix1 accepts `chat_template_kwargs` from the DSH Chat Completions adapter, including thinking mode, effort, thinking-token budget and history preservation. Explicit request settings override deployment defaults; non-null top-level fields take precedence over matching nested aliases. Template rendering and reasoning-stream parsing use the same effective thinking mode. `/health.chat_capabilities.chat_template_kwargs` reports support.

Use Node.js and `npx @deepseek-ai/dsh@0.2.1-rc.5 --profile headless --help`, then merge the [DSH configuration example](https://github.com/iMMIQ/orinfer/blob/v0.3.1-fix1/examples/dsh-cordis.patch.yml) into the headless profile. Set the service URL, matching model ID and `ORINFER_API_KEY` environment variable. The example uses the OpenAI Chat Completions provider with `input: [text, image]` and dynamically maps DSH thinking settings. See the [installation and thinking guide](https://github.com/iMMIQ/orinfer/blob/v0.3.1-fix1/docs/deepseek-harness.md).

The actual DSH CLI, through a byte-preserving recording proxy, completed five short tasks against native A2 TP8 inference: plain text, two file-tool loops in a resumed session, thinking arithmetic with a nonempty reasoning stream, and two image-tool reads with the correct red/blue ordering. All **8 API requests returned HTTP 200**, and **4 tool calls completed**. The server ended ready with zero active/queued requests and exited cleanly. This is bounded integration acceptance, not a complete agent-quality benchmark. See the [aggregate validation record](benchmarks/a2-v0.3.1-fix1-dsh-20261105.json).

For v0.3.2, retain the existing weights and install the new digest-pinned execution package with its matching model metadata. The v0.3.1 and v0.3.0 release tags remain available. Frontend image caching is bounded and supports UUID references with resend after eviction; the single-image encoding budget remains 1024 visual tokens.

## Attribution

Model weights retain the upstream MIT license in `LICENSE`; the original documentation is in `SOURCE-README.md`. Orinfer is a separate LGPL-3.0-or-later project. See the [GitHub release](https://github.com/iMMIQ/orinfer/releases/tag/v0.3.2) for engine, execution assets and source.
