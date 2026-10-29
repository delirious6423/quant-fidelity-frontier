---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: tensorfold
tags:
- deepseek
- exl3
- tensorfold
- quantization
- dgx-spark
- tp3
- 3.5bpw
- speculative-decoding
- inference-optimization
---

# DeepSeek-V4.1-Flash-1017 · EXL3 3.5 bpw · TensorFold 1.2.7

<table>
<tr>
<td width="50%"><a href="assets/1.2.6/lakyfx-code.png"><img src="assets/1.2.6/lakyfx-code.png" width="100%" alt="LakyFX TensorFold 1.2.6 Code benchmark, SparkDash run 2: C1 128.9 tok/s; C12 266.6 tok/s aggregate; thinking OFF, 400 tokens per stream." /></a></td>
<td width="50%"><a href="assets/1.2.6/lakyfx-prose.png"><img src="assets/1.2.6/lakyfx-prose.png" width="100%" alt="LakyFX TensorFold 1.2.6 Prose benchmark, SparkDash run 2: C1 95.5 tok/s; C12 235.6 tok/s aggregate; thinking OFF, 400 tokens per stream." /></a></td>
</tr>
</table>

**763B parameters. Three DGX Sparks. An open recipe you can reproduce.**
Our independent DeepSeek 1017 quantization pairs with **TensorFold recipe 1.2.7: Smart KV Cache Controller + Adaptive Speculation Controller**. The preserved **1.2.6 SparkDash run 2** reaches **266.6 tok/s code** and **235.6 tok/s prose** aggregate at C12, with **128.9 tok/s single-stream code**. The posters above show that same complete run, not a new 1.2.7 decode benchmark. Thinking was OFF for SparkDash; normal serving keeps MAX reasoning as its default.

**[Get recipe 1.2.7](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.7/docs/DS41F-1.2.7.md)** · **[GitHub release](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/releases/tag/v1.2.7)** · **[Download the weights](RELEASE-LAYOUT.md)** · [SHA-256 manifest](RELEASE-MANIFEST.json)

## Update — 29 October 2026: recipe 1.2.7 and Smart KV Cache Controller

**1.2.7 is our current published production recipe**, the final release name for the deployed X4 payload. It adds adaptive KV retention for mixed interactive sessions and autonomous agent jobs on top of the established engine and speculative controller. The same published 1017 weights are used: upgrading the recipe does not require downloading a new quantization.

### Why the Smart KV Cache Controller matters

An agent workflow is rarely one uninterrupted chat. A video job can prepare a long script, wait for a render, inspect the result and return for edits. Meanwhile, an email check may finish in one turn and not return for hours. Keeping every small recent job equally can displace a much more expensive conversation that is about to continue.

Our controller **learns from actual, verified prefix returns and the cost of recomputing them**. It uses bounded history to adapt retention and placement decisions under pressure, balancing frequently returning short jobs with long working contexts. It does not need Hermes-specific names, a manually labeled cron queue or a promise from the client about what comes next.

- **Protect useful continuations.** Reuse history, return spacing in request events and observed prefill costs inform the choice. A one-off task can age out; a genuinely returning short task can gain protection, while large reusable contexts keep explicit safeguards.
- **Use the existing memory more effectively.** The operating envelope stays at **7M configured KV positions and 3 GiB prompt/SWA checkpoints**. Both physical KV and a compatible checkpoint must still exist; free pool space alone does not make an old prefix reusable.
- **Keep control work bounded.** The policy runs on the CPU at allocation/pressure decisions, with no new per-token GPU work or background compactor. Any eligible relocation remains pressure-only, with at most one donor group and a **320 MiB per-rank cap/credit**. The policy cannot retain every idle session indefinitely.
- **Two separate adaptive controllers.** KV retention chooses which reusable states to preserve. Adaptive speculation chooses useful draft work within the **R48 shared verification ceiling**. They learn runtime estimates, not model weights; target verification and sampling remain authoritative.

**Real mixed Hermes workflow, returning requests:**

| Session group | Requests | Prefix cache hit | Median prefill |
|---|---:|---:|---:|
| My work sessions | 201 | **99.14%** | **0.641 s** |
| Autonomous cron sessions | 613 | **96.83%** | **0.672 s** |
| Combined work + cron | 814 | **97.73%** | **0.661 s** |
| Long work contexts, >128K input; subset of work | 121 | **99.63%** | **0.656 s** |

These are live observations from **28 October, approximately 2h12m**, with 937 successful requests overall. The table excludes first observed session loads; including first loads and other traffic, the overall hit was **95.09%**. It is not an equal-input decode comparison.

Across two periods of that running version, the **uncached input share fell by 47%**. The flyer estimates **+67% effective prefill throughput** using 2K tok/s computed input and 200K tok/s effective cache reuse. That estimate also includes warmup and changing task mix; it is not a measured decode or complete-workflow speedup.

[![LakyFX 1.2.7 Smart KV Cache Controller: live work/cron reuse, 47% lower uncached input share and estimated effective prefill gain](assets/1.2.7/live-kv-cache.png)](assets/1.2.7/live-kv-cache.png)

**[Controller design and exact recipe](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.7/docs/DS41F-1.2.7.md)** · [Live measurement method and evidence](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.7/docs/LIVE-CACHE-1.2.7.md) · [Machine-readable live results](benchmarks/1.2.7/LIVE-WORKFLOW.json) · [X1–X4 development history](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.7/docs/X1-X4-DEVELOPMENT.md)

## SparkDash 1.2.6 — preserved second-run results

The two opening posters and the table below use **run 2 of three consecutive stock SparkDash campaigns**, measured on **24 October 2026** on the same continuously loaded model/cache/controller. This is one complete run, not a best-of-three selection by cell. Original collectors and formulas; thinking **OFF**, temperature **0**, top_p **1**, **400 tokens per stream**.

| Concurrency | Code aggregate tok/s | Code stream median tok/s | Prose aggregate tok/s | Prose stream median tok/s |
|---|---:|---:|---:|---:|
| C1 | 128.92 | 128.92 | 95.52 | 95.52 |
| C2 | 179.94 | 91.47 | 129.07 | 66.54 |
| C3 | 190.06 | 64.24 | 153.30 | 51.60 |
| C4 | 209.58 | 53.83 | 165.04 | 41.82 |
| C5 | 222.02 | 46.23 | 177.76 | 36.55 |
| C6 | 228.80 | 39.57 | 190.86 | 33.22 |
| C8 | 243.85 | 32.63 | 212.85 | 27.89 |
| C10 | 260.74 | 27.53 | 220.31 | 22.78 |
| C12 | 266.58 | 24.21 | 235.55 | 20.35 |

Aggregate measures all streams over their shared post-first-token decode window; stream median uses each request's own decode window. The medians are not multiplied by concurrency to manufacture aggregate throughput. All **102/102 decode streams** in this run completed their 400-token budgets. Prefill and all three runs remain available with their original labels in the full report.

**[Run-2 machine-readable data](benchmarks/1.2.6/RESULTS.json)** · **[All three campaigns, prefill, acceptance and exact formulas](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.6/docs/SPARKDASH-20261024-1.2.6.md)**

## Preserved 1.2.5 release — 23 October 2026

The complete TP3 weights are ready, and **1.2.5 was published on 23 October 2026**: native source, adaptive speculation, prefix-cache improvements, production-matched gateway, tests and benchmark receipts. It is the public name for the measured **1.2.4k2a / K2a** payload; the version rename does not change the measured binaries or model weights. Existing users of these 1017 weights can update the runtime recipe without downloading a new quantization.

## Preserved SparkDash 1.2.5 results — code and prose

The original 1.2.5 posters and measured values are retained below.

<table>
<tr>
<td width="50%"><a href="assets/1.2.5/lakyfx-code.png"><img src="assets/1.2.5/lakyfx-code.png" width="100%" alt="LakyFX TensorFold 1.2.5 Code benchmark: 3 DGX Sparks, TP3, EXL3 3.51 bpw; C1 126.1 tok/s, C12 246.2 tok/s aggregate; thinking OFF, 400 tokens per stream." /></a></td>
<td width="50%"><a href="assets/1.2.5/lakyfx-prose.png"><img src="assets/1.2.5/lakyfx-prose.png" width="100%" alt="LakyFX TensorFold 1.2.5 Prose benchmark: 3 DGX Sparks, TP3, EXL3 3.51 bpw; C1 87.7 tok/s, C12 183.6 tok/s aggregate; thinking OFF, 400 tokens per stream." /></a></td>
</tr>
</table>

**3 × NVIDIA DGX Spark 128 GB / GB10 · TP3 · EXL3 ~3.51 bpw routed experts.** Original unmodified SparkDash collectors, thinking **OFF**, temperature **0**, top_p **1**, **400 generated tokens per stream**. One stock wave at each concurrency and prompt type, measured on 23 October 2026.

All values below are **tokens per second**. **C** means concurrent streams; **Aggregate** is their combined decode-window throughput; **Stream mean** is the arithmetic mean of individual stream decode rates.

| Concurrency | Code aggregate tok/s | Code stream mean tok/s | Prose aggregate tok/s | Prose stream mean tok/s |
|---|---:|---:|---:|---:|
| C1 | 126.06 | 126.06 | 87.71 | 87.71 |
| C2 | 169.36 | 88.39 | 111.79 | 60.93 |
| C3 | 174.00 | 64.20 | 123.04 | 45.27 |
| C4 | 187.34 | 50.73 | 138.33 | 38.38 |
| C5 | 207.70 | 44.90 | 146.40 | 33.50 |
| C6 | 218.05 | 38.86 | 154.10 | 29.37 |
| C8 | 226.19 | 30.94 | 164.43 | 23.92 |
| C10 | 241.80 | 26.44 | 169.73 | 20.00 |
| C12 | 246.21 | 23.12 | 183.57 | 18.00 |

**102/102 measured decode streams completed their 400-token budgets.** These are throughput tests, not completed-code correctness scores. Aggregate uses the shared window from the first stream's first token to the last stream's last token, counting `completion_tokens - 1` per stream. Stream mean uses each stream's own decode window; multiplying it by C does not reproduce aggregate throughput. C12 is the concurrency cap, not a fixed GPU batch-size claim.

**[Machine-readable benchmark data](benchmarks/1.2.5/RESULTS.json)** · [Full measurements and exact formulas](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.5/docs/SPARKDASH-20261023.md) · [Pinned upstream SparkDash](https://github.com/MiaAI-Lab/sparkDash/tree/9e3f01729a4e63dccc2b6b608e9e67d3168891b6)

### Prefill

| Target | Actual prompt tokens | Effective prefill tok/s | Cached tokens | Measurement |
|---|---:|---:|---:|---|
| 1K | 1,042 | 1,922.05 | 0 | Low confidence |
| 2K | 2,065 | 2,224.18 | 0 | 3/3 samples |
| 4K | 4,117 | 2,097.48 | 0 | 3/3 samples |
| 8K | 8,212 | 2,228.93 | 0 | 3/3 samples |

Three samples per target; all **12/12** passed. SparkDash's stock calibrated-TTFT fallback is used, with zero cached tokens. The 1K row retains its upstream low-confidence flag. This is effective request-level prefill, not isolated GPU-kernel throughput. Decode warmups and existing caches were retained; the prose C1 first-token observation is warm-prefix, not a cold-start guarantee. No new inference was run for this card update.

## What makes our recipe different

- **Smart adaptive speculation.** Our online controller learns draft acceptance and actual verification cost for the current load and context, then allocates useful speculative work inside the shared **R48** budget. It can spend more where drafts pay off and less where verification would cost more than it saves. Target-model verification and sampling remain in charge. The learner updates runtime estimates, not model weights. [Design, boundaries and AdaSpark inspiration](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.5/docs/ADAPTIVE-SPECULATION.md).
- **Less wasted work per decode round.** Shape-aware verification pricing, skipping unusable sampling rows, CPU/GPU overlap, bounded timing probes and refresh of stale cost estimates are combined with turns for existing streams between new admissions. [Scheduling and speculation recipe](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.5/docs/DS41F-1.2.4.md).
- **Prefix reuse for long-running agents.** K2/K2a places snapshots around useful retained prefixes, protects borrowed resume entries and reclaims confirmed physically dead entries first under pressure. Per-depth acceptance, reuse diagnostics and corrected KV occupancy make the behavior inspectable. Separate RigMark tests observed **42–46 ms median immediate prefix-replay TTFT** at 8K/32K/64K; replay is cache reuse, not fresh prefill speed. [Receipts and definitions](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.5/docs/BENCHMARK-1.2.5.md).
- **Lower prompt-formatting overhead.** The head's direct-output template avoids repeatedly copying growing conversation history. Its rendered text and token IDs matched the reference on 192 Python and 192 native test cases. The exact overlay is included in the recipe; published model weights stay unchanged.
- **Tools and vision with lifecycle control.** The published gateway includes tool handling, queued image requests, finite deadlines and native cancellation receipts. The separate unchanged MAX-reasoning agent workflow completed **78/78 tasks**; its scope and repaired tool errors are [documented](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.5/docs/BENCHMARK-1.2.5.md).

The online controller builds on TensorFold's existing speculative machinery and was inspired by AdaSpark. Credit for the engine belongs to **[TensorFold](https://tensorfold.dev/), Ash Hart and contributors**; our contribution is the quantization, TP3 recipe and the published tuning/integration work.

## Run recipe 1.2.7 on three Sparks

| Setting | Current recipe 1.2.7 |
|---|---|
| Model | DeepSeek-V4.1-Flash-1017; 763B total parameters in the original model |
| Weights | LakyFx mixed-precision EXL3; ~3.51 bpw main routed experts, K6 head, K4 DSpark draft projections, FP8 Engram tables |
| Hardware / parallelism | 3 × DGX Spark 128 GB; TensorFold native CUDA, TP3 |
| Streams / verification | C12 streams; R48 shared verification-row cap; adaptive speculation |
| KV / snapshots | 7,000,000 configured shared KV tokens; 3 GiB prompt-state snapshots |
| Context / prefill chunk | 401,408 tokens / 2,048 tokens |
| Normal serving | MAX reasoning by default; explicit client overrides supported |
| Build / launcher selectors | `--recipe v127` / `--variant 1.2.7` |

Start with the **[1.2.7 build, head-template preparation and launch guide](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.7/docs/DS41F-1.2.7.md)**, [host requirements](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.5/docs/HOST-ENVIRONMENT.md) and [TP3 installation](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.5/docs/TENSORFOLD-INSTALL.md). Download the matching rank cache and sparse Engram archive per node using the [artifact layout](RELEASE-LAYOUT.md). The tested native path does not require a duplicate copy of all root EXL3 layers on every Spark. The latest recipe adds a separate **head-only chat-template overlay**; followers retain the original model metadata.

Use `--variant 1.2.7` for the current recipe. The [1.2.5 guide](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.5/docs/DS41F-1.2.5.md) and its results remain available as history. The original 1.2.0 instructions remain historical weight-preparation documentation. Configured context and KV capacity are settings, not proof that every maximum-length concurrent workload fits. Local rebuilds still need qualification on their own hosts.

## Original SparkDash share cards

These are exports from SparkDash's unchanged upstream share-card renderer, generated from the same 1.2.5 measurements as the preserved 1.2.5 LakyFX posters above. Click an image for full resolution.

<table>
<tr>
<td width="50%"><a href="assets/1.2.5/sparkdash-original-code.png"><img src="assets/1.2.5/sparkdash-original-code.png" width="100%" alt="Original SparkDash code decode share card for LakyFX 1.2.5, C1 through C12." /></a></td>
<td width="50%"><a href="assets/1.2.5/sparkdash-original-prose.png"><img src="assets/1.2.5/sparkdash-original-prose.png" width="100%" alt="Original SparkDash prose decode share card for LakyFX 1.2.5, C1 through C12." /></a></td>
</tr>
</table>

[![Original SparkDash prefill share card for LakyFX 1.2.5, 1K through 8K prompts.](assets/1.2.5/sparkdash-original-prefill.png)](assets/1.2.5/sparkdash-original-prefill.png)

## Quantization, provenance and acknowledgements

This is an independent quantization of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), generated from official revision [`1bae0e1088993b9b82de08f59de5be5a34e42170`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170). **3.51 bpw describes the main routed experts**, not every tensor or a whole-model average. Other projections retain their specified precisions. No older-revision weight values are reused. The base model and weights are MIT licensed.

**About the Hub's model-size badge:** its automatic `0.5B params / BF16` figure counts the auxiliary `vision.safetensors`, not the full 763B-parameter model or its EXL3/TP3 caches.

The earlier TensorFold TF351 setup combined main routed-expert weights from [bot-lab-21's DeepSeek-V4.1-Flash EXL3 3.5bpw Pollard checkpoint](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) with other components, including the DSpark draft, from [Mia-AiLab's DeepSeek-V4.1-Flash EXL3 release](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw). The expert precision map and quantization approach for this 1017 release draw on that prior work. All weights here are generated from the pinned DeepSeek 1017 source; no older model weights are reused.

Thank you to **bot-lab-21** for openly publishing the 3.51 bpw expert weights, layer allocation, calibration tools, and measurements; **WestWaters** for the [Pollard Weights](https://github.com/WestWaters/pollard-weights) method; **Mia-AiLab** for the earlier EXL3 export; **turboderp** for [ExLlamaV3 and EXL3](https://github.com/turboderp-org/exllamav3); and **tonyd2wild and Kai** for the [DGX Spark serving groundwork](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark). DeepSeek created the base model, and TensorFold supplies this release's TP3 runtime.

**Special thanks to the [TensorFold project](https://tensorfold.dev/) and its creator, [Ash Hart (@ashxhart)](https://x.com/ashxhart).** Ash, your persistence in making demanding models run on real hardware inspired us to build these weights for our three DGX Sparks. Thank you for continuing to push TensorFold forward and for sharing the work openly. We hope these weights become a useful contribution to the community around your project.

**Historical measurements, attributed to bot-lab-21's older V4.1-Flash checkpoint:** on 39 in-domain calibration sequences of 2048 tokens, its published reference NLL was 1.3209 and its 3.51 bpw expert build was 1.3177 (paired difference -0.0032 +/- 0.0055). Those sequences were also used for expert calibration, so this is not an independent held-out comparison. The same model card reports perplexity 4.043 on six held-out texts (4,210 tokens) and HumanEval / HumanEval+ greedy pass@1 of 95.1% / 92.1%, without matching unquantized baselines for those tests. These results document the predecessor; **they are not quality measurements of the new 1017 weights**. Results for this release will be published separately after its own verification.

## Weight integrity and dated qualification

All **40 main EXL3 layers**, configuration, tokenizer, vision wrapper, **three TP3 rank caches** and **three rank-specific sparse Engram archives** are complete and have component SHA-256 receipts. [Release manifest](RELEASE-MANIFEST.json) · [Per-node download layout](RELEASE-LAYOUT.md).

The original **18 October 2026** artifact qualification passed three-rank GPU loading, basic MAX arithmetic, automatic tool calls and tool-result roundtrip, vision, prefix reuse, DSpark drafting and 12 simultaneous short requests. Its [machine-readable receipt](verification/PRODUCTION-QUALIFICATION.json) and [scope](verification/SERVING-QUALIFICATION.md) remain dated evidence of that earlier runtime. Its 8M KV / 2 GiB / R32 settings and original template do not describe the current recipe 1.2.7. The [current 1.2.7 qualification](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.7/results/20261028-1.2.7/QUALIFICATION.json), along with the [23 October recipe parity receipt](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.5/results/20261023-1.2.5/PRODUCTION-PARITY.json) and [1.2.5 qualification](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.5/results/20261023-1.2.5/QUALIFICATION.json) cover the newer runtime; the weight artifacts are unchanged.

Original rank readback and Engram owned-range checks were reused with fresh path/file-state/header and wrapper checks; the large payloads were not rescanned against busy production for this documentation update. Engrams are TP3-specific **GNU sparse archives**, about **67.6 GB per rank / 203 GB combined**. Preserve holes on extraction and copy, keep ranks separate, and follow the required regular cache hardlink instructions in the layout.

The root and `models/TF351-1017/` original `chat_template.jinja` sidecars remain intact, with their original 56-case encoder qualification. The current recipe supplies its qualified head-only overlay rather than overwriting those released files. Historical CPU-stage pending-GPU fields and old smoke settings are retained as historical records. This update changes documentation and artwork; it does not change weights, services or boot configuration.
