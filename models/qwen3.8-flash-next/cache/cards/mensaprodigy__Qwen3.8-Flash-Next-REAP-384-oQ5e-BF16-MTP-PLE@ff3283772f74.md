---
license: other
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- qwen4_exp
- moe
- mtp
- omlx
- apple-silicon
- benchmarked
---

# Qwen3.8-Flash-Next REAP-384 oQ5e + Native MTP + PLE

This Qwen3.8-Flash-Next derivative has a 384-expert target MoE bank and retains the native one-layer, 512-expert MTP predictor in the published configuration.[1] It is packaged for `qwen4_exp`-compatible runtimes.[1]

## Release facts — inspected from this repository

| Field | Published value |
| --- | --- |
| Architecture | `Qwen4ExpForConditionalGeneration` / `qwen4_exp` |
| Target transformer layers | 48 |
| Target experts per layer | 384 |
| Target routing width | top-10 |
| Native MTP | 1 layer, 512 experts, top-10 routing |
| Context configuration | 262,144 positions |
| Safetensor shards | 32 |
| Indexed tensors | 3,448 total; 32 `mtp.*` tensors |
| Primary quantization configuration | 5-bit, group size 64, with 303 named overrides |

The repository name is retained as the release identifier.[1] The authoritative artifact metadata is `config.json`: it contains a mixed-precision quantization configuration, so this release must not be described as a uniformly BF16 checkpoint.[1]

## Reproducible oMLX benchmark envelope

The charts below use only published oMLX runs for **this exact 384-expert oQ5e artifact**.[9][10][12][13] Accuracy runs used an M4 Max (40-core GPU), 128 GB unified memory, macOS 26.5.1, oMLX `v0.7.0.dev4`, 5-bit model loading, batch size 4, seed 42 stratified sampling, TurboQuant KV at 4-bit, and Lightning MTP.[9][10][12][13] Each reported accuracy result is a 100-question sample, not a full-dataset result.[9][10][12][13]

**Comparison boundary:** this card deliberately excludes Sh0wie REAP-288 results.[1][2] That model has a different active-expert count and build, so placing it on the same chart would not be apples-to-apples.[1][2] A matching Mensa REAP-288 card can carry a clearly labeled 288-to-288 comparison after its own release and matching measurements are published.[unverified]

## Community comparison — scoped, not a leaderboard

This matrix preserves useful community context while keeping unlike builds, harnesses, and hosts out of a single winner-ranking chart.[1][7][27] The **384-expert** Sh0wie build is the closest community artifact by active-expert count; Jundot's oQ4e MTP package is a 512-expert, 4-bit community baseline.[7][27]

| Published build | Active experts | Package | HumanEval evidence | LiveCodeBench evidence | Published throughput snapshot | Comparison scope |
| --- | ---: | --- | --- | --- | --- | --- |
| **Mensa REAP-384 oQ5e** | 384 | 5-bit mixed configuration | 97/100 (Thinking No); 96/100 (Thinking Yes) | 61/100 (No); 62/100 (Yes) | M4 Max 40c, 4k: 679.1 PP / 53.9 TG tok/s | Exact release; its own oMLX runs |
| **Sh0wie REAP-384 MLX** | 384 | 4-bit MLX | 92.1% on 164 HumanEval problems | No value used here | Not included here | Same active-expert count; different package and publisher harness |
| **Jundot oQ4e MTP** | 512 | 4-bit mixed configuration | 152/164 = 92.7%, full HumanEval | 55/100 | M5 Max 40c, 4k: 1,286 PP / 62.5 TG tok/s | Different expert count, quantization, runtime, batch, and performance host |

The Mensa row is based on its exact-release oMLX records and artifact metadata.[1][9][10] The Sh0wie row is publisher-reported for its 384-expert 4-bit MLX build, using a 164-problem HumanEval harness.[7] The Jundot row is tied to its oMLX HumanEval, LiveCodeBench, and M5 Max performance records.[31][32][30]

### Closest shared-suite community context — LiveCodeBench

| Build / oMLX run | Hardware | Package | Batch | Sample | Score | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Mensa REAP-384 oQ5e, Thinking Yes | M4 Max (40c), 128 GB | 5-bit | 4 | 100/1,054 | **62.0%** | Exact release result |
| Jundot oQ4e MTP, Thinking Yes | M4 Max (40c), 128 GB | 4-bit | 1 | 100/1,054 | **55.0%** | Same suite and sample size; not a controlled head-to-head |

Both LiveCodeBench rows use the same benchmark family, 100-question sample size, M4 Max (40c) / 128 GB host class, seed 42 stratification, TurboQuant KV, and Lightning MTP.[13][32] They still differ in active-expert count, quantization, oMLX version, macOS version, and batch size, so this table intentionally provides no score delta, winner label, or common bar scale.[13][32]

### Throughput context — do not compare across chip generations

| 4k published run | Chip / RAM | Package | PP tok/s | TG tok/s | Peak memory | Scope |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Mensa REAP-384 oQ5e | M4 Max (40c) / 128 GB | 5-bit | **679.1** | **53.9** | 72.8 GB | Exact-release session |
| Jundot oQ4e MTP | M5 Max (40c) / 128 GB | 4-bit | **1,286** | **62.5** | 72.3 GB | Different 512-expert model and newer chip generation |

The two 4k throughput records are useful deployment snapshots, not a speed ranking; the M5-vs-M4 host difference alone prevents an apples-to-apples conclusion.[18][30]

## Accuracy charts — same artifact, same host, same sample size

### HumanEval

| Run label reported by oMLX | Correct / sampled | Accuracy | Visual |
| --- | ---: | ---: | --- |
| Thinking No | 97 / 100 | **97.0%** | `███████████████████░` |
| Thinking Yes | 96 / 100 | **96.0%** | `███████████████████░` |

The HumanEval sample is 100 of 164 problems.[9][10] The run labels, timing, and score records are published separately: **4.5 minutes** for the 97.0% result and **7.5 minutes** for the 96.0% result.[9][10] A one-question difference in a 100-question sample is descriptive—not a statistically established ranking between settings.[unverified]

### LiveCodeBench (not relabeled as LiveBench)

| Run label reported by oMLX | Correct / sampled | Accuracy | Visual |
| --- | ---: | ---: | --- |
| Thinking No | 61 / 100 | **61.0%** | `████████████░░░░░░░░` |
| Thinking Yes | 62 / 100 | **62.0%** | `████████████░░░░░░░░` |

These oMLX records are explicitly labeled **LiveCodeBench** and use a 100-question sample of the 1,054-question benchmark.[12][13] They are shown as LiveCodeBench, not as “LiveBench,” to avoid conflating distinct benchmarks.[12][13] The recorded run times were **38.4 minutes** (Thinking No) and **1.4 hours** (Thinking Yes).[12][13]

HumanEval and LiveCodeBench are intentionally charted separately: they are different datasets and should not be combined into a single quality score.[9][10][12][13]

## Throughput charts — exact artifact on M4 Max (40c) / 128 GB

The first table is one comparable oMLX session: same model, host, 5-bit load, and Lightning MTP.[18] `PP` is prompt-processing throughput; `TG` is token-generation throughput.[18] Throughput is workload- and configuration-specific, not a general speed guarantee.[18]

### Single-stream throughput by context

| Context | PP tok/s | TG tok/s | Peak memory | PP visual | TG visual |
| ---: | ---: | ---: | ---: | --- | --- |
| 1k | **610.0** | **54.3** | 70.0 GB | `█████████████████` | `███████████████████` |
| 4k | **679.1** | **53.9** | 72.8 GB | `███████████████████` | `███████████████████` |
| 8k | **725.7** | **56.5** | 72.9 GB | `████████████████████` | `████████████████████` |

### Generation throughput by concurrent stream batch

| Batch | TG tok/s | Relative to 1× | Visual |
| ---: | ---: | ---: | --- |
| 1× | **54.3** | 1.00× | `█████` |
| 2× | **130.9** | 2.41× | `████████████` |
| 4× | **187.9** | 3.46× | `██████████████████` |

A separately published 16k-context run on the same M4 Max (40c) / 128 GB, 5-bit configuration reports **709.2 PP tok/s**, **56.5 TG tok/s**, and **73.2 GB** peak memory.[20] It is listed separately because it was not part of the three-context session above.[18][20]

## Runtime and compatibility

This release needs a runtime that supports the `qwen4_exp` architecture and the asymmetric target/MTP expert counts recorded in `config.json`.[1] A model appearing in a server catalog is not proof of a successful load.[unverified] Memory use and speed depend on runtime version, context length, KV-cache policy, storage path, concurrency, MTP configuration, and hardware.[18][20]

## Lineage and license

The base model is [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), whose official card publishes its own base-model evaluation tables and license.[8] This derivative is distributed under the **Qwen Community License 1.0**; the verbatim upstream license is included as `LICENSE` in this repository.[1][8] Review the license before any commercial deployment.[8]

## Limitations

- All accuracy values above are 100-question stratified samples; evaluate the full workload before relying on them.[9][10][12][13]
- Do not compare these 384-expert result charts to 288-expert, quantization-different, or harness-different artifacts as though they were direct rankings.[1][2]
- Vision or multimodal quality has not been independently characterized for this release.[unverified]
- The visual bars are presentation aids; the numeric columns are the authoritative values.[unverified]

## Sources

[1] https://huggingface.co/mensaprodigy/Qwen3.8-Flash-Next-REAP-384-oQ5e-BF16-MTP-PLE
[2] https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-bf16
[7] https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-384-MLX-4bit
[8] https://huggingface.co/Qwen/Qwen3.8-Flash-Next
[9] https://omlx.ai/benchmarks/intelligence/q2adqprz
[10] https://omlx.ai/benchmarks/intelligence/rvjjhv65
[12] https://omlx.ai/benchmarks/intelligence/b6vnuvc6
[13] https://omlx.ai/benchmarks/intelligence/2tc37530
[18] https://omlx.ai/benchmarks/performance/dedea6ut
[20] https://omlx.ai/benchmarks/performance/pfeuc75o
[27] https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp
[30] https://omlx.ai/benchmarks/performance/00l1yizf
[31] https://omlx.ai/benchmarks/intelligence/kmdnbouq
[32] https://omlx.ai/benchmarks/intelligence/r6k92mrp
