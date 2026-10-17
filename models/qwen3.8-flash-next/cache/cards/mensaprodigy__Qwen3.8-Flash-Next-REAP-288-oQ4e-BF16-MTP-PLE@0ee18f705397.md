---
license: other
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- qwen4_exp
- moe
- mtp
- omlx
- apple-silicon
- reap
---

# Qwen3.8-Flash-Next REAP-288 oQ4e + Native MTP + PLE

A Qwen3.8-Flash-Next derivative with 288 target MoE experts per layer and the native one-layer MTP predictor retained in the published configuration.[1] It is packaged for `qwen4_exp`-compatible runtimes.[1]

## Release facts — inspected from this repository

| Field | Published value |
| --- | --- |
| Architecture | `Qwen4ExpForConditionalGeneration` / `qwen4_exp` |
| Target transformer layers | 48 |
| Target experts per layer | 288 |
| Target routing width | top-10 |
| Native MTP | 1 layer, 512 experts, top-10 routing |
| Context configuration | 262,144 positions |
| Safetensor shards | 28 |
| Indexed tensors | 3,492 total; 76 `mtp.*` tensors |
| Primary quantization configuration | 4-bit, group size 64, with 232 named overrides |

The repository name is retained as the release identifier.[1] The authoritative artifact metadata is `config.json`: this package has a mixed-precision 4-bit configuration and should not be described as a uniformly BF16 checkpoint.[1]

## Published oMLX tests — matching source artifact

The oMLX records use the local source identifier `Qwen3.8-Flash-Next-REAP-288-oQ4e-BF16-PLE-mtp`; its published benchmark configuration is the matching 288-expert, 4-bit MTP source whose `config.json` and safetensor index match this Hub release.[1][12][13] All accuracy runs below used M4 Max (40c), 128 GB unified memory, macOS 26.5.1, oMLX `v0.7.0.dev4`, 4-bit loading, batch size 4, seed 42 stratification, TurboQuant KV at 4-bit, and Lightning MTP.[12][13]

### Accuracy

| Benchmark | Run label reported by oMLX | Correct / sampled | Score | Run time |
| --- | --- | ---: | ---: | ---: |
| HumanEval | Thinking No | 91 / 100 of 164 | **91.0%** | 4.1 min |
| LiveCodeBench | Thinking No | 56 / 100 of 1,054 | **56.0%** | 1.1 h |

The HumanEval and LiveCodeBench results are separate 100-question samples; they are not combined into a single quality score.[12][13] The benchmark record calls the code task **LiveCodeBench**, not LiveBench.[12]

### Throughput — exact source session on M4 Max (40c) / 128 GB

`PP` is prompt-processing throughput and `TG` is token-generation throughput.[14]

| Context | PP tok/s | TG tok/s | Peak memory | PP visual | TG visual |
| ---: | ---: | ---: | ---: | --- | --- |
| 1k | **642.1** | **49.1** | 43.3 GB | `█████████████████` | `███████████████████` |
| 4k | **692.3** | **48.7** | 46.0 GB | `███████████████████` | `███████████████████` |
| 8k | **744.1** | **51.5** | 46.1 GB | `████████████████████` | `████████████████████` |

| Batch | TG tok/s | Relative to 1× | Visual |
| ---: | ---: | ---: | --- |
| 1× | **49.1** | 1.00× | `████████` |
| 2× | **75.3** | 1.53× | `████████████` |
| 4× | **107.5** | 2.19× | `██████████████████` |

These measurements are workload- and configuration-specific; treat them as a reproducible snapshot, not a hardware-independent speed guarantee.[14]

## Community context — scoped, not a leaderboard

The table below is included to make the community landscape easier to inspect, not to claim a direct ranking.[1][2][3] Sh0wie REAP-288 is the closest public comparison by active-expert count; Jundot oQ4e MTP is an unpruned 512-expert, 4-bit baseline.[2][3]

| Published build | Active experts | Package | HumanEval evidence | LiveCodeBench evidence | Published throughput snapshot | Scope |
| --- | ---: | --- | --- | --- | --- | --- |
| **Mensa REAP-288 oQ4e** | 288 | 4-bit mixed configuration | 91/100 = 91.0%, sampled HumanEval | 56/100 = 56.0% | M4 Max 40c, 4k: 692.3 PP / 48.7 TG tok/s | Exact matching source artifact |
| **Sh0wie REAP-288 MLX** | 288 | 4-bit MLX | 91.5% / 164, publisher-reported | No value used here | M3 Ultra 80c, 1k: 1,025 PP / 46.8 TG tok/s | Same active-expert count; different package, host, and run settings |
| **Jundot oQ4e MTP** | 512 | 4-bit mixed configuration | 152/164 = 92.7%, full HumanEval | 55/100 | M5 Max 40c, 4k: 1,286 PP / 62.5 TG tok/s | Different expert count, host, runtime, and batch |

The Mensa row is from the matching-source oMLX HumanEval, LiveCodeBench, and M4 Max performance records.[12][13][14] Sh0wie's HumanEval figure is from its published 288-expert card; its cited oMLX performance run uses M3 Ultra / 96 GB, 4-bit loading, 1,024-token context, and MTP disabled.[2][6] Jundot's oMLX records use its 512-expert 4-bit package; the HumanEval run is full-dataset on M4 Max / 128 GB, while the listed 4k throughput snapshot is on M5 Max / 128 GB.[10][11]

### What is comparable today

- **Architecture match:** Mensa and Sh0wie both retain 288 target experts with top-10 routing.[1][2]
- **Accuracy:** Mensa's published HumanEval result is a 100-question sample, while Sh0wie and Jundot publish 164-problem results under different packages and harnesses; show all three as context, not as a controlled ranking.[13][2][10]
- **Throughput:** the published Sh0wie and Jundot runs use different chips, contexts, runtime settings, and MTP state; do not use them as a speed prediction for this artifact.[6][11]

## Runtime and compatibility

This release needs a runtime that supports `qwen4_exp`, 288 target experts, and the asymmetric 512-expert native MTP layer recorded in `config.json`.[1] Memory use and speed depend on the runtime version, context length, KV-cache policy, storage path, concurrency, and MTP configuration.[unverified]

## Lineage and license

The base model is [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), whose official card publishes its evaluation tables and license.[4] This derivative is distributed under the **Qwen Community License 1.0**; the verbatim upstream license is included as `LICENSE` in this repository.[1][4]

## Limitations

- Evaluate the exact release on your own workload before relying on any benchmark snapshot.[12][13][14]
- Do not treat the Sh0wie or Jundot results as performance or quality guarantees for this build.[2][3][10]
- Vision or multimodal quality has not been independently characterized for this release.[unverified]
- The model name is not authoritative evidence of runtime precision; use `config.json` when selecting a compatible runtime.[1]

## Sources

[1] https://huggingface.co/mensaprodigy/Qwen3.8-Flash-Next-REAP-288-oQ4e-BF16-MTP-PLE
[2] https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit
[3] https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp
[4] https://huggingface.co/Qwen/Qwen3.8-Flash-Next
[6] https://omlx.ai/benchmarks/performance/2xs5i144
[10] https://omlx.ai/benchmarks/intelligence/r6k92mrp
[11] https://omlx.ai/benchmarks/performance/00l1yizf
[12] https://omlx.ai/benchmarks/intelligence/b7jdhtsf
[13] https://omlx.ai/benchmarks/intelligence/jwmf2z73
[14] https://omlx.ai/benchmarks/performance/z4hn1h6n
