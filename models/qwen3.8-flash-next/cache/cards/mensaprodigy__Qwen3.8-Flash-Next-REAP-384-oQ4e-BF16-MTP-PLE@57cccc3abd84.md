---
license: other
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- qwen4_exp
- moe
- mtp
- ple
- omlx
- apple-silicon
- benchmarked
---

# Qwen3.8-Flash-Next REAP-384 oQ4e + Native MTP + PLE

This Qwen3.8-Flash-Next derivative has a 384-expert target MoE bank and a native MTP predictor with 512 experts. It is packaged for `qwen4_exp`-compatible runtimes.[1]

## Release facts — inspected from this repository

| Field | Published value |
| --- | --- |
| Architecture | `Qwen4ExpForConditionalGeneration` / `qwen4_exp` |
| Target transformer layers | 48 |
| Target experts per layer | 384 |
| Target routing width | top-10 |
| Native MTP predictor | 512 experts, top-10 routing |
| Context configuration | 262,144 positions |
| Safetensor shards | 29 |
| Indexed tensors | 3,448 total; 32 `mtp.*` tensors |
| Primary quantization configuration | 4-bit affine, group size 64, with 260 named overrides |

The repository identifier is retained as the release name.[1] The authoritative metadata is `config.json`: it describes a mixed 4-bit package with named 5-, 6-, and 8-bit overrides, so this release is not a uniformly BF16 checkpoint.[1]

## Reproducible oMLX benchmark envelope

The benchmark tables below use published oMLX runs for this exact 384-expert oQ4e artifact: HumanEval,[3][4] LiveCodeBench,[5][6] and performance.[7][8][9] The accuracy runs used an M4 Max (40-core GPU), 128 GB unified memory, macOS 26.5.1, oMLX `v0.7.0.dev4`, 4-bit model loading, batch size 4, seed 42 stratified sampling, TurboQuant KV at 4-bit, and Lightning MTP.[3][4] The LiveCodeBench records report the same host, software, sampling, and acceleration envelope.[5][6] Each accuracy result is a 100-question sample, not a full-dataset result.[3][5]

## Accuracy charts — same artifact, same host, same sample size

### HumanEval

| Run label reported by oMLX | Correct / sampled | Accuracy | Visual |
| --- | ---: | ---: | --- |
| Thinking No | 91 / 100 | **91.0%** | `██████████████████░░` |
| Thinking Yes | 95 / 100 | **95.0%** | `███████████████████░` |

The HumanEval sample is 100 of 164 problems. The recorded run times were 3.5 minutes for Thinking No and 8.5 minutes for Thinking Yes.[3][4] A four-question difference in a 100-question sample is descriptive, not a controlled ranking between settings.[unverified]

### LiveCodeBench (not relabeled as LiveBench)

| Run label reported by oMLX | Correct / sampled | Accuracy | Visual |
| --- | ---: | ---: | --- |
| Thinking No | 55 / 100 | **55.0%** | `███████████░░░░░░░░░` |
| Thinking Yes | 56 / 100 | **56.0%** | `███████████░░░░░░░░░` |

These records are explicitly labeled **LiveCodeBench** and use a 100-question sample of the 1,054-question benchmark. They are not relabeled as “LiveBench,” which is a different benchmark.[5][6] The recorded run times were 35.9 minutes for Thinking No and 2.5 hours for Thinking Yes.[5][6]

HumanEval and LiveCodeBench are intentionally shown separately because they measure different datasets.[3][4][5][6]

## Throughput charts — exact artifact on M4 Max (40c) / 128 GB

`PP` is prompt-processing throughput; `TG` is token-generation throughput. The records below are one oMLX session using a Code (Python) context, 4-bit loading, TurboQuant KV at 4-bit, and Lightning MTP.[7][8][9] They are workload-specific deployment snapshots, not general speed guarantees.[7][8][9]

### Single-stream throughput by context

| Context | PP tok/s | TG tok/s | Peak memory | PP visual | TG visual |
| ---: | ---: | ---: | ---: | --- | --- |
| 1k | **615.6** | **67.7** | 58.7 GB | `█████████████████` | `████████████████████` |
| 4k | **689.5** | **59.3** | 61.5 GB | `███████████████████` | `██████████████████` |
| 8k | **737.7** | **39.2** | 61.6 GB | `████████████████████` | `███████████` |

### Generation throughput by concurrent stream batch

| Batch | TG tok/s | Relative to 1× | Visual |
| ---: | ---: | ---: | --- |
| 1× | **67.7** | 1.00× | `██████████` |
| 2× | **114.9** | 1.70× | `████████████████` |
| 4× | **138.3** | 2.04× | `████████████████████` |

The batch table comes from the same published performance session as the context table.[7][8][9]

## Runtime and compatibility

This release needs a runtime that supports the `qwen4_exp` architecture and its asymmetric target/MTP expert counts recorded in `config.json`.[1] A catalog listing alone is not proof of a successful load.[unverified] Memory use and speed depend on runtime version, context length, KV-cache policy, storage path, concurrency, MTP configuration, and hardware.[7][8][9]

## Lineage and license

The base model is [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), whose official card documents the base architecture and licensing.[2] This derivative is distributed under the **Qwen Community License 1.0**; the verbatim upstream license is included as `LICENSE` in this repository.[1][2] Review the license before commercial deployment.[2]

## Limitations

- All accuracy values above are 100-question stratified samples; evaluate the full workload before relying on them.[3][4][5][6]
- Do not treat HumanEval and LiveCodeBench as a combined quality score.[3][4][5][6]
- Vision or multimodal quality has not been independently characterized for this release.[unverified]
- The visual bars are presentation aids; the numeric columns are authoritative.[unverified]

## Sources

[1] https://huggingface.co/mensaprodigy/Qwen3.8-Flash-Next-REAP-384-oQ4e-BF16-MTP-PLE
[2] https://huggingface.co/Qwen/Qwen3.8-Flash-Next
[3] https://omlx.ai/benchmarks/intelligence/nnw0eqdx
[4] https://omlx.ai/benchmarks/intelligence/026c9etj
[5] https://omlx.ai/benchmarks/intelligence/h5hbdt8q
[6] https://omlx.ai/benchmarks/intelligence/moonsh71
[7] https://omlx.ai/benchmarks/performance/ybqngdig
[8] https://omlx.ai/benchmarks/performance/ab61ivk2
[9] https://omlx.ai/benchmarks/performance/5sqxj1ug
