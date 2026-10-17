---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: exllamav3
pipeline_tag: text-generation
tags:
- exl3
- exllamav3
- qwen3_5
- quantized
---

# Qwen3.8-27B EXL3 4.0bpw HB8

This repository contains a local EXL3 quantization of Qwen/Qwen3.8-27B for ExLlamaV3.

## Quantization

- Base model: Qwen/Qwen3.8-27B
- Format: EXL3
- Decoder bitrate: 4.0 bpw
- Head and MTP bitrate: 8 bpw
- Codebook: mul1
- Output scales: always
- Calibration: 8 rows x 512 columns
- Shards: 3 safetensors files
- Target hardware: one NVIDIA RTX 3090
- Native context in config: 262144 tokens

The reduced calibration set was required because larger calibration allocations exceeded the available host memory during conversion. Benchmark this artifact before using it as a production replacement for higher-calibration quantizations.

## Verification

- All 64 language-model layers quantized successfully.
- Complete artifact loaded on one RTX 3090 with ExLlamaV3.
- A cache allocation and unload test passed.
- HumanEval: 153/164, pass@1 93.29 percent.

## Context speed benchmark

Measured with ExLlamaV3 on one RTX 3090, one request per fresh process, temperature 0, top-p 1, and 127 generated tokens per run. The reported rate is decode throughput, not prompt prefill throughput.

| Prompt context | Prefill seconds | Decode tok/s |
| ---: | ---: | ---: |
| 128 | 0.941 | 43.19 |
| 2,048 | 1.868 | 41.37 |
| 8,192 | 8.594 | 40.30 |
| 32,768 | 32.546 | 36.96 |
| 65,536 | 76.878 | 33.28 |

The 128-token run used ExLlamaV3's minimum 2048-token cache workspace. The 64K run is a measured prefill and generation test, not a claim that the model fits at the native 256K limit on a 24 GB card.

## Current single-RTX-3090 deployment

The current deployment uses TabbyAPI with ExLlamaV3 on one RTX 3090. The RTX 3090 is not used as a desktop/display GPU on the test system.

```yaml
max_seq_len: 196608
cache_size: 196608
cache_mode: Q8
max_batch_size: 1
chunk_size: 2048
draft_mode: mtp
```

The model loaded successfully with `n_ctx: 196608`. Observed NVIDIA memory after loading was 22,011 MiB used and 2,114 MiB free out of 24,576 MiB. GPU 0 in the server process is the RTX 3090 selected by `CUDA_VISIBLE_DEVICES`, not the RTX 5070 Ti.

### Live long-context speed

These measurements were made against the running TabbyAPI service after the Q8/196,608-token configuration was loaded. Temperature was 0, thinking was disabled, one request was active, and the completion was limited to 64 tokens. Prompt token counts are the actual TabbyAPI counts.

| Prompt context | Prefill tok/s | Decode tok/s | Generation |
| ---: | ---: | ---: | ---: |
| 130,012 | 624.52 | 30.13 | 20 tokens |
| 190,012 | 368.16 | 23.87 | 20 tokens |

The server is configured with `max_batch_size: 1`; parallel requests are queued rather than decoded concurrently. Long-context throughput varies with prompt content, cache reuse, draft acceptance, and system load.

### Latest live gateway logs

The following additional values are taken from current TabbyAPI generation logs during normal Hermes requests using MTP drafting. These are observational production measurements rather than isolated controlled benchmark runs.

| Context | Prefill tok/s | Decode tok/s | Draft acceptance |
|---:|---:|---:|---:|
| 70,578 | 323.94 | 46.03 | 837 / 1,743 |
| 72,184 | 438.62 | 47.23 | 1,157 / 2,300 |
| 74,213 | 351.18 | 64.95 | 932 / 1,147 |
| 76,599 | 509.54 | 32.13 | 989 / 3,722 |
| 80,739 | 502.59 | 48.45 | 803 / 1,445 |
| **81,585** | **272.33** | **43.24** | **882 / 1,830** |

The latest complete entry was 81,585 context tokens at 43.24 decode tok/s. Recent observed decode speed ranged from 32.13 to 64.95 tok/s, primarily due to variable MTP draft acceptance and request shape.

## Files and provenance

- Original source: https://huggingface.co/Qwen/Qwen3.8-27B
- Quantization source: https://github.com/turboderp-org/exllamav3
- HumanEval raw samples and scoring were generated locally with the official evaluator.
- The local benchmark report is included in the project workspace used to create this model card.

## License

Apache-2.0, subject to the base model license and terms.
