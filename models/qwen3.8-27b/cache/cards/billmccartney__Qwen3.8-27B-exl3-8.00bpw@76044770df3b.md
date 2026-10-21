---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: Bill McCartney
tags:
  - exl3
  - exllamav3
  - qwen3_5
  - 8bpw
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B EXL3 8.0bpw

## What this is

This is an **8.0bpw EXL3 quantized derivative of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**. It exists because, at the
time of packaging, no public 8bpw EXL3 release was available: the
[reference Qwen3.8 EXL3 repository](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)
topped out at `6.00bpw`. That is the main reason to choose this release over the existing
public branches.

The language-model weights were transformed from the upstream BF16 checkpoint into EXL3.
The model's MTP component and BF16 vision tower were retained. This card replaces the
upstream card so that the modification, serving constraints, and local measurements are
explicit.

## Quantization details

These values come from the release's own `quantization_config.json`, not from a target
recipe or filename:

| Field | As built |
| --- | --- |
| Quantization method | `exl3` |
| `bits` | `8.0` |
| `head_bits` | `8` |
| `mtp_bits` | `4` |
| `codebook` | `mul1` |
| `out_scales` | `always` |
| Calibration | stock exllamav3 calibration, `rows: 250`, `cols: 2048` |
| exllamav3 used to build and test | `1.4.2` (`1.4.2+cu128.torch2.9.0`) |

The exact exllamav3 build above loaded the artifact successfully. Compatibility with older
exllamav3 releases is not claimed. Calibration used the converter's packaged stock corpus
with no custom calibration source.

## How to load

For TabbyAPI, place the downloaded branch below your model directory and use the following
per-model settings. The model path is deliberately a placeholder; the remaining values come
from the configuration used for the measurements in this card.

```yaml
model:
  model_dir: /path/to/models
  model_name: Qwen3.8-27B-exl3-8.0bpw
  backend: exllamav3
  tool_format: qwen3_coder
  reasoning: true
  reasoning_start_token: "<think>"
  reasoning_end_token: "</think>"
  max_seq_len: 212992
  cache_mode: "4,4"
  gpu_split_auto: true
  autosplit_reserve: [500, 96]
  chunk_size: 2048
  max_batch_size: 1

draft_model:
  draft_mode: mtp
```

`draft_model` is a top-level TabbyAPI block, not a child of `model`. Sampling is supplied by
the request or caller: retain `temperature: 1.0`, `top_p: 0.95`, and `top_k: 20`. Image input
also requires TabbyAPI's `vision` option; leaving it off keeps the retained tower out of VRAM.

For plain exllamav3, the included argument-driven generator is both a load check and a
reproducible sample harness:

```bash
python scripts/generate_exl3_release_samples.py \
  --model_dir /path/to/Qwen3.8-27B-exl3-8.0bpw \
  --output-dir ./catbench \
  --model-label Qwen3.8-27B-exl3-8.0bpw \
  --bitrate-label 8.00bpw \
  --cache_size 16384 --cache_quant 4,4 --mtp \
  --require-reasoning-split
```

The script applies the Hugging Face chat template with thinking enabled, uses the recommended
sampling triple, enables model-owned MTP, and rejects truncated or unclosed reasoning samples.
Its model and output paths are arguments, so it can be reused unchanged for another EXL3 quant.

## MTP is preserved

The model-owned Multi-Token Prediction component is present: the artifact index contains
39 `mtp.*` tensors, including eight EXL3 linears stored at 4bpw. TabbyAPI can use it for
self-speculative decoding without a separate draft model:

```yaml
draft_model:
  draft_mode: mtp
```

At 196,608-token configured context, using `temperature: 1.0`, `top_p: 0.95`, and
`top_k: 20`, an interleaved three-run A/B on the same multi-step agentic task measured:

| Measurement | MTP off | MTP on |
| --- | ---: | ---: |
| End-to-end wall-time range | 76.69–110.06 s | 56.94–60.12 s |
| Median end-to-end wall time | 109.87 s | 58.72 s |
| Independently recounted decode rate | 22.90–24.24 tok/s | 79.54–99.70 tok/s |
| Objective task result | 3/3 pass | 3/3 pass |
| Peak total VRAM | 32,346 MiB | 34,182 MiB |

The MTP-on median was **1.87× faster**, all six tasks passed, and 2,576 of 3,997 drafted
tokens were accepted (64.45%). MTP added 1,836 MiB at the measured peak, so its speedup is
not free.

## Vision tower

The complete vision tower is retained as **333 BF16, unquantized tensors totaling about
858 MiB**. It supports the base model's image/video architecture and accounts for disk space
even in a text-only deployment.

TabbyAPI does not load the tower unless `vision` is enabled. Text-only users therefore pay
the disk cost, not the vision tower's VRAM cost.

## Serving guidance — do not use greedy decoding

> [!WARNING]
> **Do not serve this model greedy.** Use `temperature: 1.0`, `top_p: 0.95`, and
> `top_k: 20` for reasoning and agentic traffic.

The warning is based on an identical long-reasoning prompt across the available backends:

| Build and sampling | Distinct 12-gram ratio | Worst repeated 12-gram | Outcome |
| --- | ---: | ---: | --- |
| Q8_0 GGUF, vendor sampling | `0.9734` | 5× | visible answer, `stop` |
| Q8_0 GGUF, greedy | `0.2167` | **399×** | no visible answer, `length` |
| This EXL3 8bpw build, greedy | `0.9204` | 16× | no visible answer, `length` |

The EXL3 build **did not reproduce the GGUF repetition collapse**: its 0.9204 ratio and
16× worst repeat stayed outside the predeclared collapse threshold. Greedy still consumed
the full 8,192-token generation budget without closing the reasoning trace or producing a
visible answer, so it remains an unusable serving regime for long reasoning.

Short tool-call probes passed under both greedy and vendor sampling. A green short tool
sweep therefore does **not** clear greedy serving; it does not exercise the long-turn failure.

## Measured performance and VRAM

All GPU measurements below used an **NVIDIA RTX 4070 12 GB plus RTX 3090 24 GB**, a
nominal **36,864 MiB combined**. Rates are specific to the named task, prompt depth,
sampling regime, MTP setting, and software build; they are not generic maximum-throughput
claims.

| Measurement | Result | Conditions |
| --- | ---: | --- |
| Complete artifact on disk | **29,366,005,409 bytes (27.349224 GiB)** | 19 files, including four weight shards, retained MTP, and retained vision |
| Controlled cold load | **30.83 s** | 196,608 context, MTP off, force-unloaded baseline, preload through the serving router |
| Initial peak VRAM | **31,910 MiB** | 196,608 context, MTP off; 11,418 MiB on the 4070 + 20,492 MiB on the 3090 |
| MTP A/B peak VRAM | **34,182 MiB** | 196,608 context, MTP on; maximum across three identical multi-step task runs |
| Final-configuration peak VRAM | **34,634 MiB (33.82 GiB)** | 212,992 context, MTP on; head-to-head maximum, 10,643 MiB on the 4070 + 23,991 MiB on the 3090 |
| Shallow agentic decode | **59.09 tok/s median** | 17,074-token control depth, 212,992 configured context, MTP on, vendor sampling, one paired control run |
| Deep agentic decode | **44.09–50.39 tok/s median per run** | 87,424–87,945-token depth, 212,992 configured context, MTP on, vendor sampling, three runs |

The final serving ceiling was separately validated at **212,992 tokens** with MTP enabled.
A 15-cell retrieval grid was exact through 206,862 raw prompt tokens. Although a 229,376-token
cache could start, real prefills failed above 212,992 on this hardware; the base model's
262,144-token native declaration is not a claim that this quant can serve that window within
the measured VRAM budget.

The only controlled cold figure is the 30.83-second, 196,608-context, MTP-off measurement.
Loads of the final 212,992-context/MTP-on configuration were observed with a warm page cache,
so they are deliberately not reported as cold-start results.

## Verification collateral

The BF16 checkpoint remained available, so quantization fidelity was measured directly with
the pinned exllamav3 1.4.2 qbench rather than estimated from bitrate:

| Weights | Mean KL div | PPL r-100 | Weight shards (GiB) |
| --- | ---: | ---: | ---: |
| `8` | `0.00142020` | `6.94884191` | `27.326998410746455` |
| `bf16` | `0.0` | `6.94746610` | `51.747082524001598` |

`PPL r-100` uses 100 non-overlapping WikiText2 test rows of 2,048 tokens with stride 2,048
and no chat wrapper: 204,700 next-token targets. KL compares 204,800 quantized logit positions
directly with the cached BF16 distributions. The optional BF16 self-noise-floor pass was not
run, so mean KL is unadjusted. Its median was `0.00036012` and p90 was `0.00115159`.
The `GiB` column follows the reference CSV convention and reports safetensors shard bytes,
not runtime VRAM. The publishable source table is
[`Qwen_Qwen3.8-27BMeasured.csv`](Qwen_Qwen3.8-27BMeasured.csv).

Three fixed-prompt samples are under [`catbench/`](catbench/). They use the vendor sampling
triple and model-owned MTP, close their reasoning traces, produce visible answers, and stop
naturally. The short explanation is 416 tokens (distinct-12-gram `0.992593`), the concurrency
review is 2,434 tokens (`0.924887`), and the long constructive proof is 3,319 tokens
(`0.983676`, worst repeated 12-gram 4×). The argument-driven generator and reusable qbench
helpers are included under [`scripts/`](scripts/).

## Limitations

- Correctness was measured with a small agentic suite that both this model and the comparison
  model saturated: **9/9 versus 9/9** on the three-task mini-benchmark and **3/3 versus 3/3**
  on the approximately 85k-token deep probe. The instrument was too easy to resolve fine
  capability differences; the tie does not establish equal capability.
- Under the established correctness-first decision rule, this model's head-to-head outcome
  against `qwen3.6-27b-8bpw-exl` was **`lose`**: the `py-hello` latency ranges were disjoint
  in the comparison model's favor, while the other two mini-task ranges overlapped, and this
  model took 1.353–1.470× as long on the three deep runs. It did, however, **displace** the
  separately measured thinking alias, with 9/9 correctness on both sides and non-overlapping
  speed wins on all three mini tasks.
- The KL/PPL result is a single WikiText2 distribution-fidelity measurement, not a reasoning,
  tool-use, vision, or instruction-following benchmark. qbench's optional BF16
  self-noise-floor pass was not run; the reported mean KL is direct and unadjusted.
- The retained vision tensors were inventoried, but the performance and correctness results
  above are text/agentic measurements, not a visual-quality evaluation.
- The containment check observed one successful stochastic agentic run with no out-of-sandbox
  file changes. That pass cannot prove that a rare future failure is impossible.
- The exact exllamav3 build named above was tested. Other exllamav3, TabbyAPI, driver, GPU,
  context, cache, batching, or sampling combinations may perform differently.

## License

This quantized derivative is distributed under **Apache License 2.0**, matching the base
model. The included `LICENSE` file carries the applicable terms and notices. Qwen names and
marks remain the property of their respective owners; this release does not imply upstream
endorsement.
