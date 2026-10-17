---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
language:
  - en
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: mlx
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - mlx
  - mlx-vlm
  - omlx
  - oq
  - mtp
  - speculative-decoding
  - qwen
  - qwen3.8
  - qwen4-exp
  - mixture-of-experts
  - vision-language
  - quantized
  - apple-silicon
  - 3.5-bit
  - oQ
  - oQe
datasets:
  - cais/mmlu
  - HuggingFaceH4/HellaSwag-samples
  - HuggingFaceH4/truthfulQA-multi-choice-valid
  - openai/openai_humaneval
  - google-research-datasets/mbpp
  - livecodebench/livecodebench
  - AI-safety-promise/SafetyBench
model-index:
  - name: Qwen3.8-Flash-Next-oQ3.5e-fp16-mtp
    results:
      - task:
          type: image-text-to-text
          name: Text Generation (Qwen3.8-Flash-Next)
        dataset:
          type: mmlu
          name: MMLU (sample 1000/14042, no-thinking)
          config: all
          split: test
        metrics:
          - type: accuracy
            value: 85.4
            name: MMLU Accuracy (sample 1000/14042, no-thinking)
        source:
          name: Self-reported benchmark (oMLX 0.7.0rc1, M1 Ultra 128GB)
          url: https://github.com/jundot/omlx
      - task:
          type: image-text-to-text
        dataset:
          type: hellaswag
          name: HellaSwag (sample 300/10042, no-thinking)
          split: validation
        metrics:
          - type: accuracy
            value: 94.3
            name: HellaSwag Accuracy (sample 300/10042, no-thinking)
        source:
          name: Self-reported benchmark (oMLX 0.7.0rc1, M1 Ultra 128GB)
          url: https://github.com/jundot/omlx
      - task:
          type: image-text-to-text
        dataset:
          type: truthfulqa
          name: TruthfulQA (sample 300/817, no-thinking)
          config: mc
          split: validation
        metrics:
          - type: accuracy
            value: 95.3
            name: TruthfulQA Accuracy (sample 300/817, no-thinking)
        source:
          name: Self-reported benchmark (oMLX 0.7.0rc1, M1 Ultra 128GB)
          url: https://github.com/jundot/omlx
      - task:
          type: image-text-to-text
        dataset:
          type: humaneval
          name: HumanEval (full 164, no-thinking)
          split: test
        metrics:
          - type: pass_at_1
            value: 91.5
            name: HumanEval pass@1 (full 164, no-thinking)
        source:
          name: Self-reported benchmark (oMLX 0.7.0rc1, M1 Ultra 128GB)
          url: https://github.com/jundot/omlx
      - task:
          type: image-text-to-text
        dataset:
          type: mbpp
          name: MBPP (sample 200/500, no-thinking)
          split: test
        metrics:
          - type: pass_at_1
            value: 87.5
            name: MBPP pass@1 (sample 200/500, no-thinking)
        source:
          name: Self-reported benchmark (oMLX 0.7.0rc1, M1 Ultra 128GB)
          url: https://github.com/jundot/omlx
      - task:
          type: image-text-to-text
        dataset:
          type: livecodebench
          name: LiveCodeBench (sample 100/1055, no-thinking)
        metrics:
          - type: pass_at_1
            value: 48.0
            name: LiveCodeBench pass@1 (sample 100/1055, no-thinking)
        source:
          name: Self-reported benchmark (oMLX 0.7.0rc1, M1 Ultra 128GB)
          url: https://github.com/jundot/omlx
      - task:
          type: image-text-to-text
        dataset:
          type: safetybench
          name: SafetyBench (sample 300/11435, no-thinking)
          split: test
        metrics:
          - type: accuracy
            value: 86.7
            name: SafetyBench Accuracy (sample 300/11435, no-thinking)
        source:
          name: Self-reported benchmark (oMLX 0.7.0rc1, M1 Ultra 128GB)
          url: https://github.com/jundot/omlx
---

> [!WARNING]
<p align="center">
  <img src="https://img.shields.io/badge/Qwen-Qwen3.8-615CED?style=for-the-badge&logo=qwen&logoColor=white" alt="Qwen Qwen3.8">
  <img src="https://img.shields.io/badge/Apple_Silicon-MLX-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Apple silicon MLX">
  <img src="https://img.shields.io/badge/Native_MTP-Included-22C55E?style=for-the-badge" alt="Native MTP included">
</p>

# Qwen3.8-Flash-Next-oQ3.5e-fp16-mtp

A self-quantized MLX checkpoint of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) with native MTP (Lightning MTP) support, and intact vision tower, in FP16. Built as a **heavy-hitter local coding model** that leaves plenty of RAM free for parallel use on my M1 Ultra 128GB Mac Studio (my main dev machine).

Custom imatrix distribution by Robot Haus, augmented from oMLX stock corpus to ensure enhanced activation spread.

## Fastest Configuration Ranking

Across all four speed benchmarks, ranked by sustained target TPS (tokens/second) across prompt sizes:

| Speed Rank | Configuration | Settings | Avg Target TPS (pp1024 to pp200k) | Peak Memory | Continuous Batch Speedup (8x) |
|:--:|---|---|---:|---:|---:|
| Rank 1 | Run 2 | N-gram Offload: OFF + MTP ON + SpecPrefill ON | 28.0 (32.5 -> 20.2) | ~98 GB | 
| Rank 2 | Run 3 | N-gram Offload: ON + MTP ON + SpecPrefill OFF | 21.4 (22.2 -> 18.8) | 112.5 GB | 
| Rank 3 | Run 1 | N-gram Offload: ON + MTP ON + SpecPrefill ON | 20.5 (19.9 -> 18.3) | 95.6 GB | 
| Rank 4 | Run 4 | N-gram Offload: ON + MTP OFF + SpecPrefill OFF | 11.5 (11.6 -> 11.3) | 94.1 GB | 

**Run 2** delivers the fastest sustained throughput -- but it loads the N-gram table into RAM, Leaving NGRAM offloaded to disk ran fastest from my tb4 NVMe enclosure (gen 5) rather than from internal storage. For most users without that hardware (or those who want to conserve RAM), **Run 1** is the recommended configuration: SSD offload ON with MTP + SpecPrefill ON. **Run 4** proves that turning MTP off is the single worst decision -- throughput halves.

> [!TIP]
> **Bottom line:** Benchmark Run 1 = recommended for most users (SSD ON, MTP ON, SpecPrefill ON).
> Run 2 = fastest absolute speed (SSD OFF, MTP ON, SpecPrefill ON).
> Avoid Run 4 at all costs -- turning MTP off kills the model.

## Intelligence Benchmark Results

| Benchmark | Sample Size | Accuracy | Correct | Total | Time(s) | Think? |
|---|---|---:|---:|---:|---:|:--:|
| MMLU | 1000 / 14042 | 85.4% | 854 | 1000 | 2821.1 | No |
| HellaSwag | 300 / 10042 | 94.3% | 283 | 300 | 461.2 | No |
| TruthfulQA | 300 / 817 | 95.3% | 286 | 300 | 369.7 | No |
| HumanEval (pass@1) | 164 (full) | 91.5% | 150 | 164 | 754.7 | No |
| MBPP (pass@1) | 200 / 500 | 87.5% | 175 | 200 | 846.5 | No |
| LiveCodeBench (pass@1) | 100 / 1055 | 48.0% | 48 | 100 | 3798.0 | No |
| SafetyBench | 300 / 11435 | 86.7% | 260 | 300 | 365.7 | No |

> Self-reported, partial-sample, no-thinking runs on a single M1 Ultra 128GB host via oMLX 0.7.0rc1. Not leaderboard-verified. The `model-index` block in the YAML front matter encodes these for the Hub widget.

## Speed Benchmark Results

All speed benchmarks were run on an M1 Ultra Mac Studio with 128GB RAM using oMLX 0.7.0rc1.

### What the SSD N-gram offload setting does

The N-gram table stores token frequency statistics that guide prefill; a massive lookup table that would normally live in RAM. With SSD offload enabled, oMLX writes this table to disk and reads only the rows it needs at inference time. This dramatically reduces peak memory footprint, but adds I/O latency during context shifts. Loading the full table into RAM instead gives you pure speed at the cost of more VRAM -- something only available when paired with an external NVMe enclosure (Patriot Viper Gen5 on Thunderbolt 4). For most users who do not have that kind of external hardware, offloading to SSD is the only practical way to run this model at all.

> Note: in my testing, SSD N-gram offload ran faster from the Thunderbolt 4 NVMe enclosure ( in my case, a Patriot Viper Gen5). The internal drive of the M1 Ultra Mac Studio is not included in that measurement, but anecdotal throughput held consistently lower for me. (Run 4).

### Run 1 -- SSD N-gram Offload: ON | MTP: ON | SpecPrefill: ON (**recommended**)

**Configuration:**
- Adaptive max depth: 3 tokens (default) -- automatically adjusts draft depth up to the selected maximum
- SpecPrefill: ON -- attention-based sparse prefill for MoE/hybrid models (see [paper](https://arxiv.org/abs/2502.02789))
- Draft model: [sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16)
- Keep rate: 20% -- balanced (~3x, recommended)
- Threshold (tokens): 8192

| Test | TTFT(ms) | TPOT(ms) | Prefill TPS | Target TPS | E2E(s) | Throughput | Peak Mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| pp1024/tg128 | 3198.3 | 50.75 | 320.2 | 19.9 | 9.656 | 119.3 | 64.49 GB |
| pp4096/tg128 | 11197.2 | 62.76 | 365.8 | 17.4 | 11.978 | 343.0 | 71.68 GB |
| pp8192/tg128 | 20328.4 | 48.94 | 403.0 | 20.6 | 26.554 | 313.3 | 74.71 GB |
| pp16384/tg128 | 41862.0 | 48.94 | 391.4 | 20.6 | 41.960 | 390.5 | 75.38 GB |
| pp32768/tg128 | 84043.7 | 49.45 | 389.9 | 20.5 | 88.472 | 371.4 | 80.87 GB |
| pp65536/tg128 | 169275.7 | 46.75 | 387.2 | 21.6 | 175.242 | 374.7 | 82.81 GB |
| pp131072/tg128 | 345070.3 | 44.45 | 379.8 | 22.7 | 350.738 | 374.1 | 87.63 GB |
| pp200000/tg128 | 565249.8 | 55.20 | 353.8 | 18.3 | 572.289 | 349.7 | 95.59 GB |

**Continuous Batching -- pp1024 / tg128:**

| Batch | Target TPS | Speedup | Prefill TPS | Per-request TPS | TTFT(ms) | E2E(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1x | 19.9 | 1.00x | 320.2 | 320.2 | 3198.3 | 9.656 |
| 2x | 18.5 | 0.93x | 231.0 | 115.5 | 6635.4 | 15.996 |
| 4x | 34.0 | 1.71x | 207.4 | 51.9 | 13537.2 | 30.600 |
| 8x | 52.7 | 2.65x | 198.0 | 24.8 | 28901.3 | 58.462 |

### Run 2 -- SSD N-gram Offload: OFF | MTP: ON | SpecPrefill: ON

| Test | TTFT(ms) | TPOT(ms) | Prefill TPS | Target TPS | E2E(s) | Throughput | Peak Mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| pp1024/tg128 | 2047.8 | 31.02 | 500.0 | 32.5 | 5.999 | 192.0 | 86.85 GB |
| pp4096/tg128 | 6745.1 | 33.86 | 607.3 | 29.8 | 11.053 | 382.2 | 89.40 GB |
| pp8192/tg128 | 13836.4 | 33.86 | 592.1 | 29.8 | 13.836 | 592.1 | 89.49 GB |
| pp16384/tg128 | 27817.5 | 33.86 | 589.0 | 29.8 | 27.915 | 587.0 | 89.78 GB |
| pp32768/tg128 | 56407.1 | 38.52 | 580.9 | 28.3 | 56.933 | 575.8 | 90.35 GB |
| pp65536/tg128 | 114927.0 | 38.17 | 570.2 | 26.6 | 117.399 | 558.8 | 91.49 GB |
| pp131072/tg128 | 232222.1 | 41.16 | 564.4 | 24.5 | 237.519 | 552.4 | 93.84 GB |
| pp200000/tg128 | 360635.6 | 49.86 | 554.6 | 20.2 | 367.054 | 545.2 | 98.09 GB |

**Continuous Batching -- pp1024 / tg128:**

| Batch | Target TPS | Speedup | Prefill TPS | Per-request TPS | TTFT(ms) | E2E(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1x | 32.5 | 1.00x | 500.0 | 500.0 | 2047.8 | 5.999 |
| 2x | 47.0 | 1.45x | 326.6 | 163.3 | 4549.2 | 11.721 |
| 4x | 67.4 | 2.07x | 268.6 | 67.2 | 10434.4 | 22.120 |
| 8x | 99.7 | 3.07x | 240.9 | 30.1 | 23824.6 | 42.871 |

### Run 3 -- SSD N-gram Offload: ON | MTP: ON | SpecPrefill: OFF

| Test | TTFT(ms) | TPOT(ms) | Prefill TPS | Target TPS | E2E(s) | Throughput | Peak Mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| pp1024/tg128 | 2956.4 | 45.47 | 346.4 | 22.2 | 7.639 | 147.3 | 64.49 GB |
| pp4096/tg128 | 10943.5 | 49.92 | 374.3 | 20.3 | 14.430 | 288.6 | 71.68 GB |
| pp8192/tg128 | 20261.5 | 51.93 | 404.3 | 19.6 | 22.922 | 359.7 | 74.71 GB |
| pp16384/tg128 | 42284.9 | 51.93 | 387.5 | 19.6 | 42.385 | 386.6 | 75.38 GB |
| pp32768/tg128 | 84301.0 | 72.86 | 388.7 | 14.6 | 85.539 | 383.3 | 80.87 GB |
| pp65536/tg128 | 170328.1 | 45.13 | 384.8 | 22.3 | 176.077 | 372.9 | 82.81 GB |
| pp131072/tg128 | 352030.9 | 43.83 | 372.3 | 23.0 | 357.614 | 366.9 | 87.63 GB |
| pp200000/tg128 | 801987.1 | 53.69 | 249.4 | 18.8 | 808.839 | 247.4 | 112.54 GB |

**Continuous Batching -- pp1024 / tg128:**

| Batch | Target TPS | Speedup | Prefill TPS | Per-request TPS | TTFT(ms) | E2E(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1x | 22.2 | 1.00x | 346.4 | 346.4 | 2956.4 | 7.639 |
| 2x | 30.3 | 1.36x | 232.8 | 116.4 | 6528.8 | 16.817 |
| 4x | 41.4 | 1.86x | 206.1 | 51.5 | 13650.4 | 31.107 |
| 8x | 53.6 | 2.41x | 189.5 | 23.7 | 30584.2 | 59.287 |

### Run 4 -- SSD N-gram Offload: ON | MTP: OFF | SpecPrefill: OFF (slowest; avoid)

| Test | TTFT(ms) | TPOT(ms) | Prefill TPS | Target TPS | E2E(s) | Throughput | Peak Mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| pp1024/tg128 | 3194.1 | 86.71 | 320.6 | 11.6 | 14.226 | 81.0 | 62.93 GB |
| pp4096/tg128 | 10908.1 | 86.34 | 375.5 | 11.7 | 21.883 | 193.0 | 69.68 GB |
| pp8192/tg128 | 19460.8 | 89.28 | 420.9 | 11.3 | 27.596 | 300.1 | 72.49 GB |
| pp16384/tg128 | 39895.2 | 127.44 | 410.7 | 11.8 | 40.256 | 407.1 | 73.03 GB |
| pp32768/tg128 | 79527.0 | 93.95 | 412.0 | 11.8 | 80.478 | 407.3 | 78.35 GB |
| pp65536/tg128 | 159158.5 | 87.06 | 411.8 | 11.7 | 164.407 | 399.0 | 80.13 GB |
| pp131072/tg128 | 325480.8 | 87.02 | 402.7 | 11.6 | 336.554 | 389.8 | 85.21 GB |
| pp200000/tg128 | 528132.2 | 89.29 | 378.7 | 11.3 | 539.497 | 371.0 | 94.14 GB |

**Continuous Batching -- pp1024 / tg128:**

| Batch | Target TPS | Speedup | Prefill TPS | Per-request TPS | TTFT(ms) | E2E(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1x | 11.6 | 1.00x | 320.6 | 320.6 | 3194.1 | 14.226 |
| 2x | 20.9 | 1.80x | 265.3 | 132.7 | 5702.8 | 14.786 |
| 4x | 30.1 | 2.59x | 230.2 | 57.5 | 12168.4 | 29.877 |
| 8x | 52.7 | 4.54x | 210.7 | 26.3 | 27347.9 | 54.338 |

<details>
<summary><b>The weeds</b> (technical details, template sections) - click to expand</summary>

# Model Card for Qwen3.8-Flash-Next-oQ3.5e-fp16-mtp

A sensitivity-guided, mixed-precision MLX quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (rebuilt directly from the official BF16 checkpoint), with the model's matching native MTP (Lightning MTP) head and vision tower **preserved** and verified. FP16 precision, built as a heavy-hitter local coding engine that leaves RAM available for parallel work on my M1 Ultra 128GB Mac Studio.

## Model Details

### Model Description

This is a custom, self-quantized MLX checkpoint of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), performed entirely on a single M1 Ultra Mac Studio using [oMLX](https://github.com/jundot/omlx) 0.7.0rc1 (release candidate). It is not a fine-tune: weights are never trained here, only requantized.

Why this exists: HuggingFace has been flooded with Qwen3.x uploads that claim MTP retention and an intact vision tower. In my experience that is rarely the case, because conversion pipelines silently strip those components, and there is no standard way to verify from the outside without opening the checkpoint. I quantized directly from the official original weights myself so it could be certain, then verified MTP and vision tensors after quantization by inspecting the safetensors index directly.

- **Developed by:** Robot Haus (personal project; self-quantized, single operator)
- **Funded by:** Self-funded
- **Shared by:** Robot Haus
- **Model type:** Vision-language model (VLM), mixture-of-experts, quantized MLX checkpoint with native MTP head
- **Language(s) (NLP):** en (and multilingual input, per base model)
- **License:** [Qwen Community License 1.0](LICENSE) (`license: other` on the Hub)
- **Finetuned from model:** None. Quantized from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (BF16 original).

### Model Sources

- **Base model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Qwen overview:** [qwen.ai blog](https://qwen.ai/blog?id=qwen3.8-flash-next)
- **Quantization / inference engine:** [oMLX](https://github.com/jundot/omlx)
- **MLX-VLM:** [ml-explore/mlx-vlm](https://github.com/ml-explore/mlx-vlm)
- **MTP drafter model:** [sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16)
- **SpecPrefill paper:** [arXiv:2502.02789](https://arxiv.org/abs/2502.02789)

## Uses

### Direct Use

Run it as a local **heavy-coding engine** on my M1 Ultra 128GB Mac Studio (main dev machine) via [oMLX](https://github.com/jundot/omlx): multi-step agentic coding, tool use, shell/systems work, and demanding code generation with the rest of the machine left available for parallel dev use. It is intentionally kept off the critical path (offload to SSD on the base config) so coding throughput does not crowd out RAM.

### Downstream Use [optional]

Suitable as the base for further local tooling: agentic coding workflows, batch inference via oMLX continuous batching, and Apple Intelligence style integrations on-device. The preserved MTP head makes it usable with any speculative-decoding setup that supports Qwen3.8-Flash-Next MTP.

### Out-of-Scope Use

Not a general chatbot for production or high-availability use, and not verified against every benchmark -- scores above are partial samples (see *Evaluation*). Do not assume HF-wide leaderboard alignment. Hardware floor is real: the base checkpoint is very large; smaller Apple Silicon machines may be unable to load it at all regardless of quantization.

## Bias, Risks, and Limitations

Inherited from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): standard LLM risks -- factual errors, hallucination, training-data bias, and prompt injection when used with tools. Quantization (3-bit, custom calibration) can shift behavior on distribution-tail inputs; it is not equivalent to the BF16 original.

### Recommendations

Review outputs in any consequential or high-stakes context. The benchmark numbers are partial, self-reported, and no-thinking-mode; do not treat them as leaderboard-grade. See *Evaluation > Summary* for the caveats and how to reproduce a full run yourself.

## How to Get Started with the Model

Requires Apple Silicon and [oMLX](https://github.com/jundot/omlx) 0.7.0rc1 (or compatible; target hardware M1 Ultra 128GB).

```bash
# Pull the repo (from Hugging Face) and load it in oMLX, or:
pip install "omlx>=0.7.0rc1"   # see https://github.com/jundot/omlx for supported install paths
```

Then point oMLX at this model repo. Enable Lightning MTP and (optionally) SpecPrefill for speed:

```bash
# oMLX / launch flags (typical "Run 1" recommended config)
--mtp --spec-prefill --ngram-ssd-offload --keep-rate 0.2 --threshold 8192
# drafter: sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16
# ngram-ssd-offload requires an external NVMe (Thunderbolt 4) to stay fast
```

> [!WARNING]
> Do **not** run with MTP disabled (the "Run 4" configuration): decode throughput halves to ~11 tok/s. See *Evaluation > Speed Benchmarks*.

## Training Details

### Training Data

**No training.** This model is quantized, not fine-tuned or trained. The only "training data" involved is the calibration corpus used by oMLX's imatrix step to decide which weight channels get precision-preserving treatment. That corpus (3,629 samples, custom-built by Robot Haus) is described in *Calibration Corpus* below; it extends oMLX's stock Hermes/chat/reasoning mix rather than replacing it.

### Training Procedure

#### Preprocessing [optional]

Calibration samples are plain text/code prompts (chat, code, reasoning, tool-calling, shell, prose; see calibration table). No document packing or tokenization changes beyond oMLX defaults.

#### Training Hyperparameters

- **Training regime:** N/A -- quantization only (no gradient training). Quantization config: `qwen4_exp`, 3-bit, group size 64, FP16 weights retained where required.

#### Speeds, Sizes, Times [optional]

Quantization runs on one M1 Ultra 128GB Mac Studio; wall-clock time varies with checkpoint size and calibration set. Full speed/throughput measurements are in the *Speed Benchmark Results* section above.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Benchmarks are **partial samples** (not full sets): MMLU 1000/14042, HellaSwag 300/10042, TruthfulQA 300/817, HumanEval full 164, MBPP 200/500, LiveCodeBench 100/1055, SafetyBench 300/11435. All run in **no-thinking** mode unless noted otherwise.

#### Factors

Disaggregated by: precision/config (MTP on/off, SpecPrefill on/off, SSD n-gram offload on/off), prompt size (pp1024..pp200000), and batch size (1x..8x).

#### Metrics

- `accuracy`: percent correct on multiple-choice benchmarks (MMLU, HellaSwag, TruthfulQA, SafetyBench).
- `pass_at_1`: pass@1 on code-generation benchmarks (HumanEval, MBPP, LiveCodeBench).
- Speed: TTFT, TPOT, prefill/tgt TPS, throughput, peak memory.

### Results

The intelligence and speed tables are at the top of this card (not hidden in the weeds). See *Intelligence Benchmark Results* and *Speed Benchmark Results*. The full `model-index` YAML block at the top of this file encodes those scores for the Hub widget; they are self-reported, partial-sample, no-thinking runs and **not** leaderboard-verified.

## Model Examination [optional]

MTP heads and vision tensors were verified *after* quantization by direct inspection of the safetensors index (the components are present in the saved checkpoint, not just claimed). The numbers above are measured outcomes of running this exact checkpoint on this exact machine, not theoretical or derived from paper specs; this card is a self-documenting artifact of that work.

## Environmental Impact

- **Hardware Type:** Apple M1 Ultra Mac Studio, 128 GB unified memory
- **Hours used:** Not precisely metered (personal, single-machine workflow)
- **Cloud Provider:** None (local only)
- **Compute Region:** Local
- **Carbon Emitted:** Not measured

Emissions can be estimated with the [ML Impact calculator](https://mlco2.github.io/impact#compute) if precise usage hours are logged.

## Technical Specifications [optional]

### Model Architecture and Objective

Vision-language mixture-of-experts (Qwen3.8-Flash-Next) with a native Lightning MTP head, preserved through quantization. Format: MLX safetensors.

### Compute Infrastructure

#### Hardware

Target hardware: Apple M1 Ultra, 128 GB unified memory. Large-context runs (pp > ~32k) push peak memory toward 90-112 GB and benefit from (or require, without enough RAM) SSD N-gram offload backed by external NVMe over Thunderbolt 4.

#### Software

[oMLX](https://github.com/jundot/omlx) 0.7.0rc1 (release candidate) for quantization and inference; MLX / mlx-vlm underneath. SpecPrefill per [arXiv:2502.02789](https://arxiv.org/abs/2502.02789).

### Quantization details

- **Model type:** qwen4_exp
- **Bits:** 3 (mixed precision; FP16 retained where required)
- **Group size:** 64
- **Format:** MLX safetensors
- **Precision:** FP16, chosen for best performance on Apple Silicon hardware
- **Calibration corpus:** custom imatrix distribution by Robot Haus, augmented from oMLX stock corpus to ensure enhanced activation spread

## Calibration Corpus

Custom imatrix distribution (3,629 samples) rather than oMLX's stock calibration. It extends the stock Hermes/chat/reasoning mix instead of replacing it, targeting the registers this model is actually used for: tool orchestration, shell/systems reasoning, professional prose, and Swift/AppleScript -- while keeping the general diversity that research on Qwen3 shows is already a strong calibration baseline.

| Category | Samples | Source / License |
|---|---:|---|
| Retained originals (chat, code, reasoning, tool-calling) | 2,166 | oMLX stock |
| Professional writing (essays, resumes, cover letters) | 311 | InferencePrince555 + own docs |
| Agentic tool-use traces | 300 | lambda/hermes-agent-reasoning-traces, Apache 2.0 |
| Shell / sysadmin | 200 | ajibawa-2023/Shell-Code-Large, MIT |
| HTML + React | 150 | Own repos + kenhktsui/github-code-permissive-sample |
| World knowledge (Wikipedia) | 150 | wikimedia/wikipedia, CC-BY-SA |
| Swift + AppleScript | 178 | Own repos + HelloImSteven dataset, MIT |
| Organizational / multi-step | 100 | Own docs + hand-authored |
| Structured instruction-writing skills | 23 | Own skill files (curated) |
| Creative | 21 | lemon07r/VellumK2T-Fiction-SFT-01, MIT |
| **Total** | **3,629** | Apache 2.0 / MIT / CC-BY-SA or equivalent |

## Glossary [optional]

- **MTP (Multi-Token Prediction / Lightning MTP):** the model's built-in head that drafts several tokens per decode step for speculative decoding; up to ~1.5x faster decoding here.
- **SpecPrefill:** attention-based sparse prefill for MoE/hybrid models ([paper](https://arxiv.org/abs/2502.02789)).
- **TTFT / TPOT:** time-to-first-token / time-per-output-token.
- **pp / tg:** prompt-processing size / token-generation size in benchmark labels.
- **imatrix / calibration:** sample set guiding which weight channels receive precision-preserving treatment during quantization.
- **Keep rate:** fraction of rows kept (offloaded) in the SSD N-gram configuration; 20% here (~3x, recommended).

## More Information [optional]

- **Draft model used in all MTP runs:** [sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16) (bf16).
- **Engine:** oMLX "Auto", context = Code (Python) across all benchmark runs.
- **Future plans:** MTPLX and BF16 variants of this model, plus an FP16 variant of the draft model, as time allows. oMLX is preferred day-to-day for batch processing even though [MTPLX](https://github.com/youssofal/MTPLX) is faster.

## Model Card Authors [optional]

Robot Haus (written by the model from the author's benchmark notes and run logs).

## Model Card Contact

Via [oMLX](https://github.com/jundot/omlx) / repo owner.

</details>
