---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: llama.cpp
language:
  - en
  - zh
tags:
  - gguf
  - uncensored
  - qwen3.8
  - mtp
  - speculative-decoding
  - imatrix
  - quantized
---

# Qwen3.8-27B-Uncensored-GGUF

Uncensored [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), published as GGUF
quantizations with the multi-token prediction (MTP) head retained and verified.

Refusal behaviour has been **substantially reduced**, not eliminated — see Measured
behaviour for the numbers. Capabilities, training data, and architecture are otherwise
unchanged.

> **MTP tensors verified, not assumed.** Abliteration drops the `mtp.*` tensors: the model
> is re-saved through transformers, which does not carry the MTP head, while `config.json`
> still advertises it. They are grafted back from the base checkpoint and every file is
> inspected after quantization — see Method and Verification.

## Method

- Refusal directions removed with [Heretic](https://github.com/p-e-w/heretic), which
  co-minimizes refusal count against KL divergence from the base model. No hand-written
  refusal-removal code, no fine-tuning, no additional training data.
- Abliteration runs at bf16 (no 4-bit quantization); the resulting LoRA is merged into the
  bf16 base, so the published weights are not a quantized round trip.
- `mtp.*` tensors are copied verbatim from the base checkpoint after merging. Abliteration
  never touches them — it modifies `attn.o_proj` and `mlp.down_proj` in the main stack.
- The draft head was trained against the unmodified model, so acceptance rate may fall
  slightly. Speculative decoding verifies every token against the target, so output quality
  is unaffected.
- imatrix is computed directly from the f16, not from an intermediate quantization, so
  calibration sees the real weights.

## What's here

| Family | Files | Use when |
|---|---|---|
| Fused | `Qwen3.8-27B-Uncensored-<QUANT>.gguf` | One file. MTP rides inline as a built-in draft. |
| Target + draft | `Qwen3.8-27B-Uncensored-noMTP-<QUANT>.gguf` + `mtp-*-Q8_0.gguf` | Your runtime wants an explicit `--model-draft`. |
| Vision | `mmproj-*-f16.gguf` | Image input, if the base model ships a vision tower. |

The draft head stays at Q8_0 in every configuration. It is small relative to the target, and
quantizing it harder costs draft acceptance rate for almost no disk saving.

## Overview

| | |
|---|---|
| Base | Qwen/Qwen3.8-27B |
| Architecture | `Qwen3_5ForConditionalGeneration` |
| Layers | 64 |
| Vocab | 248320 |
| MTP layers | 1 |
| Vision | yes |
| Context | 262144 |
| Quants | IQ4_XS, Q4_K_M, Q5_K_M, Q6_K, Q8_0 |
| imatrix | wikitext-2 raw, 200 chunks |
| Converted with | llama.cpp `a94d563ed` |

## Files

| File | Size | MTP | PPL (wikitext-2) |
|---|---|---|---|
| `Qwen3.8-27B-Uncensored-IQ4_XS.gguf` | 15.3 GB | yes | - |
| `Qwen3.8-27B-Uncensored-Q4_K_M.gguf` | 16.8 GB | yes | PPL = 7.1814 +/- 0.25227 |
| `Qwen3.8-27B-Uncensored-Q5_K_M.gguf` | 19.5 GB | yes | - |
| `Qwen3.8-27B-Uncensored-Q6_K.gguf` | 22.4 GB | yes | - |
| `Qwen3.8-27B-Uncensored-Q8_0.gguf` | 29.0 GB | yes | - |
| `Qwen3.8-27B-Uncensored-draft-Q8_0.gguf` | 3.2 GB | - | - |
| `Qwen3.8-27B-Uncensored-noMTP-IQ4_XS.gguf` | 15.1 GB | no | - |
| `Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf` | 16.5 GB | no | - |
| `Qwen3.8-27B-Uncensored-noMTP-Q5_K_M.gguf` | 19.2 GB | no | - |
| `Qwen3.8-27B-Uncensored-noMTP-Q6_K.gguf` | 22.1 GB | no | - |
| `Qwen3.8-27B-Uncensored-noMTP-Q8_0.gguf` | 28.6 GB | no | - |

## Usage

```bash
llama-server -m Qwen3.8-27B-Uncensored-Q4_K_M.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 -c 8192
```

Target plus explicit draft:

```bash
llama-server -m Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf \
  --spec-type draft-mtp \
  --model-draft mtp-Qwen3.8-27B-Uncensored-draft-Q8_0.gguf \
  -ngl 99 -c 8192
```

`--spec-draft-n-max` defaults to 3. Throughput depends on your hardware, so sweep it —
measurements across draft lengths are in
[qwen3.8-spec-decode-bench](https://huggingface.co/datasets/JonathanColetti/qwen3.8-spec-decode-bench).

## Verification

Each artifact was checked post-quantization for MTP tensor survival rather than inferred from
the conversion flag:

```bash
python quantize.py inspect Qwen3.8-27B-Uncensored-Q4_K_M.gguf
```

This reports metadata keys, declared `block_count`, and blocks actually present. A fused file
whose present-block count does not exceed its declared count did not retain the MTP block.

| File | MTP | blocks |
|---|---|---|
| `Qwen3.8-27B-Uncensored-f16.gguf` | True | 65/65 |
| `Qwen3.8-27B-Uncensored-noMTP-f16.gguf` | False | 64/64 |
| `Qwen3.8-27B-Uncensored-IQ4_XS.gguf` | True | 65/65 |
| `Qwen3.8-27B-Uncensored-noMTP-IQ4_XS.gguf` | False | 64/64 |
| `Qwen3.8-27B-Uncensored-Q4_K_M.gguf` | True | 65/65 |
| `Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf` | False | 64/64 |
| `Qwen3.8-27B-Uncensored-Q5_K_M.gguf` | True | 65/65 |
| `Qwen3.8-27B-Uncensored-noMTP-Q5_K_M.gguf` | False | 64/64 |
| `Qwen3.8-27B-Uncensored-Q6_K.gguf` | True | 65/65 |
| `Qwen3.8-27B-Uncensored-noMTP-Q6_K.gguf` | False | 64/64 |
| `Qwen3.8-27B-Uncensored-Q8_0.gguf` | True | 65/65 |
| `Qwen3.8-27B-Uncensored-noMTP-Q8_0.gguf` | False | 64/64 |

## Measured behaviour

Benchmarked against the unmodified base model on identical settings. The delta is the
figure that matters: it isolates what the weight edit cost.

| Task | Base | Uncensored | Δ |
|---|---|---|---|
| MMLU | 83.4 | 83.3 | -0.2 |
| ARC-Challenge | 58.9 | 57.7 | -1.2 |
| HellaSwag | 82.8 | 82.9 | +0.1 |
| Winogrande | 76.1 | 75.3 | -0.8 |
| **Mean** | | | **-0.5** |

0-shot via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
bf16, both models scored in the same session. Every delta is within or close to the
reported standard error (MMLU ±0.30, ARC ±1.44, HellaSwag ±0.38, Winogrande ±1.21), so
none is clearly separable from run-to-run noise.

**These are 0-shot and are not comparable to Qwen's published scores**, which use few-shot
prompting. They are directly comparable to each other, which is the point. Note also that
ARC-Challenge is low for a model at this MMLU — the base scores 58.9 under the same
settings, so that is format sensitivity in a reasoning-tuned model, not abliteration
damage.

What the benchmarks do **not** cover: no generative evaluation (GSM8K, HumanEval), no
math or code, no multilingual, and the harness loads the text stack only — nothing here
measures the vision tower or MTP speculative decoding.

| Measurement | Base model | This model |
|---|---|---|
| Refusals (100 held-out harmful prompts) | 98/100 | **12/100** |
| KL divergence vs base (first-token) | 0 | 0.1191 |

Search: 200 Heretic trials, 23 non-dominated points. The published model is the marked row.

| refusals | KL divergence | |
|---|---|---|
| 12/100 | 0.1191 | ← published |
| 13/100 | 0.1052 | |
| 19/100 | 0.0722 | |
| 23/100 | 0.0635 | |
| 26/100 | 0.0507 | |
| 27/100 | 0.0410 | |
| 35/100 | 0.0406 | |
| 36/100 | 0.0387 | |
| 41/100 | 0.0366 | |
| 44/100 | 0.0352 | |
| 46/100 | 0.0334 | |
| 48/100 | 0.0331 | |
| 51/100 | 0.0321 | |
| 52/100 | 0.0294 | |
| 60/100 | 0.0290 | |
| 76/100 | 0.0280 | |
| 77/100 | 0.0247 | |
| 83/100 | 0.0204 | |
| 86/100 | 0.0193 | |
| 91/100 | 0.0170 | |
| 96/100 | 0.0146 | |
| 97/100 | 0.0044 | |
| 98/100 | 0.0004 | |

### How to read these

**Refusal rate** is the count of refusals over 100 held-out prompts from
[`mlabonne/harmful_behaviors`](https://huggingface.co/datasets/mlabonne/harmful_behaviors)
(test split) — explicitly harmful requests, not benign ones. So this number is *not* an
over-refusal rate: it does not tell you how often the model declines legitimate work. It
tells you how much of the original safety behaviour on harmful requests remains.

**KL divergence** is measured against the unmodified base model over first-token
distributions, and is the optimizer's proxy for "how much did we damage the model". Lower is
closer to base. It is a proxy, not a capability measurement — a low KL does not certify that
reasoning or coding ability survived, and nothing here does certify that.

The two trade off against each other. Heretic searches a Pareto front between them; the
published point is one choice on that front, not a global optimum.

### Caveats that matter

- **Refusals were measured in non-thinking mode.** This model's chat template opens a
  `<think>` block, so the evaluation closes it explicitly to score answers rather than
  reasoning traces. With thinking enabled the refusal rate may differ, in either direction.
- **The measurement is 100 prompts from one dataset.** It generalizes to that distribution
  of harmful requests and no further. Refusal behaviour on other topics is uncharacterized.
- **Perplexity is wikitext-2 only** (see the Files table). It detects gross quantization
  damage. It does not detect capability loss on reasoning, code, or multilingual work.
- **Quantization compounds everything above.** The measurements were taken on the bf16
  merge; the files you download are quantized.

## Requirements

MTP speculative decoding landed in llama.cpp PR #22673. Builds older than that will load
these files and silently ignore the MTP tensors.

## Limitations

- Refusals are reduced, not eliminated, and not redirected. This model attempts many requests
  the original declines, but a meaningful fraction still get refused — see Measured behaviour.
- Behaviour near the old refusal boundary is less stable than the base model.
- Lower quants compound that. Evaluate behaviour on Q6_K or Q8_0, not IQ4_XS.
- Capability benchmarks show a 0.5-point mean drop vs base across MMLU, ARC-Challenge,
  HellaSwag and Winogrande. See Measured behaviour. No generative, math, code, or
  multilingual evaluation was run.

## Intended use

Local inference. Not intended for deployment to third parties without your own safety layer.

## License

Apache 2.0, inherited from Qwen/Qwen3.8-27B. The base model's license and acceptable use
policy still apply to your use of this derivative.

## Speculative decoding, measured on this model

| prompt | spec_type | n_max | tok/s | vs baseline |
|---|---|---|---|---|
| prose | none | - | 74.8 | 1.00x |
| prose | draft-mtp | 1 | 89.0 | 1.19x |
| prose | draft-mtp | 2 | 85.7 | 1.15x |
| prose | draft-mtp | 3 | 72.0 | 0.96x |
| prose | draft-mtp | 4 | 71.1 | 0.95x |
| prose | draft-mtp | 5 | 62.9 | 0.84x |
| prose | draft-mtp | 6 | 53.8 | 0.72x |
| prose | draft-mtp | 7 | 49.9 | 0.67x |
| prose | draft-mtp | 8 | 59.9 | 0.80x |
| code | none | - | 74.7 | 1.00x |
| code | draft-mtp | 1 | 95.4 | 1.28x |
| code | draft-mtp | 2 | 92.9 | 1.24x |
| code | draft-mtp | 3 | 82.6 | 1.11x |
| code | draft-mtp | 4 | 74.9 | 1.00x |
| code | draft-mtp | 5 | 67.4 | 0.90x |
| code | draft-mtp | 6 | 59.4 | 0.80x |
| code | draft-mtp | 7 | 55.6 | 0.74x |
| code | draft-mtp | 8 | 70.9 | 0.95x |
| chat | none | - | 74.7 | 1.00x |
| chat | draft-mtp | 1 | 90.6 | 1.21x |
| chat | draft-mtp | 2 | 84.2 | 1.13x |
| chat | draft-mtp | 3 | 76.1 | 1.02x |
| chat | draft-mtp | 4 | 70.4 | 0.94x |
| chat | draft-mtp | 5 | 64.3 | 0.86x |
| chat | draft-mtp | 6 | 55.2 | 0.74x |
| chat | draft-mtp | 7 | 50.2 | 0.67x |
| chat | draft-mtp | 8 | 54.1 | 0.72x |
