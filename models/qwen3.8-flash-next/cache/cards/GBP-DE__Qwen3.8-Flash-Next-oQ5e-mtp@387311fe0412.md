---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - omlx
  - qwen
  - qwen3.8
  - oq
  - oq5
  - oq5e
  - 5-bit
  - quantized
  - mtp
  - vision
---

# Qwen3.8-Flash-Next-oQ5e-mtp

<div class="rounded-r-lg border-l-4 border-blue-500 p-4 not-italic" style="background-color: rgba(59, 130, 246, 0.13); border-left: 4px solid #3b82f6; padding: 16px 20px; border-radius: 0 8px 8px 0; font-style: normal;">

  <p><strong>🔄 UPDATE — 2026-10-13: New WLSRefit versions available!</strong></p>

  <p>This is an earlier oQ5e quantization. The newer WLSRefit versions use an improved quantization method designed to reduce quantization error and preserve more of the original model's quality.</p>
  
  <p><strong>For new downloads, please use one of these updated versions:</strong></p>

  <ul>
    <li><a href="https://huggingface.co/GBP-DE/Qwen3.8-Flash-Next-oQ5e-WLSRefit-MTP"><strong>oQ5e WLSRefit MTP</strong></a> — Recommended replacement at the same quantization level.</li>
    <li><a href="https://huggingface.co/GBP-DE/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP"><strong>oQ4e WLSRefit MTP</strong></a> — Smaller alternative with lower memory requirements and noticeably improved reasoning consistency over our previous oQ4e conversion in local testing.</li>
  </ul>

  <p>This earlier version remains available for compatibility and comparison.</p>

</div>

**A high-quality oQ5e quantization of Qwen3.8-Flash-Next, targeting improved reasoning robustness and generation stability over oQ4e while still fitting comfortably on a 128 GB Apple Silicon system with SSD N-gram offload.**

This model was quantized directly from the official **Qwen/Qwen3.8-Flash-Next** weights using **oMLX 0.6.4**.

The goal of this quantization is to retain more of the original model's reasoning quality and generation stability than lower-bit oQ4e variants, while remaining practical for local inference on a 128 GB Apple Silicon Mac.

## Quantization

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen3.8-Flash-Next` |
| Quantization | **oQ5e** |
| Quantizer | **oMLX 0.6.4** |
| Enhanced / imatrix mode | Yes |
| Nominal group size | 64 |
| Non-quantized dtype | BF16 |
| MTP preserved | Yes |
| Vision preserved | Yes |
| Local model size | **125.7 GB** |
| Sensitivity model | `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` |

The oQ5e weights were quantized **directly from the original BF16 source model**.

`Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` was used only as the sensitivity model for determining the mixed-precision allocation. Its quantized weights are **not** the source weights of this model.

The included `oq_imatrix_report.json` contains the quantization/imatrix report generated during the process.

> **Note:** oQ mixed-precision quantization can use different effective settings for selected tensors/modules. The nominal group size used for this quantization was 64.

## Hardware and memory

Primary testing was performed on:

- **MacBook Pro**
- **Apple M5 Max**
- **128 GB Unified Memory**
- oMLX
- SSD N-gram Offload enabled
- Lightning MTP enabled

During active generation under similar everyday system conditions:

| Model | Total system memory |
|---|---:|
| `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` | ~93 GB |
| **This oQ5e model** | **~102 GB** |

The observed difference was approximately **9 GB**.

These are **total system memory measurements**, not model-only resident-memory figures. They include macOS, oMLX and other running applications such as Chrome.

Qwen3.8-Flash-Next contains a very large N-gram/PLE component. With **SSD N-gram Offload**, this does not all need to remain resident in unified memory, making this 125.7 GB model practical on a 128 GB Apple Silicon system.

Long-context memory requirements will of course increase with context length and KV-cache usage.

## Recommended oMLX settings

For a 128 GB Apple Silicon Mac:

- **SSD N-gram Offload: ON**
- **Lightning MTP: ON**

These were the settings used for the comparisons below.

TurboQuant KV, SpecPrefill and DFlash were disabled during the benchmark runs unless otherwise stated.

## Why oQ5e?

The motivation for creating this model was generation and reasoning stability.

In testing, an existing oQ4e quantization occasionally showed severe generation instability on difficult prompts — including one runaway self-correction loop on an otherwise simple question.

The oQ5e model has so far behaved more robustly in several demanding reasoning and constraint-solving tests.

**In my testing, oQ5e provides more reasoning and generation stability on difficult prompts than oQ4e, while remaining practical on a 128 GB Apple Silicon system.**

This is not intended to claim that oQ5e wins every benchmark. Some tests resulted in ties, and oQ4e was faster on some workloads. The examples below are included to show the observed trade-off rather than only the best result.

## Selected Q4e vs oQ5e tests

### 8-person constraint puzzle — Thinking OFF

A constraint-solving test involving:

- 8 people
- 8 unique times
- 8 unique colors
- 24 constraints
- checksum verification

**oQ5e**

- Prefill: 276.9 tok/s
- Generation: 69.7 tok/s
- Result: **Exact solution**
- All assignments correct
- Checksum correct

**oQ4e**

- Prefill: 226.3 tok/s
- Generation: 65.8 tok/s
- Result: **Failed**
- Duplicate time assignment
- Multiple constraint violations
- Incorrect checksum

This was one of the clearest non-thinking quality differences observed.

### Five-trigger state-machine benchmark — Thinking ON

A multi-stage simulation involving four reservoirs, scheduled rate changes, five state-dependent trigger events, six transfers, exact event times, final states and verification values.

**oQ5e**

- Prefill: 220.2 tok/s
- Generation: 43.4 tok/s
- Thinking: 505 s
- Total: 520.3 s

The model correctly determined:

- all five trigger times
- all trigger-state amounts
- all transfers
- all final reservoir levels
- the correct first-empty reservoir
- the correct empty time

Two synthetic post-hoc verification values (checksum and signature) were calculated incorrectly.

**oQ4e**

- Prefill: 308.8 tok/s
- Generation: 42.9 tok/s
- Thinking: 761.1 s
- Total: 771.6 s

The visible reasoning/output remained correct through the intermediate trigger states shown, but the model did **not complete the required answer**.

### Python debugging / scheduling benchmark — Thinking ON

A production-style Python debugging task tested:

- code comprehension
- sweep-line algorithms
- heaps
- half-open interval semantics
- tie-breaking
- edge cases
- complexity analysis
- manual verification

**oQ5e**

- Prefill: 145.4 tok/s
- Generation: 37.5 tok/s
- Thinking: 504 s
- Total: 588.4 s

The model:

- completed the response
- correctly identified the important bugs
- derived the correct expected schedule
- calculated all four verification values correctly

However, its proposed implementation contained a merge-state bug: it updated a local `last_segment` value without updating the corresponding tuple already stored in the returned `result` list.

Therefore this test is counted as a **coding failure despite a correct analysis and correct expected results**.

**oQ4e**

- Prefill: 233.7 tok/s
- Generation: 37.2 tok/s
- Thinking: 880 s
- Total before termination: 885.6 s

The reasoning had reached the correct expected verification values, but the model remained in extended self-review and **never produced the required final answer**.

This is counted as a failure due to non-completion.

### Counterexample: four-tank state-machine test

Not every test favored oQ5e.

On another independently verified state-machine problem, **both models produced the exact correct answer**.

| | oQ4e | oQ5e |
|---|---:|---:|
| Prefill | 296.5 tok/s | 188.6 tok/s |
| Generation | 44.4 tok/s | 41.0 tok/s |
| Thinking | **362.0 s** | 544.2 s |
| Total | **370.9 s** | 557.1 s |
| Result | Exact | Exact |

On this test, oQ4e was substantially faster while achieving identical correctness.

### Boolean/SAT logic test

A separate verified 10-variable Boolean constraint problem with a unique solution produced an exact result from both models.

| | oQ4e | oQ5e |
|---|---:|---:|
| Prefill | 186.4 tok/s | 130.2 tok/s |
| Generation | 52.9 tok/s | 50.6 tok/s |
| Thinking | 29.5 s | **23.1 s** |
| Total | 35.0 s | **30.2 s** |
| Result | Exact | Exact |

This illustrates that the difference between the quantizations is often not visible on smaller reasoning tasks.

## Benchmark interpretation

These are practical local tests rather than a standardized academic evaluation suite.

The observed pattern so far is:

- On normal or moderately difficult prompts, **oQ4e and oQ5e can be very similar**.
- On several harder constraint and state-tracking problems, **oQ5e showed better completion and generation stability**.
- oQ5e is **not universally faster**.
- oQ4e can be substantially faster on some reasoning workloads.
- oQ5e uses approximately **9 GB more total system memory** in the tested configuration.
- Higher precision does not eliminate reasoning or coding errors.

The main reason to choose this oQ5e build over oQ4e is therefore **quality and robustness rather than speed or minimum memory usage**.

## Model architecture

This is a quantized version of **Qwen3.8-Flash-Next**.

The base model uses a sparse Mixture-of-Experts architecture with approximately:

- 125B language-model parameters
- 6B activated parameters
- 51B N-gram embedding parameters
- 4B MTP parameters
- 48 layers
- 512 experts
- 10 routed experts + 1 shared expert activated

For full architecture details, capabilities, supported context lengths and usage information, refer to the official Qwen model card.

## Credits

### Base model

**Qwen/Qwen3.8-Flash-Next**

https://huggingface.co/Qwen/Qwen3.8-Flash-Next

All credit for the base model architecture, training and original weights belongs to the Qwen team.

### Quantization

Quantized with **oMLX 0.6.4** using oQ5e enhanced/imatrix quantization.

### Sensitivity model

Sensitivity analysis used:

**Jundot/Qwen3.8-Flash-Next-oQ4e-mtp**

https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp

The sensitivity model was used to guide mixed-precision allocation only. Final weights were quantized directly from the official BF16 source.

## License

This quantized model is distributed under the **Qwen Community License 1.0**, inherited from the base model.

Please read the full license before using or redistributing the model:

https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/LICENSE