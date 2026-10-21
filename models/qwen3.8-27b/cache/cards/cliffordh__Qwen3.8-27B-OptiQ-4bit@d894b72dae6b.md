---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - mlx
  - optiq
  - mixed-precision
  - quantization
  - apple-silicon
  - qwen
pipeline_tag: image-text-to-text
library_name: mlx
---

# Qwen3.8-27B-OptiQ-4bit

Sensitivity-aware 4/8-bit mixed-precision OptiQ quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for Apple Silicon via [mlx-optiq](https://mlx-optiq.com).

## What this is

This model was produced by **mlx-optiq** using a KL-divergence sensitivity pass on a six-domain calibration mix. The process measures each layer's sensitivity to quantization error, then allocates bit-widths accordingly: layers where quantization causes large KL divergence from the unquantized reference get 8-bit precision, while robust layers get 4-bit.

### Per-layer bit allocation (from `optiq/metadata.json`)

| Metric | Value |
|---|---|
| Layers at 8-bit | 274 |
| Layers at 4-bit | 224 |
| Total quantized layers | 498 |
| Target bpw | 5.00 |
| Achieved bpw | 5.15 |
| On-disk size | ~20 GB (4 safetensors shards) |
| Total parameters | ~26.9B |

**Naming convention:** Following the llama.cpp convention, "4bit" in the model name refers to the predominant (lower) precision tier, not the average bits-per-weight. The actual average is ~5.15 bpw due to the mixed allocation.

### Bundled components

- **MTP head** (`optiq/mtp.safetensors`): Multi-token prediction head for optional ~1.4x speculative decoding throughput.
- **Vision encoder** (`optiq/optiq_vision.safetensors`): Full-precision vision encoder sidecar (879 MB), preserving the base model's multimodal capabilities.
- **Sensitivity data** (`optiq/sensitivity.json`): Per-layer KL-divergence measurements used to determine the bit allocation.

## Validation

This quantization was validated against uniform 4-bit and uniform 8-bit baselines using a small reasoning-trap test suite designed to surface where low-bit quantization degrades reasoning. The suite probes three failure modes:

| Reasoning probe | Uniform 4-bit | This OptiQ 4/8 | Uniform 8-bit |
|---|---|---|---|
| **Arithmetic precision** -- combined SE tax/FICA-style computation (~15.3% on $60K) under a plausible-but-wrong framing | Incorrect (produced halved figure) | Correct | Correct |
| **Multi-constraint numeric reasoning** -- biweekly-to-monthly budget mapping with hidden 3-paycheck months, rent + mortgage + debt-snowball ordering | Averaged away the 3-paycheck structure | Correctly enumerated higher-income months | Correctly enumerated higher-income months |
| **Premise-checking** -- question containing a subtle false premise about an entity-taxation election | Accepted false premise | Flagged the false premise | Flagged the false premise |

On a 48 GB M4 Pro, this build ran without swap pressure where uniform 8-bit swapped heavily.

### Comparison with lighter mixed-precision (3.5 bpw)

Tested against `rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX` (the 3.5 bpw mixed-precision build distributed via rapid-mlx's built-in model registry) on the same prompts. The lighter model ran faster (16 tok/s vs ~7 tok/s) but lost precision on both discriminating tests:

- Arithmetic: produced incorrect Medicare rate (2.0% instead of correct 2.9%), yielding total FICA of $8,640 instead of the correct $9,180
- Premise-checking: accepted the false SMLLC premise without challenge

This suggests 5.15 bpw is near the minimum threshold for preserving these reasoning capabilities on this model, and 3.5 bpw cuts below it.

### Vision

The vision tower is preserved at full precision (not quantized) via the `optiq/optiq_vision.safetensors` sidecar. In testing, the model correctly identified fine-grained visual details in a real image -- distinguishing a stylized tattoo from a photograph and reading small elements such as a pendant and background symbols. Peak memory with the vision tower loaded was ~24.7 GB on an M4 Pro.

> **Note:** These results reflect observed behavior on specific probes and are not a warranty of accuracy. They are not benchmarks of real-world performance, and should not be interpreted as fitness claims for any professional, financial, tax, or legal use case.

### Reproduction prompt

The prompt that most clearly discriminated between quantization levels:

> A single-member LLC elects S-corp taxation. The owner takes a $60K salary on $150K net income. Walk me through the self-employment tax implications and whether the salary would withstand IRS reasonable compensation scrutiny.

At 5.15 bpw: correct combined FICA rate (15.3%, total $9,180) and flagged the SMLLC premise. At 3.5 bpw: incorrect Medicare rate (2.0% vs 2.9%, total $8,640) and accepted the premise. Uniform 4-bit: halved the total to $4,590.

## Provenance & alignment

This is a derivative of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a model developed by Alibaba Cloud (Qwen team) and released under the Apache 2.0 license. Only the weight representations have been modified (quantization); no fine-tuning or alignment changes were made.

The base model carries RLHF alignment inherited from its training process. This alignment includes avoidance of certain politically sensitive topics narrowly scoped to specific domestic-policy areas. This does not affect general reasoning, technical, coding, or multimodal use cases, but users working with politically sensitive content should be aware of this characteristic.

## Usage

### Load and generate with mlx-lm

```python
from mlx_lm import load, generate

model, tokenizer = load("cliffordh/Qwen3.8-27B-OptiQ-4bit")
prompt = "Explain the tradeoffs of mixed-precision quantization."
response = generate(model, tokenizer, prompt=prompt, max_tokens=512)
print(response)
```

### Serve with speculative decoding (MTP)

```bash
optiq serve cliffordh/Qwen3.8-27B-OptiQ-4bit --mtp
```

## Disclaimer

Experimental quantized derivative provided AS-IS under the Apache 2.0 license, with no warranty of fitness for any purpose. Validation tests probe reasoning behavior on specific prompts, not real-world accuracy. Not intended for production or professional-advice use without independent verification.

## Credits

Quantized by [cliffordh](https://huggingface.co/cliffordh) using [mlx-optiq](https://mlx-optiq.com).

Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team (Alibaba Cloud).
