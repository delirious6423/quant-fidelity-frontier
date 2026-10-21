---
license: mit
language: en
library_name: mlx
pipeline_tag: text-generation
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
- mlx
- 3-bit
- quantized
- qwen3_5_moe
- apple-silicon
---

# Chronos-Ornith-1.5-35B-A3B — MLX 3-bit (text-only, fits 24 GB Macs)

A 3-bit MLX conversion of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), made so the model runs on a 24 GB Apple Silicon Mac with room to spare.

**Text-only.** The vision tower is not included, which is what keeps this build at **15.2 GB** (about 3 GB smaller than the multimodal 3-bit builds). If you need image input, use a multimodal build instead.

## Conversion

Plain affine quantization with `mlx_lm` 0.31.3, with no calibration or mixed-precision recipe:

```bash
python -m mlx_lm convert --hf-path ornith-ai/Ornith-1.5-35B-A3B \
  -q --q-bits 3 --q-group-size 64
```

The result averages 3.5 bits per weight. `mlx_lm` keeps the MoE router gates at 8-bit.

## Usage

```bash
pip install mlx-lm
mlx_lm.generate --model Lifeboat78/Chronos-Ornith-1.5-35B-A3B-MLX-3bit --prompt "Hello"
```

```python
from mlx_lm import load, generate
model, tokenizer = load("Lifeboat78/Chronos-Ornith-1.5-35B-A3B-MLX-3bit")
print(generate(model, tokenizer, prompt="Hello", max_tokens=200))
```

## Measured on an M5 MacBook Pro, 24 GB

These numbers come from a private 36-case suite for a domain-specific assistant covering routing, domain vocabulary, JSON output, reasoning, instruction following, false-premise handling and 32k-token retrieval. The comparison model is Unsloth's `Qwen3.6-35B-A3B-UD-MLX-3bit`, run with the same harness on the same day.

| | This build | Qwen3.6-35B-A3B UD 3-bit |
|---|---|---|
| Graded score | 33/36 (91.7%) | 34/36 (94.4%) |
| Generation speed (median) | **72.7 tok/s** | 46.1 tok/s |
| Median thinking turn | **9.4 s** | 14.8 s |
| Peak memory at a 32k context | **16.1 GB** | 17.4 GB |
| Load time | 3.1 s | 4.0 s |

**Known weakness:** in one ungraded summarization case (a count broken into categories), this build miscounted the totals and gave inconsistent numbers, while the Qwen build got them right. Check its arithmetic on anything that matters. A plain 3-bit quantization without calibration may be part of the reason.

## License and credit

The weights are under the MIT license from the base model. All credit for the model goes to [ornith-ai](https://huggingface.co/ornith-ai). This repository only changes the quantization.
