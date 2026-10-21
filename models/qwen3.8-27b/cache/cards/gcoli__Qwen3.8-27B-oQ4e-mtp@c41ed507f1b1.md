---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- oqe
- quantized
- 4-bit
- apple-silicon
- qwen3_5
- mtp
- vision-language
---

# Qwen3.8-27B-oQ4e-mtp

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) quantized for Apple Silicon with
**oQe** ([oQ](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md) enhanced
quantization, oMLX v0.5.7). The format is MLX safetensors.

- **Base precision**: 4-bit affine, group size 64
- **Effective size**: 4.70 bits per weight over the language model, 4.89 over the whole repository
- **Download**: 17.0 GB in 4 shards
- **Vision encoder**: kept, in bfloat16 (0.92 GB, not quantized)
- **MTP heads**: kept and quantized (29 tensors)

> [!IMPORTANT]
> This quantization was uploaded on **2026-08-14** and replaces a previous version.
> Re-download the weights if you pulled this repository before that date.

## Variants

| Model | Base bits | bits/weight (LM) | Size | Notes |
| --- | --- | --- | --- | --- |
| [oQ3.5e-mtp](https://huggingface.co/gcoli/Qwen3.8-27B-oQ3.5e-mtp) | 3-bit | 4.06 | 14.8 GB | smallest footprint; the highest quantization error of the four |
| [oQ4e-mtp](https://huggingface.co/gcoli/Qwen3.8-27B-oQ4e-mtp) **(this repo)** | 4-bit | 4.70 | 17.0 GB | the balanced default for 24-32 GB machines |
| [oQ5e-mtp](https://huggingface.co/gcoli/Qwen3.8-27B-oQ5e-mtp) | 5-bit | 5.67 | 20.3 GB | low quantization error at a moderate size |
| [oQ6e-mtp](https://huggingface.co/gcoli/Qwen3.8-27B-oQ6e-mtp) | 6-bit | 6.67 | 23.7 GB | the closest to bfloat16 of the four; the largest download |

Sizes are decimal GB, as Hugging Face reports them. "bits/weight" covers the language model only,
including the scales and biases of the quantized tensors. The bfloat16 vision encoder adds
0.92 GB to every variant.

## Run it

### oMLX

```bash
omlx serve gcoli/Qwen3.8-27B-oQ4e-mtp --port 8000
```

The server exposes an OpenAI-compatible API. The menu-bar app loads the same repository ID.

### mlx-lm

```bash
pip install -U mlx-lm
mlx_lm.generate --model gcoli/Qwen3.8-27B-oQ4e-mtp --prompt "Explain affine quantization."
```

*Unverified: mlx-lm support for the `qwen3_5` vision-language architecture and for the MTP heads
is not tested for this card. The language weights are standard MLX safetensors.*

### Sampling parameters

Qwen recommends these values for Qwen3.8-27B. The `generation_config.json` in this repository
carries the thinking-mode values.

| Mode | temperature | top_p | top_k | presence_penalty |
| --- | --- | --- | --- | --- |
| Thinking (default) | 1.0 | 0.95 | 20 | 0.0 |
| Instruct / non-thinking | 0.7 | 0.80 | 20 | 1.5 |

Do not use greedy decoding. It produces repetition in this model family.

### Memory

Plan for about 24 GB of unified memory: the weights, the KV cache and the vision encoder share it.
The context window is 262,144 tokens natively, so a long context adds several GB of KV cache.
*This is an estimate, not a measurement.*

## What oQe does

Standard affine quantization treats every weight in a group as equally important. oQe adds an
**importance matrix (imatrix)**: oMLX runs calibration inference, records the activation
magnitude each weight actually sees, and weights the quantization error by that importance.
Weights that drive large activations receive the smaller error.

oQ then allocates bits per tensor instead of per model. It measures the sensitivity of each
tensor as `MSE(float_output, quantized_output) / mean(float_output^2)` and raises the bit width
of the sensitive tensors above the base, while the total stays near the target.

## Quantization recipe

These values are read back from the published `config.json` and `oq_imatrix_report.json`.

| Setting | Value |
| --- | --- |
| Tool | oMLX v0.5.7, oQ enhanced quantization |
| Mode | affine |
| Base bits / group size | 4 / 64 |
| Per-tensor boosts | 166 tensors at 5-bit |
| Embeddings (`embed_tokens`) | 4-bit (base) |
| Output head (`lm_head`) | 4-bit (base) |
| Quantized tensors | 505 |
| Vision tower (`vision_tower.*`) | 333 tensors, bfloat16, not quantized |
| Non-quantized dtype | bfloat16 (norms, scales, biases) |
| MTP heads | preserved, quantized |
| Text-only mode | off, so the vision encoder ships with the model |

### Calibration

| Setting | Value |
| --- | --- |
| Dataset | `oqe_code_multilingual` |
| Samples x sequence length | 128 x 512, adaptive sampling enabled |
| imatrix entries | 504 |
| imatrix cache | collected fresh for this run |
| Tensors with imatrix applied | 503 of 505 |
| Tensors without a match | `language_model.lm_head`, `language_model.model.embed_tokens` |
| Strict coverage | off |

Strict coverage was off, so the two unmatched tensors fell back to standard oQ affine
quantization instead of failing the run. The calibration set is code and multilingual text.
Expect the importance weighting to favour those domains over, for example, creative prose.

## Limitations

- **No benchmarks were run.** This card reports no perplexity, no KL divergence and no task
  scores. The ordering in the variants table follows bit width, not measured quality.
- The vision encoder is unquantized, so image and video input costs the same memory in every
  variant.
- MTP heads only speed up generation on a runtime that implements multi-token prediction. On
  other runtimes they occupy disk and memory without an effect.
- Quantization below 4 bits degrades long-context and multi-step reasoning first. Test
  `oQ3.5e-mtp` on your own task before you rely on it.

## Reproduce

Quantize the base model in oMLX with these options:

1. Enable **oQe**.
2. Enable **Reuse imatrix cache**, and leave the cache path empty.
3. Leave **Strict imatrix coverage** off.
4. Leave **Text only** off.
5. Enable **Preserve MTP**.
6. Set the non-quant dtype to **bfloat16**.
7. Select the **oQ4** level and start the quantization.

## License

Apache 2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The
quantization changes the weights only. All model capabilities and restrictions come from the
base model.

## Credits

- [Qwen team](https://huggingface.co/Qwen) for Qwen3.8-27B.
- [jundot/omlx](https://github.com/jundot/omlx) for oMLX and the oQ quantization system.
