---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- mlx
- oq
- quantized
- mtp
---

# Qwen3.8-Flash-Next-oQ6-mtp

This model is an MLX quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), made with [oQ](https://github.com/jundot/omlx) (oMLX v0.7.0) mixed-precision quantization.

## Quantization details

- **Base model**: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Model type**: qwen4_exp
- **oQ level**: 6
- **Default bits**: 6
- **Group size**: 64 (default; some modules use 32 or 128)
- **Effective size**: ~6.69 bits per weight, 150 GB
- **dtype**: bfloat16 (non-quantized weights, scales and biases)
- **MTP**: multi-token prediction head preserved (`mtp_num_hidden_layers: 1`)
- **Vision encoder**: included
- **N-gram embedding table**: included
- **Format**: MLX safetensors

Share of quantized parameters by bit width:

| Bits | Share |
|---|---|
| 8-bit | 0.9% |
| 6-bit | 99.1% |

## Sensitivity model

oQ's layer-sensitivity map was measured on [Jundot/Qwen3.8-Flash-Next-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp) (128 samples x 256 tokens, `code_multilingual` calibration set) instead of the full bf16 checkpoint, which does not fit in memory on a 128 GB Mac.

## Requirements

About 150 GB of weights: this model needs a Mac with more unified memory than that (e.g. 192 GB).

## License

This model inherits the [Qwen Community License 1.0](LICENSE) from the base model [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
