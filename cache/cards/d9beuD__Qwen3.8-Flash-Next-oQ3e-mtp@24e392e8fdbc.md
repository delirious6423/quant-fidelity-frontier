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
- oqe
- imatrix
- quantized
- mtp
---

# Qwen3.8-Flash-Next-oQ3e-mtp

This model is an MLX quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), made with [oQe](https://github.com/jundot/omlx) (oMLX v0.7.0) imatrix-weighted mixed-precision quantization.

See also the standard oQ version: [d9beuD/Qwen3.8-Flash-Next-oQ3-mtp](https://huggingface.co/d9beuD/Qwen3.8-Flash-Next-oQ3-mtp).

## Quantization details

- **Base model**: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Model type**: qwen4_exp
- **oQ level**: 3
- **Default bits**: 3
- **Group size**: 64 (default; some modules use 32 or 128)
- **Effective size**: ~3.75 bits per weight, 84 GB
- **dtype**: bfloat16 (non-quantized weights, scales and biases)
- **MTP**: multi-token prediction head preserved (`mtp_num_hidden_layers: 1`)
- **Vision encoder**: included
- **N-gram embedding table**: included
- **Format**: MLX safetensors

Share of quantized parameters by bit width:

| Bits | Share |
|---|---|
| 8-bit | 0.8% |
| 6-bit | 0.0% |
| 5-bit | 0.6% |
| 4-bit | 1.4% |
| 3-bit | 97.1% |

## Importance matrix (oQe)

- **Calibration set**: `oqe_code_multilingual`, 1,024 samples x 512 tokens (adaptive: 128 to 1,024 samples, 8 rounds)
- **Collection**: streamed one decoder layer at a time from the bf16 checkpoint (no proxy model)
- **Expert coverage**: 75,240 / 75,264 routed experts activated (99.97%); the 24 experts that received no calibration tokens use standard oQ quantization
- Full details in [`oq_imatrix_report.json`](oq_imatrix_report.json)

## Sensitivity model

oQ's layer-sensitivity map was measured on [Jundot/Qwen3.8-Flash-Next-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp) (128 samples x 256 tokens, `code_multilingual` calibration set) instead of the full bf16 checkpoint, which does not fit in memory on a 128 GB Mac.

## Requirements

About 84 GB of weights: this model needs a Mac with more unified memory than that (e.g. 128 GB).

## License

This model inherits the [Qwen Community License 1.0](LICENSE) from the base model [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
