---
license: mit
base_model:
  - zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
quantization: exl3
language:
  - en
  - zh
tags:
  - glm
  - glm-5
  - exl3
  - exllamav3
  - quantized
  - 4bit
  - moe
  - mul1
---

# GLM-5.3-Flash EXL3 4 bpw MUL1

An EXL3 quantization of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), created with the `mul1` codebook for ExLlamaV3.

This checkpoint is intended for efficient local inference with ExLlamaV3, with particular interest in single-stream decode performance on large MoE models.

The main language-model weights use 4.0 bpw. The vision component and `lm_head` use 6 bpw.

## Quantization details

- Base model: `zai-org/GLM-5.3-Flash`
- Format: EXL3
- Main language-model bitrate: 4.0 bpw
- Vision bitrate: 6 bpw
- `lm_head` bitrate: 6 bpw
- MTP bitrate: 4 bpw
- Codebook: `mul1`
- EXL3 metadata version: `1.4.4`
- Output scales: `always`
- Quantizer: ExLlamaV3
- Calibration metadata: `rows=250`, `cols=2048`

The calibration dataset, prompts, random seed, and exact ExLlamaV3 commit were not recorded in the checkpoint.

The checkpoint also preserves the original model-side quantization metadata:

- Original format: FP8
- FP8 format: `e4m3`
- Activation scheme: dynamic
- Weight block size: `[128, 128]`

## Why MUL1?

The `mul1` codebook is supported by ExLlamaV3's fused INT8-activation GEMV path.

For supported tensors, GPUs, and tensor shapes, this path may improve single-stream decode performance. Actual performance depends on GPU architecture, context length, tensor parallel configuration, runtime version, and workload.

To test the INT8 GEMV path:

```bash
EXL3_INT8_GEMV=2
```

To disable it:

```bash
EXL3_INT8_GEMV=0
```

The current ExLlamaV3 runtime uses the INT8 GEMV path by default where supported. See the [ExLlamaV3 environment-variable documentation](https://github.com/turboderp-org/exllamav3/blob/master/doc/env_vars.md).

## Usage

Use a recent ExLlamaV3 build with GLM-5.3-Flash and `mul1` support.

Load the complete repository directory, including:

- `config.json`
- `quantization_config.json`
- `model.safetensors.index.json`
- all `model-*.safetensors` files
- tokenizer and processor files

This checkpoint is intended for ExLlamaV3-compatible local inference and is not an official release from Z.ai or ExLlamaV3.

## Credits

- Original model: [Z.ai — GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
- Quantization and runtime: [ExLlamaV3](https://github.com/turboderp-org/exllamav3)

## License

The original GLM-5.3-Flash model is released under the MIT License. Please refer to the included `LICENSE` file and the original model repository for the applicable terms.
