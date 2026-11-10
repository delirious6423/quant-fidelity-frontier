---
library_name: transformers
license: mit
pipeline_tag: image-text-to-text
base_model:
  - zai-org/GLM-5.3-Flash
datasets:
  - CohereLabs/aya_dataset
  - interstellarninja/hermes_reasoning_tool_use
tags:
  - quantization
  - nvfp4
---

# GLM-5.3-Flash-NVFP4

An NVFP4-quantized version of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).

This model was independently quantized to **NVFP4 W4A4** using **LLM Compressor** and saved in the **compressed-tensors** format.

## Model Details

* **Base model:** zai-org/GLM-5.3-Flash
* **Architecture:** Glm5NextForConditionalGeneration
* **Quantization:** NVFP4 W4A4
* **Serialization format:** nvfp4-pack-quantized
* **Compression backend:** compressed-tensors
* **Calibration samples:** 1,024
* **Maximum calibration sequence length:** 4,096 tokens

## Calibration Data

The following datasets were used for calibration:

* [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset)
* [interstellarninja/hermes_reasoning_tool_use](https://huggingface.co/datasets/interstellarninja/hermes_reasoning_tool_use)

The calibration dataset consists of:

* 512 general multilingual instruction samples from Aya
* 512 reasoning and tool-use conversation samples from Hermes

## Quantization Configuration

```yaml
default_stage:
  default_modifiers:
    GPTQModifier:
      config_groups:
        group_0:
          targets: ['re:.*mlp\.experts\..*(gate|up|down)_proj$']
          weights:
            num_bits: 4
            type: float
            symmetric: true
            group_size: 16
            strategy: tensor_group
            dynamic: false
            actorder: static
            scale_dtype: torch.float8_e4m3fn
            observer: nvfp4_expanded_imatrix
            observer_kwargs: {strict: true}
          input_activations: {num_bits: 4, type: float, symmetric: true, group_size: 16, strategy: tensor_group,
            dynamic: local, scale_dtype: torch.float8_e4m3fn, observer: static_minmax}
          output_activations: null
          format: null
      targets: [Linear]
      ignore: ['re:.*visual.*', lm_head, 're:.*mlp\.gate$', 're:.*self_attn\.indexer\..*']
      bypass_divisibility_checks: false
      requires_calibration_data: true
      block_size: 128
      dampening_frac: 0.01
      actorder: static
      batched_quantization: auto
```

## Reproducibility

The quantization code and calibration preprocessing are available at:

[glm-5.3-flash-nvfp4-quantization](https://github.com/yasu-oh/glm-5.3-flash-nvfp4-quantization)

The quantization run used:

```text
transformers:        5.22.2
llmcompressor:       0.15.1a20261106
compressed-tensors:  0.20.2a20261103
```

