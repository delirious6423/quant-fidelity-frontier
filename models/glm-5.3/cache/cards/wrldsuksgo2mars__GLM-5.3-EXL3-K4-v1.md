---
language:
- en
- zh
license: other
license_name: glm-5.3
base_model: zai-org/GLM-5.3
pipeline_tag: text-generation
tags:
- glm
- glm-5.3
- exl3
- 4-bit
- moe
---

# GLM-5.3 EXL3 K4

This is a calibrated 4-bit EXL3 quantization of `zai-org/GLM-5.3`.

The routed expert `gate_proj`, `up_proj`, and `down_proj` tensors in base-model
layers 3 through 77 use the EXL3 K4 representation. Other tensors, including
the native MTP overlay, retain their source representation. The repository
contains 46 safetensors shards and the complete quantization metadata in
`quantize_config.json`.

## Compatibility

This is an EXL3 checkpoint, not a drop-in GPTQ or standard Transformers
checkpoint. Use a serving runtime that supports GLM-5.3 MoE/DSA and this EXL3
tensor layout. Check runtime compatibility before downloading the full model.

## Tool-use result

The best observed `tool-eval-bench` run scored **91/100** (125/138 raw points)
with fixed DFlash2 draft width 5. This is a single local evaluation result and
is not an official upstream GLM-5.3 benchmark.

## License

This derivative checkpoint is distributed under the GLM-5.3 model license.
Review and comply with the license terms of the original `zai-org/GLM-5.3`
model before use.
