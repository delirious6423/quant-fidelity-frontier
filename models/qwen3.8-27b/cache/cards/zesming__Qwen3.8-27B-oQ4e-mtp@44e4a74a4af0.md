---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - mlx
  - omlx
  - oq
  - oqe
  - 4-bit
  - mtp
  - qwen
  - vision-language
  - multimodal
---

# Qwen3.8-27B-oQ4e-mtp

This is an MLX oQ4e quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced with oMLX 0.5.7.

## Quantization

- Quantizer: oMLX oQ4e
- Base precision target: 4-bit affine, group size 64
- Effective mixed-precision budget reported by oMLX: approximately 4.9 bpw
- Calibration: oQe imatrix-weighted affine quantization
- Non-quantized tensor dtype: bfloat16
- Native MTP weights and configuration retained
- Vision tower retained; this is not a text-only conversion
- Output: four safetensors shards, approximately 15.83 GB

oQ4e uses oMLX layer-sensitivity planning together with activation-importance calibration. Architecture protection rules leave the vision encoder unquantized and protect output-critical tensors.

## Multimodal configuration note

The official base checkpoint includes both preprocessor_config.json and video_preprocessor_config.json. oMLX 0.5.7 copied the image preprocessor configuration during quantization but did not copy the separate video preprocessor sidecar. The original video_preprocessor_config.json from Qwen/Qwen3.8-27B was therefore added unchanged to this repository so that the official image/video processor metadata is preserved.

No vision weights were reconstructed or borrowed from another checkpoint.

## Validation

The converted checkpoint was validated locally with oMLX 0.5.7:

- Loaded successfully as a VLM.
- Native Lightning MTP was selected and activated at runtime.
- Text generation completed successfully with MTP draft acceptance recorded by oMLX.
- An OpenAI-compatible multimodal API request correctly identified the main subject of a test image.
- The checkpoint index contains the retained language_model.mtp.* tensors and the full vision_tower.* tensor set.

## Suggested oMLX settings

For thinking mode, the Qwen model card recommends:

- temperature=1.0
- top_p=0.95
- top_k=20
- min_p=0.0
- presence_penalty=0.0
- repetition_penalty=1.0

This conversion supports enable_thinking, preserve_thinking, and reasoning_effort (xhigh, medium, or low). Native Lightning MTP can be enabled in the oMLX model settings.

## License and attribution

The base model is licensed under Apache 2.0. See the included LICENSE file and the [official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) for details.
