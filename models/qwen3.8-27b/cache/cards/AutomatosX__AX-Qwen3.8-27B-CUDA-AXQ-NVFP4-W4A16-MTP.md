---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- axquant
- axq
- nvfp4
- compressed-tensors
- development
pipeline_tag: image-text-to-text
---
# AX-Qwen3.8-27B-CUDA-AXQ-NVFP4-W4A16-MTP

Experimental AXQuant NVFP4 W4A16 checkpoint, exported using the AXQuant 2.2.1 encoder from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0). Source revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

## Storage and precision

AXQuant-owned RTN encoder; no activation calibration or measured sensitivity. Packed E2M1 weights, E4M3 scales per 16 values, FP32 inverse global scale. Activations stay BF16. The container uses compressed-tensors NVFP4A16. Conversion backend: `numpy-reference` (CPU reference encoder on Mac Studio M2 Ultra 192 GB); a CUDA pack name describes the intended runtime format, not the conversion device.

496 language attention/MLP matrices are packed; 703 other tensors match the source exactly. Vision, embeddings, normalization, output head, and packaged MTP weights are protected. The artifact manifest binds every exported weight and configuration file; static-validation.json records the complete protected-tensor comparison and bounded row sampling.

## Runtime status

No runtime load or generation result is available for this checkpoint. Runtime compatibility is unverified. Reference BF16 reconstruction and text generation passed with PyTorch CPU on Mac Studio M2 Ultra 192 GB; see reference-runtime-smoke.json. This does not verify native CUDA or FP4 execution.

The -MTP suffix means the original MTP tensors and configuration are packaged and preserved. MTP decoding is not verified or certified; keep speculative decoding disabled.

The manifest intentionally remains `status=development`, `runtime_verified=false`, `quality_certified=false`. Any separate smoke receipt applies only to its exact manifest digest. Vision and audio generation, long context, numerical parity, task quality, speed, and memory savings during execution have not been qualified. This is not an AX Engine MLX pack.

## Licensing

The model weights retain the upstream Apache-2.0 license; see LICENSE and NOTICE. Preserve any included upstream usage policy. The AXQuant toolkit license does not replace the source model license.
