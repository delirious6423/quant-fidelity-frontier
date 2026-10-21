---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
license: mit
tags:
- gguf
- quantized
- mixture-of-experts
- deepseek-v4
---

# DeepSeek-V4-Flash-0731 role-splice experiment

This is an experimental role-spliced GGUF for `deepseek-ai/DeepSeek-V4-Flash-0731`. It is not an official DeepSeek release.

The file combines already-quantized tensor payloads from four revision-pinned GGUFs. No tensor was dequantized or requantized during assembly.

## Expert recipe

| Routed-expert role | Tensors | Type | Donor |
| --- | ---: | --- | --- |
| `ffn_down_exps.weight` | 43 | `IQ3_S` | [TacoTakumi](https://huggingface.co/TacoTakumi/DeepSeek-V4-Flash-0731-GGUF/tree/ee3c05f1f0a89306a9c31664e3c0e10895c2d661) |
| `ffn_gate_exps.weight` | 43 | `IQ1_S` | [6block](https://huggingface.co/6block/DeepSeek-V4-Flash-0731-GGUF/tree/de73b664d87a24d117dc2b315832423a1e7d9b50) |
| `ffn_up_exps.weight` | 43 | `IQ1_M` | [AtomicChat](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF/tree/b744b2af48c0008788234b5421122c62b6811b2e) |

The other 1,199 tensors come from [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf/tree/e7f04037032990db0346398d249baf9fb9df1ccc), including its Q8/F16/F32 attention, shared-expert, router, output, and auxiliary tensors.

Complete tensor-type census:

| Type | Tensors |
| --- | ---: |
| `F32` | 492 |
| `F16` | 359 |
| `Q8_0` | 345 |
| `IQ3_S` | 43 |
| `IQ1_S` | 43 |
| `IQ1_M` | 43 |
| `I32` | 3 |

## File and verification

- File: `DeepSeek-V4-Flash-0731-DIQ3S-GIQ1S-UIQ1M.gguf`
- Size: 86,720,111,200 bytes (80.76 GiB)
- Tensors: 1,328
- SHA-256: `7cd6c76a910df9a05567e07ad3955e93ec9373fddc2c5732382d9214261593e9`
- Manifest SHA-256: `c0c5c28c4cc48e3845368686c63649e462afb9a07ce597cf4fc306dbb9495de1`

`splice-manifest.json` records every source repository revision, LFS SHA-256, source byte range, tensor type, shape, and output offset. `verification.json` records structural verification and the runtime smoke test.

The assembled file passed two checks:

1. A complete parser check matched all 1,328 tensor names, dimensions, types, and offsets against the manifest and calculated the whole-file SHA-256.
2. CPU-only `llama-cli` from llama.cpp revision `2f56fc3431f47fe042bf3825e4d5523bdddda993` loaded the model, initialized a 128-token context, evaluated `Hello`, generated one token, and exited successfully.

## Important limitation

This splice has not been quality-benchmarked. The three expert donors used different importance matrices and calibration corpora. GGUF tensor blocks are independently decodable, but combining them does not prove that the resulting model preserves the quality of any donor release.

Treat this as an experimental candidate until it has been compared with the Antirez, Unsloth, and uniform baselines on the same KL/perplexity and task harness.

The file does not include an MTP or DSpark sidecar.

## Runtime

Use a llama.cpp build with `deepseek4` support. The smoke test used revision `2f56fc3`.

```bash
llama-cli \
  -m DeepSeek-V4-Flash-0731-DIQ3S-GIQ1S-UIQ1M.gguf \
  -c 4096 \
  -p "Hello"
```

Configure GPU offload and context size for your hardware. The GGUF alone occupies 80.76 GiB before runtime buffers and KV cache.

## Credits and license

The base model and inherited license come from [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731). Quantized payloads are credited to Antirez, TacoTakumi, 6block, and AtomicChat at the pinned revisions above.

License: MIT.