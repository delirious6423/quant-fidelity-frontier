---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
tags:
- rocmfp4
- rocm
- llama.cpp
- gguf
- gfx1100
---

# Qwen3.8-27B-ROCmFPX-GGUF

**ROCmFPX GGUF quantization of `Qwen/Qwen3.8-27B`**

AMD focused model weight formats. Tested on AMD 7900 XTX

Requires ROCmFPX llama.cpp build to run. See link below

**Quantizations available**

| File | ROCmFPX preset | Size |
| --- | --- | ---: |
| [`Qwen3.8-27B-ROCmFP4.gguf`](https://huggingface.co/hugobugo34/Qwen3.8-27B-ROCmFP4/resolve/main/Qwen3.8-27B-ROCmFP4.gguf?download=true) | `Q4_0_ROCMFP4` | 17.7 GB |
| [`Qwen3.8-27B-ROCMFP4-COHERENT.gguf`](https://huggingface.co/hugobugo34/Qwen3.8-27B-ROCmFP4/resolve/main/Qwen3.8-27B-ROCMFP4-COHERENT.gguf?download=true) | `Q4_0_ROCMFP4_COHERENT` | 15.7 GB |
| [`Qwen3.8-27B-ROCmFP4-FAST.gguf`](https://huggingface.co/hugobugo34/Qwen3.8-27B-ROCmFP4/resolve/main/Qwen3.8-27B-ROCMFP4-FAST.gguf?download=true) | `Q4_0_ROCMFP4_FAST` | 14.6 GB |

## Related links

- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)  *(Qwen Repo)*
- [ROCmFPX](https://github.com/charlie12345/ROCmFPX)   *(Github Repo)*