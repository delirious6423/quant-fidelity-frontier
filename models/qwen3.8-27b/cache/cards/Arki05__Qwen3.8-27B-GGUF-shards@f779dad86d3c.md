---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
---

# Qwen3.8-27B, every block tensor at every type

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at 18 llama.cpp types, each block tensor at each type,
for assembling GGUFs with a type per tensor ([qalloc](https://arki05.com/projects/qalloc/)). Prebuilt files:
[Arki05/Qwen3.8-27B-GGUF](https://huggingface.co/Arki05/Qwen3.8-27B-GGUF).

- `source/`: Unsloth's BF16 GGUF ([unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)), unchanged.
- `types/<T>-hq.gguf`: every linear block tensor at llama.cpp type `T` (IQ1_M … Q8_0, IQ4_NL, MXFP4, NVFP4), rebuilt
  from the BF16 weights Hessian-quantized. Same type, format and size as llama.cpp's: plain GGUF types, for stock
  llama.cpp (a GGUF assembled from them holds the plain type `T`); modified versions of the original weights.
- `types/<T>.gguf` (Q2_K … Q8_0): token_embd, output, the multi-token-prediction layer and the linear attention's small
  `ssm_alpha` / `ssm_beta` tensors at llama.cpp's own quantization.
- `manifest.json` (`qcore-store/1`), `skeleton.gguf`, `store.json`, `records/`: what qalloc reads, with every tensor's
  offset and sha256.

The earlier version of this store (every tensor at ~41 types by llama.cpp's and ik_llama.cpp's quantizers) is at
revision [`0090490`](https://huggingface.co/Arki05/Qwen3.8-27B-GGUF-shards/tree/0090490f).

Apache-2.0, as the original model.
