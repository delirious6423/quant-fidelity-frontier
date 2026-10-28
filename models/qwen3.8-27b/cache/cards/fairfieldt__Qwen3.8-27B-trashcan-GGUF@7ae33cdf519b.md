---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- trashcan
---

# Qwen3.8-27B GGUF for trashcan

The quant [trashcan](https://github.com/fairfieldt/trashcan) runs Qwen3.8-27B with, tensor parallel over two or four
2013 Mac Pros (FirePro D700s, Vulkan). Pure Q4_0 with a Q6_K output head, a Q8_0 token embedding and f32 DeltaNet
gates (`ssm_alpha`, `ssm_beta`): the tensor types trashcan's kernels read. Other community quants mix in types they
don't.

```sh
llama-quantize --pure --imatrix imatrix_unsloth.gguf --output-tensor-type q6_k --token-embedding-type q8_0 \
  --tensor-type ssm_alpha=f32 --tensor-type ssm_beta=f32 Qwen3.8-27B-BF16.gguf Qwen3.8-27B-trashcan-Q4_0.gguf q4_0
```

BF16 source: [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF). Importance matrix:
[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).

`mtp-Qwen3.8-27B-trashcan-Q4_0.gguf` is the multi-token-prediction layer for speculative decoding (`--mtp`), quantized
the same way from ggml-org's `mtp-Qwen3.8-27B-BF16.gguf`, without the importance matrix (it has no entries for that
layer).

| File | Bytes | SHA-256 |
|---|---|---|
| `Qwen3.8-27B-trashcan-Q4_0.gguf` | 16,193,669,792 | `92349815af2ba00133f9342aaf12dcbe9c6311e6d198414f8a3ac8c90b9c7284` |
| `mtp-Qwen3.8-27B-trashcan-Q4_0.gguf` | 2,643,753,632 | `22ba20696d734700a683640a887bf14f0cbf4e2f92fd6ee0c56ba01ef4beb210` |

Single-stream decode on trashcan: 28.4 tok/s on two Macs (39.5 with MTP), 40.7 on four (52.0 with MTP). Recipes
`2-qwen27b` and `4-qwen27b`: [docs/deploy.md](https://github.com/fairfieldt/trashcan/blob/main/docs/deploy.md).
