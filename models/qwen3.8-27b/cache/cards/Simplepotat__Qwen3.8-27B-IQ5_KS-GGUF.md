---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: text-generation
library_name: gguf
tags:
- gguf
- imatrix
- conversational
- qwen3_5
- ik_llama.cpp
---

# Qwen3.8-27B-IQ5_KS-GGUF

MTP capable and smaller non MTP IQ5_KS quants of Qwen 3.8 27B [for use with ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), created through reuse of [Ubergarm's tensor recipe for Qwen 3.6 27B](https://huggingface.co/ubergarm/Qwen3.6-27B-GGUF) with his updated Qwen 3.8 imatrix.

Creation of this was possible due to the more or less identical structure of 3.8 27B to its predecessor, and if the recipe is equally effective (which it should be) this should maintain SOTA performance for full 24GB offload with room for medium length full precision context windows.

## Files

| File | Purpose | Size | SHA-256 |
|---|---|---:|---|
| `Qwen3.8-27B-MTP-IQ5_KS.gguf` | MTP-capable build | 18.963 GiB | `F879626127706403EF9B75C8E3F5E48D78B89FAD32D07A59127F2AFDD485C067` |
| `Qwen3.8-27B-no-MTP-IQ5_KS.gguf` | Smaller build with the MTP tensors removed | 18.543 GiB | `07A86EDD155B9CB1321BBF65FB36D47B5856AE148C1869EB33A89895308D5F62` |

The no-MTP file is 430.413 MiB smaller. With MTP disabled, the tested ik_llama.cpp loader ignores the embedded predictor tensors, so both files use the same VRAM but 430MiB less disk usage is 430MiB less disk usage.

## MTP usage

```bash
./llama-cli \
  -m Qwen3.8-27B-MTP-IQ5_KS.gguf \
  -ngl 999 \
  --spec-type mtp:n_max=1,p_min=0.0
```

## Chat Template

Both GGUFs contain an exact match of the latest official Qwen3.8-27B chat template at the time of publication.

## Sources and tooling

- BF16 GGUF Source: [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)
- Tensor Recipe: [ubergarm/Qwen3.6-27B-GGUF](https://huggingface.co/ubergarm/Qwen3.6-27B-GGUF)
- Importance Matrix: [ubergarm/Qwen3.8-27B-GGUF](https://huggingface.co/ubergarm/Qwen3.8-27B-GGUF)
- Individual-tensor quantizer: [Thireus/ik_llama.cpp build 5263](https://github.com/Thireus/ik_llama.cpp/releases/tag/th-quantize_individual_tensors-b5263-8cbc1f7)