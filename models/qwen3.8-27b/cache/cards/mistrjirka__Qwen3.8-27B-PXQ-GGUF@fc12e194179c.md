---
quantized_by: mistrjirka
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
tags:
- gguf
- pxa
- pxq
- quantized
---

# PXA PXQ GGUF quantizations of Qwen3.8-27B

PXA/PXQ GGUF quantizations derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

## Runtime support

These are **GGUF containers** containing PXA-specific PXQ tensor types; stock upstream llama.cpp does not support these PXQ codecs.

- **PXA** — https://github.com/poisonxa16/pxa — the reference PXQ runtime. Its primary optimized targets are NVIDIA Pascal (SM60/SM61) and Volta (SM70). PXA also documents wide CUDA builds including Ampere (SM86) and Ada (SM89); those newer architectures use safe generic paths where a Pascal/Volta-specific kernel does not apply and are not claimed to have the same tuning coverage.
- **`mistrjirka/llama.cpp` `v100-optimized`** — https://github.com/mistrjirka/llama.cpp/tree/v100-optimized — a current llama.cpp-based runtime focused on NVIDIA Volta (SM70) and Turing (SM75), tested primarily on Tesla V100 and RTX 2080 Ti. The branch implements the complete published PXQ tensor family, not just PXQ4.

| PXQ tier | PXA | `mistrjirka/llama.cpp` `v100-optimized` |
| --- | --- | --- |
| PXQ1 | Supported | Supported |
| PXQ2 | Supported | Supported |
| PXQ3 | Supported | Supported |
| PXQ4 | Supported | Supported |
| PXQ4-HQ | Supported | Supported |
| PXQ6 | Supported | Supported |

Both runtimes also understand mixed files containing multiple PXQ tensor tiers. Performance and fast-path coverage still depend on GPU architecture and tensor shape; "supported" here means the PXQ wire type and execution path exist, not that every tier is equally optimized on every GPU.

## Which quant should I choose?

`PXQ4` is the normal default. `PXQ4-HQ` and `PXQ6` trade more space for fidelity; `PXQ3` and `PXQ2` trade fidelity for smaller files. `PXQ1` is only published if PXA actually emits PXQ1 tensors for this architecture.

The codec bpw describes the PXQ tensors, not necessarily the complete-file effective bpw: sensitive or structurally unsuitable tensors may deliberately remain Q6_K, Q8_0, F16/F32, etc.

## Available files

| Filename | Tier | PXQ codec | File size | Description |
| --- | --- | --- | ---: | --- |
| [`Qwen3.8-27B-PXQ2-GGUF.gguf`](https://huggingface.co/mistrjirka/Qwen3.8-27B-PXQ-GGUF/resolve/main/Qwen3.8-27B-PXQ2-GGUF.gguf?download=true) | PXQ2 | ~2.27 bpw | 9.42 GB | Aggressive LM4 × E16-row tier. |
| [`Qwen3.8-27B-PXQ3-GGUF.gguf`](https://huggingface.co/mistrjirka/Qwen3.8-27B-PXQ-GGUF/resolve/main/Qwen3.8-27B-PXQ3-GGUF.gguf?download=true) | PXQ3 | ~3.27 bpw | 12.44 GB | Lower-bit LM8 bit-plane × E16-row tier. |
| [`Qwen3.8-27B-PXQ4-GGUF.gguf`](https://huggingface.co/mistrjirka/Qwen3.8-27B-PXQ-GGUF/resolve/main/Qwen3.8-27B-PXQ4-GGUF.gguf?download=true) | PXQ4 | ~4.27 bpw | 15.46 GB | Recommended core 4-bit PX16 × E16-row tier. |
| [`Qwen3.8-27B-PXQ4-HQ-GGUF.gguf`](https://huggingface.co/mistrjirka/Qwen3.8-27B-PXQ-GGUF/resolve/main/Qwen3.8-27B-PXQ4-HQ-GGUF.gguf?download=true) | PXQ4-HQ | ~4.52 bpw | 16.22 GB | Higher-fidelity PXQ4 with finer bs8 sub-scales. |
| [`Qwen3.8-27B-PXQ6-GGUF.gguf`](https://huggingface.co/mistrjirka/Qwen3.8-27B-PXQ-GGUF/resolve/main/Qwen3.8-27B-PXQ6-GGUF.gguf?download=true) | PXQ6 | ~5.27 bpw | 18.48 GB | Quality-first 5-bit LM32 × E16-row tier. |

All model filenames explicitly contain `GGUF`.

## Validation and publishing

Each artifact is processed independently. Publishing is pipelined with a maximum of **one upload + one quantization at a time**:

1. dry-run and named-tier composition check
2. full quantization
3. GGUF reopen/type census
4. SHA256
5. upload using the precomputed SHA256; while this uploads, the next tier may quantize
6. remote Hugging Face size + LFS SHA256 verification
7. local quantized GGUF deletion only after successful remote verification

The pipeline is disk-bounded: it never intentionally gets more than one uploading quant plus one next quant ahead.

PXA revision used by this queue job: `25d34eeab1a418fd21411158400a40d057ab063f`.

## Skipped tiers

- **PXQ1**: dry-run produced zero pxq1 tensors; refusing a misleading PXQ1 file
