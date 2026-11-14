---
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- gemma4
- local-model-access
---

# Gemma 4 26B-A4B-it, UD-IQ4_XS, for Local Model Access

This repository holds one model file: Unsloth's UD-IQ4_XS quantization of Google's Gemma 4 26B-A4B-it, in the exact bytes on which [Local Model Access](https://github.com/ArtistJoshHughes/local-model-access) (LMA) 1.4 built and tested its whole-number form of the model.

| | |
|---|---|
| File | `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` |
| Size | 13,597,175,584 bytes |
| SHA-256 | `534e1bdfffdb62dff1d3636fedb41607a234b3b31abb7089947bac2aef00a960` |
| Model | Gemma 4 26B-A4B-it by Google, the text model: 30 layers, each with 128 experts, of which every token goes through 8 |
| Weights | Unsloth's UD-IQ4_XS mix of IQ3_S, IQ4_NL, IQ4_XS and Q8_0 (13.5 GB); norms, routers and scales in F32 |
| Licence | Apache 2.0 |

The vision projector is not included.

## Why a copy

LMA names a model file by its SHA-256 and size, so that every site that approves a model, and every device that runs it, has the same bytes. No file in Unsloth's repository has this SHA-256:

- Their file of this name from 4 May to 17 July 2026 ([revision b19ae87](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/blob/b19ae878a3d8d38352d69c370321f94ff98023a0/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf), SHA-256 `694a7d2a3b994655…`) has the same size, the same 60 entries in its header and the same table of 658 tensors at the same offsets. Six of those entries come in another order.
- Their current file (SHA-256 `babd1e389d386352…`) also has a newer chat template in its header.

This repository keeps the tested bytes at a fixed revision. The file is a standard GGUF: llama.cpp, Ollama and other GGUF tools read it as they read Unsloth's.

## With Local Model Access

LMA lets websites use a model file that the visitor keeps on their own device. Its whole-number form of this model ([SPEC.md](https://github.com/ArtistJoshHughes/local-model-access/blob/main/SPEC.md), section 18) gives the same tokens, and the same numbers in every layer, on every device and graphics card that passes its checks, so a site can check text jobs done on visitors' computers to the token (section 19). It came out the same to the byte on Chrome's software adapter, on NVIDIA graphics cards (Blackwell, Turing and Pascal) on Windows and Linux, and on Intel discrete (Battlemage) and integrated (Arrow Lake) GPUs on Linux. On chat answers in the model's own format, it picks the same next token as the model worked out in 64-bit floats 99.69% of the time.

It needs about 14.2 GB of graphics memory, so LMA gives its jobs only to computers that pass a gate, and never to phones or tablets. To approve the file, copy its entry from LMA's [registry/models.json](https://github.com/ArtistJoshHughes/local-model-access/blob/main/registry/models.json): it names this repository, the commit, the file, its SHA-256 and its size.

## Credits

- Gemma 4 26B-A4B-it: Google, [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), Apache 2.0.
- The UD-IQ4_XS quantization: Unsloth, [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF), Apache 2.0.
- Hosted for Local Model Access (LMA) by artist Josh Hughes.
