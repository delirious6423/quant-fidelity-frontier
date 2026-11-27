---
license: llama3.1
base_model:
- meta-llama/Llama-3.1-8B-Instruct
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- ik_llama.cpp
- imatrix
---


**These GGUF files load only in [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) built with the IQ4_NT prototype PR ([2978](https://github.com/ikawrakow/ik_llama.cpp/pull/2978))!**

**Note that you need to download the 'v2' files for the current version of the format in the PR linked above.**

IQ4_NT is a 4-bit nibble-trellis prototype. The PR description has the format details and the measurements.


## Qwen3.8-27B

| file | size | contents |
|---|---|---|
| `Qwen3.8-27B-IQ4_NT-v2.gguf` | 14.40 GB (14,401,017,440 B), 4.22 bits per weight overall | **format v2**: 487 tensors IQ4_NT (480 with exact-column companions), `attn_v` IQ5_K, `output` Q6_K, `token_embd` IQ4_KS |
| `Qwen3.8-27B-IQ4_NT.gguf` | 14.48 GB (14,477,520,512 B), 4.24 bits per weight overall | format v1 (superseded): 487 tensors IQ4_NT, `attn_v` IQ5_K, `output` Q6_K, `token_embd` IQ4_KS |
| `qwen38-27b-agentic.imatrix` | 13.6 MB | Joel Farthing's imatrix from [ji-farthing/Qwen3.8-27B-ik-llama-GGUF](https://huggingface.co/ji-farthing/Qwen3.8-27B-ik-llama-GGUF/blob/main/qwen38-27b-agentic.imatrix) Thanks!|

## Llama-3.1-8B-Instruct

| file | size | contents |
|---|---|---|
| `Llama-3.1-8B-Instruct-IQ4_NT-v2.gguf` | 4.31 GB (4,309,272,800 B), 4.29 bits per weight overall | **format v2**: 192 tensors IQ4_NT (all with exact-column companions), `attn_v` IQ5_K, `output` Q6_K, `token_embd` IQ4_KS |
| `Llama-3.1-8B-Instruct-IQ4_NT.gguf` | 4.34 GB (4,336,245,120 B), 4.32 bits per weight overall | format v1 (superseded): 192 tensors IQ4_NT, `attn_v` IQ5_K, `output` Q6_K, `token_embd` IQ4_KS |
| `imatrix_unsloth.dat` | 5.0 MB | imatrix by Unsloth from [unsloth/Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct-GGUF) |

## How to test

```
git clone https://github.com/ikawrakow/ik_llama.cpp && cd ik_llama.cpp
git fetch origin pull/2978/head:iq4-nt && git checkout iq4-nt
cmake -B build -DGGML_NATIVE=ON                                                     # AVX-512 VNNI CPU only
cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89"   # with Ampere/Ada GPU
cmake --build build --config Release -j
```
