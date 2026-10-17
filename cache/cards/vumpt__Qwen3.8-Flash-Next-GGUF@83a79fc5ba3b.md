---
library_name: llama.cpp
license: other
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - qwen
  - qwen4-exp
  - moe
  - text-to-text
  - image-text-to-text
---

# Qwen3.8-Flash-Next — GGUF (Q4_K_M)

GGUF conversion of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), the Qwen4-experimental hybrid model (Gated DeltaNet + Qwen Sparse Attention + 512-expert MoE + n-gram "PLE" embedding table).

- **125B total MoE params (6B active)** + **51B n-gram table** + **4B MTP** = 180B params
- Converted with the in-flight [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) (`qwen4exp` architecture support)

## Quantization layout

`Q4_K_M` recipe. Because several tensor shapes in this architecture aren't divisible by 256, llama.cpp's type-fallback applies per-tensor (this is a feature of PR #27742's quantizer fixes, not an error):

| Component | Quant |
|---|---|
| MoE experts / attention / FFN | Q4_K_M (fallback to Q5_0/q8_0 where ncols % 256 ≠ 0) |
| Token embedding + output | Q6_K |
| N-gram (PLE) hash table (51B params) | Q5_0 (160-col layout → falls back from Q6_K) |

## Files

| File | Quant | Size |
|---|---|---|
| `qwen3.8-flash-next-Q4_K_M.gguf` | Q4_K_M | ~120 GB |

## Usage (llama.cpp)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch origin pull/27742/head:qwen4exp && git checkout qwen4exp
cmake -B build && cmake --build build -j --target llama-cli
./build/bin/llama-cli -m qwen3.8-flash-next-Q4_K_M.gguf -p "Hello" -ngl 99
```

> **Note:** This architecture is only supported on the PR #27742 branch. Mainline `llama.cpp` (as of this writing) does **not** load `qwen4_exp`. Use the branch above.

## Source

- Weights: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (bf16, 360 GB)
- License: [qwen-community-1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
