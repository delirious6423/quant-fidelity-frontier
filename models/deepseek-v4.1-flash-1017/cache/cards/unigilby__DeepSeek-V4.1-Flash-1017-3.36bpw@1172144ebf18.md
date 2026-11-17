---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - tensorfold
  - apple-silicon
  - metal
  - quantized
  - mixed-precision
  - moe
---

# DeepSeek-V4.1-Flash-1017-3.36bpw

Per-tensor, imatrix-weighted mixed-bit quantization of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (revision `1bae0e1088993b9b82de08f59de5be5a34e42170`) sized to run on a **256 GB Apple Silicon Mac** (such as an M5 Ultra 256 GB) with a full **1,048,576-token** context window, the DSpark MTP drafter, and the vision tower loaded.

## Highlights

- **3.36 bpw** across the GPU-wired backbone (`216.14 GiB`, `551.88B` weights; `3.36 bpw` / `215.56 GiB` across the 46,429 re-quantizable matrices, down from `4.33 bpw` in the release checkpoint)
- **3.31 bpw** stored across the 46,080 routed expert matrices (`209.60 GiB`, `543.58B` weights) — **3.77 bpw active** when weighted by calibration routing frequency, because 56.8% of routed token traffic hits experts kept in lossless release MXFP4
- **3.40 bpw** across all 47 `.safetensors` shards (`224.44 GiB` of tensors, `240.98 GB` on disk with 256-byte tensor alignment, including the unwired `7.39 GiB` DSpark drafter and `0.90 GiB` vision tower)

## Quantization Method

1. **Calibration:** 800,239-token importance matrix and per-expert routing counts from [`unsloth/DeepSeek-V4.1-Flash-1017-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (`imatrix_unsloth.gguf_file`).
2. **Bit allocation (`bitmap.json`):** Multiple-choice knapsack optimization over exact GPU-measured imatrix-weighted relative error across all 46,429 re-quantizable matrices under a `215.555 GiB` wired-weight budget.
3. **Affine fit:** Re-quantized matrices use MLX-style LSB-packed unsigned codes (`weight = scale * q + bias` with `BF16` `scales` and `biases` at group size 64 or 32), initialized with min-max and refined with two imatrix-weighted least-squares refits. All `keep` tensors are byte-identical copies from `deepseek-ai/DeepSeek-V4.1-Flash-1017`.
4. **Layout (`tensorfold-dsv41-mixedbit` v1):** Documented in `tensorfold_mixedbit.json` and `config.json` (`quantization_config.quant_method = "tensorfold_mixedbit"`). Every tensor starts at a 256-byte-aligned offset for zero-copy Metal buffer mapping, and shards break cleanly before `mtp.*` and `vision.*` so drafter and vision weights are not wired into the backbone residency set.

### Per-Component Breakdown

| Component | Weights | Release Format | Mixed-Bit Assignment | Avg bpw | Size (GiB) |
| :--- | ---: | :--- | :--- | ---: | ---: |
| **Routed experts** (`w1, w2, w3`) | 543.58B | MXFP4 (4.25 bpw) | 12,795 `keep` MXFP4 · 15,030 `a3g64` · 18,246 `a2g64` · 9 `a2g32` | **3.312** *(3.775 active)* | 209.60 |
| **Attention** | 5.13B | FP8 / BF16 | 100 `keep` · 120 `a6g64` · 4 `a8g64` · 3 `a5g64` | **6.620** | 3.95 |
| **Shared experts** | 1.42B | FP8 | 116 `keep` FP8 · 4 `a6g64` | **7.958** | 1.31 |
| **`embed.weight` + `head.weight`** | 1.32B | BF16 | 2 `a4g64` | **4.500** | 0.69 |
| **Kept backbone** (Engram dense, routers, norms, `hc_*`) | 0.43B | FP8 / BF16 / F32 | `keep` | **11.648** | 0.59 |
| **DSpark drafter (`mtp.*`, unwired)** | 14.23B | MXFP4 / FP8 / BF16 / F32 | `keep` | **4.461** | 7.39 |
| **Vision tower (unwired)** | 0.49B | BF16 | `keep` | **16.000** | 0.90 |

## Usage Notes (TensorFold Metal)

- **Engram tables:** The ~189 GB host-resident Engram lookup tables (`.engram.embed.*`) are unchanged from `deepseek-ai/DeepSeek-V4.1-Flash-1017` and are not duplicated here (`"engram_tables": "release"` in `tensorfold_mixedbit.json`). Set `TF_DSV41_ENGRAM_RELEASE` to the local `deepseek-ai/DeepSeek-V4.1-Flash-1017` directory and `TF_DSV41_ENGRAM_DIR` to the packed Engram directory.
- **256 GB unified memory setup:** Raise the macOS GPU wired limit (`sudo sysctl iogpu.wired_limit_mb=249036`) and pass `DSV41_LOAD_LIMIT_GB=235` when serving so the `216.5 GiB` backbone residency set stays wired above the default 70%-of-RAM threshold.

## Validation & Performance

- **Logit fidelity vs. release** (teacher-forced across 38 prompts, M3 Ultra 512 GB): mean **KL = 0.0789**, mean **top-1 agreement = 94.1%**.
- **Quality checks vs. release** (same binary, greedy, M5 Ultra 512 GB): MMLU-200 **86.5%** (`173/200`) vs. **89.0%** (`178/200`) release; single-tool function calling **20/20**; long-context needle recall **9/9** across 32K, 131K, and 500K, plus verified at **1,011,610 tokens** on an M5 Ultra 256 GB.
- **Throughput** (M5 Ultra 256 GB, TensorFold Metal with DSpark speculative decoding, temperature 0, 1 stream): **85.7 tok/s** code, **68.1 tok/s** chat (`2.3 s` warm load).

## License & Attribution

Inherits the MIT license of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). Importance matrix calibration statistics derived from [`unsloth/DeepSeek-V4.1-Flash-1017-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF).
