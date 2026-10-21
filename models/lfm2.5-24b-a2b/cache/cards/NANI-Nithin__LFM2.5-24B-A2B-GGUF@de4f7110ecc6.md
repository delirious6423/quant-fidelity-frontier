---
base_model: LiquidAI/LFM2.5-24B-A2B
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - quantized
---

# LFM2.5-24B-A2B GGUF

GGUF quantizations of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), covering 31 files (365.9 GB total).

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| [LFM2.5-24B-A2B-BF16.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-BF16.gguf) | `BF16` | 44.67 GB | Full precision source. Every quant below is cut from this file. |
| [LFM2.5-24B-A2B-Q8_0.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q8_0.gguf) | `Q8_0` | 23.74 GB | Effectively lossless. Use when disk and RAM are not the constraint. |
| [LFM2.5-24B-A2B-Q6_K.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q6_K.gguf) | `Q6_K` | 18.34 GB | Near-lossless; the last stop before quality becomes measurable. |
| [LFM2.5-24B-A2B-Q5_K_M.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q5_K_M.gguf) | `Q5_K_M` | 15.86 GB | Very good quality, noticeably smaller than Q6_K. |
| [LFM2.5-24B-A2B-Q5_K_S.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q5_K_S.gguf) | `Q5_K_S` | 15.41 GB | Slightly smaller than Q5_K_M for a slight quality cost. |
| [LFM2.5-24B-A2B-Q5_1.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q5_1.gguf) | `Q5_1` | 16.78 GB | Legacy. Prefer Q5_K_M. |
| [LFM2.5-24B-A2B-Q5_0.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q5_0.gguf) | `Q5_0` | 15.41 GB | Legacy. Prefer Q5_K_M. |
| [LFM2.5-24B-A2B-Q4_K_M.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q4_K_M.gguf) | `Q4_K_M` | 13.53 GB | The usual default. Best quality-per-byte for most people. |
| [LFM2.5-24B-A2B-Q4_K_S.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q4_K_S.gguf) | `Q4_K_S` | 12.72 GB | A little smaller than Q4_K_M, a little worse. |
| [LFM2.5-24B-A2B-IQ4_NL.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ4_NL.gguf) | `IQ4_NL` | 12.65 GB | Non-linear 4-bit; good on hardware without fast K-quant kernels. |
| [LFM2.5-24B-A2B-IQ4_XS.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ4_XS.gguf) | `IQ4_XS` | 11.96 GB | Best sub-4.5bpw option; usually beats Q4_K_S at a smaller size. |
| [LFM2.5-24B-A2B-Q4_1.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q4_1.gguf) | `Q4_1` | 14.03 GB | Legacy. Prefer Q4_K_M. |
| [LFM2.5-24B-A2B-Q4_0.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q4_0.gguf) | `Q4_0` | 12.65 GB | Legacy round-to-nearest. Prefer Q4_K_M unless a runtime needs this. |
| [LFM2.5-24B-A2B-MXFP4_MOE.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-MXFP4_MOE.gguf) | `MXFP4_MOE` | 12.39 GB | MoE-only 4-bit microscaling format for the expert tensors. |
| [LFM2.5-24B-A2B-Q3_K_L.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q3_K_L.gguf) | `Q3_K_L` | 11.57 GB | Small, with real quality loss. Usable when RAM is tight. |
| [LFM2.5-24B-A2B-Q3_K_M.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q3_K_M.gguf) | `Q3_K_M` | 10.68 GB | Smaller again; noticeable degradation. |
| [LFM2.5-24B-A2B-IQ3_M.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ3_M.gguf) | `IQ3_M` | 9.80 GB | Strong at ~3.7bpw, clearly better than Q3_K_M. |
| [LFM2.5-24B-A2B-IQ3_S.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ3_S.gguf) | `IQ3_S` | 9.71 GB | Slightly smaller than IQ3_M. |
| [LFM2.5-24B-A2B-Q3_K_S.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q3_K_S.gguf) | `Q3_K_S` | 9.71 GB | Aggressive. Prefer IQ3_M at a similar size. |
| [LFM2.5-24B-A2B-IQ3_XS.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ3_XS.gguf) | `IQ3_XS` | 9.18 GB | Aggressive but coherent. |
| [LFM2.5-24B-A2B-IQ3_XXS.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ3_XXS.gguf) | `IQ3_XXS` | 8.65 GB | Very aggressive; the last coherent step down. |
| [LFM2.5-24B-A2B-Q2_K.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q2_K.gguf) | `Q2_K` | 8.20 GB | Very small, heavily degraded. For experimentation. |
| [LFM2.5-24B-A2B-IQ2_M.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ2_M.gguf) | `IQ2_M` | 2.02 GB | The smallest size most people find usable. |
| [LFM2.5-24B-A2B-Q2_K_S.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q2_K_S.gguf) | `Q2_K_S` | 7.61 GB | Smaller than Q2_K, at a further quality cost. |
| [LFM2.5-24B-A2B-IQ2_S.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ2_S.gguf) | `IQ2_S` | 6.67 GB | Below the usual usability line. |
| [LFM2.5-24B-A2B-IQ2_XS.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ2_XS.gguf) | `IQ2_XS` | 6.60 GB | Experimental. |
| [LFM2.5-24B-A2B-IQ2_XXS.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ2_XXS.gguf) | `IQ2_XXS` | 5.93 GB | Experimental. |
| [LFM2.5-24B-A2B-Q2_0.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q2_0.gguf) | `Q2_0` | 6.44 GB | Extreme, group-64. Included for completeness. |
| [LFM2.5-24B-A2B-IQ1_M.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ1_M.gguf) | `IQ1_M` | 5.10 GB | Extreme. Expect substantial degradation. |
| [LFM2.5-24B-A2B-IQ1_S.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ1_S.gguf) | `IQ1_S` | 4.59 GB | Extreme. Expect substantial degradation. |
| [LFM2.5-24B-A2B-Q1_0.gguf](https://huggingface.co/NANI-Nithin/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q1_0.gguf) | `Q1_0` | 3.33 GB | Extreme. Included for completeness. |

## Which one should I download?

Pick the largest file that leaves a couple of gigabytes of headroom on the device you will run it on — the model has to fit in RAM (or VRAM, if you are offloading) alongside the KV cache and the OS.

- Plenty of memory: **Q6_K** or **Q8_0**.
- The usual choice: **Q4_K_M**.
- Tight on memory: **IQ4_XS**, then **IQ3_M**, then **IQ2_M**.
- The `IQ*` files generally beat a `Q*` file of similar size, at the cost of slightly slower inference on some hardware.

## Usage

```bash
llama-cli -hf NANI-Nithin/LFM2.5-24B-A2B-GGUF:Q4_K_M -p "Hello"
```

Or download one file and point at it directly:

```bash
huggingface-cli download NANI-Nithin/LFM2.5-24B-A2B-GGUF LFM2.5-24B-A2B-Q4_K_M.gguf --local-dir .
llama-cli -m LFM2.5-24B-A2B-Q4_K_M.gguf -p "Hello"
```

---

Quantized with [llama.cpp](https://github.com/ggml-org/llama.cpp) by AgentQuantix on 2026-10-14. Licensing follows the base model.
