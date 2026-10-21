---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
---

# Qwen3.8-27B IQ4_XS 量化模型（适配 16GB 显存）

本模型基于 **Qwen3.8-27B** 进行 **IQ4_XS** 量化（4‑bit），文件体积为 **12.9 GiB**，专为 **16GB 显存** 的显卡优化，在保持较低困惑度的同时，兼顾推理速度和显存占用。

与同体积的 **UD_IQ3_K_XL**（12.5 GiB）量化方案进行了全面对比，评估指标如下。

---

## 📊 量化质量对比

| 评估指标 | **IQ4_XS (本模型)** | **UD_IQ3_K_XL (对比)** |
|----------|---------------------|------------------------|
| **文件大小** | **12.9 GB** | 12.5 GB |
| **量化精度** | IQ4_XS (4‑bit) | UD_IQ3_K_XL (约 3‑bit?) |
| **量化模型困惑度 (Mean PPL)** | 7.0875 ± 0.0459 | 7.1117 ± 0.0459 |
| **与基座模型 PPL 相关性** | 99.28% | 99.31% |
| **平均 KL 散度 (Mean KLD)** | 0.03168 ± 0.00029 | 0.03130 ± 0.00032 |
| **最大 KL 散度 (Max KLD)** | **12.089**（更小） | 21.409 |
| **99.9% KL 分位数** | 1.134 | 1.219 |
| **Top‑1 一致率 (Same top p)** | 91.845% ± 0.071% | 92.419% ± 0.069% |
| **平均概率变化 (Mean Δp)** | **-0.381% ± 0.013%**（更接近 0） | -0.738% ± 0.013% |
| **RMS 概率变化 (RMS Δp)** | **4.893% ± 0.041%**（更小） | 5.120% ± 0.046% |

在不启用MTP的情况下可以做到16Gib净空VRAM（不作为Windows的显示显卡）的情况下110k上下文
开启MTP大概80k上下文


# Qwen3.8-27B IQ4_XS Quantized Model (Optimized for 16GB VRAM)

This model is a **IQ4_XS** quantization (4‑bit) of **Qwen3.8-27B**. The file size is **12.9 GB**, and it is specifically optimized for graphics cards with **16GB VRAM**, offering a good balance between low perplexity, inference speed, and memory usage.

We have conducted a comprehensive comparison with the **UD_IQ3_K_XL** quantization scheme (12.5 GB) of similar size. The evaluation metrics are shown below.

---

## 📊 Quantization Quality Comparison

| Metric | **IQ4_XS (This Model)** | **UD_IQ3_K_XL (Comparison)** |
|--------|--------------------------|-------------------------------|
| **File Size** | **12.9 GB** | 12.5 GB |
| **Quantization Precision** | IQ4_XS (4‑bit) | UD_IQ3_K_XL (~3‑bit?) |
| **Quantized Model Perplexity (Mean PPL)** | 7.0875 ± 0.0459 | 7.1117 ± 0.0459 |
| **Correlation with Base PPL** | 99.28% | 99.31% |
| **Mean KL Divergence (Mean KLD)** | 0.03168 ± 0.00029 | 0.03130 ± 0.00032 |
| **Maximum KL Divergence (Max KLD)** | **12.089** (smaller) | 21.409 |
| **99.9% KLD Quantile** | 1.134 | 1.219 |
| **Top‑1 Agreement (Same top p)** | 91.845% ± 0.071% | 92.419% ± 0.069% |
| **Mean Probability Change (Mean Δp)** | **-0.381% ± 0.013%** (closer to 0) | -0.738% ± 0.013% |
| **RMS Probability Change (RMS Δp)** | **4.893% ± 0.041%** (smaller) | 5.120% ± 0.046% |

---

### Context Length with 16GB VRAM

- **Without MTP** (Multi‑Token Prediction): Can achieve a context length of **110k** tokens when there is 16GB free VRAM (and the GPU is not used as the primary display adapter for Windows).
- **With MTP enabled**: Approximately **80k** tokens.
