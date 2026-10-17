---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
---
---
## Qwen 3.8 27B Heretic‑Ara IQ4_XS 量化模型（适配 16GB 显存）

本模型基于 **Qwen 3.8 27B Heretic‑Ara BF16** 进行 IQ4_XS 量化（4‑bit），文件体积为 **12.7-13.3 GiB**，专为 16GB 显存的显卡优化

8/22 更新修复思考的问题，模型性能没有变化
大概还需要1天我会更新MTP的版本

8/22 Updated and fixed the issues with thinking; model performance remains unchanged
It will probably take about another day for me to update the MTP version.

8/23 更新，对模型本身的性能进行一定优化，体积略微变大，无MTP版本13Gib，MTP版本13.3Gib
8/23 Update: The model's performance has been slightly optimized, resulting in a slight increase in size
The non-MTP version is 13GiB, while the MTP version is 13.3GiB



与同体积的 **Heretic‑Ara‑Q3_K_M**（12.4 GiB）量化方案进行了全面对比

本模型使用Heretic Arbitrary-Rank Ablation做到的无审查

📊 量化质量对比

| 评估指标 | Heretic-Ara BF16 (base) | IQ4_XS-3.0 | IQ4_XS-2.0| Heretic-Ara-Q3_K_M|
|----------|-------------------------|-----|---------------------|---------------------------|
| **文件大小** | 50.1 GiB | 13 GiB (带MTP 13.3 GiB) | 12.7 GiB | 12.4 GiB |
| **量化精度** | BF16 | IQ4_XS (4‑bit) | IQ4_XS (4‑bit) | Q3_K_M (约 3‑bit) |
| **模型困惑度 (Mean PPL)** | 7.008212 ± 0.045362 | 7.046980 ± 0.045498 | 7.102940 ± 0.046017 | 7.403971 ± 0.048924 |
| **与基座模型 PPL 相关性** | 100% | 99.34% | 99.26% | 98.31% |
| **平均 KL 散度 (Mean KLD)** | 0 | 0.027832 ± 0.000324 | 0.033398 ± 0.000308 | 0.076034 ± 0.000554 |
| **最大 KL 散度 (Max KLD)** | 0 | 18.317436 | 15.094215 | 17.866985 |
| **99.9% KL 分位数** | 0 | 1.162850 | 1.130034 | 2.448278 |
| **Top‑1 一致率 (Same top p)** | 100% | 92.867% ± 0.067% | 91.619% ± 0.072% | 88.152% ± 0.084% |
| **平均概率变化 (Mean Δp)** | 0% | -0.243% ± 0.012% | -0.306% ± 0.013% | -0.490% ± 0.020% |
| **RMS 概率变化 (RMS Δp)** | 0% | 4.538% ± 0.045% | 4.952% ± 0.041% | 7.560% ± 0.054% |

> *注：基座（BF16）的 KL 散度、Δp 等指标均为 0（自身对比），一致率为 100%。*

在不启用 MTP 的情况下，IQ4_XS 模型在 16 GiB 无显存占用（不作为 Windows 显示显卡）下可支持约 **110k** 上下文

开启 MTP 后约为 **80k**




Qwen 3.8 27B Heretic‑Ara IQ4_XS Quantized Model (Optimized for 16 GB VRAM)
This model is quantized from Qwen 3.8 27B Heretic‑Ara BF16 using the IQ4_XS scheme (4‑bit), with a file size of 12.8 GiB, specifically designed for graphics cards with 16 GB of VRAM.

It has been comprehensively compared against the similarly sized Heretic‑Ara‑Q3_K_M (12.4 GiB) quantization variant.

This model achieves uncensored behavior through Heretic's Arbitrary-Rank Ablation.

📊 Quantization Quality Comparison

| Evaluation Metric | Heretic-Ara BF16 (base) | IQ4_XS-3.0 | IQ4_XS-2.0| Heretic-Ara-Q3_K_M (comparison) |
|-------------------|-------------------------|-----|-------------------------|--------------------------------|
| **File Size** | 50.1 GiB | 13 GiB (with MTP 13.3 GiB) | 12.7 GiB | 12.4 GiB |
| **Quantization Precision** | BF16 | IQ4_XS (4‑bit) | IQ4_XS (4‑bit) | Q3_K_M (~3‑bit) |
| **Model Perplexity (Mean PPL)** | 7.008212 ± 0.045362 | 7.046980 ± 0.045498 | 7.102940 ± 0.046017 | 7.403971 ± 0.048924 |
| **PPL Correlation with Base Model** | 100% | 99.34% | 99.26% | 98.31% |
| **Mean KL Divergence (Mean KLD)** | 0 | 0.027832 ± 0.000324 | 0.033398 ± 0.000308 | 0.076034 ± 0.000554 |
| **Max KL Divergence (Max KLD)** | 0 | 18.317436 | 15.094215 | 17.866985 |
| **99.9% KL Quantile** | 0 | 1.162850 | 1.130034 | 2.448278 |
| **Top‑1 Agreement Rate (Same top p)** | 100% | 92.867% ± 0.067% | 91.619% ± 0.072% | 88.152% ± 0.084% |
| **Mean Probability Change (Mean Δp)** | 0% | -0.243% ± 0.012% | -0.306% ± 0.013% | -0.490% ± 0.020% |
| **RMS Probability Change (RMS Δp)** | 0% | 4.538% ± 0.045% | 4.952% ± 0.041% | 7.560% ± 0.054% |

> *Note: For the base model (BF16), KL divergence, Δp, etc. are all 0 (self‑comparison), and the agreement rate is 100%.*

With MTP (Multi‑Token Prediction) disabled, the IQ4_XS model supports approximately 110k context length on a 16 GiB GPU with no VRAM reserved for display (i.e., not used as the primary display adapter).
With MTP enabled, the context length is approximately 80k.