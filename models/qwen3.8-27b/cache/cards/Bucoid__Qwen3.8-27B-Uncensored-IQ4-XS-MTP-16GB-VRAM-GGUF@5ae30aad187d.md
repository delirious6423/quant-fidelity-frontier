---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
---

此模型可能短期更新，我发布了一个基于Heretic Arbitrary-Rank Ablation的无审查版本模型，性能更好且体积更小
链接：
https://huggingface.co/Bucoid/Qwen3.8-27B-Heretic-Ara-IQ4-XS-16GB-VRAM-GGUF
这个模型可能过一段时间我会更新让他不那么菜，如果你需要无审查版本的模型，基于下载这个Ara的

This model may receive short-term updates. 
I have released an uncensored version based on Heretic Arbitrary-Rank Ablation,
which offers better performance and a smaller file size.
Link: https://huggingface.co/Bucoid/Qwen3.8-27B-Heretic-Ara-IQ4-XS-16GB-VRAM-GGUF
This model may be updated in a while to make it less underwhelming
If you need an uncensored version, please download this Ara-based one instead.



Qwen3.8-27B Uncensored IQ4_XS 量化模型（适配 16GB 显存）
本模型基于 Qwen3.8-27B Uncensored 进行 IQ4_XS 量化（4‑bit），文件体积为 12.9 GiB，专为 16GB 显存 的显卡优化，在保持较低困惑度的同时，兼顾推理速度和显存占用。

与同体积的 UD_IQ3_K_XL（12.5 GiB）量化方案进行了全面对比，评估指标如下。

📊 量化质量对比

| 评估指标 | IQ4_XS (本模型) | UD_IQ3_K_XL (对比) |
|----------|----------------|---------------------|
| 文件大小 | 12.9 GB | 12.5 GB |
| 量化精度 | IQ4_XS (4‑bit) | UD_IQ3_K_XL (约 3‑bit?) |
| 量化模型困惑度 (Mean PPL) | 7.1481 ± 0.0465 | 7.1117 ± 0.0459 |
| 与基座模型 PPL 相关性 | 99.28% | 99.31% |
| 平均 KL 散度 (Mean KLD) | 0.03268 ± 0.00030 | 0.03130 ± 0.00032 |
| 最大 KL 散度 (Max KLD) | 16.017（更小） | 21.409 |
| 99.9% KL 分位数 | 1.075 | 1.219 |
| Top‑1 一致率 (Same top p) | 91.655% ± 0.072% | 92.419% ± 0.069% |
| 平均概率变化 (Mean Δp) | -0.343% ± 0.013%（更接近 0） | -0.738% ± 0.013% |
| RMS 概率变化 (RMS Δp) | 4.986% ± 0.039%（更小） | 5.120% ± 0.046% |

在不启用MTP的情况下可以做到16GiB净空VRAM（不作为Windows的显示显卡）的情况下110k上下文

开启MTP大概80k上下文。


---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
---
Qwen3.8-27B Uncensored IQ4_XS Quantized Model (Optimized for 16GB VRAM)
This model is based on Qwen3.8-27B Uncensored and quantized with IQ4_XS (4‑bit), with a file size of 12.9 GiB. It is tailored for GPUs with 16GB VRAM, balancing low perplexity, inference speed, and memory usage.

We conducted a comprehensive comparison against the UD_IQ3_K_XL quantization scheme (12.5 GiB, roughly 3‑bit) of the same model size. The evaluation metrics are as follows.

📊 Quantization Quality Comparison

| Metric | IQ4_XS (this model) | UD_IQ3_K_XL (baseline) |
|--------|---------------------|-------------------------|
| File size | 12.9 GB | 12.5 GB |
| Quantization precision | IQ4_XS (4‑bit) | UD_IQ3_K_XL (~3‑bit) |
| Mean perplexity (quantized) | 7.1481 ± 0.0465 | 7.1117 ± 0.0459 |
| Correlation with base model PPL | 99.28% | 99.31% |
| Mean KL divergence | 0.03268 ± 0.00030 | 0.03130 ± 0.00032 |
| Maximum KL divergence | 16.017 (lower) | 21.409 |
| 99.9% KL quantile | 1.075 | 1.219 |
| Top‑1 agreement rate | 91.655% ± 0.072% | 92.419% ± 0.069% |
| Mean probability change (Mean Δp) | -0.343% ± 0.013% (closer to 0) | -0.738% ± 0.013% |
| RMS probability change (RMS Δp) | 4.986% ± 0.039% (lower) | 5.120% ± 0.046% |

With MTP disabled, the model can achieve ~110k context length while keeping ~16 GiB free VRAM (when not used as the primary display GPU on Windows). With MTP enabled, the context length is around 80k.