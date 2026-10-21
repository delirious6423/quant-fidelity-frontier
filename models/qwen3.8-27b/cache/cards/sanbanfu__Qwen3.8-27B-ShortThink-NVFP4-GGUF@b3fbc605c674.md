---
license: apache-2.0
language:
  - zh
  - en
tags:
  - qwen3.5
  - qwen3.8
  - short-thinking
  - efficient-reasoning
  - nvfp4
  - quantization
  - gguf
  - rtx5090
library_name: llama.cpp
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
---

# Qwen3.8-27B-ShortThink

> 短思考版 Qwen3.8-27B —— 保留推理质量，压缩思考长度
> Short-thinking Qwen3.8-27B — keep reasoning quality, compress thinking length

## 简介 / Overview

**ShortThink** 是一个基于 Qwen3.8-27B 的短思考后训练版本（post-training）。

目标是解决 Qwen3.8 的一个痛点：**xhigh 模式思考质量高但太长，medium 模式质量下降明显**。


**ShortThink** is a short-thinking post-trained version of Qwen3.8-27B.

It addresses a pain point of Qwen3.8: **xhigh reasoning is high-quality but verbose; medium drops quality noticeably**.


## 量化方案 / Quantization

MLP → **NVFP4**（4-bit，MSE 校准 128 条，W4A16，Blackwell FP4 张量核心加速）
Attention → **Q4_K / Q5_K / Q6_K / Q8_0**（多档可选）
lm_head / embeddings / MTP 末层 → **Q8_0**（保两头质量）
GDN 核心（ssm_alpha/beta）→ **F16**（注意力质量关键）

| 文件 | 大小 | Attention | 定位 |
|:----|:----|:----|:----|
| `Qwen3.8-27B-ShortThink-NVFP4-MLP-Q4K-attn.gguf` | 15.9 GB | Q4_K | 🚀 最小最快 |
| `Qwen3.8-27B-ShortThink-NVFP4-MLP-Q5K-attn.gguf` | 16.6 GB | Q5_K | 速度优先 |
| `Qwen3.8-27B-ShortThink-NVFP4-MLP-Q6K-attn.gguf` | 17.4 GB | Q6_K | ⚖️ 平衡推荐 |
| `Qwen3.8-27B-ShortThink-NVFP4-MLP-Q8_0-attn.gguf` | 19.1 GB | Q8_0 | 质量优先 |
| `mmproj-F16.gguf` | 0.86 GB | — | 视觉投影（多模态用） |

MLP → **NVFP4** (4-bit, MSE-calibrated 128 samples, W4A16, Blackwell FP4 tensor core)
Attention → **Q4_K / Q5_K / Q6_K / Q8_0** (multiple tiers)
lm_head / embeddings / MTP last layer → **Q8_0** (preserve quality at both ends)
GDN core (ssm_alpha/beta) → **F16** (critical for attention quality)

## 使用 / Usage

### llama.cpp

```bash
# 文本推理
llama-server -m Qwen3.8-27B-ShortThink-NVFP4-MLP-Q6K-attn.gguf \
  --jinja -c 16384 -ngl 99 -fa on

# 多模态推理（加视觉投影）
llama-server -m Qwen3.8-27B-ShortThink-NVFP4-MLP-Q6K-attn.gguf \
  --mmproj mmproj-F16.gguf --jinja -c 16384 -ngl 99 -fa on
```

支持 `reasoning_effort` 档位（low/medium/high/xhigh），与 Qwen3.8 官方行为一致。

### 硬件要求

- **推荐**：RTX 5090 / 5080（Blackwell，NVFP4 硬件加速）
- **兼容**：任何支持 NVFP4 的 llama.cpp 构建（≥ v10360）

## 效果 / Results

- 思考长度较 xhigh 平均压缩约 **40-50%**（编程/数学/常识多领域）
- 答案正确性经测试用例验证，与 xhigh 持平
- 适用场景：编程、数学推理、日常问答（中文/英文）

## 致谢与许可 / Credits & License

- **Base**: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0)
- **Quantization**: NVIDIA ModelOpt + llama.cpp

本项目基于 Apache-2.0 许可发布。使用 Qwen3.8-27B 请遵守其 Apache-2.0 许可。
Published under Apache-2.0. Base model Qwen3.8-27B is Apache-2.0 licensed.

---

