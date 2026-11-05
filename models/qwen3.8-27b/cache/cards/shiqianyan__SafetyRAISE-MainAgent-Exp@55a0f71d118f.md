---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language:
- zh
tags:
- gguf
- llama.cpp
- experimental
---

# SafetyRAISE-MainAgent-Exp

这是一个**实验仓库**，存放 SafetyRAISE 主智能体实验产出的量化模型（GGUF）。按实验分子目录，每个子目录里有模型文件和配套的运行文件（聊天模板、思考预算提示语、启动示例），细节见各子目录的 README。

| 目录 | 内容 |
| --- | --- |
| `report-exp-v1/` | 报告实验第一版：Qwen3.8-27B 经 LoRA 微调与偏好优化后合并量化（Q4_K_M），附重训的 MTP 头 |
| `report-exp-v2/` | 报告实验第二版：在第一版权重上做 MLP 方向编辑后重新量化（Q4_K_M），附在编辑后模型上重训的 MTP 头 |

**暂不开放原始权重**：LoRA 适配器、单独的 MTP 头权重、编辑参数和全精度权重都不在这里。

训练与评测代码、数据说明和技术报告见 GitHub 仓库：<https://github.com/QianYan-Art/SafetyRAISE-Lora>

按 Apache License 2.0 发布（见 `LICENSE`、`NOTICE`），与基座 Qwen3.8-27B 相同。
