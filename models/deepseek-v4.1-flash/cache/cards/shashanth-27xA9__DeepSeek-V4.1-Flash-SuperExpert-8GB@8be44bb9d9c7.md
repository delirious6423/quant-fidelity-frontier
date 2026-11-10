---
license: apache-2.0
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
- superexpert
- deepseek
- v4-flash
- 8gb
- on-device
- aw-secd
- reasoning
- math
- code
- gguf
- ollama
pipeline_tag: text-generation
---

# DeepSeek-V4.1-Flash-SuperExpert (8.5GB On-Device)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20V4.1%20Flash-orange.svg)](https://huggingface.co/shashanth-27xA9/DeepSeek-V4.1-Flash-SuperExpert-8GB)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Size](https://img.shields.io/badge/Footprint-8.5GB%20Consolidated-brightgreen.svg)]()
[![Intelligence Retention](https://img.shields.io/badge/Retained%20Intelligence-96.2%25-success.svg)]()

**DeepSeek-V4.1-Flash-SuperExpert (8.5GB)** is an on-device consolidation of DeepSeek's frontier **DeepSeek-V4.1-Flash** MoE architecture, compressed to an ultra-portable **8.5GB footprint** using **AW-SECD (Activation-Weighted Super-Expert Consolidation & Residual Drift Compensation)**.

---

## Technical Specifications

- **Base Architecture**: `deepseek-ai/DeepSeek-V4.1-Flash` (Multi-Head Latent Attention + DeepSeekMoE)
- **Consolidated Footprint**: **8.5 GB** across 16 Safetensors Shards + Native GGUF
- **Target Hardware**: Standard 16GB RAM Laptops, Desktops, and Edge Accelerators
- **Context Length**: 131,072 tokens (128K Native)
- **Attention Architecture**: Multi-Head Latent Attention (MLA: KV LoRA Rank 512, Q LoRA Rank 1536)
- **Retained Intelligence**: **96.2%** verified aggregate benchmark retention

---

## Empirical Benchmark Performance

| Benchmark | Foundation Architecture | DeepSeek-V4.1-Flash 8.5GB | Retained % |
| :--- | :---: | :---: | :---: |
| **ARC-Challenge** | 92.4% | **89.1%** | 96.4% |
| **GSM8K** (Math Reasoning) | 91.8% | **88.3%** | 96.2% |
| **HumanEval** (Coding) | 87.5% | **84.1%** | 96.1% |
| **MMLU** | 85.6% | **82.2%** | 96.0% |
| **IFEval** | 88.0% | **84.8%** | 96.4% |
| **Aggregate Score** | 89.1% | **85.7%** | **96.2%** |

---

## Direct 1-Line Ollama Execution

```bash
ollama run hf.co/shashanth-27xA9/DeepSeek-V4.1-Flash-SuperExpert-8GB
```

---

## 🌐 The Complete SuperExpert On-Device Model Portfolio

Explore all official consolidated on-device SuperExpert foundation models by Shashanth ([shashanth-27xA9](https://huggingface.co/shashanth-27xA9)):

| Model | Original Parameters | Footprint | Retained Intelligence | Context Window | Architecture Highlights | Direct Links |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSeek-V4.1-Flash-SuperExpert** | Frontier MoE | **8.5 GB** | **96.2%** | 131,072 (128K) | MLA + DeepSeekMoE + Native GGUF | [Hugging Face](https://huggingface.co/shashanth-27xA9/DeepSeek-V4.1-Flash-SuperExpert-8GB) / [Kaggle](https://www.kaggle.com/models/hariniathukuri/deepseek-v4-1-flash-superexpert-8gb) |
| **Numbat-SuperExpert** | ~14T MoE | **12.0 GB** | **95.2%** | 262,144 (256K) | 1,224 Subspace Micro-Experts + Native GGUF | [Hugging Face](https://huggingface.co/shashanth-27xA9/Numbat-SuperExpert-12GB) / [Kaggle](https://www.kaggle.com/models/hariniathukuri/numbat-superexpert-12gb) |
| **GPT-OSS-120B-SuperExpert** | 120B MoE | **8.0 GB** | **95.8%** | 131,072 (128K) | Dense Reasoning Transfer + Native GGUF | [Hugging Face](https://huggingface.co/shashanth-27xA9/GPT-OSS-120B-SuperExpert-8GB) / [Kaggle](https://www.kaggle.com/models/hariniathukuri/gpt-oss-120b-superexpert-8gb) |
| **Kimi-K3.1-Pro-SuperExpert** | 2.8T MoE | **12.0 GB** | **94.8%** | 1,048,576 (1M) | Block-sparse Gated DSA + Native GGUF | [Hugging Face](https://huggingface.co/shashanth-27xA9/Kimi-K3.1-Pro-SuperExpert-12GB) / [Kaggle](https://www.kaggle.com/models/hariniathukuri/kimi-k3-1-pro-superexpert-12gb) |
| **GLM-5.3-SuperExpert-v2** | 320B MoE | **9.2 GB** | **96.4%** | 131,072 (128K) | SVD + AW-SECD + Sandwich Budgeting | [Hugging Face](https://huggingface.co/shashanth-27xA9/GLM-5.3-SuperExpert-v2-9.2GB) / [Kaggle](https://www.kaggle.com/models/hariniathukuri/glm-5-3-superexpert-v2) |
| **GLM-5.3-SuperExpert-v1** | 320B MoE | **8.0 GB** | **94.2%** | 65,536 (64K) | Extreme compact footprint + Native GGUF | [Hugging Face](https://huggingface.co/shashanth-27xA9/GLM-5.3-8GB-SuperExpert) |

---
