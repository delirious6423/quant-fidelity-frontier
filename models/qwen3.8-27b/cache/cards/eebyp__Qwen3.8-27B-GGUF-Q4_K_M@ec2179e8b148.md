---
base_model: Qwen/Qwen3.8-27B
library_name: llama.cpp
tags: [gguf, q4_k_m, quantized, llama.cpp]
language: [zh, en]
---

# Qwen3.8-27B GGUF Q4_K_M

本仓库为 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 的 Q4_K_M 量化版本（GGUF 格式），可直接用 llama.cpp 或 Ollama 加载推理。

## 转换信息

| 项目 | 值 |
|------|-----|
| **源模型** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)（BF16） |
| **量化工具** | llama.cpp（convert_hf_to_gguf.py + llama-quantize） |
| **量化档位** | Q4_K_M（4.92 BPW） |
| **文件大小** | 15.7 GB |
| **转换日期** | 2026-10-11 |
| **转换环境** | Colab G4 · RTX PRO 6000 Blackwell Server Edition (97887 MiB) |

## 转换流程

```
步骤 1：下载原始模型（BF16 safetensors）     → 52 GB
步骤 2：格式转换（safetensors → GGUF FP16）  → 50.9 GB
步骤 3：权重量化（GGUF FP16 → Q4_K_M）       → 15.7 GB
```

### 实测耗时

| 步骤 | 耗时 |
|------|------|
| 下载模型 | 2 分 38 秒 |
| 格式转换 | 3 分 0 秒 |
| 量化 | 1 分 29 秒 |
| **合计** | **约 7 分钟** |

## 使用方法

### llama.cpp 直接加载

```bash
# 下载本仓库
hf download eebyp/Qwen3.8-27B-GGUF-Q4_K_M \
  Qwen3.8-27B-Q4_K_M.gguf \
  --local-dir ./models/

# 启动推理服务
llama-server \
  -m ./models/Qwen3.8-27B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -c 8192
```

### Ollama 使用

```dockerfile
# Modelfile
FROM ./Qwen3.8-27B-Q4_K_M.gguf
```

```bash
ollama create qwen3.8-27b-q4 -f Modelfile
ollama run qwen3.8-27b-q4
```

### Python 调用（OpenAI 兼容接口）

```python
import requests

url = "http://localhost:8080/v1/chat/completions"
payload = {
    "model": "default",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 512,
    "temperature": 0.7,
    "chat_template_kwargs": {"enable_thinking": False}
}
resp = requests.post(url, json=payload)
print(resp.json()["choices"][0]["message"]["content"])
```

## 实测性能

> 测试环境：Colab G4 · RTX PRO 6000 Blackwell Server Edition (97887 MiB)

| 指标 | 值 |
|------|-----|
| 单请求速度 | 55.7~60.4 toks/s |
| 显存占用 | ~20 GB |

## 部署效率

| 项目 | 值 |
|------|-----|
| 量化产出全流程耗时 | 约 7 分钟 |
| 从本仓库下载量化模型（16 GB） | **9.4 秒** |
| 对比：下载原始模型（52 GB） | 2 分 38 秒 |
| 节省时间（含转换量化） | **约 7 分钟** |
| 本次会话计算单元消耗 | 5.30 个 |

## 注意事项

- Q4_K_M 量化对输出质量影响极小，六项实用能力测试（写作/翻译/代码/分析/提取/推理）与 FP16 无显著差异
- 该模型对 Qwen3.8 的 chat template 默认开启 reasoning 保留，做速度测试时需传 `chat_template_kwargs: {"enable_thinking": False}` 关闭思考模式，否则速度数据会失真
- 建议 `-c` 参数按 slot 数设置总上下文（如 8 slot 设 `-c 16384`，每 slot 分 2048）

## 相关仓库

- [eebyp/Qwen3.8-27B-GGUF-Q4_K_M](https://huggingface.co/eebyp/Qwen3.8-27B-GGUF-Q4_K_M) — 本仓库
- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — 原始模型
  

## 👤 作者

- **Yiping Bai**（白毅平）
- ORCID：[0009-0004-2842-8241](https://orcid.org/0009-0004-2842-8241)
- Website：[byp.com.cn](https://byp.com.cn)
- Hugging Face：[eebyp](https://huggingface.co/eebyp)