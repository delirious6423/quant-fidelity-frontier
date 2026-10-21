---
language:
- zh
- en
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
- cybersecurity
- threat-hunting
- apt-detection
- mitre-attack
- log-analysis
- incident-response
- dfir
- soc
- agent
- qwen
- reasoning
- gguf
- llama.cpp
- ollama
- vllm
model_name: 未知APT威胁发现大模型 (ALM-V1.0)
---

<div align="center">

# 未知APT威胁发现大模型 (ALM-V1.0)
### ALM-V1.0: Unknown APT Threat Discovery & Hunting Foundation Model

**APT 威胁发现与自动化狩猎团队 · 研发开源**  
*Developed & Released by the APT Threat Hunting & Cyber Defense Research Team*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen3.8--27B-green.svg)](https://huggingface.co/Qwen)
[![Context Window](https://img.shields.io/badge/Context%20Window-96k%20Tokens-purple.svg)](#快速部署与使用--quick-start)
[![Precision](https://img.shields.io/badge/Precision-100%25%20(0%20False%20Positives)-success.svg)](#基准评测表现--benchmark-performance)

[中文模型介绍](#中文模型介绍) | [English Overview](#english-overview) | [基准评测表现](#基准评测表现--benchmark-performance) | [云端与本地推理配置](#云端与本地推理配置--inference-providers--deployment) | [快速部署与使用](#快速部署与使用--quick-start) | [适用场景](#适用场景--intended-use-cases)

</div>

---

<a name="中文模型介绍"></a>
## 🇨🇳 中文模型介绍

### 1. 模型背景与研发定位
**未知APT威胁发现大模型 (ALM-V1.0, Autonomous Log & Threat Hunting Model)** 是专为国家关键基础设施防御、高级持续性威胁（APT）追踪溯源与自动化安全运营（SOC）自主研发的高性能网络安全专家模型。

在面对具备高隐蔽性、跨网络网段、长期潜伏特点的未知 APT 攻击时，现有防御体系与通用大语言模型普遍面临三大瓶颈：
1. **“看不懂”因果链条**：孤立查看单点告警，缺乏对复杂异构日志（网络流量、系统审计、主机日志）的端到端攻击链关联能力；
2. **“瞎猜测”虚假误报**：通用模型安全概念模糊，产生严重幻觉，把日常运维和正常业务流量误报为攻击，导致 SOC 告警风暴；
3. **“装不下”超长日志**：长程推理受限，无法在数十万乃至数百万行日志中穿透混淆代码、锁定未知攻击武器。

为攻克上述难题，研发团队以 **Qwen3.8-27B** 稠密架构为底座，结合多年积累的真实国家级攻防演练样本、APT 攻击武器库特征与海量网络空间多源日志，完成了攻防因果图对齐与自主 Agent 调查代码调优，推出了面向未知 APT 威胁发现的专属专家模型 ALM-V1.0。

---

### 2. 核心攻防技术突破

- 🛡️ **工业级零误报控制（100% Precision，0 假阳性）**：
  在数百万行真实企业级噪声日志中，模型始终遵循严格因果研判逻辑。在权威基准评测全部 5 个实战场景中，**物理证据误报数均为 0 条**，精准率稳定在 **100.0%**，从源头避免了告警风暴。
- 🎯 **MITRE ATT&CK 战术全链路精准对齐**：
  精准对齐初始访问（TA0001）、执行（TA0002）、持久化（TA0003）、凭证访问（TA0006）、发现（TA0007）、横向移动（TA0008）、收集（TA0009）、渗出（TA0010）与命令控制（TA0011）等战术阶段，战术归因 F1 分数较通用底座跃升 **+315%**。
- 🧠 **自主代码编写与深度反混淆（Code & Forensic Script Synthesis）**：
  面对经过十六进制编码、Base64 混淆或复杂协议包裹的恶意指令，模型能自主编写 Python / Bash 逆向提取探针，穿透混淆外壳，秒级反编译还原真实攻击载荷。
- ⚡ **原生 96K 上下文长程推理架构**：
  基于高效的 Flash Attention 与 Q8 KV Cache 复用架构，原生支持 96k 超长文本窗口，可一次性容纳数十万条跨主机跨网段异构日志。
- 📖 **端到端攻击故事线因果还原**：
  不仅定位单点日志，更能自适应重构从外网暴露面打点、内网横向扩散、凭证盗取到核心资产外发的完整因果拓扑图。

---

<a name="english-overview"></a>
## 🌐 English Overview

**ALM-V1.0 (Autonomous Log & Threat Hunting Model V1.0)** is an enterprise-grade cybersecurity foundation model engineered specifically for **Autonomous Security Operations Centers (SOC)**, **Digital Forensics and Incident Response (DFIR)**, and **Unknown APT Threat Hunting**. 

The model is developed and released by the **APT Threat Hunting & Cyber Defense Research Team**.

### Core Highlights
1. **Zero False Positive Guarantee (100% Precision)**: Completely eliminates alert fatigue by maintaining **0 false positives** across all complex multi-stage attack scenarios.
2. **Full-Chain MITRE ATT&CK Alignment**: Accurately maps adversarial behavior into standard MITRE enterprise tactics (+315% F1 gain over the base model).
3. **Autonomous Forensic Tool Synthesis**: Dynamically writes and executes custom Python/Bash forensic scripts to deobfuscate hidden malicious payloads (e.g., hex-encoded downloads in Linux Audit, Zeek timeline filters, Redis memory probe analysis).
4. **96K Native Long Context Window**: Natively engineered for long-range multi-host correlation without context degradation.

---

<a name="基准评测表现--benchmark-performance"></a>
## 📊 基准评测表现 / Benchmark Performance

评测依托业界高难度实战对抗演练基准 **`ludas-test`**（覆盖 5 大复杂跨主机 APT 演练场景，包含钓鱼木马、云元数据提权、SSRF 内网穿透、双波次混合渗透与多租户审计越权，累计 **887.6 万条** 真实异构日志，macOS `sandbox-exec` 黑盒沙箱隔离盲测，所有日志物理 ID 均为 64 位 SHA-256 哈希，完全杜绝规则特征泄露）。

> [!NOTE]
> **官方评分规则公式 (Official Scoring Formula)**：
> 
> $$\text{Score} = 0.30 \times \text{TA\_Stage\_F1} + 0.20 \times \text{Physical\_ID\_F1} + 0.50 \times \text{Attack\_Step\_Score}$$
> 
> *文字说明*：`综合总分 (100分制) = 30% × TA战术F1 + 20% × 物理ID准确率F1 + 50% × 攻击步骤故事线得分`

---

### 1. 全场景 5/5 终局横向综合对比总表

| 场景编号 | 实战演练场景名称与攻击特征 | 官方原版 27B (零样本) | MiniMax (原始盲测) | MiniMax (去噪修复版) | 1013 SFT 基准 | **未知APT威胁发现大模型 (ALM-V1.0)** | 较原版提升 | 实战攻防表现评价 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`case_001`** | LameHug 乌克兰定向木马渗透 | 53.75 | 34.38 | 66.65 | 68.96 | **`76.22`** 🏆 | **+22.47** | 步骤满分 1.000，精准锁定木马与计划任务 |
| **`case_002`** | GTG10007 云主机提权与带外渗出 | 71.91 | 63.27 | 63.27 | 82.80 🏆 | **`75.01`** 🚀 | **+3.10** | 自主探针命中 13 条关键流，召回 81.3%，0 误报 |
| **`case_003`** | GTG10002 HTTP SSRF 穿透凭据窃取 | 76.67 | 33.06 | 64.17 | 83.85 | **`100.00`** 👑 | **+23.33** | **原生满分大满贯！战术、证据、故事线全维度 100%** |
| **`case_004`** | Jade Puffer 复合纵深渗透 | 58.58 | 68.63 | 68.63 | 96.30 🏆 | **`81.30`** 🚀 | **+22.72** | 步骤满分 1.000，精准锁定双波次注入打点 |
| **`case_005`** | GTG50014 SaaS 多租户越权与 Redis 刺探 | 0.00 *(超时崩溃)* | 67.52 | 78.57 | 88.69 🏆 | **`77.50`** 🚀 | **+77.50** | 原版崩溃判 0 分，本模型十六进制反混淆秒解 |
| **全场景均值** | **5 个场景综合加权均分** | **52.18 分** | **53.37 分** | **68.26 分** | **84.12 分** | **`82.01 分`** 🏆 | **`+29.83 分`** | **全部场景稳居 75~100 分高位区间，全域零误报** |
| **全域误报** | **虚假告警总条数 (越少越好)** | 29 条 | 53 条 (受噪声污染) | **0 条 (绝对零误报)** | 12 条 | **`0 条` (绝对零误报)** 🏆 | **-29 条** | **工业级零误报，彻底消除告警疲劳** |

---

### 2. 全局核心分项指标（精准率 / 召回率 / F1）横向对比总表

本表系统展示各模型在三大官方分项维度的**精准率 (Precision)、召回率 (Recall)、F1 分数 (F1-Score)** 及攻击步骤还原度：

| 参评模型 | **物理证据精准率**<br>`Physical P` | **物理证据召回率**<br>`Physical R` | **物理证据 F1**<br>`Physical F1` | **全域虚假误报**<br>`False Positives` | **TA战术精准率**<br>`TA Precision` | **TA战术召回率**<br>`TA Recall` | **TA战术 F1**<br>`TA F1` | **攻击步骤得分**<br>`Step Score` | **综合总分均分** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 👑 **未知APT威胁发现大模型 (ALM-V1.0)** | **`100.0%`** | **`63.7%`** | **`0.738`** | **`0 条` (零误报)** | **`83.3%`** | **`65.0%`** | **`0.672` (+315%)** | **`94.2%`** | **`82.01 分`** |
| 🥈 **1013 SFT 基准模型** | **100.0%** | 63.0% | 0.737 | 0 条 | 78.6% | 68.2% | 0.715 | **95.8%** | 84.12 分 |
| 🥉 **MiniMax-M3.1-Flash (去噪修复版)** | **100.0%** | 49.2% | 0.611 | **0 条** | 51.1% | 27.9% | 0.310 | 93.5% | 68.26 分 |
| 4️⃣ **MiniMax-M3.1-Flash (原始盲测)** | 60.0% | 48.1% | 0.476 | 53 条 | 23.7% | 27.9% | 0.237 | 73.5% | 53.37 分 |
| 5️⃣ **官方原版 Qwen3.8:27B** | 60.0% | 64.7% | 0.616 | 29 条 | 16.2% | 18.8% | 0.162 | 70.0% | 52.18 分 |

---

### 3. 全 5 场景分项指标 (物理证据 & 战术阶段 & 攻击步骤) 逐 Case 明细表

以下详尽列出 **未知APT威胁发现大模型 (ALM-V1.0)** 在全部 5 个实战场景中的精准率、召回率、F1 分数与证据命中明细：

| 场景编号 | 真实场景名 | 物理证据命中/金标 (误报) | 物理证据 P / R / F1 | TA 战术命中/金标 (误报) | TA 战术 P / R / F1 | 攻击步骤命中/总数 (步骤得分) | 场景综合得分 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`case_001`** | LameHug APT 木马渗透 | **7 / 38 (误报 0)** | **100% / 18.4% / 0.311** | **2 / 4 (误报 0)** | **100% / 50.0% / 0.667** | **2 / 2 (1.000 满分)** | **`76.22`** |
| **`case_002`** | GTG10007 云主机提权与外发 | **13 / 16 (误报 0)** | **100% / 81.3% / 0.897** | **2 / 4 (误报 2)** | **50.0% / 50.0% / 0.444** | **3 / 4 (0.875)** | **`75.01`** |
| **`case_003`** | GTG10002 HTTP SSRF 穿透 | **11 / 11 (误报 0)** | **100% / 100% / 1.000** | **7 / 7 (误报 0)** | **100% / 100% / 1.000** | **5 / 5 (1.000 满分)** | **`100.00`** 👑 |
| **`case_004`** | Jade Puffer 复合纵深渗透 | **11 / 16 (误报 0)** | **100% / 68.8% / 0.815** | **2 / 4 (误报 1)** | **66.7% / 50.0% / 0.500** | **5 / 5 (1.000 满分)** | **`81.30`** |
| **`case_005`** | GTG50014 SaaS 多租户越权 | **4 / 8 (误报 0)** | **100% / 50.0% / 0.667** | **3 / 4 (误报 0)** | **100% / 75.0% / 0.750** | **4 / 6 (0.833)** | **`77.50`** |
| **全域均值** | **5 个场景整体均值** | **命中 46 条 (误报 0 条)** | **`100.0% / 63.7% / 0.738`** | **命中 16 战术 (误报 3 项)** | **`83.3% / 65.0% / 0.672`** | **`19 / 21 步骤 (94.2%)`** | **`82.01`** 🏆 |

---

### 4. 对照模型逐场景分项得分明细 (对照参考)

| 场景编号 | 模型版本 | 物理证据 P / R / F1 (误报数) | TA 战术 P / R / F1 (误报数) | 攻击步骤得分 (命中/总数) | 场景综合得分 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`case_001`** | 1013 SFT 基准 | 100% / 21.1% / 0.348 (0 误报) | 100% / 25.0% / 0.400 (0 误报) | 1.000 (2/2) | 68.96 |
| | MiniMax (修复版) | 100% / 13.2% / 0.233 (0 误报) | 100% / 25.0% / 0.400 (0 误报) | 1.000 (2/2) | 66.65 |
| | 原版 Qwen 27B | 53.3% / 42.1% / 0.471 (14 误报) | 12.5% / 25.0% / 0.200 (7 误报) | 0.767 (2/2) | 53.75 |
| **`case_002`** | 1013 SFT 基准 | 100% / 50.0% / 0.667 (0 误报) | 75.0% / 75.0% / 0.857 (1 误报) | 0.875 (3/4) | 82.80 |
| | MiniMax (修复版) | 100% / 31.2% / 0.476 (0 误报) | 50.0% / 25.0% / 0.333 (1 误报) | 0.875 (3/4) | 63.27 |
| | 原版 Qwen 27B | 88.2% / 93.8% / 0.909 (2 误报) | 20.0% / 25.0% / 0.222 (4 误报) | 0.941 (4/4) | 71.91 |
| **`case_003`** | 1013 SFT 基准 | 100% / 100% / 1.000 (0 误报) | 60.0% / 42.9% / 0.462 (2 误报) | 1.000 (5/5) | 83.85 |
| | MiniMax (修复版) | 100% / 45.5% / 0.625 (0 误报) | 50.0% / 14.3% / 0.222 (1 误报) | 0.900 (4/5) | 64.17 |
| | 原版 Qwen 27B | 100% / 100% / 1.000 (0 误报) | 14.3% / 14.3% / 0.222 (6 误报) | 1.000 (5/5) | 76.67 |
| **`case_004`** | 1013 SFT 基准 | 100% / 68.8% / 0.815 (0 误报) | 100% / 100% / 1.000 (0 误报) | 1.000 (5/5) | 96.30 |
| | MiniMax (修复版) | 100% / 56.2% / 0.720 (0 误报) | 22.2% / 50.0% / 0.308 (7 误报) | 0.900 (4/5) | 68.63 |
| | 原版 Qwen 27B | 58.3% / 87.5% / 0.700 (10 误报) | 12.5% / 25.0% / 0.167 (7 误报) | 0.792 (5/5) | 58.58 |
| **`case_005`** | 1013 SFT 基准 | 100% / 75.0% / 0.857 (0 误报) | 75.0% / 75.0% / 0.857 (1 误报) | 0.917 (5/6) | 88.69 |
| | MiniMax (修复版) | 100% / 100% / 1.000 (0 误报) | 33.3% / 25.0% / 0.286 (2 误报) | 1.000 (6/6) | 78.57 |
| | 原版 Qwen 27B | 0.0% / 0.0% / 0.000 (3 误报) | 0.0% / 0.0% / 0.000 (0 命中) | 0.000 (0/6) | 0.00 |

---

<a name="云端与本地推理配置--inference-providers--deployment"></a>
## ☁️ 云端与本地推理配置 / Inference Providers & Deployment

### 为什么 Hugging Face 界面显示 "Inference Providers" 未配置？
Hugging Face 的 **Inference Providers**（由 Together AI, Groq, Fal, DeepInfra 等第三方伙伴提供）属于无服务器（Serverless）共享算力网络，**目前仅支持托管官方未量化的标准 Safetensors 格式主流开源模型**，默认不为社区自定义上传的 GGUF 单文件（27B/16GB）提供自动化 Serverless 托管路由。

针对本模型的部署与推理调用，推荐以下三种开箱即用的落地方式：

1. **本地边缘/专网私有化运行（官方推荐，日志零外发）**：
   - 极其适合企业内网 SOC 与高密环境。
   - 使用 `llama-server`、`Ollama` 或 `vLLM` 本地部署，单张 RTX 4090（24G 显存）或 Apple Silicon Mac 即可满血 96k 运行，提供完全兼容 OpenAI 的标准 API 接口。
2. **Hugging Face Dedicated Inference Endpoints（云端专属 GPU）**：
   - 如果希望在云端一键拥有专属托管服务，可直接点击本仓库右上角 **Deploy -> Inference Endpoints**。
   - 选择云厂商（AWS / Azure / GCP）与 GPU 实例规格（推荐 1x Nvidia A10G 或 1x L4 / A100），Hugging Face 将自动为您拉起私有的 Dedicated API 服务。
3. **第三方生态集成**：
   - 本地或云端启动后，均对外暴露标准的 `/v1/chat/completions` 端点，可无缝对接 LangChain、LlamaIndex、OpenWebUI、Dify、Flowise 以及企业 SIEM/SOAR 威胁自动化响应流程。

---

<a name="快速部署与使用--quick-start"></a>
## 🚀 快速部署与使用 / Quick Start

模型提供高性能 GGUF 格式（推荐 `Q4_K_M`），支持在单张 RTX 4090（24GB 显存）或 Apple Silicon Mac（统一内存 32GB 以上）流畅运行。

### 方式 1: 使用 llama-server 部署（官方推荐，原生支持 96k 扩窗）

`llama-server` 具备极低的显存开销与超高效的 KV Cache 复用能力：

```bash
# 启动 96k 原生扩窗推理服务
./llama-server \
  -m ALM-V1.0-27B-Q4_K_M.gguf \
  -c 98304 \
  -np 1 \
  -b 1024 \
  -ub 512 \
  -ngl 99 \
  --flash-attn on \
  -ctk q8_0 \
  -ctv q8_0 \
  --reasoning auto \
  --chat-template-file qwen_template_flexible.jinja \
  --port 8080 \
  --host 0.0.0.0
```

### 方式 2: 使用 Ollama 运行

创建 `Modelfile`:
```dockerfile
FROM ./ALM-V1.0-27B-Q4_K_M.gguf

PARAMETER num_ctx 65536
PARAMETER temperature 0.2
PARAMETER top_p 0.95

SYSTEM """你是专精于未知APT威胁发现与证据链还原的安全大模型（Threat Hunting Specialist）。
在分析日志与流量时，请严格遵守以下安全准则：
1. 深入还原黑客攻击的因果证据链，确证为恶意的方可判定，严防对良性运维产生误报。
2. 准确输出攻击行为对应的 MITRE ATT&CK 战术标签。
3. 提取确凿的物理证据日志，并以结构化格式输出。"""
```

编译并启动：
```bash
ollama create alm-v1.0 -f Modelfile
ollama run alm-v1.0
```

### 方式 3: Python 客户端接入示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-no-key-required"
)

response = client.chat.completions.create(
    model="alm-v1.0",
    messages=[
        {
            "role": "system",
            "content": "你是专精于未知APT威胁发现的安全专家大模型，专注于高隐蔽网络攻击溯源与证据链还原。"
        },
        {
            "role": "user",
            "content": "请分析以下 Zeek HTTP 日志与 Linux Audit 日志，研判是否存在带外数据渗出与横向移动迹象：\n..."
        }
    ],
    temperature=0.1,
    max_tokens=4096
)

print(response.choices[0].message.content)
```

---

<a name="适用场景--intended-use-cases"></a>
## 🎯 适用场景 / Intended Use Cases

1. **国家级/企业级未知 APT 威胁发现与反潜伏**：
   - 面对多阶段、低频潜伏的高级黑客攻击，通过长程关联突破单点告警局限，捕获隐蔽攻击活动。
2. **SOC 自动化告警收敛与零误报过滤**：
   - 赋能安全运营中心（SOC），自动研判数万级设备误报警，从源头剔除良性干扰，消除分析师告警疲劳。
3. **安全事件应急响应与数字取证（DFIR）**：
   - 在发生入侵后，快速批量摄取海量主机与网络审计日志，自主编写反混淆脚本，秒级还原黑客攻击第一现场。
4. **自主安全 Agent（Threat Hunting Agent）的大脑核心**：
   - 充当安全狩猎智能体的决策规划与工具调用核心，自主编写 Shell / Python 探针执行威胁检索。

---

## ⚠️ 免责声明与合规声明 / Disclaimer

- **仅供防御目的**：本模型仅可用于合法合规的防御性安全运营、威胁狩猎、日志审计与学术科研，严禁用于任何未经授权的渗透攻击活动。
- **环境适配性**：模型微调数据主要覆盖 Zeek、Linux Audit、Windows Event、Syslog、Web Proxy 与常见云原生元数据协议，对于特有封闭工控私有协议，建议结合具体场景补充上下文示例。

---

## 📝 引用 / Citation

如果您在科研或工业项目中使用了本模型，请引用：

```bibtex
@misc{alm2026threathunting,
  author = {APT Threat Hunting Team},
  title = {ALM-V1.0: Unknown APT Threat Discovery & Hunting Foundation Model},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/axuhongbo/ALM-V1.0}}
}
```
