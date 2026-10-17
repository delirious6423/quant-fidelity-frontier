---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - gsq
  - rco
  - qwen3.8
  - t4
  - kaggle
  - openai-compatible
---

# nqwen27 — Qwen3.8-27B GSQ-RCO IQ2_XS · T4 一键部署套件

**模型**: [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)（奥地利 IST-DASLab，GSQ + RCO 梯度搜索动态非均匀量化）的 **IQ2_XS（8.42 GB）** 量化版，基座 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。

**引擎**: [llama.cpp](https://github.com/ggml-org/llama.cpp) `b11175`（源码编译，`GGML_CUDA=ON`，`CUDA_ARCHITECTURES=75`）。
> 为什么不是 NInfer？[NInfer](https://github.com/Neroued/ninfer) 是面向 RTX 5090（sm_120a）的单卡引擎，构建强制 Blackwell，且只加载私有 `.ninfer` 格式（不加载 GGUF），T4（sm_75）无法使用；GSQ-RCO 官方 README 亦声明其 GGUF "runs unmodified in llama.cpp, Ollama, and LM Studio"。

**服务**: `llama-server` 提供 OpenAI 兼容 API（`/v1/chat/completions` 等），模型名 `nqwen27`，API Key `1234`，**上下文 128k（CTX=131072，模型原生支持 262144）**，KV cache q8_0 量化 + FlashAttention，并通过 [aitun.cc](https://aitun.cc) 子域名隧道暴露公网。

## 文件

| 文件 | 说明 |
|---|---|
| `deploy_oneclick.sh` | 一键部署：引擎（预编译优先，GLIBC 不兼容自动源码编译）→ 权重 → API → 隧道 |
| `start_server.sh` | 单独启动/重启 API 服务（key=1234，128k，显存不足自动降级） |
| `start_tunnel.sh` | 单独启动 aitun 子域名隧道 |
| `rescue_restart.sh` | 现场救援：一键重启 128k API + 公网隧道（服务/隧道掉线时用） |
| `kaggle_oneclick_cell.py` | Kaggle Notebook 单 cell 引导脚本 |
| `engine/llama-b11175-cu128-sm75-t4.tar.gz` | **T4 专用预编译引擎包（54MB）**：llama.cpp b11175 (4de0926), CUDA 12.8, sm_75, glibc 2.35，含全部 .so，免 40 分钟源码编译 |
| `weights/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf` | 模型权重（8.42 GB，标准 GGUF） |

## Kaggle T4 快速开始

新建 Kaggle Notebook（开 GPU T4 ×2、Internet 开启），粘贴运行 `kaggle_oneclick_cell.py` 内容；或任一 Linux + CUDA 机器：

```bash
git clone https://huggingface.co/tchbcb/nqwen27 && cd nqwen27
AITUN_API_KEY=<你的aitun_key> bash deploy_oneclick.sh
```

脚本自动完成：llama.cpp 引擎（默认优先拉本仓库 T4 预编译包，54MB 数十秒；不兼容时官方预编译包 → 源码编译约 40 分钟）→ 从本仓库 `weights/` 下载权重（失败自动回退 ISTA-DASLab 原仓库）→ 启动 API（127.0.0.1:8080，key=1234，128k 上下文）→ aitun 隧道输出公网 `https://<子域名>.aitun.cc`。全程约 10-15 分钟（权重下载占大头）。

## API 调用

已部署示例（服务存活期间）：`https://qwenapi.aitun.cc`

```bash
curl https://qwenapi.aitun.cc/v1/chat/completions \
  -H "Authorization: Bearer 1234" \
  -H "Content-Type: application/json" \
  -d '{"model":"nqwen27","messages":[{"role":"user","content":"用一句话介绍维也纳"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="https://<你的子域名>.aitun.cc/v1", api_key="1234")
r = client.chat.completions.create(
    model="nqwen27",
    messages=[{"role": "user", "content": "你好"}],
)
print(r.choices[0].message.content)
```

- 关闭思维链（Qwen3.8 默认开启 thinking）: 请求体加 `"chat_template_kwargs": {"enable_thinking": false}`
- 流式输出: `"stream": true`
- 视觉多模态: 本套件默认纯文本；如需图片理解，另行下载原仓库 `mmproj-Qwen3.8-27B-BF16.gguf` 并给 llama-server 加 `--mmproj` 参数

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `AITUN_API_KEY` | 空 | aitun 注册用户 key（子域名隧道必需） |
| `API_KEY` | `1234` | OpenAI 兼容 API 的 Bearer 鉴权 key |
| `PORT` | `8080` | 本机服务端口 |
| `CTX` | `131072` | 上下文长度（128k；最大可至模型原生 262144，但 T4 显存有限） |
| `P1_CTK`/`P1_CTV` 等 | `q8_0`/`q4_0` | KV cache 量化预设链，见下方 FAQ |
| `SPEC_FLAGS` | 空 | llama-server 额外参数透传（投机解码实验等），见下方性能章节 |
| `WORK` | `/root/nqwen27` | 部署目录 |
| `SKIP_PREBUILT` | `0` | 置 1 跳过预编译尝试，直接源码编译 |

## FAQ：128k 上下文怎么放进 16G 显存？

llama.cpp 的 KV cache 是显存大头，`start_server.sh` 用 **KV cache 量化（`-ctk/-ctv`）+ FlashAttention（`-fa on`）** 压缩 KV，并按显存自动降档：

| 预设 | 层卸载 | KV 量化 | 适用 |
|---|---|---|---|
| P1（默认） | `-ngl 99` 全入 GPU | `q8_0` | 2×T4（层切分后 KV 约 8-9G/卡） |
| P2 | `-ngl 99` 全入 GPU | `q4_0` | 显存紧张 |
| P3 | `-ngl 35` 部分卸载 | `q4_0` | 单卡 16G 兜底 |

启动时先试 P1，检测到 OOM 自动换 P2、P3，成功的档位会打印在日志里。KV q8_0 接近无损，q4_0 有轻微精度损失但完全可用。

## 性能基准与提速说明（2×T4 实测）

用 `llama-bench`（p1024/tg128，b11175）实测五种方案后的结论——**当前默认配置（`-sm layer` + 默认 ubatch 512）就是 2×T4 上的最优解**：

| 方案 | prefill tok/s | decode tok/s | 结论 |
|---|---|---|---|
| **`-sm layer`（当前默认）** | **313** | **11.5** | 双卡层切分接力，T4 无 NVLink 下最快 |
| `-sm none`（单卡全放） | 223 | 8.0 | 两卡合计带宽 > 单卡，即使有接力开销 |
| `-sm tensor`（张量并行） | 181 | 5.0 | 逐层 PCIe 同步开销远超收益 |
| `-sm row` | — | — | 本量化 + FA 组合下启动失败 |
| `-ub 2048` | 224 | 10.9 | 大 ubatch 反而降速，保持默认 512 |

实测解码约 **11-13 tok/s、prefill 约 275-310 tok/s**，即 27B 模型 @ 2bpw 在 2×T4（2018 年数据中心卡，无 NVLink）上的硬件极限水平。

**关于 MTP**：b11175 引擎支持 `--spec-type draft-mtp`，但 **GSQ-RCO 量化版 GGUF 不含 MTP 层**（启动时报 `context type MTP requested but model doesn't contain MTP layers`），因此无法开启；需要换带 MTP 头的权重才行。

**投机解码实验**（已实测，默认不开启）：

- `ngram-simple`（免草稿模型）：普通文本 +5% 以内，重复文本 ±0，不值得；
- `draft-simple` + [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) 草稿：普通对话 **8.8 tok/s（变慢 20%）**，高度重复文本 12.2 tok/s（+12%），仅对重复型生成（批量模板、代码补全）有微弱收益。

如需在重复型工作负载上试验：

```bash
# 草稿模型已可自动下载（约1.3GB），或手动放入 $WORK/draft/
SPEC_FLAGS="--spec-type draft-simple -md /root/nqwen27/draft/Qwen3.5-2B-Q4_K_M.gguf -ngld 99 -devd CUDA0 -ctkd q4_0 -ctvd q4_0" \
  bash start_server.sh
# 或免草稿的 ngram 模式
SPEC_FLAGS="--spec-type ngram-simple" bash start_server.sh
```

启动失败会自动沿 P1→P2→P3 降档兜底；去掉 `SPEC_FLAGS` 重跑即恢复默认配置。

## 安全提示

仓库内脚本**不包含任何真实密钥**（aitun key / HF token 均通过环境变量传入）。公网服务请务必保留 `API_KEY` 鉴权，必要时把默认的 `1234` 改成强 key。

## 来源与致谢

- 基座模型: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)（Apache-2.0）
- 量化: [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)（GSQ: arXiv 2604.18556；RCO: arXiv 2605.00649）
- 引擎: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
