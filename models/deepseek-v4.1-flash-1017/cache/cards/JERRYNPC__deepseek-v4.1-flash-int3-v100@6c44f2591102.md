---
language:
- zh
- en
license: other
license_name: deepseek-v4.1-flash-license
license_link: https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: pytorch
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- deepseek
- moe
- quantized
- int3
- int8
- v100
- custom-code
- text-generation
---

# DeepSeek-V4.1-Flash-1017 · int3 for single V100-32GB

**非官方量化权重。** 把 762B 的 DeepSeek-V4.1-Flash-1017 压到单张 Tesla V100-PCIE-32GB
（sm_70）+ 64GB 内存上运行：int8 稠密权重、int3（3.125 bit）路由专家、2-bit engram，
配合「显存常驻 dense + 宿主内存 LRU + NVMe 冷读」的专家流式加载。

- 推理代码：`GitHub 源码仓库（链接待补）`（MIT）
- 上游模型：[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- 本卡对应：**384 专家/层全量版**；另有 nano 实验版和共享 engram 仓

> 这不是官方实现，也不是 81 tok/s 的方案。32GB V100 上实测稳态 **1.2–1.35 tok/s**，
> 首 token 十几秒到两分钟，取决于提示长度和缓存冷热。质量边界见下文。

---

## 仓库组成

本 repo **只含 dense + 路由专家**。engram 表约 55GB，单独放在
[`JERRYNPC/deepseek-v4.1-flash-engram-v100`](https://huggingface.co/JERRYNPC/deepseek-v4.1-flash-engram-v100)，
full 和 nano 共用。

| 文件 | 大小 | 说明 |
|---|---:|---|
| `manifest.json` | 几 KB | 权重清单；路径全部是相对路径 |
| `dense-00000..00007.safetensors` | 8.45GB | int8 per-channel + fp16/fp32 小张量 |
| `experts3-L0.bin .. experts3-L39.bin` | 40 × 5.31GB = 212.3GB | int3 专家，定长记录 |
| `experts3-MTP0..2.bin`（可选） | 3 × 1.77GB | DSpark MTP 专家 |
| `mtp-dense-00000..00002.safetensors`（可选） | 0.65GB | DSpark MTP 稠密 |
| `tokenizer.json` / `tokenizer_config.json` | 几 MB | tokenizer |
| `engram_token_map.npy` | 0.5MB | engram 压缩 token 映射 |
| `_experts3_sha256.json` | 4KB | 40 个专家分片的 sha256 |
| `SHA256SUMS` | 几 KB | 分片校验和 |
| `engram/engram-L1.bin`、`engram/engram-L14.bin` | 55.3GB | **在 engram 仓**，下载到同一目录 |

无 MTP 合计约 **276GB**（dense 8.47 + experts 212.34 + engram 55.3）；
含可选 MTP 约 **282GB**（+ 5.95GB）。

### 量化格式

- **路由专家**：对称 int3，值域 `[-4, 3]`，group=128，fp16 scale，3 字节存 8 个值
  = **3.125 bit/权重**；单专家 `w1/w3 [2304,5120]`、`w2 [5120,2304]`，
  定长记录 `13,824,000 B`。40 层 × 384 专家。
- **稠密**：int8 per-output-channel（attention/shared expert/head），norm 等小张量保留
  fp32/fp16；加载时反量化为 fp16，约占 **14.92GiB** 显存（无 MTP）。
- **engram**：`rowcb2`，每行 64 字节 2-bit 索引 + 4 个 fp16 质心，256 维，
  行数 384,006,168（L1）/ 384,016,682（L14）；只按 token 查 24 行，不整表驻留。
- **MTP/DSpark**（可选）：3 层，每层 128 专家 top-3。实测收益不稳定，默认不建议开。

---

## 硬件要求

| 项目 | 要求 |
|---|---|
| GPU | **Tesla V100 32GB（sm_70）**；24GB 卡装不下 dense + 专家 slab |
| 内存 | **64GB**（40GB 专家 LRU + engram 页缓存） |
| 磁盘 | **NVMe，≥300GB 空闲**；SATA SSD 能跑但冷读明显更慢，USB 机械盘不可用 |
| 系统 | Windows 10/11 x64（已验证）；Linux 理论可行但未测试 |
| Python | 3.10–3.13（推荐 3.11；3.13.14 已验证） |
| PyTorch | **2.6.0 + cu124**；不要用 2.8+（不再提供 sm_70 轮子） |
| CUDA | 驱动支持 CUDA 12.4；NVRTC 可选，缺省自动回退 LUT 解包 |

V100 没有 FP8/FP4 Tensor Core，也没有 Triton。实现是纯 PyTorch 算子 + 可选 NVRTC
int3 解包 kernel + 可选 CPU AVX2 GEMV 动态库。

---

## 快速开始

### 1. 代码与环境

```bash
git clone https://github.com/GitHub 源码仓库（链接待补）.git v41-v100
cd v41-v100

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. 下载权重（两个 repo 解压到同一目录）

```bash
pip install -U huggingface_hub

hf download JERRYNPC/deepseek-v4.1-flash-int3-v100 \
  --local-dir checkpoints/v41-flash-int3

hf download JERRYNPC/deepseek-v4.1-flash-engram-v100 \
  --local-dir checkpoints/v41-flash-int3
```

目录应长这样：

```
checkpoints/v41-flash-int3/
├── manifest.json
├── dense-00000.safetensors ... dense-00007.safetensors
├── experts3-L0.bin ... experts3-L39.bin
├── engram/engram-L1.bin
├── engram/engram-L14.bin
├── engram_token_map.npy
└── tokenizer.json / tokenizer_config.json
```

### 3. 路径便携化（只需一次）

```bash
python tools/make_manifest_portable.py \
  checkpoints/v41-flash-int3/manifest.json --in-place
rm -f checkpoints/v41-flash-int3/manifest.abs.json
```

### 4. 冒烟

```bash
export PYTHONPATH=src        # Windows cmd: set PYTHONPATH=src

python -m v41v100.selftest   # 不需要权重，验证代码链路

python -m v41v100.generate \
  --ckpt checkpoints/v41-flash-int3 \
  --quant checkpoints/v41-flash-int3 \
  --prompt "用一句话解释 MoE" \
  --max-new-tokens 16 --cache-gb 40 --workers 4
```

### 5. OpenAI 兼容服务

```bash
python serve_v41.py \
  --ckpt checkpoints/v41-flash-int3 \
  --quant checkpoints/v41-flash-int3 \
  --manifest checkpoints/v41-flash-int3/manifest.json \
  --host 127.0.0.1 --port 18222 \
  --api-key sk-v41-local \
  --model-id deepseek-v4.1-flash-int3 \
  --cache-gb 40 --workers 4 --max-seq-len 32768 \
  --no-spec
```

Windows 可以直接用仓库里的 `start_v41_server.bat`（自动找
`checkpoints/v41-flash-int3` 或 `ckpt`）。

客户端：

```bash
curl http://127.0.0.1:18222/v1/chat/completions \
  -H "Authorization: Bearer sk-v41-local" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4.1-flash-int3","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

模型 ID 也支持 `deepseek-v4.1-flash-int3-thinking`。完整字段、流式、工具调用见代码仓的
`docs/05-serving.md`。

---

## 实测性能

测试机：Tesla V100-PCIE-32GB + i5-10400F + 64GB DDR4 + NVMe，Windows，贪心解码。

| 场景 | 结果 |
|---|---|
| 33-token 编程提示，32 token 解码 | 平均 0.985 s/token；稳态 **0.74–0.83 s/token（1.2–1.35 tok/s）** |
| 首 token（冷缓存） | 约 20s；2.5k token agent 提示 80–115s |
| 512-token prefill | 约 2 分钟（每层碰 300+ 专家，读约 165GB） |
| 显存峰值 | decode 26.9GiB；prefill 27.0GiB（reserved 27.35GiB） |
| 宿主专家缓存 | 40GB，命中约 66%；显存 10GB slab 命中约 94% |
| DSpark（MTP） | 长程接受率约 63%，端到端 0.67–1.9 s/token，收益不稳定 |

不同提示长度、缓存冷热、是否 thinking 会让数字差很多；不要拿「累计 tok/s」和
「稳态 tok/s」直接比较。复现方法见 `docs/07-benchmarks.md`。

---

## 发布前能力抽检（2026-10-22）

同一份 full int3 权重（384 专家/层，无 MTP，`manifest-int3.json`），V100-32GB，贪心解码，
chat 模板；每题都用断言/数值判分，不是人工看文本。

| 类别 | 题目 | 结果 |
|---|---|---|
| 数学 | 23+48、7×8、144÷12、2x+5=17、80 元七五折 | **5/5** |
| 编程基础 | add / is_even / max_of_three / count_vowels / fib | **5/5** |
| 编程加试 | binary_search / two_sum / valid_parentheses | **3/3** |
| 合计 | 13 题 | **13/13** |

这组题只验证最低可用性，不是 benchmark。项目级代码、长上下文和工具调用仍需自行验证；
nano 版本未参加本次抽检，且已知代码能力不可靠。

---

## 质量与限制

- **代码**：33-token 编程提示下，全量 384 专家能生成结构完整、可运行的快速排序；
  这是「可用」而不是「和原模型完全一致」。
- **工具调用**：OpenAI `tools` + DSML 解析已实现；全量版的工具调用可用，但未做大规模评测。
- **通用能力**：量化有损失；int3 专家的 rel_l2 约 0.29（group=128），长链推理和冷门知识
  可能退化。重要任务请与官方权重对照。
- **上下文**：默认 32K；滑动窗口 128，长上下文靠压缩位置 top-512。KV cache 用 fp16，
  与官方 fp8/fp4 有微小差异。
- **速度**：这不是高吞吐方案；单请求单槽，约为 1 tok/s 量级。4 个并发请求会排队。
- **精度**：不要期待 81 tok/s，也不要把 nano 的速度当成这个版本的速度。

如果你只有 50–100GB 磁盘，可以看
[`JERRYNPC/deepseek-v4.1-flash-nano-v100`](https://huggingface.co/JERRYNPC/deepseek-v4.1-flash-nano-v100)，
但 nano 的代码/工具能力不可靠，只适合演示。

---

## 许可与来源

- 本仓库**权重**是 DeepSeek-V4.1-Flash-1017 的量化衍生作品，**不是官方发布**。
  使用、再分发与商用请遵循上游模型页面的许可；**不是 MIT**。
- 推理**代码**是 MIT；代码仓里有 `LICENSE`。
- 量化格式与实测记录见代码仓 `docs/02-checkpoints.md`、`docs/03-quantization.md`、
  `docs/07-benchmarks.md`。

```text
Derived from deepseek-ai/DeepSeek-V4.1-Flash-1017.
Not an official DeepSeek release.
Quantization: int3 (g128) routed experts, int8 dense, 2-bit row-codebook engram.
Quantized by JERRYNPC, 2026-10.
```

如果这份权重帮你跑通了模型，欢迎在代码仓提 issue 报告你的硬件、速度和问题。
