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
- reap
- pruning
- v100
- custom-code
- experimental
---

# DeepSeek-V4.1-nano · 80 experts/layer for single V100-32GB

**实验版，非官方。** 用 REAP 把 DeepSeek-V4.1-Flash-1017 每层 384 个路由专家剪到
**80 个**，其中 26 个常驻显存、54 个留在宿主内存按需 H2D。目标是在 32GB V100 上
让专家不再逐 token 读盘，把解码稳定在 1 tok/s 以上。

> **质量警告：** nano 的代码生成和工具调用**不可靠**。K=80 的快速排序会输出语法错误、
> 循环退化成重复 token；DSML 工具块可能不完整。它适合演示、链路验证、研究剪枝边界，
> **不要用于真实编程或 agent 任务**。要写代码请用全量 int3 或 p192。

- 推理代码：`GitHub 源码仓库（链接待补）`（MIT）
- 上游模型：[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- 全量版：[`JERRYNPC/deepseek-v4.1-flash-int3-v100`](https://huggingface.co/JERRYNPC/deepseek-v4.1-flash-int3-v100)
- 共享 engram：[`JERRYNPC/deepseek-v4.1-flash-engram-v100`](https://huggingface.co/JERRYNPC/deepseek-v4.1-flash-engram-v100)

---

## 仓库组成

| 文件 | 大小 | 说明 |
|---|---:|---|
| `manifest.json` | 几 KB | `keep` / `resident_keep` / `reap` 元数据 |
| `dense-00000..00007.safetensors` | 8.45GB | 与全量版相同的 int8 dense |
| `experts3-L0-p80.bin .. L39-p80.bin` | 40 × 1.11GB = 44.2GB | 每层 80 个 int3 专家 |
| `tokenizer.json` / `tokenizer_config.json` | 几 MB | tokenizer |
| `engram_token_map.npy` | 0.5MB | engram 压缩 token 映射 |
| `SHA256SUMS` | 几 KB | 分片校验和 |
| `engram/engram-L1.bin`、`engram/engram-L14.bin` | 55.3GB | **在 engram 仓**，下载到同一目录 |

模型目录约 **53GB**；加 engram 后约 **108GB**（`du` 口径约 101GiB）。

专家文件内部按「常驻优先」排序：前 26 个是 `resident_keep`，启动时装进显存；
后 54 个在宿主 LRU 里，用时通过独立 CUDA 流 H2D。gate 仍是 384 维，
`expert_mask` 把 304 个被剪专家置 `-inf`，`keep_pos` 映射到文件内位置。

---

## 硬件要求

| 项目 | 要求 |
|---|---|
| GPU | **Tesla V100 32GB（sm_70）**；nano 实测占用约 **29.1GiB**，余量很小 |
| 内存 | **64GB**；`--cache-gb 30` 时 RSS 约 33GB |
| 磁盘 | **NVMe，≥120GB 空闲**（模型 53GB + engram 55GB） |
| 系统 | Windows 10/11 x64（已验证）；Linux 未测试 |
| Python | 3.10–3.13（推荐 3.11；3.13.14 已验证） |
| PyTorch | **2.6.0 + cu124**；不要用 2.8+ |

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
pip install -U huggingface_hub
```

### 2. 下载权重（两个 repo 解压到同一目录）

```bash
hf download JERRYNPC/deepseek-v4.1-flash-nano-v100 \
  --local-dir checkpoints/v41-nano

hf download JERRYNPC/deepseek-v4.1-flash-engram-v100 \
  --local-dir checkpoints/v41-nano
```

### 3. 路径便携化

```bash
python tools/make_manifest_portable.py \
  checkpoints/v41-nano/manifest.json --in-place
rm -f checkpoints/v41-nano/manifest.abs.json
```

### 4. 启动服务

```bash
python serve_v41.py \
  --ckpt checkpoints/v41-nano \
  --quant checkpoints/v41-nano \
  --manifest checkpoints/v41-nano/manifest.json \
  --host 127.0.0.1 --port 18222 \
  --api-key sk-v41-local \
  --model-id deepseek-v4.1-nano \
  --cache-gb 30 --workers 2 --max-seq-len 8192 \
  --no-spec
```

Windows 可直接用仓库里的 `start_v41_nano.bat`。

```bash
curl http://127.0.0.1:18222/v1/chat/completions \
  -H "Authorization: Bearer sk-v41-local" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4.1-nano","messages":[{"role":"user","content":"写一个 hello world"}],"max_tokens":64}'
```

模型 ID 也支持 `deepseek-v4.1-nano-thinking`。

---

## 实测性能

测试机：Tesla V100-PCIE-32GB + i5-10400F + 64GB DDR4 + NVMe，Windows，贪心解码。

| 指标 | 结果 |
|---|---|
| 显存 | dense 14.92GiB + 常驻专家 13.39GiB + KV/激活 ≈ **29.1GiB**（reserved 29.6GiB） |
| 宿主内存 | `--cache-gb 30` 时约 33GB RSS |
| 解码 | **约 1.0–1.25 tok/s**（96–260 token 多次实测） |
| 首 token | 3–55s，取决于提示长度和是否 thinking |
| 加载 | 数十秒（dense + 1040 个常驻专家） |

速度比全量版快，是因为 26 个常驻专家完全不走 H2D、54 个非驻留专家全部能装进宿主内存，
稳态几乎不读盘。代价是质量。

---

## 质量边界（重要）

REAP 校准使用 33 条编程/工具/数学/CoT 提示；K=80 保留约 96.6% 的校准显著度。
但显著度不等于能力。同一进程、贪心、48 token 上限的 K 扫描：

```
K=28  1+1 -> "1"        错误
K=32  1+1 -> "1"        错误
K=36  1+1 -> "1"        错误
K=40  1+1 -> "1"        错误
K=48  1+1 -> "1"        错误
K=56  1+1 -> "2"        正确
K=64  快速排序           代码不完整
K=80  快速排序           语法错误、循环退化成重复 token
K=96  快速排序           仍然语法错误
K=192 编程域接近可用
K=384 快速排序           正确、结构完整
```

所以：

- nano（K=80）能生成代码框架，但细节错误，**不能当编程助手**；
- 工具调用未通过测试，DSML 可能不完整；
- 简单问答和演示可用，数学/推理能力明显弱于全量；
- 想提高质量只能增加 K，但 32GB 显存放不下更多常驻专家；
  K=56 全驻留的显存总量约 43.7GiB，已经超卡。

---

## 量化与剪枝格式

- **专家**：从全量 int3 文件按专家整块复制，**不重新量化**；格式与全量版逐位一致：
  对称 int3、group=128、fp16 scale、3.125 bit/权重，单专家 `13,824,000 B`。
- **剪枝**：`tools/reap_calibrate.py` 统计「路由权重 × 专家输出范数」，
  `tools/reap_prune.py 80` 按 `np.lexsort((tie, -count, -score))` 排序保留 top-80。
- **dense**：与全量版相同的 int8 per-channel + fp32/fp16 小张量，加载后 14.92GiB。
- **engram**：与全量版相同的 rowcb2 表，只按 token 查 24 行，不整表驻留。
- **MTP**：nano 不含 DSpark；`manifest.nano.mtp = "omitted"`。

构建脚本和完整账目见代码仓 `docs/06-nano.md`。

---

## 限制

- **不是通用模型**：领域剪枝 + 80 专家，通用能力、长链推理、冷门知识都会退化。
- **不是高吞吐**：单槽、约 1 tok/s；4 个并发请求会排队。
- **显存余量小**：29.1GiB used / 32GiB；更换驱动、增大 `--max-seq-len` 或提高
  `--cache-gb` 都可能 OOM。
- **不要用于生产**：代码/工具调用不可靠，输出需要人工检查。
- **许可**：权重跟随上游模型许可，**不是 MIT**；代码是 MIT。

---

## 许可与来源

```text
Derived from deepseek-ai/DeepSeek-V4.1-Flash-1017.
Not an official DeepSeek release.
REAP-pruned to 80 routed experts/layer, int3 (g128), int8 dense,
2-bit row-codebook engram. 26 experts/layer resident in VRAM.
Quantized/pruned by JERRYNPC, 2026-10.
```

使用、再分发与商用前请阅读上游模型页面的许可条款。这份权重不附带任何正确性保证。
