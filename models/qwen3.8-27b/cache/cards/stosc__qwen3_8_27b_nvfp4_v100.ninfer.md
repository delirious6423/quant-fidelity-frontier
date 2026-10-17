---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - unsloth/Qwen3.8-27B-NVFP4
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - nvfp4
  - fp8
  - w4a4
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
model-index:
  - name: Qwen3.8-27B-nvfp4-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: IFBench
          type: ifbench
        metrics:
          - type: accuracy
            value: 77.00
            name: Prompt-level strict (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2025
          type: aime25
        metrics:
          - type: accuracy
            value: 96.67
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2026
          type: aime26
        metrics:
          - type: accuracy
            value: 96.67
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: GPQA-Diamond
          type: gpqa_diamond
        metrics:
          - type: accuracy
            value: 90.40
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: image-text-to-text
          name: Image Text to Text
        dataset:
          name: ERQA
          type: erqa
        metrics:
          - type: accuracy
            value: 66.25
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: image-text-to-text
          name: Image Text to Text
        dataset:
          name: RealWorldQA
          type: real_world_qa
        metrics:
          - type: accuracy
            value: 83.53
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
---

# 专为 NInfer 构建的 Qwen3.8-27B NVFP4 模型



> **本仓库的模型文件是从[neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)克隆而来，该模型配合[stosc/ninfer-v100-sm70](https://github.com/stosc/ninfer-v100-sm70)在Tesla V100 32G PCIE接口的显卡先运行稳定。为方部署，故将模型单独存一份。以下是原本模型仓库配套的文档。文档中的推理引擎运行参数仅供参考，是无法直接使用的。**


该仓库包含 [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 的一种 NVFP4 / FP8 混合表示。它将官方 BF16 检查点与 [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) 提供的、已打包定长的文本（Text）权重，以 NInfer 原生的 `.ninfer` 构件格式合并而成。该构件**仅适用于 NInfer**；它不是一个 Transformers 检查点、Safetensors 发行包，也不是 GGUF 文件。
该构件采用 Qwen3.5 Dense（稠密）架构。文本层 0–55 使用 NVFP4 的 MLP 权重；而词元嵌入（token embedding）、注意力输入/输出投影、GDN 的 Q/K/V/Z 及输出投影、完整输出头（output head），以及文本层 56–63 的 MLP 权重，则使用行缩放（row-scaled）FP8。控制权重（control weights）使用 BF16，并单独保存了 MTP、Vision（视觉）与 DFlash2 权重。

## 构件（Artifact）

| 字段 | 值 |
|---|---|
| 文件名 | `qwen3_8_27b_nvfp4.ninfer` |
| 大小 | 23,719,715,844 字节（22.09 GiB） |
| SHA-256 | `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82` |
| 容器版本 | 3 |
| 架构 | `Qwen3_5ForCausalLM` |
| 公开模型名 | `qwen3.8-27b` |
| 对话模板 | [qwen3_8.jinja](https://github.com/Neroued/ninfer/blob/98dada0e03cb073fd07f905400b5904bc6e82759/tools/chat_templates/qwen3_8.jinja)；可用 `--chat-template FILE` 覆盖 |
| 模板默认值 | 思维（thinking）开启；努力程度 `xhigh`；保留闭环（closed-turn）推理 |
| 存储对象数 | 1,246（1,240 个张量 + 6 个资源） |
| NVFP4 张量数 | 112 |
| 行缩放 FP8 张量数 | 146 |

该文件包含 Text、Vision、MTP、DFlash2、优化后的候选头（proposal head，即投机解码头）以及前端资源。视觉与投机权重仅在启动时显式选择后才会加载。来源派生的 NVFP4 与 FP8 权重被原样保留，未作解码与重新量化；仅官方 BF16 词元嵌入被本地编码为行缩放 FP8。

可用以下命令校验已下载文件：

```bash
printf '%s  %s\n' \
  '74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82' \
  'qwen3_8_27b_nvfp4.ninfer' | sha256sum --check
```

本次发布包含来自 `z-lab/Qwen3.8-27B-DFlash2`、修订号 `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` 的完整 DFlash2 配套权重。使用 `--spec dflash2 --draft-tokens 7 --lm-head-draft` 进行选择；支持 draft 数量 1..15。DFlash2 需要下文所列的运行时修订版本。已有的性能与评测表格仍保留其所声明的 MTP 配置与修订版本。

## 运行要求

- [NInfer](https://github.com/Neroued/ninfer) 修订 [`04350ba9`](https://github.com/Neroued/ninfer/commit/98dada0e03cb073fd07f905400b5904bc6e82759) 或更高版本，须从源码编译；
- 64 位 Linux；
- NVIDIA GeForce RTX 5090（`sm_120a`）；
- CUDA Toolkit 13.1 或更新版本。

已拥有官方 v2 文件？可在[本地直接升级](https://github.com/Neroued/ninfer/blob/master/docs/weight-conversion.md#upgrade-an-existing-v2-artifact)，无需重新下载权重。

NInfer 不提供安装目标（install target）或打包二进制文件。源码编译所需的依赖请参见[仓库 README](https://github.com/Neroued/ninfer#quick-start)。

## 下载并运行 CLI 示例

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "用三句话解释预填充（prefill）与解码（decode）。" \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

关于图像、视频与结构化对话历史，请参阅 [CLI 指南](https://github.com/Neroued/ninfer/blob/master/docs/cli.md)。

## 启动本地服务器

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

每个请求的逻轓上限为 240,000 个词元。共享的 240,000 词元 Device KV 池在"两个请求的完成预留量之和能容纳"时可接纳两个并发活动请求；任一请求在独占运行时均可使用整个池。两个额外的 Device 检查点槽（checkpoint slots）、八个固定的 Host State 槽，以及 8 GiB 固定的 Host KV，可在资源压力下保留可复用的续接状态。

API 接口请参阅 [HTTP 服务指南](https://github.com/Neroued/ninfer/blob/master/docs/serving.md)，缓存与准入（admission）语义请参阅[资源调度参考](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/resource-scheduling-and-context-cache.md)。

## 支持的功能

该构件支持：

- 思维（thinking）与非思维（non-thinking）两种模式下的文本生成；
- 图像、多图像、视频及混合多模态消息；
- MTP（多令牌预测）投机解码，draft 窗口 1 至 5；
- DFlash2，draft 窗口 1 至 15，使用内置的 DFlash2 配套权重（`--spec dflash2 --draft-tokens 7`，可选 `--lm-head-draft`）；
- BF16、INT8、FP8、NVFP4 以及 K8V4 的 KV 缓存；
- CUDA Graph 解码与兼容前缀（compatible-prefix）复用；
- 受启动容量约束的小规模并发服务，具备真正的批处理解码（batched decode）；
- NInfer 命令行（CLI）；
- OpenAI Responses Core、OpenAI Chat Completions 与 Anthropic Messages 三种服务接口。

## 性能

下列 MTP0 测量数据在 NInfer 修订 [`f08597d`](https://github.com/Neroued/ninfer/commit/f08597d6eaafce5b875934aaa85854fcd5426df8) 下采集，MTP3 测量数据在修订 [`32c9881`](https://github.com/Neroued/ninfer/commit/32c9881b6783949df4999422a764b3dcaa111b13) 下采集。两轮实验均使用一张 NVIDIA GeForce RTX 5090、CUDA 13.1 编译/运行时、CUDA 驱动 API 13.3、随机采样（stochastic sampling）、INT8 group-64 KV、CUDA Graphs、1,024 词元的预填充分块（prefill chunk），并关闭前缀复用。MTP0 未使用投机后端，上下文上限 262,144 词元；MTP3 每个请求的上下文上限为 131,072 词元，draft token 数为 3。

### 并发 MTP=3 语料总耗时（makespan）

固定语料包含 3 个长推理（long-reasoning）与 12 个跨场景（cross-scenario）样例，每个样例 5 个随机种子，共 75 个请求。每个并发点都启动全新服务器，使用相同的打乱种子（shuffle seed）与有序 HTTP 发送序列。C=1 为串行单请求语料。总耗时包含预填充、解码、负载切换（workload transitions）与最终排空（drain）。

| 并发数 C | 请求数 | 解码词元数 | 总耗时（秒） | 请求/秒 | 语料解码（词元/秒） | 平均批大小 | MTP 接受率 | 加速比 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 752,160 | 4,670.27 | 0.0161 | 161.1 | 1.00 | 60.8% | 1.00× |
| 2 | 75 | 739,951 | 2,510.78 | 0.0299 | 294.7 | 1.98 | 59.2% | 1.86× |
| 4 | 75 | 713,384 | 1,647.74 | 0.0455 | 432.9 | 3.29 | 58.0% | 2.83× |
| 8 | 75 | 723,602 | 2,164.90 | 0.0346 | 334.2 | 2.36 | 57.6% | 2.16× |

全部 300 个请求均完成，未出现请求失败、CUDA 错误或显存溢出（OOM）。C=4 给出最短的完整语料总耗时。C=8 受显存压力限制，其完整语料结果比 C=4 更慢。采样是随机的，因此固定提示词与种子并不保证跨并发数值路线产生词元完全相同的续接；上表给出了确切的解码词元总数。

### 长上下文基线（关闭 MTP）

每个数值为服务器预热后，5 个固定种子下算术平均 ± 样本标准差。

| 提示词词元数 | 预填充阶段（词元/秒） | 服务端 TTFT（毫秒） | 解码阶段（词元/秒） |
|---:|---:|---:|---:|
| 7,680 | 8,340.4 ± 13.0 | 931.6 ± 1.6 | 71.2 ± 0.1 |
| 64,512 | 5,297.9 ± 259.2 | 12,281.1 ± 561.5 | 65.7 ± 0.8 |
| 130,048 | 3,544.7 ± 25.3 | 36,853.5 ± 259.4 | 59.6 ± 0.9 |
| 260,096 | 2,203.1 ± 13.4 | 118,354.8 ± 717.2 | 52.9 ± 2.3 |

### MTP=3 单请求长推理解码

C=1 点为每个样例提供 5 个样本。数值来自服务器阶段计时与投机计数器，为算术平均 ± 样本标准差。

| AIME 2026 样例 | 完成词元数 | 解码阶段（词元/秒） | MTP 接受率 | MTP 每轮词元数 |
|---|---:|---:|---:|---:|
| 第 1 题 | 1,465.4 ± 417.3 | 195.2 ± 4.6 | 76.0% ± 2.4% | 3.28 ± 0.07 |
| 第 15 题 | 65,414.4 ± 271.9 | 151.4 ± 2.0 | 56.2% ± 1.1% | 2.69 ± 0.03 |
| 第 30 题 | 50,023.4 ± 14,839.1 | 167.5 ± 23.7 | 64.6% ± 14.9% | 2.94 ± 0.45 |

### MTP=3 单请求跨场景解码

每个类别含 3 个样例，每个样例 5 个种子，共 15 个样本。

| 类别 | 解码阶段（词元/秒） | MTP 接受率 | MTP 每轮词元数 |
|---|---:|---:|---:|
| 代码（Code） | 194.3 ± 6.1 | 76.4% ± 3.9% | 3.29 ± 0.12 |
| 故事（Story） | 126.1 ± 10.9 | 37.4% ± 5.8% | 2.12 ± 0.17 |
| 翻译（Translation） | 192.3 ± 11.9 | 75.0% ± 6.5% | 3.25 ± 0.19 |
| 结构化输出（Structured output） | 219.8 ± 8.6 | 90.8% ± 5.1% | 3.72 ± 0.15 |

指标定义与精确复现命令，请参阅[完整方法与结果](https://github.com/Neroued/ninfer/blob/master/docs/performance/qwen3.8-27b.md)。

## 评测

该构件通过 NInfer 的 OpenAI 兼容服务路由进行评测，开启思维、MTP=3、INT8 group-64 KV。EvalScope 1.9.0 使用 0-shot 提示词、基于规则的评分（rule-based scoring），每题 1 个样本，采样参数为：温度 1.0、top-p 0.95、top-k 20、presence penalty 0.0、种子 42。文本套件在 252,928 词元上下文上限下运行；多模态套件以 `--vision` 在 81,920 词元上限下运行。

| 基准测试 | NInfer NVFP4 | 正确 / 总数 | 官方 Qwen3.8-27B BF16 |
|---|---:|---:|---:|
| IFBench（提示词级严格） | 77.00% | 231 / 300 | 79.5 |
| AIME 2025 | 96.67% | 29 / 30 | — |
| AIME 2026 | 96.67% | 29 / 30 | — |
| GPQA-Diamond | 90.40% | 179 / 198 | 89.2 |
| ERQA | 66.25% | 265 / 400 | 65.5 |
| RealWorldQA | 83.53% | 639 / 765 | 85.9 |

全部 1,723 个已配置样本均完成并评分。IFBench 另报告：指令级严格 80.50%、提示词级宽松 80.33%、指令级宽松 83.50%。以上均为单样本结果，并非 pass@k。

官方 Qwen3.8-27B BF16 数据来自[上游模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)；其采样设置与 IFBench 指标层级未在上游声明，故最后一列并非同协议对比。NVFP4 在四个重叠基准上的差值均落在 ±2.5 分以内，且上游模型卡未报告任何 AIME 结果。

## 限制

- NInfer 在单张 RTX 5090 与单个 CUDA 设备上执行，每个引擎（Engine）启动时可固定配置 1–8 个活动请求的容量。
- 它不提供大规模或抢占式连续批处理（continuous batching）、优先级/QoS 调度、多 GPU 执行、CPU/GPU 卸载或分布式服务。
- 上下文分配受 GPU 显存与所选 KV 缓存类型约束。
- NInfer 不执行生成的工具调用（tool calls）。

## 来源（Provenance）

| 字段 | 值 |
|---|---|
| 基础仓库 | `Qwen/Qwen3.8-27B` |
| 基础修订号 | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| 基础下载源 | `modelscope.cn/models/Qwen/Qwen3.8-27B` |
| 量化来源仓库 | `unsloth/Qwen3.8-27B-NVFP4` |
| 量化来源修订号 | `60e813d4dbbdc5d64cf3f5a8caf2897bedf03679` |
| 转换配方（recipe） | `qwen3_8_27b_nvfp4` |
| 嵌入编码器 | `fp8_row_maxabs` |
| 转换工具仓库 | `https://github.com/Neroued/ninfer` |
| 最低运行时修订号 | `98dada0e03cb073fd07f905400b5904bc6e82759` |
| 排序输入 SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

构件标识、汇总的对象清单与转换来源发布于 [`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/blob/main/artifact-manifest.json)。精确的存储契约维护于 [v3 容器参考](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/artifact-container.md)。

## 许可证

本 NInfer 构件以 Apache License 2.0 分发。[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 基础仓库与[量化来源仓库](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)同样以 Apache-2.0 许可。用户仍须负责遵守许可证及适用法律法规。
