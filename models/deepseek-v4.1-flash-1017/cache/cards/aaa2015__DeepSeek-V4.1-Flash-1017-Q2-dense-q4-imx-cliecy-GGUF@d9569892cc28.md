---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - vision
  - multimodal
  - image-text-to-text
  - deepseek
  - deepseek-v4.1
  - moe
  - apple-silicon
  - metal
  - myds
  - imatrix
  - dense-q4
  - q2
---

# DeepSeek-V4.1-Flash-1017 · Q2 dense-q4 (imatrix, cliecy)

这是 **DeepSeek-V4.1-Flash-1017** 的 GGUF 量化权重，由
[**myds**](https://github.com/aaa2015/myds) 推理引擎的生产配置直接导出。

| 项 | 值 |
|---|---|
| 基座模型 | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) |
| 许可 | **MIT**（继承基座） |
| 文件 | `DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf` |
| 大小 | **362,142,629,888 字节**（337.2 GiB） |
| SHA256 | `43948333c8eb0172613384abb5d5e1fe2fd02f40a1ede88afc1cbf22bef1cb68` |
| 量化形态 | MoE 专家 **Q2_K**；dense 部分 **Q4_K**（imatrix 校准） |

## 这是什么量化

DeepSeek-V4.1-Flash 是 MoE 架构：**384 个路由专家，每 token 激活 6 个**，
另有 1 个共享专家。这个文件对不同部分用了不同位宽：

- **路由专家（占体积绝大部分）**：`Q2_K`
- **dense 路径**（MLA 的 Q/K/V/Out、共享专家 FFN、LM head）：`Q4_K`
- 校准：imatrix（`cliecy` 校准集）

这样做的理由：decode 阶段每 token 必须读完整个权重面（dense 5.17 GB + MoE 2.39 GB
= 7.55 GB），位宽直接决定速度；而 dense 部分对精度更敏感，所以单独提到 Q4_K。

## 多模态：需要配套的 Vision 编码器

这是**多模态模型**（DeepSeek-V4.1-Flash 原生支持图片输入）。文本与视觉权重分开存放：

| 文件 | 用途 | 大小 |
|---|---|---|
| `DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf` | 语言模型本体 | 337.2 GiB |
| `DeepSeek-V4.1-Flash-1017-Vision.gguf` | **视觉编码器**（DeepSeek-ViT + 2 层 MLP projector） | 925.6 MiB |

**纯文本用途只下第一个即可**；要处理图片必须两个都下，并用 `--vision` 指定第二个：

```bash
./bin/myds -m DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf \
           --vision DeepSeek-V4.1-Flash-1017-Vision.gguf \
           --port 8080 --ctx-size 327680 --alias myds
```

视觉路径的预处理与 ds4 逐字节对拍（见 myds 仓库 `docs/VISION.md`）。

## 用于 myds

这个文件是 [myds](https://github.com/aaa2015/myds) 的**生产配置**，
在该引擎上实测：

| 指标 | 值 |
|---|---|
| prefill | 350–600 t/s（随 prompt 长度变化） |
| decode | **约 38–44 t/s** |
| GPU | Apple M2 Ultra（76 核 GPU，192 GiB 统一内存） |

启动：

```bash
./bin/myds -m DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf \
           --port 8080 --ctx-size 327680 --alias myds
```

## 权重是 8 个分片

本仓库的权重超过托管方单文件上限，因此按**固定字节长度**切分为 8 片
（不是按张量切）。每片 42.16 GiB，最后一片略小：

```
parts/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf.part01 .. part08
parts/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf.parts    ← 清单，58 字节
```

**8 片之和 = 362,142,629,888 字节（337.2 GiB），sha256 = `43948333c8eb0172613384abb5d5e1fe2fd02f40a1ede88afc1cbf22bef1cb68`。**

清单内容：

```
myds-split 1
parts 8
chunk 45268074496
total 362142629888
```

### 用法一：直接读（推荐）

[myds](https://github.com/aaa2015/myds) 支持字节分段，**指定任意一片即可**，
无需合并，也无需 337 GB 的额外磁盘空间：

```bash
./bin/myds -m parts/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf.part01 \
           --port 8080 --ctx-size 327680 --alias myds
```

引擎会把 8 片映射进一段连续虚拟地址，对上层等价于一个文件。
**必须保留 `.parts` 清单**：它是布局的唯一依据，缺失会直接拒绝加载——
少了任何一片都会让后续数据整体错位，静默加载出的是"看起来正常、数据全错"的模型。

### 用法二：合并成单文件

给其他 llama.cpp 系工具用：

```bash
cat parts/*.gguf.part0? > DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
shasum -a 256 DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
# 应得到 43948333c8eb0172613384abb5d5e1fe2fd02f40a1ede88afc1cbf22bef1cb68
```

合并后需要 337 GiB 可用磁盘，加上分片本身共约 675 GiB。

## 硬件要求

**需要约 150 GB 可用统一内存**（实测常驻 148–150 GB）。这意味着：

- Apple Silicon：**M2/M3 Ultra 192 GB** 或更大
- 普通 32/64/128 GB 机器装不下，会 OOM

⚠️ 加载时请确保系统有足够余量：内存不足会在 Metal 分配阶段失败，
表现为 `Insufficient Memory` + `prefill failed`。同一台机器上**不要同时加载两份**。

## 精度说明

量化必然带来损失。这个文件在 **myds** 上的实测行为：

- 7 类核心安全问题识别与 BF16 一致（代码审计对照）
- 长距离检索（大海捞针探针）正常

**但这是 Q2 级别的 MoE 量化，不适合作为精度基准。** 需要精度基线请用官方 safetensors。

## 与 myds 的 0-ULP 概念的关系

myds 有一个「0-ULP」的概念，指的是**同一权重下、引擎内部改动是否逐位等价**，
与本权重的量化误差是**两件不同的事**：

- 量化误差：这个文件相对 BF16 的损失（不可避免）
- 0-ULP：引擎优化相对引擎基线的损失（可通过开关控制）

详见 myds 仓库的 `docs/ULP_SWITCHES.md`。

## 许可

MIT。基座模型 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 的 README 明确写明
「this repository and the model weights are licensed under the MIT License」。

再分发请保留本文件与 myds 的 `LICENSE`。

## 引用

```bibtex
@misc{deepseekai2026deepseekv41flash,
      title={DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression},
      author={DeepSeek-AI},
      year={2026},
}
```
