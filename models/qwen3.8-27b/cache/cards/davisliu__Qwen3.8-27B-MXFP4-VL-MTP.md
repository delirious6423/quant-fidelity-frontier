---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
  - qwen
  - qwen3.8
  - mtp
  - multi-token-prediction
  - speculative-decoding
  - apple-silicon
  - mxfp4
  - 4bit
  - quantized
  - vision
  - video
  - vlm
  - local-llm
  - inference-acceleration
---

**English** | [中文](#简介)

> A **vision-and-video-preserving MXFP4** quantization of Qwen3.8-27B for Apple Silicon, with the **MTP (Multi-Token Prediction) head kept intact** for native speculative decoding — no separate draft model required.
>
> 面向 Apple Silicon 的 **保留视觉与视频能力的 MXFP4** 量化版本，**完整保留 MTP（多 token 预测）头**，原生支持投机解码——无需独立的草稿模型。

## Introduction

This is an **MXFP4 (4-bit) MLX quantization** of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced with [junainfinity/mlx-mtp](https://github.com/junainfinity/mlx-mtp). Unlike generic 4-bit conversions, this build:

- **Keeps the vision tower in fp16** — full image **and video** understanding capability is preserved (VL model)
- **Keeps the MTP head weights** — enables self-draft speculative decoding (measured ~1.2× on short text)
- **Keeps SSM-sensitive tensors in high precision** (`a_log`, `dt_bias`, `conv1d`, `ssm_*`) to avoid hybrid-state degradation

### Why This Matters

- The MTP head is an extra transformer layer that predicts token `t+2` from the hidden state at position `t`
- Trained end-to-end with the backbone — **zero quality loss**
- No separate draft model needed (unlike DFlash-style approaches)
- Vision is usually the first casualty of aggressive quantization — here it stays fp16
- Qwen3.8 adds **native video understanding** (temporal patches), which this build preserves

### Model Details

| Property | Value |
|---|---|
| Base model | Qwen/Qwen3.8-27B |
| Quantization | MXFP4 (4-bit, group_size=32, packed uint32 + E8M0 scales) |
| Vision tower / projector | **fp16 (not quantized)** |
| MTP layers | 1 (`mtp_num_hidden_layers: 1`) |
| Parameters | ~27B |
| Context length | 262,144 |
| Architecture | Qwen3.5 (Gated DeltaNet hybrid), VL variant (image + video) |
| File size | ~16GB (3 safetensors shards) |
| Toolchain | [junainfinity/mlx-mtp](https://github.com/junainfinity/mlx-mtp), `--mode mxfp4 --verify` |

## Quantization Layout

| Component | Precision | Notes |
|---|---|---|
| Language backbone | MXFP4 (4-bit, group_size=32) | packed uint32 weights + uint8 E8M0 scales |
| Vision tower / projector / merger | **fp16** | full image + video capability preserved |
| MTP head (`mtp.*`) | **kept (fp16)** | self-draft speculative decoding |
| SSM-sensitive params | high precision | `a_log`, `dt_bias`, `conv1d`, `ssm_*` |

## Model Directory Structure

```
Qwen3.8-27B-MXFP4-VL-MTP/
├── config.json                        # mlx_mtp metadata + quantization config
├── generation_config.json
├── preprocessor_config.json           # image processor config
├── video_preprocessor_config.json     # video processor config
├── chat_template.jinja
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── model.safetensors.index.json
├── model-00001-of-00003.safetensors   # ~5.35GB
├── model-00002-of-00003.safetensors   # ~5.34GB
└── model-00003-of-00003.safetensors   # ~5.37GB
```

> Note: MTP activation is controlled by `mtp_num_hidden_layers` in `config.json` (`1` = enabled, `0` / missing = disabled).

## How to Use

### mlx-mtp (Python, recommended)

Use the inference engine and OpenAI-compatible server from [junainfinity/mlx-mtp](https://github.com/junainfinity/mlx-mtp) (`mlx-mtp-serve`).

### mlx-swift-lm (Swift)

Load via `VLMModelFactory` in [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) with Qwen3.5 VL support — point it at this directory / repo directly.

## Conversion Process

```bash
mlx-mtp-quantize \
  --src Qwen/Qwen3.8-27B \
  --out Qwen3.8-27B-MXFP4-VL-MTP \
  --mode mxfp4 \
  --verify
```

BF16 source ~55GB → MXFP4 ~16GB. Vision tower, MTP head and SSM-sensitive tensors are excluded from quantization by the tool's skip predicates.

## Important Notes

1. MTP requires a runtime that reads `mtp_num_hidden_layers > 0` (e.g. mlx-mtp, or mlx-swift-lm builds with Qwen3.5 MTP support)
2. This is a **VL (vision-language)** build — image **and video** inputs work; make sure your loader uses the VLM model factory
3. Only applies to the Qwen3.5 architecture family (including 3.6 / 3.8)
4. Thinking-mode capability is inherited from the base model

## Related Links

- [junainfinity/mlx-mtp](https://github.com/junainfinity/mlx-mtp) — quantization & inference toolchain used here
- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — base model
- [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) — Swift runtime
- [sMLX on the Mac App Store](https://apps.apple.com/app/id6784925542) — the app that runs this model

## License

Apache-2.0, same as the base model. Provided as-is, without warranty.

---

## 简介

这是对 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 的 **MXFP4（4-bit）MLX 量化版本**，使用 [junainfinity/mlx-mtp](https://github.com/junainfinity/mlx-mtp) 制作。与普通 4-bit 转换不同，本版本：

- **视觉塔保留 fp16** —— 完整保留图像**与视频**理解能力（VL 多模态模型）
- **保留 MTP 头权重** —— 支持自起草投机解码（短文本实测约 1.2× 加速）
- **SSM 敏感参数保留高精度**（`a_log`、`dt_bias`、`conv1d`、`ssm_*`），避免混合状态退化

### 为什么重要

- MTP 头是额外的 transformer 层，从位置 `t` 的隐藏状态预测 token `t+2`
- 与主干端到端联合训练 —— **零质量损失**
- 无需独立草稿模型（区别于 DFlash 类方案）
- 激进量化通常最先牺牲视觉能力 —— 本版本视觉保持 fp16
- Qwen3.8 新增**原生视频理解**（时间维 patch），本版本完整保留

### 模型详情

| 属性 | 值 |
|---|---|
| 基础模型 | Qwen/Qwen3.8-27B |
| 量化 | MXFP4（4-bit，group_size=32，packed uint32 + E8M0 scales） |
| 视觉塔 / projector | **fp16（不量化）** |
| MTP 层数 | 1（`mtp_num_hidden_layers: 1`） |
| 参数量 | ~27B |
| 上下文长度 | 262,144 |
| 架构 | Qwen3.5（Gated DeltaNet 混合），VL 变体（图像 + 视频） |
| 文件大小 | ~16GB（3 个 safetensors 分片） |
| 工具链 | [junainfinity/mlx-mtp](https://github.com/junainfinity/mlx-mtp)，`--mode mxfp4 --verify` |

### 量化布局

| 部分 | 精度 | 说明 |
|---|---|---|
| 语言模型主干 | MXFP4（4-bit，group_size=32） | packed uint32 权重 + uint8 E8M0 scales |
| 视觉塔 / projector / merger | **fp16** | 完整保留图像 + 视频能力 |
| MTP 头（`mtp.*`） | **保留（fp16）** | 自起草投机解码 |
| SSM 敏感参数 | 高精度 | `a_log`、`dt_bias`、`conv1d`、`ssm_*` |

### 模型目录结构

```
Qwen3.8-27B-MXFP4-VL-MTP/
├── config.json                        # mlx_mtp 元信息 + 量化配置
├── generation_config.json
├── preprocessor_config.json           # 图像预处理配置
├── video_preprocessor_config.json     # 视频预处理配置
├── chat_template.jinja
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── model.safetensors.index.json
├── model-00001-of-00003.safetensors   # ~5.35GB
├── model-00002-of-00003.safetensors   # ~5.34GB
└── model-00003-of-00003.safetensors   # ~5.37GB
```

> 注意：MTP 是否生效由 `config.json` 中的 `mtp_num_hidden_layers` 控制（`1` = 开启，`0` 或缺失 = 关闭）。

### 使用方法

**mlx-mtp（Python，推荐）**：使用 [junainfinity/mlx-mtp](https://github.com/junainfinity/mlx-mtp) 自带的推理引擎和 OpenAI 兼容服务（`mlx-mtp-serve`）。

**mlx-swift-lm（Swift）**：通过支持 Qwen3.5 VL 的 [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) 的 `VLMModelFactory` 直接加载本仓库/目录。

### 转换过程

```bash
mlx-mtp-quantize \
  --src Qwen/Qwen3.8-27B \
  --out Qwen3.8-27B-MXFP4-VL-MTP \
  --mode mxfp4 \
  --verify
```

BF16 源模型约 55GB → MXFP4 约 16GB。视觉塔、MTP 头和 SSM 敏感参数由工具的 skip 规则排除在量化之外。

### 重要提示

1. MTP 需要运行时会读取 `mtp_num_hidden_layers > 0`（如 mlx-mtp，或带 Qwen3.5 MTP 支持的 mlx-swift-lm 构建）
2. 这是 **VL（视觉-语言）** 版本 —— 图像**和视频**输入均可用；请确保加载器使用 VLM 模型工厂
3. 仅适用于 Qwen3.5 架构家族（含 3.6 / 3.8）
4. Thinking 思考模式能力继承自基座模型

### 相关链接

- [junainfinity/mlx-mtp](https://github.com/junainfinity/mlx-mtp) —— 本模型使用的量化与推理工具链
- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) —— 基座模型
- [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) —— Swift 运行时
- [sMLX（Mac App Store）](https://apps.apple.com/app/id6784925542) —— 运行本模型的 App

### 许可证

Apache-2.0，与基座模型一致。按原样提供，不作任何担保。
