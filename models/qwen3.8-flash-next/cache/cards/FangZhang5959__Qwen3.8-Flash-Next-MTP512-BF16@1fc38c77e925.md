---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - safetensors
  - vllm
  - mtp
  - speculative-decoding
  - qwen4_exp
language:
  - multilingual
---

# Qwen3.8-Flash-Next-MTP512-BF16

> [GitHub 主页](https://github.com/FangZhangTNS/Qwen3.8-Flash-Next-ZH-Writing-RP) | [魔搭 ModelScope（国内直连）](https://modelscope.cn/models/FangZhangTNS/Qwen3.8-Flash-Next-MTP512-BF16) | [English Technical Overview](#english-technical-overview)

从 **Qwen3.8-Flash-Next** 原生无损提取的独立 **512 专家 BF16 MTP 投机解码草稿权重**（包含全部 31 个 `mtp.*` 张量 + `model.language_model.embed_tokens.weight` + `lm_head.weight`），专为在**未魔改的原版 `vllm==0.34.0`** 上给剪枝主干模型（如 `k192` / `k128`）提供异构 MTP 加速而构建。

本权重完整保留原生 512 专家且不绑定语言，适用于当前 `ZH-Writing-RP` 全部 4 个 Safetensors 档位以及后续 V2 中文/英文/多语言剪枝版本。

## 权重规格

| 属性 | 数值 |
| --- | --- |
| 张量内容 | 33 个张量（31 个 `mtp.*` + `embed_tokens` + `lm_head`），与 BF16 源权重字节级 100% 一致 |
| 分片与体积 | 2 分片（张量净重 `7.224 GiB`，目录总计 `7.246 GiB`） |
| 路由专家数 | 512（未剪枝，与主干模型专家数解耦） |
| 零补丁兼容机制 | 在 `config.json` 中写入 `quantization_config` 并设 `exclude_modules: ["*"]`，使全部 33 个 MTP 张量保持纯 BF16（`TRITON Unquantized MoE backend`），同时绕过上游 `vllm==0.34.0` 在非量化路径下加载 `GatedResidual` 12 行对齐补零 shard 时触发的 `pop_missing_shards()` 严格报错 |

## 启动命令（`vllm==0.34.0`）

可与任意剔除了 `mtp.*` 且保留 `text_config.mtp_num_hidden_layers=1` 的剪枝主干（`W4A16_NVFP4` Marlin 或 `W3A16` Humming）直接搭配：

```bash
vllm serve ./Qwen3.8-Flash-Next-ZH-Writing-RP-K192-NVFP4 \
  --dtype bfloat16 \
  --moe-backend marlin \
  --speculative-config '{"method":"mtp","model":"./Qwen3.8-Flash-Next-MTP512-BF16","num_speculative_tokens":1}'
```

## 校验状态、已知边界与上游许可

- **真机闸门**：已在原生未改版 `vllm==0.34.0`（RTX 5070 Ti 16 GiB WSL2，3 层同构主干）通过 `W4A16_NVFP4` Marlin 联跑（`34.68 tok/s`）与 `W3A16` Humming 联跑（`84.73 tok/s`）。
- **离线审计**：全部 33 个张量与 BF16 源权重字节级一致，全分片 SHA-256 记录于 `sha256_manifest.json`。
- **ROCm 边界**：上游 `vllm/models/qwen4_exp/nvidia/mtp.py:407` 的 `maybe_fuse_shared_experts` 读取了主干 `num_experts`，NVIDIA CUDA 路径默认不触发，ROCm 融合路径可能受限。
- **上游来源**：基于 `Qwen/Qwen3.8-Flash-Next` 及 `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2` 非专家补丁派生，受 **`Qwen Community License 1.0`** 约束（全文见本仓 [LICENSE](LICENSE)；个人/本地使用、修改与二次分发须保留版权声明，商业 MaaS 与商业 AI 编程/办公助手产品须另获 Qwen 授权）。本权重仅面向本地中文创作、角色扮演与科研用途，使用者须遵守适用法律法规。

---

## English Technical Overview

- **Purpose**: Language-neutral, untouched 512-expert BF16 standalone MTP draft checkpoint (`31 mtp.*` + `embed_tokens` + `lm_head`, 2 shards, `7.246 GiB`) for zero-patch heterogeneous speculative decoding with pruned `Qwen3.8-Flash-Next` backbones on unmodified `vllm==0.34.0` (`_make_draft_vllm_config()`).
- **Zero-Patch Config**: `quantization_config` with `exclude_modules: ["*"]` keeps all 33 tensors pure BF16 (`TRITON Unquantized MoE backend`) while bypassing upstream `vllm==0.34.0`'s strict `pop_missing_shards()` error on `GatedResidual`'s 12-row padding shard.
- **Verification**: Hardware-verified with both `W4A16_NVFP4` Marlin (`34.68 tok/s`) and `W3A16` Humming (`84.73 tok/s`) on unmodified `vllm==0.34.0`, plus 100% byte-exact BF16 audit. ROCm caveat: `maybe_fuse_shared_experts` reads target `num_experts` (inactive on NVIDIA CUDA).
