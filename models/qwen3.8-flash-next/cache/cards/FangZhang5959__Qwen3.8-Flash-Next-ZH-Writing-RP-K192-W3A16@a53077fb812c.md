---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - safetensors
  - vllm
  - w3a16
  - humming
  - moe-pruning
  - reap
  - chinese
  - roleplay
  - creative-writing
  - qwen4_exp
language:
  - zh
---

# Qwen3.8-Flash-Next-ZH-Writing-RP-K192-W3A16

> [GitHub 主页](https://github.com/FangZhangTNS/Qwen3.8-Flash-Next-ZH-Writing-RP) | [魔搭 ModelScope（国内直连）](https://modelscope.cn/models/FangZhangTNS/Qwen3.8-Flash-Next-ZH-Writing-RP-K192-W3A16) | [English Technical Overview](#english-technical-overview)

**Qwen3.8-Flash-Next** 的中文写作与角色扮演（RP）均衡 Q3 剪枝 Safetensors 权重（512 -> 192 专家，top-10 不变，`v3_M2d` 掩码，底座为 `Qwen/Qwen3.8-Flash-Next` 叠加 `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2` 非专家补丁），面向 `vllm==0.34.0` + Humming 后端部署。

> **定位声明**：垂直中文文学写作与 RP 专精版（与 GGUF `v1b` 同掩码；同掩码盲评中文写作 Elo **1626** / 文学分 **6.84**、中文 RP Elo **1592** / **7.91**）。领域外能力大幅退化（代码 Pass@1 = **0.0%**、中文百科 = **18.3%**、LogiQA2 = **52.4%**），请勿用于编程、理工百科或通用英文任务。

## 权重规格

| 属性 | 数值 |
| --- | --- |
| 架构 | `Qwen4ExpForConditionalGeneration`（48 层，192 路由专家，top-10） |
| 专家量化 | 校准 `W3A16`（`uint3b4`，group-64，Humming）：基于 `corpus_v3` 激活二阶矩加权执行 48 点宽区间裁剪 `[0.22..1.05]` + 2 步 Lloyd-Max WLS 尺度精修（激活加权 MSE 较裸 RTN 降低 **75.3%**） |
| PLE / Engram 词表 | `vllm==0.34.0` 原生 `PLE NVFP4`（26.822 GiB，UVA Linux 默认 `cpu_offload=True`） |
| 敏感层保护 | 字节级原始 BF16（`lm_head`、`embed_tokens`、`model.visual.*`、`hyper_connection`、`linear_attn`、`self_attn`、`mlp.gate`、`shared_expert`） |
| 分片与体积 | 21 分片 / 84,604 张量（张量净重 `53.940 GiB`，目录总计 `53.981 GiB`） |
| GPU 主干显存占用 | PLE 放内存后约 **27.1 GiB**（搭配独立 `MTP512-BF16` 另加 7.2 GiB） |
| 独立 MTP 搭配 | 直接搭配 `Qwen3.8-Flash-Next-MTP512-BF16`（原版 `vllm==0.34.0` 零补丁异构投机解码） |

## 启动命令（`vllm==0.34.0`）

环境要求：`vllm==0.34.0`、`transformers>=5.22.2`、`compressed-tensors>=0.17.0`、`humming-kernels>=0.1.19`。

```bash
vllm serve ./Qwen3.8-Flash-Next-ZH-Writing-RP-K192-W3A16 \
  --dtype bfloat16 \
  --moe-backend humming \
  --speculative-config '{"method":"mtp","model":"./Qwen3.8-Flash-Next-MTP512-BF16","num_speculative_tokens":1}'
```

推荐中文写作与 RP 采样参数：`temperature=0.7, top_p=0.8, top_k=20, min_p=0.1, repetition_penalty=1.1`。

## 校验状态与上游许可

- **真机闸门**：在未修改原版 `vllm==0.34.0`（Humming + PLE NVFP4 + 独立 `MTP512-BF16`，3 层同构主干）通过实跑验证（`114.5 s`，`8.95 GiB` VRAM，`84.73 tok/s`），同 prompt 贪婪解码首 7 token 与 `W4A16_NVFP4` Marlin 逐字符一致。
- **全量离线审计**：全部 21 分片通过 `sha256_manifest.json` 与 BF16 源字节级核验。
- **上游来源**：基于 `Qwen/Qwen3.8-Flash-Next` 及 `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2` 非专家补丁派生，受 **`Qwen Community License 1.0`** 约束（全文见本仓 [LICENSE](LICENSE)；个人/本地使用、修改与二次分发须保留版权声明，商业 MaaS 与商业 AI 编程/办公助手产品须另获 Qwen 授权）。本权重仅面向本地中文创作、角色扮演与科研用途，使用者须遵守适用法律法规。

---

## English Technical Overview

- **Scope**: Chinese literary writing & RP specialist only (`v3_M2d` 512 -> 192 experts, top-10 preserved; Coding `0.0%`, Chinese-SimpleQA `18.3%`, LogiQA2 `52.4%`). English tracks are planned for V2.
- **Format**: Activation-calibrated `W3A16` (`uint3b4`, group-64, Humming; `-75.3%` activation-weighted MSE vs bare RTN) + native `vllm==0.34.0` `PLE NVFP4` (`26.822 GiB`) + byte-exact BF16 sensitive layers. 21 shards / 84,604 tensors (`53.981 GiB` total; ~27.1 GiB GPU backbone).
- **Verification**: Passed unmodified `vllm==0.34.0` Humming + `MTP512-BF16` hardware gate (`84.73 tok/s`, first 7 greedy tokens bit-matched Marlin) and 100% shard SHA-256 audit. Full method: [README_EN.md](https://github.com/FangZhangTNS/Qwen3.8-Flash-Next-ZH-Writing-RP/blob/main/README_EN.md).
