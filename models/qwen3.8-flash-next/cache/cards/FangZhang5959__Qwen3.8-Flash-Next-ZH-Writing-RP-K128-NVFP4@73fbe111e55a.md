---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - safetensors
  - vllm
  - nvfp4
  - moe-pruning
  - reap
  - chinese
  - roleplay
  - creative-writing
  - qwen4_exp
language:
  - zh
---

# Qwen3.8-Flash-Next-ZH-Writing-RP-K128-NVFP4

> [GitHub 主页](https://github.com/FangZhangTNS/Qwen3.8-Flash-Next-ZH-Writing-RP) | [魔搭 ModelScope（国内直连）](https://modelscope.cn/models/FangZhangTNS/Qwen3.8-Flash-Next-ZH-Writing-RP-K128-NVFP4) | [English Technical Overview](#english-technical-overview)

**Qwen3.8-Flash-Next** 的轻量 Q4 中文写作与角色扮演（RP）剪枝 Safetensors 权重（512 -> 128 专家，剪枝率 75%，top-10 不变，`v3_M2d` 掩码，底座为 `Qwen/Qwen3.8-Flash-Next` 叠加 `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2` 非专家补丁），面向显存更紧凑的 `vllm==0.34.0` 服务器部署。

> **定位声明**：垂直中文文学写作与 RP 专精版（与 GGUF `v1c` 同掩码；同掩码盲评中文写作 Elo **1535** / 文学分 **7.16**、中文 RP Elo **1532** / **8.02**）。领域外能力大幅退化（代码 Pass@1 = **0.0%**、中文百科 = **11.7%**、LogiQA2 = **52.4%**、英文不可用），请勿用于编程、理工百科或英文任务。

## 权重规格

| 属性 | 数值 |
| --- | --- |
| 架构 | `Qwen4ExpForConditionalGeneration`（48 层，128 路由专家，top-10） |
| 专家量化 | 基于剪枝后原生 `k128` 统计校准的 `W4A16_NVFP4`（group-16，E2M1 + E4M3 尺度，共享 `gate_proj`/`up_proj` FP32 `weight_scale_2`） |
| PLE / Engram 词表 | `vllm==0.34.0` 原生 `PLE NVFP4`（26.822 GiB，UVA Linux 默认 `cpu_offload=True`） |
| 敏感层保护 | 字节级原始 BF16（`lm_head`、`embed_tokens`、`model.visual.*`、`hyper_connection`、`linear_attn`、`self_attn`、`mlp.gate`、`shared_expert`） |
| 分片与体积 | 21 分片 / 75,388 张量（张量净重 `52.607 GiB`，目录总计 `52.646 GiB`） |
| GPU 主干显存占用 | PLE 放锁页内存后约 **25.8 GiB**（双 16G 或单 32G+ 显存卡可部署；搭配独立 `MTP512-BF16` 另加 7.2 GiB） |
| 独立 MTP 搭配 | 直接搭配 `Qwen3.8-Flash-Next-MTP512-BF16`（原版 `vllm==0.34.0` 零补丁异构投机解码） |

## 启动命令（`vllm==0.34.0`）

环境要求：`vllm==0.34.0`、`transformers>=5.22.2`、`compressed-tensors>=0.17.0`。

```bash
vllm serve ./Qwen3.8-Flash-Next-ZH-Writing-RP-K128-NVFP4 \
  --dtype bfloat16 \
  --moe-backend marlin \
  --speculative-config '{"method":"mtp","model":"./Qwen3.8-Flash-Next-MTP512-BF16","num_speculative_tokens":1}'
```

推荐中文写作与 RP 采样参数：`temperature=0.7, top_p=0.8, top_k=20, min_p=0.1, repetition_penalty=1.1`。

## 校验状态与上游许可

- **全量离线审计**：全部 21 分片通过张量偏移/形状连续性、全量 SHA-256（`sha256_manifest.json`）、共享 `weight_scale_2` 与未量化 BF16 层原字节比对。
- **上游来源**：基于 `Qwen/Qwen3.8-Flash-Next` 及 `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2` 非专家补丁派生，受 **`Qwen Community License 1.0`** 约束（全文见本仓 [LICENSE](LICENSE)；个人/本地使用、修改与二次分发须保留版权声明，商业 MaaS 与商业 AI 编程/办公助手产品须另获 Qwen 授权）。本权重仅面向本地中文创作、角色扮演与科研用途，使用者须遵守适用法律法规。

---

## English Technical Overview

- **Scope**: Chinese literary writing & RP specialist only (`v3_M2d` 512 -> 128 experts, 75% pruned, top-10 preserved; Coding `0.0%`, Chinese-SimpleQA `11.7%`, English unsuitable). English tracks are planned for V2.
- **Format**: Post-pruning native-calibrated `W4A16_NVFP4` (group-16) + native `vllm==0.34.0` `PLE NVFP4` (`26.822 GiB`) + byte-exact BF16 sensitive layers. 21 shards / 75,388 tensors (`52.646 GiB` total; ~25.8 GiB GPU backbone).
- **Verification**: 100% shard header/offset/shape/SHA-256 and BF16 source byte-exact audit PASS. Full method: [README_EN.md](https://github.com/FangZhangTNS/Qwen3.8-Flash-Next-ZH-Writing-RP/blob/main/README_EN.md).
