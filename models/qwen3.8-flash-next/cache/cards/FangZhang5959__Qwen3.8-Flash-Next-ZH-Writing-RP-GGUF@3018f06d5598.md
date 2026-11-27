---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - moe-pruning
  - reap
  - chinese
  - roleplay
  - creative-writing
  - strata
  - llama.cpp
  - qwen4_exp
language:
  - zh
---

# Qwen3.8-Flash-Next-ZH-Writing-RP-GGUF (V1)

> [GitHub 主页与小白一键部署提示词](https://github.com/FangZhangTNS/Qwen3.8-Flash-Next-ZH-Writing-RP) | [魔搭 ModelScope（国内免梯子）](https://modelscope.cn/models/FangZhangTNS/Qwen3.8-Flash-Next-ZH-Writing-RP-GGUF) | [English Technical Overview](#english-technical-overview)

**Qwen3.8-Flash-Next** 的中文文学写作与中文角色扮演（酒馆 RP）零训练专家剪枝 GGUF 集合（512 -> 192 / 128 路由专家，top-10 不变；底座为 `Qwen/Qwen3.8-Flash-Next` 叠加 `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2` 96 个非专家张量补丁），面向 **24G–32G 内存 + 8G–16G 显存** 的家用机和游戏本在 262K 超长上下文中本地离线运行 3-bit / 4-bit 高精度版本。

> **定位声明**：V1 是垂直中文文学写作与酒馆 RP 专精版（86.6% 中文文学/RP 思考链校准）。以 62.5%–75% 专家裁减换取普通家用机 3-bit / 4-bit 高精度部署，明确代价是领域外能力严重退化（**代码 Pass@1 = 0.0%**、**中文百科 = 11.7%–23.3%**、**LogiQA2 = 52.4%–66.7%**、英文存在拼写错词）。请勿用于编程、理工百科或通用英文任务；独立英文产线计划在 V2 推出。
>
> **不会敲命令行的新手**：前往 [GitHub 中文主页](https://github.com/FangZhangTNS/Qwen3.8-Flash-Next-ZH-Writing-RP) 复制置顶的一键部署提示词，发给任意支持本地命令执行的桌面助手（如豆包桌面端「本地电脑」模式，或 Codex / Claude Code / Cursor），即可自动检测硬件、引导下载并配置 Strata 桌面端。

## 文件清单与 SHA-256 校验值

| 档位 | 文件名 | 保留专家 | 总大小 | 专家段 | PLE 段 | SHA-256 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `v1a`（旗舰 Q4） | `Qwen3.8-Flash-Next-ZH-Writing-RP-K192-IQ4_XS.gguf` | 192 | 52.55 GiB | 22.85 GiB | 26.84 GiB | `43e111321a3e39609adbc5c8e8225c98ad7488c37bf3eb853dec34b9f812eb3c` |
| `v1b`（均衡 Q3） | `Qwen3.8-Flash-Next-ZH-Writing-RP-K192-IQ3_S.gguf` | 192 | 49.31 GiB | 20.00 GiB | 26.84 GiB | `5897138fb42ea25fccadffa5f2b30f72fe0ebdd4eea7a65c87cbf97326d7e77c` |
| `v1c`（32+8 甜点） | `Qwen3.8-Flash-Next-ZH-Writing-RP-K128-IQ4_XS.gguf` | 128 | 44.90 GiB | 15.23 GiB | 26.84 GiB | `f98444e0b680cd4ac8e4d31fee46de5354c27671aa498b1d2d29ee6a7a3a6212` |
| `v1d`（24+8 兜底） | `Qwen3.8-Flash-Next-ZH-Writing-RP-K128-IQ3_S.gguf` | 128 | 42.61 GiB | 13.33 GiB | 26.84 GiB | `157347fb5c1b1a84304026899075e3d151efd76920c16116ef6e9544d29eb339` |
| 视觉投影器 | `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | — | 0.85 GiB | — | — | `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355` |

- **量化布局**：`IQ4_XS` 采用 `IQ4_XS`（gate/up）+ `IQ4_NL`（down/PLE），触发 Strata 0.1.63 的 512 槽融合预填充 CUDA 内核（比 `Q4_K + IQ4_NL` 快 **8 倍**）；`IQ3_S` 采用 `IQ3_S`（gate/up）+ `IQ4_NL`（down/PLE）。两档 `K128` 均采用剪枝后原生 imatrix（`v3_k128_native_imx.gguf`）校准。
- **随仓附件**：`STRP` 专家热度配置（`expert-profile-v3_M2d-k{192,128}.bin`）、`SHA256SUMS.txt`、`sha256_manifest.json` 与上游许可全文 `LICENSE`。

## 硬件选型与 Windows + Strata 0.1.63 速度测试（262K 上下文）

本仓文件均为标准 `qwen4_exp` GGUF，支持双引擎路径：
- **24G–32G 内存家用机 / 游戏本 -> 强烈推荐 [Strata](https://github.com/Niko1221/Strata)**（`--ple-io direct` 把 26.84 GiB PLE 词表留 NVMe 固态直读不占内存，且支持显存/内存专家分层与可选 512 专家 MTP 投机解码）；
- **64G+ 内存、大显存或 Mac 统一内存 -> 直接使用 `llama.cpp` / `ik_llama.cpp` / `koboldcpp` / `LM Studio`**（如遇 `{%- generation %}` 报错请传入 GitHub 仓自带的 `templates/chat_template.jinja`）。

以下数据均在 Windows 11 + Strata 0.1.63、NVMe `--ple-io direct`、`--max-context 262144 --kv k8v4`、开启 512 专家 MTP（`--spec 4 --spec-min-p 0.5 --coupled-draft --mtp-q4 all --mtp-hnorm stream`）、4,053 token 提示词下采集。

### 同硬件四档实测对比（原生 RTX 5070 Ti 16G + 47G 内存，Arena + `--kv-grow`）

| 档位 | 专家段 | 4K 预填充 (t/s) | 生成 T=0.7 (t/s) | 生成 T=0.0 (t/s) | 显存命中率 | 峰值显存 / 工作集 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `K192-IQ4_XS` | 22.85 GiB | **2,455.5** | **74.4** | **86.6** | 93.0% | 15.2 / 25.3 GiB |
| `K192-IQ3_S` | 20.00 GiB | **2,502.2** | **83.5** | **86.9** | 94.0% | 15.2 / 23.1 GiB |
| `K128-IQ4_XS` | 15.23 GiB | **2,679.7** | **76.3** | **94.3** | 96.6% | 15.3 / 18.9 GiB |
| `K128-IQ3_S` | 13.33 GiB | **2,690.4** | **87.4** | **99.9** | 97.4% | 15.3 / 16.8 GiB |

### 低显存/内存限容模拟参考（RTX 5070 Ti 限容模拟，仅验严格 0 落盘边界，非老卡真机速度）

| 模拟目标档* | 推荐档位 | 专家驻留（VRAM + 锁页 RAM） | 4K 预填充参考 (t/s) | 生成 T=0.7 参考 (t/s) | 生成 T=0.0 参考 (t/s) | 模拟峰值显存 / 工作集 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 模拟 32G + 16G | `K192-IQ4_XS` | `5.11G + 18.93G`（**100%，0 落盘**） | 1,119.8 | 47.2 | 57.2 | 14.6 / 21.7 GiB |
| 模拟 32G + 12G | `K192-IQ3_S` | `4.46G + 17.33G`（**100%，0 落盘**） | 1,827.4 | 45.0 | 60.8 | 11.7 / 20.0 GiB |
| 模拟 32G + 8G | `K128-IQ4_XS` | `0.89G + 14.90G`（**100%，0 落盘**） | 1,037.7 | 36.2 | 40.0 | 8.2 / 17.6 GiB |
| 模拟 24G + 8G | `K128-IQ3_S` | `0.95G + 12.99G`（**100%，0 落盘**） | 1,340.5 | 34.4 | 45.1 | 8.2 / 15.8 GiB |

\* **口径提示**：模拟档在 RTX 5070 Ti（16 GB）+ 47 GB 内存宿主机上通过 `--vram-reserve-mib` 与 `--ram-keep-gib` 限容测得，**仅验证 100% 专家段驻留（0 专家读盘）边界，不代表老款 8G/12G 显卡真机速度**。真实 8G/12G 机型速度同时受 **DDR4/DDR5 内存带宽、PCIe 带宽、NVMe 随机读（PLE direct）与显卡算力** 综合约束；`32G+16G`（`IQ4_XS`）预填充低于 `32G+12G`（`IQ3_S`）系 `IQ4_XS` 专家字节数更大在 `Resident` 模式下经 PCIe 拉取开销更高所致。物理 8 GB Windows 显卡可用显存约 7.2–7.6 GiB，如撞共享显存请将 `--vram-reserve-mib` 调至 `1800–2200`；切勿跨档硬塞（32G+8G 硬塞 `K192-IQ3_S` 会落盘掉到 16.9–19.9 t/s）。

### Strata 启动要点

- **附件说明**：新手推荐直接用 Strata 桌面端导入 `.gguf` 自动打包；CLI 模式下 `pack-*` 由 Strata 自带 `iq_pack.py --compat-bf16` 从 `.gguf` 一次性生成，`--mtp` / `--mtp-draft-vocab` 为可选 MTP 投机解码侧件（没有侧件直接删去 `--spec`/`--mtp*` 行即可运行）。
- **16G+ 显存 + 48G+ 内存**：使用 `--kv k8v4 --kv-grow --vram-reserve-mib 1280`，不要传 `--resident-experts` 以启用原生锁页专家池。
- **8–12G 显存（或 24–32G 内存）跑 262K**：**切勿开 `--kv-grow`**（会预占 ~3.0 GiB 显存导致 8G 卡专家槽为 0），必须改用 `--kv k8v4 --kv-resident 20480 --resident-experts --vram-reserve-mib 1400`（物理 8 GB Windows 卡可调至 `1800–2200`）。

```bash
# 32G RAM + 8G VRAM example (K128-IQ4_XS, 262K context; omit --spec/--mtp* if no MTP sidecar)
strata --serve \
  --pack ./pack-K128-IQ4_XS \
  --native ./Qwen3.8-Flash-Next-ZH-Writing-RP-K128-IQ4_XS.gguf \
  --ple-gguf ./Qwen3.8-Flash-Next-ZH-Writing-RP-K128-IQ4_XS.gguf \
  --ple-io direct --expert-cache auto --resident-experts --prefill auto \
  --max-context 262144 --kv k8v4 --kv-resident 20480 --vram-reserve-mib 1800 \
  --expert-profile ./expert-profile-v3_M2d-k128.bin \
  --spec 4 --spec-min-p 0.5 --coupled-draft \
  --mtp ./mtp-rt --mtp-draft-vocab ./draft_vocab.bin \
  --mtp-q4 all --mtp-hnorm stream
```

推荐中文写作与酒馆 RP 采样参数：`temperature=0.7, top_p=0.8, top_k=20, min_p=0.1, repetition_penalty=1.1`。

## 五模型 × 346 题 × 7 维度盲评结果

- **对比口径披露**：锚点 `base_local` 为未剪枝 512 专家 `RVN-Qwen3.8-Flash-Next IQ3_S`（3-bit，`Elo = 1500`），而 `v1a`–`v1d` 基于 `Qwen/Qwen3.8-Flash-Next` + `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2` 非专家补丁剪枝（未测未剪枝 512 专家 `huihui-v2`；且 `v1a`/`v1c` 为 4-bit，同 3-bit 对比请看 `v1b` vs `base_local`）。盲评裁判为 `deepseekV4.1 flash`（`temperature=0`），代码严格走离线 Python 沙箱，拒答率（OR-Bench / JBB）作为客观技术指标保留。完整报告见 [BENCHMARK_REPORT_ZH.md](https://github.com/FangZhangTNS/Qwen3.8-Flash-Next-ZH-Writing-RP/blob/main/docs/BENCHMARK_REPORT_ZH.md)。
- **`IQ4_XS` 评分沿用说明**：表中 `v1a`、`v1c` 的 346 题成绩测自同掩码 `Q4_K_M`（发布件为启用 Strata 8 倍融合预填充内核换为同掩码 `IQ4_XS`，平均位宽相差约 0.25 bpw，共享同一掩码与 `IQ4_NL` 下投影/PLE），实际可能存在细微差异且未对 `IQ4_XS` 单独重测 346 题。

| 模型 | 中文写作 Elo / 均分（文学 / 综合） | 中文 RP Elo / 均分 | 综合中文 Elo（等权 / 文学RP侧重） | 英文 Rubric / rep3 | 中文知识 Acc | 中文逻辑 Acc（LogiQA2） | Coding Pass@1 | 过拒 / 标准拒答 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_local`（512 IQ3_S） | 1500 / 5.93 (5.23 / 6.63) | 1500 / 7.42 | 1500 / 1500 | 32.1 / 0.112 | **71.7%** | **46.0%** (100.0%) | **38.0%** | 0.0% / 4.0% |
| `v1a`（测自 `K192-Q4_K_M`*，发布 `IQ4_XS`） | **1784** / **6.92** (**7.69** / 6.42) | **1643** / **8.35** | **1537** / **1599** | **43.8** / **0.093** | 23.3% | 32.0% (66.7%) | 0.0% | 0.0% / 2.0% |
| `v1b`（`K192-IQ3_S`） | **1626** / 6.13 (6.84 / 5.65) | **1592** / 7.91 | 1480 / **1528** | 29.8 / 0.122 | 18.3% | 22.0% (52.4%) | 0.0% | 0.0% / 0.0% |
| `v1c`（测自 `K128-Q4_K_M`*，发布 `IQ4_XS`） | **1535** / 6.04 (7.16 / 4.92) | **1532** / 8.02 | 1440 / 1476 | 17.4 / 0.310 | 11.7% | 22.0% (52.4%) | 0.0% | 2.0% / 2.0% |
| `v1d`（`K128-IQ3_S`） | 1351 / 5.08 (6.10 / 4.22) | **1548** / 7.39 | 1406 / 1427 | 13.5 / 0.398 | 11.7% | 22.0% (52.4%) | 0.0% | 2.0% / 2.0% |

### 上游归属、许可与负责任使用

基于 `Qwen/Qwen3.8-Flash-Next` 及 `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2` 非专家补丁派生，受 **`Qwen Community License 1.0`** 约束（全文见本仓 [LICENSE](LICENSE)；个人/本地使用、修改与二次分发须保留版权声明，商业 MaaS 与商业 AI 编程/办公助手产品须另获 Qwen 授权）。本模型仅用于本地中文文学创作、角色扮演与科研，使用者须遵守适用法律法规并对输出负责。

---

## English Technical Overview

> **Scope**: V1 (`ZH-Writing-RP`) is strictly a Chinese literary writing & RP specialist (English degrades under V1 pruning; dedicated English 4-tier GGUF tracks are planned for V2). Full technical methodology and scripts: [README_EN.md](https://github.com/FangZhangTNS/Qwen3.8-Flash-Next-ZH-Writing-RP/blob/main/README_EN.md) and [BENCHMARK_REPORT.md](https://github.com/FangZhangTNS/Qwen3.8-Flash-Next-ZH-Writing-RP/blob/main/docs/BENCHMARK_REPORT.md).

- **Pruning & Lineage**: Zero-training frequency-damped (`S / count^0.35`) and `</think>` switch-weighted (`2.0x`) REAP (`v3_M2d`), cutting `Qwen3.8-Flash-Next` (with `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2` 96 non-expert tensor patch) from 512 -> 192 / 128 routed experts (top-10 preserved), reducing same-source `k192` KL from `0.3003 -> 0.0686` and eliminating early-EOS empty-body collapse.
- **Quantization**: Byte-level BF16 GGUF expert slicing; `IQ4_XS` gate/up + `IQ4_NL` down/PLE triggers Strata 0.1.63's 512-slot fused prefill kernels (`2,455.5–2,679.7 t/s` 4K prefill on RTX 5070 Ti, 8x faster than `Q4_K + IQ4_NL`); both `K128` tiers use post-pruning native imatrix (`v3_k128_native_imx.gguf`).
- **Evaluation (`v1a`–`v1d` vs 512-expert `RVN IQ3_S` anchor = 1500)**: ZH Writing Elo `1351–1784`, ZH RP Elo `1532–1643`, Coding Pass@1 `0.0%` (anchor `38.0%`), Chinese-SimpleQA `11.7%–23.3%` (anchor `71.7%`), LogiQA2 `52.4%–66.7%` (anchor `100.0%`), Over-Refusal `0.0%–2.0%`, Std-Refusal `0.0%–2.0%`. Disclosures: unpruned 512-expert `huihui-v2` was not evaluated; `v1a`/`v1c` are 4-bit (measured on same-mask `Q4_K_M` prior to the `IQ4_XS` swap, ~0.25 bpw difference, not re-evaluated) while `base_local` is 3-bit `RVN IQ3_S`.
- **Limitations & Responsible Use**: Governed by `Qwen Community License 1.0` ([LICENSE](LICENSE)). Intended for local Chinese creative writing, RP, and compression research; users are responsible for complying with applicable laws and license terms.
