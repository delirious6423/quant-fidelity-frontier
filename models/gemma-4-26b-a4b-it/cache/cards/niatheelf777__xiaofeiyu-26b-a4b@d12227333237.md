---
license: gemma
base_model: google/gemma-4-26B-A4B-it
language:
  - zh
  - en
pipeline_tag: text-generation
library_name: gguf
tags:
  - gemma-4
  - moe
  - experimental
  - archive
  - roleplay
  - conversational
  - chinese
  - im-chat
  - persona
  - multi-lora
  - split-phase
  - llama-cpp
  - gguf
  - xiaofeiyu
---

# Xiaofeiyu 26B-A4B (实验性架构验证存档 / Experimental Archive)

> **⚠️ 实验性架构验证存档（不推荐日常使用 / Experimental Preview — Not Recommended）**：本仓库仅作为 `26B-A4B` MoE 架构迁移实验存档，作者团队尚未对其进行充分的日常聊天把玩与手感打磨。**日常对话体验与本地部署请直接选用主力打磨版 [`niatheelf777/xiaofeiyu-12b-v4`](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4)。**
>
> **Experimental Preview — Not Recommended for Daily Use**: This repository is preserved solely as a `26B-A4B` MoE architecture migration experiment and has **not** undergone thorough daily chat playtesting or conversational tuning. For everyday conversation quality and local deployment, please use our battle-tested flagship [`niatheelf777/xiaofeiyu-12b-v4`](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4) (`Q8_0` + 5 LoRAs on `≥ 16 GB` VRAM, or custom `UD-Q4_K_XL-xfy` on `12 GB` VRAM).

## Recommended Flagship & Family Navigation / 主力模型导航

| 组件 / 仓库 | 状态与推荐说明 | 链接 |
| :--- | :--- | :--- |
| **12B 主力打磨版（首选）** | 线上 `xfy.tospeak.top` 同款打磨版：`≥ 16 GB` 显卡首选 `Q8_0`（`~12.9 GiB`），`12 GB` 显卡用特制 `UD-Q4_K_XL-xfy`（`~8.5–8.9 GiB`） | **[`niatheelf777/xiaofeiyu-12b-v4`](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4)** |
| **E4B 通用声带翻译核** | `6–8 GB` 小显存适用（`~5.3 GiB`），外挂 DeepSeek / Claude / GPT 改写短气泡 | [`niatheelf777/xiaofeiyu-translator-e4b`](https://huggingface.co/niatheelf777/xiaofeiyu-translator-e4b) |
| **26B-A4B 实验存档** | 仅架构迁移验证存档，未打磨日常手感，不建议日常使用 | **`xiaofeiyu-26b-a4b`（当前存档仓）** |
| **在线二维性格球试玩** | 体验经过充分打磨的 12B 主力版在线效果 | [https://xfy.tospeak.top](https://xfy.tospeak.top) |
| **GitHub 开源主仓** | 一键启动脚本、内置二维性格球网页与 `DSH` 插件 | [github.com/niatheelf777/xiaofeiyu](https://github.com/niatheelf777/xiaofeiyu) |

---

## Archived Artifacts & Reference Checksums (For Reproducibility Only)

This archive preserves the `google/gemma-4-26B-A4B-it` Split-Phase LoRA cartridges and three `llama-imatrix` dynamic GGUFs (`UD-Q4_K_XL-xfy`, `UD-IQ4_XS-xfy`, `UD-IQ3_M-xfy`) created during MoE architecture verification. Launcher support remains intact in `server/launch_local.py` (`--preset 26b-a4b` / `26b-a4b-iq4xs` / `26b-a4b-12g` / `26b-a4b-q8`), though new users should deploy [`niatheelf777/xiaofeiyu-12b-v4`](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4) instead.

| File | Size (Bytes) | SHA256 Prefix | Role |
| :--- | :--- | :--- | :--- |
| `gemma-4-26B-A4B-it-UD-Q4_K_XL-xfy.gguf` | `14,346,745,216` | `1a09364d0d67` | Experimental `UD-Q4_K_XL` MoE GGUF |
| `gemma-4-26B-A4B-it-UD-IQ4_XS-xfy.gguf` | `14,053,723,520` | `81fd84bd6cb5` | Experimental `UD-IQ4_XS` MoE GGUF |
| `gemma-4-26B-A4B-it-UD-IQ3_M-xfy.gguf` | `9,700,164,992` | `96ce3a9db25f` | Experimental `UD-IQ3_M` MoE GGUF |
| `s0_xiaofeiyu_backbone.gguf` | `148,712,960` | `ee1e74e3484a` | Slot 0 Voice Core LoRA (standard launcher filename) |
| `s0_xiaofeiyu_26b_a4b.gguf` | `148,712,960` | `ee1e74e3484a` | Slot 0 Voice Core LoRA (explicit 26B alias) |
| `c1_neko_clingy.gguf` | `49,678,784` | `a7348eb62327` | Slot 1 style cartridge |
| `c2_older_tease.gguf` | `49,678,784` | `0313da7c9599` | Slot 2 style cartridge |
| `c3_brat_mesugaki.gguf` | `49,678,784` | `e4c2257d6bb1` | Slot 3 style cartridge |
| `c4_fragile_porcelain.gguf` | `49,678,784` | `72e5816f6ee2` | Slot 4 style cartridge |
| `lora_gguf_manifest.json` | `1,302` | — | Slot order & base revision manifest |

Base model usage is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
