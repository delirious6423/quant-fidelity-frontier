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
  - roleplay
  - conversational
  - chinese
  - im-chat
  - persona
  - multi-lora
  - split-phase
  - llama-cpp
  - gguf
  - companion
  - 12gb-vram
  - xiaofeiyu
---

# Xiaofeiyu 26B-A4B v2 (小肥鱼 26B-A4B 极速 MoE 版 — 16GB & 12GB Dynamic GGUF + 5-Slider Voice Suite)

> **26B 总参数、`4B` 激活参数的高吞吐中文伴侣与编程双修模型（`google/gemma-4-26B-A4B-it`）**：通过 **Split-Phase 脑声解耦**，思考链与代码/算账阶段全程 `LoRA = 0.0` 保持 26B 原厂智商（`GEN1..5 = 5/5`），进入 `<|"|>` 聊天气泡瞬间切换五滑条口语卡带（经在线反油腻/反排比 On-Policy DPO 打磨），提供适配 **`RTX 4060 Ti 16GB`**（`UD-Q4_K_XL` / `UD-IQ4_XS`）与 **`RTX 3060 / 4070 12GB`**（`UD-IQ3_M`）的三款 BF16 动态 `imatrix` 量化包。

> **English Summary**: `niatheelf777/xiaofeiyu-26b-a4b` combines `google/gemma-4-26B-A4B-it` (4B active MoE parameters per token) with five composable `.gguf` LoRA voice cartridges (`s0` + `c1..c4`) and three BF16-derived `llama-imatrix` dynamic quantizations for 12 GB, 16 GB, and 24 GB GPUs. Reasoning and code blocks run at `LoRA = 0.0`, while spoken Mandarin IM bubbles blend five persona sliders in real time.

## Xiaofeiyu (小肥鱼) Family Navigation / 家族矩阵导航

| 组件 / 仓库 | 定位与显存门槛 | 链接 |
| :--- | :--- | :--- |
| **12B 主力全能版** | `RTX 3060 / 4070 12GB` 首选单模（`UD-Q4_K_XL` `~8.9 GiB` 显存，代码推理 `5/5`） | [`xiaofeiyu-12b-v4`（12B 主力全能版）](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4) |
| **26B-A4B 极速 MoE 版** | `4B` 激活参数高吞吐 MoE（16GB `UD-Q4_K_XL` / `IQ4_XS`；12GB `UD-IQ3_M`） | **`xiaofeiyu-26b-a4b`（当前仓 · 26B-A4B 极速 MoE 版）** |
| **E4B 通用声带翻译核** | `302 ms` 零思考口语翻译旁路（`~5.3 GiB`），外挂 DeepSeek / Claude / GPT 零改代码 | [`xiaofeiyu-translator-e4b`（E4B 302ms 通用声带翻译核）](https://huggingface.co/niatheelf777/xiaofeiyu-translator-e4b) |
| **在线二维性格球试玩** | 免下载直接拖拽性格球体验五滑条实时混音与代码/算账分相 | [https://xfy.tospeak.top](https://xfy.tospeak.top) |
| **GitHub 一键本地启动仓** | 内置 `:8081` 纸墨风性格球网页 + OpenAI `/v1` 网关 + `DSH` 双模插件 | [github.com/niatheelf777/xiaofeiyu](https://github.com/niatheelf777/xiaofeiyu) |
| **HuggingFace 官方合集** | 小肥鱼全系 GGUF 底座与五槽口语卡带总览 | [Xiaofeiyu Spoken Persona Collection](https://huggingface.co/collections/niatheelf777/xiaofeiyu-split-phase-spoken-persona-family-6afb60d6cecd8097f5fb3a13) |

## Choose a GGUF for Your GPU (`16 GB` & `12 GB` Consumer Editions)

All three dynamic GGUFs are quantized directly from the exact `BF16` `google/gemma-4-26B-A4B-it` checkpoint using `llama-imatrix --parse-special` calibrated on bilingual Mandarin IM + code + native Gemma 4 tool/thought turns (keeping MoE router gates `ffn_gate_inp` in `F32`, shared dense MLP + `attn_v` + `token_embd` in `Q6_K`, and width-704 `ffn_down_exps` in 32-element `Q4_0` blocks). VRAM figures below include all 5 `.gguf` LoRA cartridges + Q8 KV cache:

| Launcher `--preset` | File | File Size | `ctx=4096` VRAM | `ctx=8192` VRAM | L4 Decode (`p50`) | `GEN1..5` | Held-Out `eval_25` | Target GPUs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`26b-a4b`** *(16GB/24GB Sweet Spot)* | `gemma-4-26B-A4B-it-UD-Q4_K_XL-xfy.gguf` | `13.36 GiB` | `14.25 GiB` | `14.29 GiB` | `62.3 tok/s` | `4/5` | `20/25` | **`RTX 4060 Ti 16GB`**, `RTX 4080 16GB`, `L4 24GB` |
| **`26b-a4b-iq4xs`** *(Windows 16GB)* | `gemma-4-26B-A4B-it-UD-IQ4_XS-xfy.gguf` | `13.09 GiB` | `13.97 GiB` | `14.02 GiB` | `67.4 tok/s` | `5/5` | `20/25` | **`RTX 4060 Ti 16GB`** (Windows WDDM desktop active) |
| **`26b-a4b-12g`** *(12GB Consumer)* | `gemma-4-26B-A4B-it-UD-IQ3_M-xfy.gguf` | `9.03 GiB` | `9.92 GiB` | `9.96 GiB` | `74.0 tok/s` | `5/5` | `17/25` | **`RTX 3060 12GB`**, **`RTX 4070 / 4070 SUPER 12GB`** |

> **Windows 12 GB WDDM Tip**: If desktop browsers/Electron apps already occupy >2.5 GiB VRAM on a 12 GB card, either close GPU-heavy windows before running `--preset 26b-a4b-12g` (`~10.3 GiB`) or run [`niatheelf777/xiaofeiyu-12b-v4`](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4) (`--preset 12b-q4`, `~8.9 GiB` VRAM).

## 3-Minute Turnkey Quickstart (Built-in `:8081` 2D Persona Orb Web UI + Dual-Mode `DSH`)

```bash
git clone https://github.com/niatheelf777/xiaofeiyu.git && cd xiaofeiyu
pip install tokenizers huggingface_hub

# 16GB+ GPU (RTX 4060 Ti 16GB / L4 24GB) — launches llama-server + gateway + http://127.0.0.1:8081/ Orb Web UI:
python server/launch_local.py --preset 26b-a4b --auto-download-llama --open

# 16GB Windows Desktop (extra WDDM headroom):
python server/launch_local.py --preset 26b-a4b-iq4xs --auto-download-llama --open

# 12GB GPU (RTX 3060 / 4070 12GB):
python server/launch_local.py --preset 26b-a4b-12g --auto-download-llama --open
```

Connect **DeepSeek Harness (`DSH`)** directly to `http://127.0.0.1:8081/v1` (`model="xiaofeiyu-v4|1.0,0,0.85,0,0"`), or install native multi-bubble rendering + the live header mixer:

```bash
node packages/dsh-plugin/scripts/install-dsh.mjs --apply --profile web --upstream-base http://127.0.0.1:8081/v1
```

## 5-Slot Voice Cartridge Reference

| Slot | File | Scope | Persona Role | Example Model Tag (`w1+w2+w3+w4 <= 1.15`) |
| :---: | :--- | :--- | :--- | :--- |
| `0` | `s0_xiaofeiyu_backbone.gguf` (`s0_xiaofeiyu_26b_a4b.gguf`) | `r=64`, `L0..29` QKV+O+SharedMLP (`141.8 MiB`) | Core Xiaofeiyu Mandarin IM rhythm (SFT + On-Policy Anti-Cliché DPO) | `xiaofeiyu-v4\|1.0,0.0,0.0,0.0,0.0` |
| `1` | `c1_neko_clingy.gguf` | `r=32`, `L10..29` QKV+O+SharedMLP (`47.4 MiB`) | Affectionate clingy catgirl (`粘人猫娘`) | `xiaofeiyu-v4\|1.0,0.85,0.0,0.0,0.0` |
| `2` | `c2_older_tease.gguf` | `r=32`, `L10..29` QKV+O+SharedMLP (`47.4 MiB`) | Playful teasing older sister (`年上御姐`) | `xiaofeiyu-v4\|1.0,0.0,0.85,0.0,0.0` |
| `3` | `c3_brat_mesugaki.gguf` | `r=32`, `L10..29` QKV+O+SharedMLP (`47.4 MiB`) | Cheeky tsundere banter (`恶劣雌小鬼`) | `xiaofeiyu-v4\|1.0,0.0,0.0,0.85,0.0` |
| `4` | `c4_fragile_porcelain.gguf` | `r=32`, `L10..29` QKV+O+SharedMLP (`47.4 MiB`) | Quiet delicate attachment (`易碎依赖`) | `xiaofeiyu-v4\|1.0,0.0,0.0,0.0,0.85` |

## Repository Files & SHA256 Checksums

| File | Size (Bytes) | SHA256 Prefix | Purpose |
| :--- | :--- | :--- | :--- |
| `gemma-4-26B-A4B-it-UD-Q4_K_XL-xfy.gguf` | `14,346,745,216` | `1a09364d0d67` | 16 GB / 24 GB dynamic `imatrix` GGUF |
| `gemma-4-26B-A4B-it-UD-IQ4_XS-xfy.gguf` | `14,053,723,520` | `81fd84bd6cb5` | 16 GB Windows WDDM headroom dynamic `imatrix` GGUF |
| `gemma-4-26B-A4B-it-UD-IQ3_M-xfy.gguf` | `9,700,164,992` | `96ce3a9db25f` | 12 GB consumer GPU dynamic `imatrix` GGUF |
| `s0_xiaofeiyu_backbone.gguf` | `148,712,960` | `ee1e74e3484a` | Slot 0 v2 Voice Core LoRA (standard launcher filename) |
| `s0_xiaofeiyu_26b_a4b.gguf` | `148,712,960` | `ee1e74e3484a` | Slot 0 v2 Voice Core LoRA (explicit 26B alias) |
| `c1_neko_clingy.gguf` | `49,678,784` | `a7348eb62327` | Slot 1 style cartridge |
| `c2_older_tease.gguf` | `49,678,784` | `0313da7c9599` | Slot 2 style cartridge |
| `c3_brat_mesugaki.gguf` | `49,678,784` | `e4c2257d6bb1` | Slot 3 style cartridge |
| `c4_fragile_porcelain.gguf` | `49,678,784` | `72e5816f6ee2` | Slot 4 style cartridge |
| `lora_gguf_manifest.json` | `1,302` | — | Slot order & base revision manifest |

Verified metrics (`UD-Q4_K_XL-xfy` / `UD-IQ4_XS-xfy` + Split-Phase gateway on `L4`): `GEN1..GEN5 = 5/5` (`UD-IQ4_XS`) / `4/5` (`UD-Q4_K_XL`), multi-turn `葱姜蒜` entity recall `5/5` across all 5 sliders, held-out `eval_25 = 20/25` (`0` comma-collapse turns, `0%` prose-wrapped Markdown fences), and `5/5` unique persona replies per prompt.

Base model usage is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
