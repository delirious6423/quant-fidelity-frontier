---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
language:
- zh
- en
pipeline_tag: text-generation
library_name: gguf
tags:
- gemma4
- traditional-chinese
- taiwan
- gguf
- llama.cpp
---

# 羲和 HeliosLM Chat 26B-A4B (GGUF, 8 GB GPU)

發佈者 Published by: **Bitstar Inc.**

羲和 Chat 26B-A4B 是以 **Google Gemma 4 26B-A4B-it** 為基礎的對話模型。我們在 Gemma 4 26B-A4B 上以台灣繁體中文、英文等公開資料做持續預訓練，再以 chat vector 方式加回 Gemma 4 26B-A4B-it，並在 chat template 內建一段羲和系統提示（自我介紹、台灣用語、爭議議題保持中立）。這個版本的思考段落較短。

HeliosLM Chat 26B-A4B is a chat model **based on Google Gemma 4 26B-A4B-it**. We continued pre-training Gemma 4 26B-A4B on public Traditional Chinese (Taiwan), English and other data, added that difference to Gemma 4 26B-A4B-it as a chat vector, and built a default HeliosLM system prompt into the chat template. This build keeps its reasoning short.

- 參數量 Parameters: 26B total, about 4B active per token (mixture of experts)
- 授權 License: Apache-2.0, same as Google Gemma 4
- 檔案 File: `helioslm-chat-26b-a4b-Q4_K_M.gguf`（約 17 GB，4-bit）

## 8 GB 顯卡怎麼跑 Running on an 8 GB GPU

26B 是 MoE 模型。用 llama.cpp 把部分 expert 層放在 CPU 記憶體，就能在 8 GB 顯卡上跑。需要約 32 GB 系統記憶體。

It is a mixture-of-experts model: with llama.cpp you can keep part of the expert layers in system RAM and run it on an 8 GB GPU. About 32 GB of system RAM is needed.

```bash
# 8 GB GPU: about 6 GB VRAM
llama-server -m helioslm-chat-26b-a4b-Q4_K_M.gguf -ngl 99 --n-cpu-moe 24 -c 8192 --jinja \
  --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}'

# 24 GB GPU or more: everything on the GPU
llama-server -m helioslm-chat-26b-a4b-Q4_K_M.gguf -ngl 99 -c 16384 --jinja
```

- 顯卡更小時改用 `--cpu-moe`（約 3 GB VRAM，速度略慢）。With less VRAM use `--cpu-moe` (about 3 GB VRAM, a bit slower).
- 不需要思考模式時，請同時加上 `--reasoning-budget 0` 與 `--chat-template-kwargs '{"enable_thinking":false}'`。To turn thinking off, pass both flags shown above.

網站 Website: https://helioslm.pages.dev/

模型可能產生錯誤或過時資訊，請自行查證重要內容。The model can make mistakes; verify important information.
