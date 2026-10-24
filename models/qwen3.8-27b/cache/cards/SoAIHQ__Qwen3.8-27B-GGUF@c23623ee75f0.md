---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: SoAI
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
tags:
- gguf
- llama.cpp
- imatrix
- soai
- multimodal
---

<div align="center">

<a href="https://soai.to"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/SoAIHQ/Qwen3.8-27B-GGUF/resolve/main/soai-logo-dark.png">
  <img alt="SoAI" src="https://huggingface.co/SoAIHQ/Qwen3.8-27B-GGUF/resolve/main/soai-logo-light.png" width="480">
</picture></a>

</div>

<h1 align="center">Qwen3.8-27B · GGUF</h1>

<div align="center">

A 27B dense model with image input, a 256K context and adjustable reasoning depth.

[![](https://img.shields.io/badge/base%20model-Qwen%2FQwen3.8--27B-86b31e?style=flat-square)](https://huggingface.co/Qwen/Qwen3.8-27B) [![](https://img.shields.io/badge/license-apache--2.0-86b31e?style=flat-square)](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) [![](https://img.shields.io/badge/GGUF-Q4__K__M%20%C2%B7%20Q8__0-86b31e?style=flat-square)](https://huggingface.co/SoAIHQ/Qwen3.8-27B-GGUF/tree/main) ![](https://img.shields.io/badge/context-256K-86b31e?style=flat-square) [![](https://img.shields.io/badge/llama.cpp-v0.5.0-86b31e?style=flat-square)](https://github.com/ggml-org/llama.cpp)

</div>

GGUF builds of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by [Qwen](https://huggingface.co/Qwen) for llama.cpp and SoAI, calibrated on real chat, code and tool calls.

---

## At a glance

| Text | Images | Audio | Reasoning | Tool calling | Context |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Yes | Yes | No | Yes | Yes | 256K |

| Specification | Details |
| :--- | :--- |
| Parameters | 27B |
| Architecture | Dense, hybrid Gated DeltaNet + attention |
| Context window | 262,144 tokens |
| Reasoning | Thinking by default, depth set by `reasoning_effort` (`low`, `medium`, `xhigh`) |
| Languages | 201 languages and dialects |
| Made by | [Qwen](https://huggingface.co/Qwen) |

## Choosing a file

| Download | Size | Quality | Best for |
| :--- | ---: | :--- | :--- |
| [**Q4_K_M**](https://huggingface.co/SoAIHQ/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf) | 16.8 GB | High | Most machines, with the best quality for its size |
| [**Q8_0**](https://huggingface.co/SoAIHQ/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q8_0.gguf) | 29.0 GB | Near-lossless | Output closest to the original, when memory allows |
| [mmproj (F16)](https://huggingface.co/SoAIHQ/Qwen3.8-27B-GGUF/resolve/main/mmproj-Qwen3.8-27B-f16.gguf) | 927.6 MB | Add-on | Needed only for image input |

> [!TIP]
> `Q4_K_M` suits most machines. It is about a third of the size of the original 16-bit weights and stays
> close to them in quality. If `Q8_0` fits in your memory, it keeps output closest to the original. SoAI
> does not publish quants below 4 bits because their quality loss is rarely worth the saved space.

The model needs memory for the file and for the context, and the context part grows with the context length you set. llama.cpp can run the model on the GPU, on the CPU, or with its layers split between them.

Images need the `mmproj-Qwen3.8-27B-f16.gguf` add-on as well, loaded with `--mmproj`. `llama-server -hf` downloads it automatically.

## Quick start

With llama.cpp, this command starts a local server with a built-in chat UI and an OpenAI-compatible API:

```bash
llama-server -hf SoAIHQ/Qwen3.8-27B-GGUF:Q4_K_M
```

Open <http://localhost:8080> to chat, or send requests to the API:

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Explain GGUF in one sentence."}],
  "chat_template_kwargs": {"reasoning_effort": "medium"}
}'
```

<details>
<summary>Download the files to a folder first</summary>

```bash
hf download SoAIHQ/Qwen3.8-27B-GGUF --include "Qwen3.8-27B-Q4_K_M.gguf" "mmproj-*" --local-dir Qwen3.8-27B-GGUF
llama-server -m Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-f16.gguf
```

</details>

## Recommended settings

| Mode | `temperature` | `top_p` | `top_k` | `min_p` | `presence_penalty` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Non-thinking | 0.7 | 0.8 | 20 | 0.0 | 1.5 |

Qwen recommends these values in the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B). Thinking switches off with `"enable_thinking": false`. `reasoning_effort` defaults to `xhigh`, and `medium` or `low` answer faster.

In llama-server, you set reasoning per request with `"chat_template_kwargs": {"reasoning_effort": "medium"}`.

## How these files are made

`Q4_K_M` is quantized with an importance matrix (imatrix), which tells llama.cpp which weights affect
the output most so it can store them with more precision. SoAI computes the imatrix from its own
calibration corpus instead of generic web text, and formats every conversation in it with this model's
chat template, including its native tool-call syntax. The model is therefore measured on the token
patterns it sees in use: multi-turn chat in 21 languages, code edits in 22 programming languages, tool
calls and their results, step-by-step math, and web prose.

`Q8_0` is quantized without the imatrix, because llama.cpp's `Q8_0` format does not use one.

Before any quantization, the build checks that the chat, thinking and tool-call markers are stored as
special tokens and that the embedded chat template matches the original. A conversion that imports those
markers as plain text breaks turns, thinking and tool calls without any error message, so the build stops
instead of publishing the file.

The imatrix file is published in this repository, and the provenance table below pins the original
revision and the llama.cpp commit, so you can rebuild these files yourself.

<details>
<summary>Calibration corpus sources</summary>

| Source | Revision | License | Records | Used for |
| --- | --- | --- | --- | --- |
| [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | `af9c13333eb9` | ODC-By 1.0 | 60 | Multilingual web prose in the 21 SoAI UI languages |
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | `87f09149ef47` | ODC-By 1.0 | 20 | English educational prose |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | `179dd21fc551` | Apache-2.0 | 70 | Multi-turn, multilingual human conversations |
| [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | `f9ea04583f02` | Apache-2.0 | 60 | Human-written instructions and answers in SoAI UI languages |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | `8862406ff857` | Apache-2.0 | 60 | Tool definitions, tool calls and tool results |
| [bigcode/commitpackft](https://huggingface.co/datasets/bigcode/commitpackft) | `fc56fe33c030` | MIT (dataset); permissively licensed source files | 88 | Code edits in 22 programming and config languages |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `740312add88f` | MIT | 40 | Step-by-step math reasoning |
</details>

<details>
<summary>All files and SHA-256 checksums</summary>

| File | Type | Size | SHA-256 |
| :--- | :--- | ---: | :--- |
| [Qwen3.8-27B-Q8_0.gguf](https://huggingface.co/SoAIHQ/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q8_0.gguf) | Q8_0 | 29.0 GB | `063af85a2bbc8238…` |
| [Qwen3.8-27B-Q4_K_M.gguf](https://huggingface.co/SoAIHQ/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q4_K_M.gguf) | Q4_K_M | 16.8 GB | `9cad144ac26eab0a…` |
| [mmproj-Qwen3.8-27B-f16.gguf](https://huggingface.co/SoAIHQ/Qwen3.8-27B-GGUF/blob/main/mmproj-Qwen3.8-27B-f16.gguf) | mmproj (F16) | 927.6 MB | `cb0a71fe9b84af25…` |
| [Qwen3.8-27B.imatrix](https://huggingface.co/SoAIHQ/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.imatrix) | imatrix | 13.6 MB | `d70c3d52c2d3d5bf…` |

The full checksums are in [`SHA256SUMS`](https://huggingface.co/SoAIHQ/Qwen3.8-27B-GGUF/blob/main/SHA256SUMS).
</details>

## Provenance

| Item | Details |
| :--- | :--- |
| Original model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by [Qwen](https://huggingface.co/Qwen), revision [`1d4bf0f2ff`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) |
| Converted with | [llama.cpp](https://github.com/ggml-org/llama.cpp) `v0.5.0`, commit [`7fe450e19`](https://github.com/ggml-org/llama.cpp/commit/7fe450e19305b828c199d602c23a8337aaa1f03b) |
| Importance matrix | SoAI calibration corpus v1, rendered with this model's chat template and computed on the Q8_0 weights |
| Weights | The original release, with no fine-tuning, merging or retraining. Only the storage format and numerical precision differ. |
| Built | 2026-09-26 |

## License and credits

These files are distributed under the original model's license (`apache-2.0`), and its full
terms continue to apply. The [license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) and the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B)
also describe the model's capabilities, training data, benchmarks and limitations.

[Qwen](https://huggingface.co/Qwen) designed and trained the model, and SoAI only repackages the
published weights. The conversion uses [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi
Gerganov and the ggml contributors.

## Run it with SoAI

Your models, your hardware, one software suite. [Download SoAI](https://soai.to/#download) for Windows,
macOS, Linux or Android.

<div align="center">

<sub>Packaged by <a href="https://soai.to">SoAI</a>.</sub>

</div>
