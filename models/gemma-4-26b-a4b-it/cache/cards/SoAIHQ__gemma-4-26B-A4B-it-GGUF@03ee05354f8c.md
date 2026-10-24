---
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
quantized_by: SoAI
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
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
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/SoAIHQ/gemma-4-26B-A4B-it-GGUF/resolve/main/soai-logo-dark.png">
  <img alt="SoAI" src="https://huggingface.co/SoAIHQ/gemma-4-26B-A4B-it-GGUF/resolve/main/soai-logo-light.png" width="480">
</picture></a>

</div>

<h1 align="center">gemma-4-26B-A4B-it · GGUF</h1>

<div align="center">

A mixture of experts with 25.2B parameters, of which 3.8B are active for each token, so each token costs about as much compute as a 4B model.

[![](https://img.shields.io/badge/base%20model-google%2Fgemma--4--26B--A4B--it-86b31e?style=flat-square)](https://huggingface.co/google/gemma-4-26B-A4B-it) [![](https://img.shields.io/badge/license-apache--2.0-86b31e?style=flat-square)](https://ai.google.dev/gemma/docs/gemma_4_license) [![](https://img.shields.io/badge/GGUF-Q4__K__M%20%C2%B7%20Q8__0-86b31e?style=flat-square)](https://huggingface.co/SoAIHQ/gemma-4-26B-A4B-it-GGUF/tree/main) ![](https://img.shields.io/badge/context-256K-86b31e?style=flat-square) [![](https://img.shields.io/badge/llama.cpp-v0.5.0-86b31e?style=flat-square)](https://github.com/ggml-org/llama.cpp)

</div>

GGUF builds of [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) by [Google](https://huggingface.co/google) for llama.cpp and SoAI, calibrated on real chat, code and tool calls.

---

## At a glance

| Text | Images | Audio | Reasoning | Tool calling | Context |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Yes | Yes | No | Yes | Yes | 256K |

| Specification | Details |
| :--- | :--- |
| Parameters | 25.2B total, 3.8B active |
| Architecture | Mixture of experts (128 experts, 8 active) |
| Context window | 262,144 tokens |
| Reasoning | Configurable thinking, switched per request |
| Languages | 35+ languages (pre-trained on 140+) |
| Made by | [Google](https://huggingface.co/google) |

## Choosing a file

| Download | Size | Quality | Best for |
| :--- | ---: | :--- | :--- |
| [**Q4_K_M**](https://huggingface.co/SoAIHQ/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-Q4_K_M.gguf) | 16.8 GB | High | Most machines, with the best quality for its size |
| [**Q8_0**](https://huggingface.co/SoAIHQ/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-Q8_0.gguf) | 26.9 GB | Near-lossless | Output closest to the original, when memory allows |
| [mmproj (F16)](https://huggingface.co/SoAIHQ/gemma-4-26B-A4B-it-GGUF/resolve/main/mmproj-gemma-4-26B-A4B-it-f16.gguf) | 1.2 GB | Add-on | Needed only for image input |

> [!TIP]
> `Q4_K_M` suits most machines. It is about a third of the size of the original 16-bit weights and stays
> close to them in quality. If `Q8_0` fits in your memory, it keeps output closest to the original. SoAI
> does not publish quants below 4 bits because their quality loss is rarely worth the saved space.

The model needs memory for the file and for the context, and the context part grows with the context length you set. llama.cpp can run the model on the GPU, on the CPU, or with its layers split between them.

Images need the `mmproj-gemma-4-26B-A4B-it-f16.gguf` add-on as well, loaded with `--mmproj`. `llama-server -hf` downloads it automatically.

## Quick start

With llama.cpp, this command starts a local server with a built-in chat UI and an OpenAI-compatible API:

```bash
llama-server -hf SoAIHQ/gemma-4-26B-A4B-it-GGUF:Q4_K_M
```

Open <http://localhost:8080> to chat, or send requests to the API:

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Explain GGUF in one sentence."}],
  "chat_template_kwargs": {"enable_thinking": true}
}'
```

<details>
<summary>Download the files to a folder first</summary>

```bash
hf download SoAIHQ/gemma-4-26B-A4B-it-GGUF --include "gemma-4-26B-A4B-it-Q4_K_M.gguf" "mmproj-*" --local-dir gemma-4-26B-A4B-it-GGUF
llama-server -m gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --mmproj gemma-4-26B-A4B-it-GGUF/mmproj-gemma-4-26B-A4B-it-f16.gguf
```

</details>

## Recommended settings

| Mode | `temperature` | `top_p` | `top_k` |
| :--- | :---: | :---: | :---: |
| All tasks | 1.0 | 0.95 | 64 |

Google recommends these values in the [original model card](https://huggingface.co/google/gemma-4-26B-A4B-it). Google uses the same sampling settings for every task, with thinking on or off.

In llama-server, you set reasoning per request with `"chat_template_kwargs": {"enable_thinking": true}`.

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
| [gemma-4-26B-A4B-it-Q8_0.gguf](https://huggingface.co/SoAIHQ/gemma-4-26B-A4B-it-GGUF/blob/main/gemma-4-26B-A4B-it-Q8_0.gguf) | Q8_0 | 26.9 GB | `2c62620ce27d81d1…` |
| [gemma-4-26B-A4B-it-Q4_K_M.gguf](https://huggingface.co/SoAIHQ/gemma-4-26B-A4B-it-GGUF/blob/main/gemma-4-26B-A4B-it-Q4_K_M.gguf) | Q4_K_M | 16.8 GB | `d13ab520dd546f63…` |
| [mmproj-gemma-4-26B-A4B-it-f16.gguf](https://huggingface.co/SoAIHQ/gemma-4-26B-A4B-it-GGUF/blob/main/mmproj-gemma-4-26B-A4B-it-f16.gguf) | mmproj (F16) | 1.2 GB | `6b77845c12b7bb12…` |
| [gemma-4-26B-A4B-it.imatrix](https://huggingface.co/SoAIHQ/gemma-4-26B-A4B-it-GGUF/blob/main/gemma-4-26B-A4B-it.imatrix) | imatrix | 56.9 MB | `bb4a6c2ec3ddae7e…` |

The full checksums are in [`SHA256SUMS`](https://huggingface.co/SoAIHQ/gemma-4-26B-A4B-it-GGUF/blob/main/SHA256SUMS).
</details>

## Provenance

| Item | Details |
| :--- | :--- |
| Original model | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) by [Google](https://huggingface.co/google), revision [`4d7ae4984b`](https://huggingface.co/google/gemma-4-26B-A4B-it/tree/4d7ae4984b7db7de8f8457170b3f1a419ee76d52) |
| Converted with | [llama.cpp](https://github.com/ggml-org/llama.cpp) `v0.5.0`, commit [`7fe450e19`](https://github.com/ggml-org/llama.cpp/commit/7fe450e19305b828c199d602c23a8337aaa1f03b) |
| Importance matrix | SoAI calibration corpus v1, rendered with this model's chat template and computed on the Q8_0 weights |
| Weights | The original release, with no fine-tuning, merging or retraining. Only the storage format and numerical precision differ. |
| Built | 2026-09-26 |

## License and credits

These files are distributed under the original model's license (`apache-2.0`), and its full
terms continue to apply. The [license](https://ai.google.dev/gemma/docs/gemma_4_license) and the [original model card](https://huggingface.co/google/gemma-4-26B-A4B-it)
also describe the model's capabilities, training data, benchmarks and limitations.

[Google](https://huggingface.co/google) designed and trained the model, and SoAI only repackages the
published weights. The conversion uses [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi
Gerganov and the ggml contributors.

## Run it with SoAI

Your models, your hardware, one software suite. [Download SoAI](https://soai.to/#download) for Windows,
macOS, Linux or Android.

<div align="center">

<sub>Packaged by <a href="https://soai.to">SoAI</a>.</sub>

</div>
