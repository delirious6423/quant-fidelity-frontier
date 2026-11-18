---
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
- qwen3.8
- vision
- image-text-to-text
---

# Qwen3.8-27B-Homebrew

Homebrew is a Qwen3.8-27B-derived model distributed in GGUF format, with BF16 and Q4_K_M options, native support for **262,144 tokens** of context, and an optional vision projector.

Both language GGUFs include their tokenizer and chat template. Choose one for text-only use; load `mmproj-F16.gguf` alongside it when using image input with a compatible runtime.

## Download

| File | Size | Format |
| --- | ---: | --- |
| Qwen3.8-27B-Homebrew-BF16.gguf | 54.66 GB | BF16 |
| Qwen3.8-27B-Homebrew-Q4_K_M.gguf | 17.01 GB | Q4_K_M |
| mmproj-F16.gguf | 0.93 GB | Vision encoder/projector |

Download the Q4_K_M model and Windows launcher:

```sh
hf download vibe-coder-001/Qwen3.8-27B-Homebrew Qwen3.8-27B-Homebrew-Q4_K_M.gguf Start-Homebrew.ps1 --local-dir Homebrew
cd Homebrew
```

For BF16, replace the Q4_K_M filename with `Qwen3.8-27B-Homebrew-BF16.gguf`.

For image input, download the projector into the same directory:

```sh
hf download vibe-coder-001/Qwen3.8-27B-Homebrew mmproj-F16.gguf --local-dir Homebrew
```

The language GGUF and projector are two components of one multimodal setup. The language GGUF also works alone for text-only use.

## Run

With a compatible llama.cpp build (text generation tested with b12497):

```sh
llama-server -m Qwen3.8-27B-Homebrew-Q4_K_M.gguf --ctx-size 262144 --alias Qwen3.8-27B-Homebrew --jinja --host 127.0.0.1 --port 49385
```

Add `--mmproj mmproj-F16.gguf` to load the vision projector. Configure GPU offload and context size for your hardware.

The Windows launcher accepts a path to your compatible `llama-server.exe`:

```powershell
.\Start-Homebrew.ps1 -ServerPath C:\llama.cpp\llama-server.exe
.\Start-Homebrew.ps1 -ServerPath C:\llama.cpp\llama-server.exe -Variant BF16
.\Start-Homebrew.ps1 -ServerPath C:\llama.cpp\llama-server.exe -Vision
```

The published launcher defaults to **262,144 tokens**, matching the model's native context length. No additional context scaling is needed at this length. Memory requirements depend on context size, cache precision, and GPU offload; use `-ContextSize 8192` for a smaller memory footprint. The API base is `http://127.0.0.1:49385/v1`, with model ID `Qwen3.8-27B-Homebrew`.

Use the embedded chat template. Thinking is enabled by default with medium reasoning effort; compatible clients can set `enable_thinking=false` for direct responses.

## Validation and limitations

Structural checks passed for both language GGUFs and the projector. Q4_K_M text-generation checks passed, and the model and projector loaded together successfully. A small local coding evaluation was also performed; it is not a general benchmark.

BF16 inference, image understanding, full-window retrieval, and MTP speculative decoding have not been validated for this release. Results depend on the runtime, settings, and task. Output is text; this is not an image-generation or voice model.

Released under Apache-2.0. Based on [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
