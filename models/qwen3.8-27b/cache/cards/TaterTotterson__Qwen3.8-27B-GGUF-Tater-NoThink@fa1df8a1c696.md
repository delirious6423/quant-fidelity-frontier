---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: llama.cpp
pipeline_tag: image-text-to-text
tags:
  - gguf
  - llama-cpp
  - qwen
  - qwen3.8
  - 27b
  - dense
  - multimodal
  - vision
  - video
  - image-text-to-text
  - tool-calling
  - speculative-decoding
  - draft-model
  - mtp
  - multi-token-prediction
  - tater
  - nothink
  - chat-template
  - tater-pick
  - tater-nothink
---

# Qwen3.8-27B GGUF — Tater NoThink

This repository packages the official Qwen3.8-27B llama.cpp artifacts with a Tater NoThink chat-template patch.

Qwen3.8-27B is a dense 27B vision-language model. It supports text, image, and video understanding, tool calling, a native 262K context window, and Multi-Token Prediction (MTP).

## What changed

Only the chat-template metadata in the main GGUF was changed. The template forces `enable_thinking = false`, including when a client requests thinking mode. Generation begins with an empty `<think></think>` block and proceeds directly to the answer.

The model tensors and quantization were not modified. The vision projector and MTP draft model are unchanged files from the pinned `ggml-org` conversion.

## Files

| File | Purpose |
| --- | --- |
| `Qwen3.8-27B-Tater-NoThink-Q4_K_M.gguf` | Main Q4_K_M NoThink model |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | Official Q8_0 vision projector |
| `mtp-Qwen3.8-27B-Q8_0.gguf` | Official Q8_0 MTP draft model for speculative decoding |
| `chat_template.jinja` | Patched template embedded in the main GGUF |

## Tater

Download the main GGUF. Tater will also offer the vision projector with it.

For speculative decoding, enable **Speculative Decoding**, select **MTP**, and choose `mtp-Qwen3.8-27B-Q8_0.gguf` as the draft model.

## llama.cpp example

```bash
llama-server \
  --model Qwen3.8-27B-Tater-NoThink-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf \
  --spec-draft-model mtp-Qwen3.8-27B-Q8_0.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 6
```

MTP performance depends on the backend, prompt, context length, and hardware. Compare it with ordinary generation on your own workload.

## Sources

- Base model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), pinned to `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- GGUF conversion: [`ggml-org/Qwen3.8-27B-GGUF`](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF), pinned to `97c30c65c8d9a3e73f9fdfb50f1d1a669e9a2827`.

The official conversion currently supplies MTP sidecars. No DFlash or DSpark draft model is included here because there is no corresponding official artifact in the pinned source repository.

## Integrity

SHA-256:

- Main NoThink GGUF: `4c844893d312b740f189c84f007e6dc015a84837f4031f59156ad758941d312c`
- Vision projector: `2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb`
- MTP draft model: `cbf60a0c48b431bb61f1d49b8948dc88ac29c398d6dbdbbb2e6e89ef77eacc9a`

The combined tensor-data SHA-256 is also checked before publishing to confirm that the main model's tensor payload is identical to the official Q4_K_M source.

## License

Qwen3.8-27B is provided under the Apache License 2.0. The upstream license is included as `LICENSE`.
