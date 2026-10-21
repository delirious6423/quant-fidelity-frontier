---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- unsloth/Qwen3.8-27B-NVFP4
language:
- es
- en
tags:
- yaya
- nvfp4
- sovereign-ai
- latam
---

<div align="center">

# yaya3.8

**Intelligence you own. Inference you control. From Peru, for the Global South.**

</div>

---

## Why this model exists

**yaya3.8** is the model that powers [Yaya Tech](https://yaya.tech). It is a 27B-parameter frontier-class open model that we run **entirely on our own hardware** — no hyperscaler in the loop, no third-party API between our users and their intelligence.

We publish it because we believe something simple:

> **Peru, LATAM, and the whole Global South deserve to own their intelligence.**

### Digital sovereignty

Digital sovereignty means that the infrastructure your life runs on — your data, your models, your inference — answers to *you*, not to a data center on another continent governed by someone else's laws, someone else's pricing, and someone else's kill switch.

For most of the world, AI today is a rented service: your words leave your country, are processed by machines you will never audit, under terms that can change tomorrow. That is not intelligence you own. That is intelligence you borrow — until it's taken away, repriced, or read.

### What it means to own your data

Owning your data is not a checkbox in a privacy policy. It means:

- **Your data never leaves infrastructure you control.** At Yaya, footage, messages, and payments are processed on our own nodes. Raw data does not get shipped to external clouds.
- **Open weights, on your disk.** A model you can download, inspect, fine-tune, and serve yourself can never be discontinued out from under you. This repository is exactly that.
- **The only secure API is a confidential API.** If you must call remote compute, the bar is *remote attestation before any data is sent* — cryptographic proof, from the firmware up, that the machine on the other end is running exactly the code it claims, inside a confidential-computing enclave. No attestation, no data. Trust is verified, never assumed.

### Be the change

We run all of our inference ourselves — not because it is easy, but because sovereignty is not something you ask for. You build it. A vending machine in Lima, a phone in Cusco, a family mail server: every one of them can be served by intelligence that lives where its users live. That is the change we want to see in the world, so we are being it.

---

## The model

yaya3.8 is [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in the NVFP4 dynamic quantization by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) (Unsloth Dynamic V3.0). The weights are unmodified from the upstream quant — open provenance is part of ownership, and standing on the shoulders of great open-weight work is exactly how the Global South catches up.

| | |
|---|---|
| Parameters | 27B |
| Quantization | NVFP4 (Unsloth Dynamic V3.0), FP8 `lm_head` |
| Extras | MTP head (`model_mtp.safetensors`) for fast/speculative decoding |
| Size on disk | ~23 GB |
| License | Apache 2.0 (inherited from Qwen3.8) |

> [!Note]
> This quant runs in **vLLM** and not SGLang — the `lm_head` is FP8, which SGLang cannot load.

### Run it yourself — that's the point

```bash
vllm serve agentyaya/yaya3.8
```

One command, your hardware, your intelligence. That is what ownership feels like.

---

## Attribution

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) © Alibaba Cloud, Apache-2.0
- Quantization: [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)

*Yaya Tech PBC — ownership and sovereignty, from the firmware up.*
