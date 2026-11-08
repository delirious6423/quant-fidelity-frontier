---
library_name: starlight
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - huihui-ai/Huihui-Qwen3.8-27B-abliterated
base_model_relation: quantized
tags:
  - starlight
  - qwen3.8
  - quantized
  - multimodal
  - mtp
  - abliterated
  - amd
  - nvidia
---

# Qwen3.8-27B IQ-A (abliterated) for Starlight

> **For Starlight only.** This repository holds a model file made for one program, the inference engine **Starlight**. It is not a Transformers or Safetensors checkpoint, and it is not meant for llama.cpp, Ollama, LM Studio or any other program: other programs are neither supported nor tested.

The abliterated twin of the IQ-A file of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for Starlight, 11.86 GiB: most weights are 3-bit and 4-bit blocks (IQ3_S and IQ4_XS), so the model fits one 16 GB graphics card of either maker. The vision encoder and the multi-token-prediction (MTP) draft head are inside the same file.

## At a glance

| | |
|---|---|
| File | `IQ-A-abliterated_f472152c.starlight` |
| Size | 12,735,418,624 bytes (11.86 GiB) |
| SHA-256 | `2fb2ab4cec284e9f51b4298ad0b4ab60277fe3d3da00a2eaba88b71b20fa9bf3` |
| MD5 | `f472152c9cdffa35ec479d4945654c86` |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (dense, 27B parameters), abliterated |
| Format | Starlight `.starlight` model file |
| Recipe | IQ-A, hybrid head: mostly IQ4_XS and IQ3_S blocks |
| Inside the file | text model, vision encoder, MTP draft head |
| Cards | one 16 GB card (AMD RX 9070 XT, NVIDIA RTX 5070 Ti or RTX 5060 Ti) or two RTX 5060 Ti |
| Licence | Apache-2.0 |

## What is different from the plain file

The refusal direction is taken out of the 70 tensors that write into the residual stream (layers 17 to 51: the down projection, the linear-attention output projection and the attention output projection), as in [huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) (revision 739e3c5b), and those tensors are quantised again with the file's own recipe. Every other byte is the plain file's. Size, speed and limits are the plain file's; the figures below were measured with the plain file and were not repeated for this one.

## Memory and speed, as measured with the plain IQ-A file

| card | mode | largest start that worked | decode speed on record |
|---|---|--:|--:|
| RX 9070 XT, one card | MTP | 174,400 tokens | no speed record |
| RX 9070 XT, one card | DFlash2 | 170,496 tokens (262,144 with cache overflow) | 137.1 tok/s (BetterBench, 24 Oct 2026) |
| RTX 5060 Ti, one card | MTP | 200,704 tokens (262,144 with cache overflow) | 69.3 tok/s (29 Oct 2026) |
| RTX 5060 Ti, one card | DFlash2 | 149,504 tokens (262,144 with cache overflow) | 113.0 tok/s (29 Oct 2026) |
| RTX 5060 Ti, one card | no speculation | 229,376 tokens (262,144 with cache overflow) | 29.7 tok/s (29 Oct 2026) |
| RTX 5060 Ti, two cards | MTP | 262,144 tokens | 122.8 tok/s (29 Oct 2026) |
| RTX 5060 Ti, two cards | DFlash2 | 262,144 tokens | 180.3 tok/s (29 Oct 2026) |

How to read the table: *Largest start that worked* is the biggest context window at which the engine booted and answered a long prompt on that card (q4_0 attention cache). *Cache overflow* means the attention cache spills into system RAM. *BetterBench* is [BetterBench v0.6.1](https://github.com/GGZ14/BetterBench) ("BB-weighted" decode tokens per second over a mix of prompts). Every figure is a single run on the date given, with the Starlight engine of that date; yours will differ with your machine and the engine build.

Speculative decoding: MTP works with this file alone, because the draft head is inside it. DFlash2 is faster still and needs a separate drafter file that is not part of this repository; the Launcher lists the drafters it knows. The model's own context limit is 262,144 tokens.

## About abliterated models

"Abliterated" means that the direction in the model's internal activations that carries refusals has been taken out of the weights. Requests that the original model would decline are answered. It is not a different or smarter model, and it has no content filter of its own: it will not warn, refuse or moderate.

- You are responsible for what you ask it and for what you do with its answers. Follow the law and the licence of the base model.
- Do not offer it to people who have not chosen an unfiltered model, and add your own safeguards where a use needs them.
- Everything the base model's card says about errors, bias and factual reliability still applies; removing refusals does not make answers more correct.

## How to use

1. Download the file into the models folder of your Starlight installation:

   ```bash
   hf download feverdoingwork/Qwen3.8-27B-IQ-A-abliterated-Starlight IQ-A-abliterated_f472152c.starlight --local-dir <models folder>
   ```

2. Open the Starlight Launcher. It lists the file as **IQ-A (abliterated)**: it recognises the file by its MD5.
3. Choose your card and press **Start**. The Launcher starts the Starlight engine program for that card (AMD or NVIDIA) and serves the model on your own machine through an OpenAI-compatible HTTP interface.

## Verify the download

```bash
echo "2fb2ab4cec284e9f51b4298ad0b4ab60277fe3d3da00a2eaba88b71b20fa9bf3  IQ-A-abliterated_f472152c.starlight" | sha256sum --check
```

The same line is in `SHA256SUMS`. The Starlight Launcher knows the file by its MD5, `f472152c9cdffa35ec479d4945654c86`.

## Limits

- Starlight only (see the notice at the top). The file is not a general-purpose checkpoint.
- Window sizes depend on the card, the cache type and what else uses video memory. The table lists what was measured; a case that is not listed was not measured.
- No quality score of this file against the BF16 model is published in this card yet. Quantisation costs some quality, and the base model's own limits apply.
- Like every language model it can state wrong things with confidence.

## License and credits

- **Licence:** Apache-2.0, the licence of the base model (the text is in `LICENSE`).
- **Base model:** [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team.
- **Refusal direction:** the one published by huihui-ai in [Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) (revision 739e3c5b, Apache-2.0), applied to the weights by the Starlight project.
- **Quantisation, packing and the Starlight engine:** the Starlight project.
