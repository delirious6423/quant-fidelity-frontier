---
library_name: starlight
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - starlight
  - qwen3.8
  - quantized
  - multimodal
  - mtp
  - amd
  - nvidia
---

# Qwen3.8-27B IQ-A for Starlight

> **For Starlight only.** This repository holds a model file made for one program, the inference engine **Starlight**. It is not a Transformers or Safetensors checkpoint, and it is not meant for llama.cpp, Ollama, LM Studio or any other program: other programs are neither supported nor tested.

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in 11.86 GiB for Starlight. The **IQ-A** recipe stores most weights as 3-bit and 4-bit blocks (IQ3_S and IQ4_XS), so the model fits one 16 GB graphics card of either maker. The vision encoder and the multi-token-prediction (MTP) draft head are inside the same file.

## At a glance

| | |
|---|---|
| File | `IQ-A_343c87a3.starlight` |
| Size | 12,735,418,624 bytes (11.86 GiB) |
| SHA-256 | `336c642521d7223f354274283c0cd659414243bdb8e1ece502f3dfc0569bc63f` |
| MD5 | `343c87a3f77f563ea3d374a1e213c3a6` |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (dense, 27B parameters) |
| Format | Starlight `.starlight` model file |
| Recipe | IQ-A, hybrid head: mostly IQ4_XS and IQ3_S blocks |
| Inside the file | text model, vision encoder, MTP draft head |
| Cards | one 16 GB card (AMD RX 9070 XT, NVIDIA RTX 5070 Ti or RTX 5060 Ti) or two RTX 5060 Ti |
| Licence | Apache-2.0 |

## What is inside

Tensor types, read from the file's header: IQ4_XS 64 tensors (5.73 GiB, 48.3 % of the file) and IQ3_S 194 tensors (5.28 GiB, 44.5 %), and the rest in small amounts: the token embedding (Q3G64; the vocabulary is pruned to 141,312 entries), 4 IQ3_XXS, 54 Q4G64, 54 Q5G64, 582 BF16 and 2 W8G32 tensors. Pictures are read by the vision encoder inside the file, so no second file is needed.

## Memory and speed, as measured

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

Speculative decoding: MTP works with this file alone, because the draft head is inside it. DFlash2 is faster still and needs a separate drafter file that is not part of this repository; the Launcher lists the drafters it knows. The model's own context limit is 262,144 tokens. The RTX 5070 Ti is a supported card, but this card holds no window figures for it.

## How to use

1. Download the file into the models folder of your Starlight installation:

   ```bash
   hf download feverdoingwork/Qwen3.8-27B-IQ-A-Starlight IQ-A_343c87a3.starlight --local-dir <models folder>
   ```

2. Open the Starlight Launcher. It lists the file as **IQ-A**: it recognises the file by its MD5.
3. Choose your card and press **Start**. The Launcher starts the Starlight engine program for that card (AMD or NVIDIA) and serves the model on your own machine through an OpenAI-compatible HTTP interface.

## Verify the download

```bash
echo "336c642521d7223f354274283c0cd659414243bdb8e1ece502f3dfc0569bc63f  IQ-A_343c87a3.starlight" | sha256sum --check
```

The same line is in `SHA256SUMS`. The Starlight Launcher knows the file by its MD5, `343c87a3f77f563ea3d374a1e213c3a6`.

## Limits

- Starlight only (see the notice at the top). The file is not a general-purpose checkpoint.
- Window sizes depend on the card, the cache type and what else uses video memory. The table lists what was measured; a case that is not listed was not measured.
- No quality score of this file against the BF16 model is published in this card yet. Quantisation costs some quality, and the base model's own limits apply.
- Like every language model it can state wrong things with confidence.

## License and credits

- **Licence:** Apache-2.0, the licence of the base model (the text is in `LICENSE`).
- **Base model:** [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team.
- **Quantisation, packing and the Starlight engine:** the Starlight project.
