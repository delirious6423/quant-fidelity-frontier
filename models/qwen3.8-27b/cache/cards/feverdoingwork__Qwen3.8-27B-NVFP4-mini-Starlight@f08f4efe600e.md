---
library_name: starlight
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - Qwen/Qwen3.8-27B-FP8
base_model_relation: quantized
tags:
  - starlight
  - qwen3.8
  - nvfp4
  - quantized
  - multimodal
  - mtp
  - blackwell
  - nvidia
---

# Qwen3.8-27B NVFP4-mini for Starlight

> **For Starlight only.** This repository holds a model file made for one program, the inference engine **Starlight**. It is not a Transformers or Safetensors checkpoint, and it is not meant for llama.cpp, Ollama, LM Studio or any other program: other programs are neither supported nor tested.

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in 12.62 GiB for Starlight on NVIDIA Blackwell cards. **NVFP4-mini** keeps the weights in NVFP4 (4-bit floating point with block scales); the gate and up projections of every layer are re-made from the BF16 checkpoint at 3 bits, which is what makes the file small enough for one 16 GB card. The vision encoder (also re-made from the BF16 checkpoint) and the multi-token-prediction (MTP) draft head are inside the same file.

## At a glance

| | |
|---|---|
| File | `NVFP4-mini_268fa4ae.starlight` |
| Size | 13,551,144,192 bytes (12.62 GiB) |
| SHA-256 | `a1475895921f2e81a91131f2ab263c46f2e80f05d5d3670d19f79d189cf98cd2` |
| MD5 | `268fa4aeebba6d6f5641a0667841e31b` |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (dense, 27B parameters); the NVFP4 weights descend from [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) |
| Format | Starlight `.starlight` model file |
| Recipe | NVFP4 body, gate/up at 3 bits (Q3) on every layer, vision encoder from BF16 |
| Inside the file | text model (198 NVFP4 weight tensors), vision encoder (333 tensors), MTP draft head |
| Cards | NVIDIA only: one 16 GB Blackwell card (RTX 5060 Ti, RTX 5070 Ti) or two RTX 5060 Ti. No AMD card. |
| Licence | Apache-2.0 |

## Memory and speed, as measured

| card | mode | largest start that worked | decode speed on record |
|---|---|--:|--:|
| RTX 5060 Ti, one card | MTP | 178,176 tokens (262,144 with cache overflow) | 70.3 tok/s (BetterBench, 30 Oct 2026) |
| RTX 5060 Ti, one card | DFlash2 | 141,000 tokens (262,144 with cache overflow) | 114.0 tok/s (30 Oct 2026) |
| RTX 5060 Ti, one card | no speculation | 179,200 tokens (262,144 with cache overflow) | 18.4 tok/s (30 Oct 2026) |
| RTX 5060 Ti, two cards | MTP | 262,144 tokens | 128.3 tok/s (30 Oct 2026) |
| RTX 5060 Ti, two cards | DFlash2 | 262,144 tokens | 176.9 tok/s (29 Oct 2026) |

How to read the table: *Largest start that worked* is the biggest context window at which the engine booted and answered a long prompt on that card (q4_0 attention cache). *Cache overflow* means the attention cache spills into system RAM. *BetterBench* is [BetterBench v0.6.1](https://github.com/GGZ14/BetterBench) ("BB-weighted" decode tokens per second over a mix of prompts). Every figure is a single run on the date given, with the Starlight engine of that date; yours will differ with your machine and the engine build.

A small agentic coding test (5 tasks, window 98,304, MTP, RTX 5070 Ti, highest reasoning effort) solved 4 of 5 on the first pass and 5 of 5 with one rerun. That is a very small sample; it says the file works for coding tasks, not how it ranks.

Speculative decoding: MTP works with this file alone, because the draft head is inside it. DFlash2 is faster still and needs a separate drafter file that is not part of this repository; the Launcher lists the drafters it knows. The model's own context limit is 262,144 tokens.

## How to use

1. Download the file into the models folder of your Starlight installation:

   ```bash
   hf download feverdoingwork/Qwen3.8-27B-NVFP4-mini-Starlight NVFP4-mini_268fa4ae.starlight --local-dir <models folder>
   ```

2. Open the Starlight Launcher. It lists the file as **NVFP4-mini**: it recognises the file by its MD5.
3. Choose your card and press **Start**. The Launcher starts the Starlight engine program for that card (AMD or NVIDIA) and serves the model on your own machine through an OpenAI-compatible HTTP interface.

## Verify the download

```bash
echo "a1475895921f2e81a91131f2ab263c46f2e80f05d5d3670d19f79d189cf98cd2  NVFP4-mini_268fa4ae.starlight" | sha256sum --check
```

The same line is in `SHA256SUMS`. The Starlight Launcher knows the file by its MD5, `268fa4aeebba6d6f5641a0667841e31b`.

## Limits

- Starlight only (see the notice at the top). The file is not a general-purpose checkpoint.
- Window sizes depend on the card, the cache type and what else uses video memory. The table lists what was measured; a case that is not listed was not measured.
- No quality score of this file against the BF16 model is published in this card yet. Quantisation costs some quality, and the base model's own limits apply.
- Like every language model it can state wrong things with confidence.

## License and credits

- **Licence:** Apache-2.0, the licence of the base model (the text is in `LICENSE`).
- **Base model:** [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team.
- **Quantisation, packing and the Starlight engine:** the Starlight project.
