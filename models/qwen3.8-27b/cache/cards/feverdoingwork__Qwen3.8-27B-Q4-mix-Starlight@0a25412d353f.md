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
---

# Qwen3.8-27B Q4-mix for Starlight (RX 7900 XTX)

> **For Starlight only.** This repository holds a model file made for one program, the inference engine **Starlight**. It is not a Transformers or Safetensors checkpoint, and it is not meant for llama.cpp, Ollama, LM Studio or any other program: other programs are neither supported nor tested.

> **For the RX 7900 XTX only.** This bake was made for and measured on one card, the AMD Radeon RX 7900 XTX (gfx1100, Navi 31, 24 GB). It is not offered for, and has no measurements on, any other card.

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in 13.29 GiB for Starlight. The **Q4-mix** recipe (lineage `mxfp4v2`) stores 93 % of the weights as 4-bit groups with 5-bit for the rest, so the model fits one 24 GB card with room for long context. The vision encoder and the multi-token-prediction (MTP) draft head are inside the same file.

## At a glance

| | |
|---|---|
| File | `Q4-mix_bd65689b.starlight` |
| Size | 14,270,973,440 bytes (13.29 GiB) |
| SHA-256 | `d6c62e72a8de4a6a18f1e8bb8923a0fdba6a6ef11097b58cb3b70ae8a0d60643` |
| MD5 | `bd65689b34fe048adcb25d9eb6ac9123` |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (dense, 27B parameters) |
| Format | Starlight `.starlight` model file (container v2) |
| Recipe | Q4-mix, groupwise 4-bit with 5-bit (weights lineage `mxfp4v2`) |
| Inside the file | text model, vision encoder, MTP draft head, tokenizer frontend with chat template |
| Card | AMD Radeon RX 7900 XTX (24 GB, gfx1100) |
| Licence | Apache-2.0 |

## What is inside

Tensor types, read from the file's header: Q4G64 315 tensors (12.37 GiB, 93.1 % of the tensor bytes), Q5G64 55 tensors (0.55 GiB, 4.1 %), one Q3G64 tensor (0.27 GiB, 2.1 %) and small amounts of BF16, FP32, W8G32 and one Q6G64 tensor. Pictures are read by the vision encoder inside the file, so no second file is needed.

## Memory and speed, as measured

Greedy evaluation (`temp=0.0`, MTP with 3 draft tokens, prefill chunk 2048), RX 7900 XTX, as documented in the project's AMD 27B mailbox:

| Context Length | Decode TPS | Prefill TPS | MTP Acceptance | TTFT |
| :--- | --: | --: | --: | --: |
| 10k (10,008) | 42.3 | 559.8 | 61.5 % (2.84 tok/rnd) | 17.88 s |
| 40k (39,984) | 43.7 | 444.8 | 73.6 % (3.21 tok/rnd) | 89.89 s |
| 80k (79,956) | 34.2 | 347.9 | 58.3 % (2.75 tok/rnd) | 229.82 s |
| 160k (159,900) | 31.6 | 241.4 | 68.6 % (3.06 tok/rnd) | 662.39 s |
| 240k (239,832) | 28.4 | 185.1 | 71.4 % (3.14 tok/rnd) | 1295.69 s |

How to read the table: every figure is a single run with the Starlight engine build of that date; yours will differ with your machine and the engine build. A ladder with the Launcher's recommended sampling (`temp=1.0, top_p=0.95, top_k=20, presence_penalty=0.0`, thinking on) is not published yet and will be added here when it is measured.

Speculative decoding: MTP works with this file alone, because the draft head is inside it. DFlash2 is faster still and needs a separate drafter file that is not part of this repository; the Launcher lists the drafters it knows.

## How to use

1. Download the file into the models folder of your Starlight installation:

   ```bash
   hf download feverdoingwork/Qwen3.8-27B-Q4-mix-Starlight Q4-mix_bd65689b.starlight --local-dir <models folder>
   ```

2. The Starlight Launcher recognises the file by its MD5 (`bd65689b34fe048adcb25d9eb6ac9123`), under any file name, in any folder it lists.

## Provenance

This file reached publication as a split export (seven parts, `SHA256SUMS_qwen3_8_27b_q4_mix_vision.txt` in the source archive). The parts were reassembled on the publishing machine and every part was verified against that manifest; the reassembled file's MD5 (`bd65689b…`) matches the artifact-registry entry for `qwen3_8_27b_q4_mix_vision`.

## Verify the download

```bash
echo "d6c62e72a8de4a6a18f1e8bb8923a0fdba6a6ef11097b58cb3b70ae8a0d60643  Q4-mix_bd65689b.starlight" | sha256sum --check
```

The same line is in `SHA256SUMS`.

## Limits

- Starlight only (see the notice at the top). The file is not a general-purpose checkpoint.
- RX 7900 XTX only (see the notice at the top).
- The speed table lists what was measured; a case that is not listed was not measured.
- No quality score of this file against the BF16 model is published in this card yet. Quantisation costs some quality, and the base model's own limits apply.
- Like every language model it can state wrong things with confidence.

## License and credits

- **Licence:** Apache-2.0, the licence of the base model (the text is in `LICENSE`).
- **Base model:** [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team.
- **Quantisation, packing and the Starlight engine:** the Starlight project.
