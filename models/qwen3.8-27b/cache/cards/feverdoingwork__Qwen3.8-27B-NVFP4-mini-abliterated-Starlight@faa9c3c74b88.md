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
  - nvfp4
  - quantized
  - multimodal
  - mtp
  - abliterated
  - blackwell
  - nvidia
---

# Qwen3.8-27B NVFP4-mini (abliterated) for Starlight

> **For Starlight only.** This repository holds a model file made for one program, the inference engine **Starlight**. It is not a Transformers or Safetensors checkpoint, and it is not meant for llama.cpp, Ollama, LM Studio or any other program: other programs are neither supported nor tested.

The abliterated twin of the NVFP4-mini file of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for Starlight on NVIDIA Blackwell cards, 12.62 GiB. The weights are NVFP4 (4-bit floating point with block scales) with the gate and up projections at 3 bits; the vision encoder and the multi-token-prediction (MTP) draft head are inside the same file.

## At a glance

| | |
|---|---|
| File | `NVFP4-mini-abliterated_cd3db230.starlight` |
| Size | 13,551,144,192 bytes (12.62 GiB) |
| SHA-256 | `44456d056b28e0c64ec838860826501af5d8b0cede00b8c0c11a900062797b34` |
| MD5 | `cd3db230ad3b35727bfc50172ee19ae1` |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (dense, 27B parameters), abliterated |
| Format | Starlight `.starlight` model file |
| Recipe | NVFP4 body, gate/up at 3 bits (Q3) on every layer, vision encoder from BF16 |
| Inside the file | text model, vision encoder, MTP draft head |
| Cards | NVIDIA only: one 16 GB Blackwell card (RTX 5060 Ti, RTX 5070 Ti) or two RTX 5060 Ti. No AMD card. |
| Licence | Apache-2.0 |

## What is different from the plain file

The refusal direction is taken out of the 70 tensors that write into the residual stream (layers 17 to 51: the down projection, the linear-attention output projection and the attention output projection), as in [huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) (revision 739e3c5b), and those tensors are quantised again with the file's own recipe. Every other byte is the plain file's. Size, speed and limits are the plain file's; the figures below were measured with the plain file and were not repeated for this one.

## How this file was made and checked

Made on 8 November 2026 from the NVFP4-mini file (MD5 `268fa4aeebba6d6f5641a0667841e31b`, in the sibling repository) by overwriting the 70 matrices that write into the residual stream (26 linear-attention output, 35 MLP down and 9 attention output matrices, layers 17 to 51) with the NVFP4 round-to-nearest quantisation of the abliterated checkpoint's tensors. Checks on the finished file:

- Size and directory are the NVFP4-mini file's. 70 ranges differ from it and every other byte is identical.
- The 70 written matrices decode back to exactly what the quantiser produced.
- The size of the removed direction in these matrices falls to about 31 % of its size in NVFP4-mini (30.8 % to 31.6 %, matrix by matrix).
- The 70 matrices are byte-for-byte the ones of the earlier abliterated NVFP4-mini build (MD5 `e56cb25255a440880e9b9440f66dea24`, made on the NVFP4-mini file with MD5 `e0497a0ee4ef864ef4c94fa3a6e0755f`). The two files differ only outside these matrices.
- Payload census: 1,252 tensors, no defects (the only all-zero pages are the pruned-vocabulary padding rows, by design).

This file has not been started in the Starlight engine on a GPU yet. The speed and window table below was measured with the NVFP4-mini file.

## Memory and speed, as measured with the plain NVFP4-mini file

| card | mode | largest start that worked | decode speed on record |
|---|---|--:|--:|
| RTX 5060 Ti, one card | MTP | 178,176 tokens (262,144 with cache overflow) | 70.3 tok/s (BetterBench, 30 Oct 2026) |
| RTX 5060 Ti, one card | DFlash2 | 141,000 tokens (262,144 with cache overflow) | 114.0 tok/s (30 Oct 2026) |
| RTX 5060 Ti, one card | no speculation | 179,200 tokens (262,144 with cache overflow) | 18.4 tok/s (30 Oct 2026) |
| RTX 5060 Ti, two cards | MTP | 262,144 tokens | 128.3 tok/s (30 Oct 2026) |
| RTX 5060 Ti, two cards | DFlash2 | 262,144 tokens | 176.9 tok/s (29 Oct 2026) |

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
   hf download feverdoingwork/Qwen3.8-27B-NVFP4-mini-abliterated-Starlight NVFP4-mini-abliterated_cd3db230.starlight --local-dir <models folder>
   ```

2. Open the Starlight Launcher. It lists the file as **NVFP4-mini (abliterated)**: it recognises the file by its MD5.
3. Choose your card and press **Start**. The Launcher starts the Starlight engine program for that card (AMD or NVIDIA) and serves the model on your own machine through an OpenAI-compatible HTTP interface.

## Launcher entry

The Starlight Launcher recognises a model file by its MD5. This file's MD5 is not in the Launcher's built-in list yet, so until a Launcher build lists it, save the entry below as `~/.config/starlight-launcher/supported.json` (if that file exists, add the entry to its `models` list) and restart the Launcher. It replaces the built-in entry of the same id and keeps the older file's MD5 as well.

<details>
<summary>supported.json entry</summary>

```json
{
  "models": [
    {
      "id": "27b-c-nv-abl",
      "group": "NVFP4-mini",
      "family": "27b",
      "title": "NVFP4-mini (abliterated)",
      "files": [
        "NVFP4-mini-abliterated_cd3db230.starlight",
        "NVFP4-mini-abliterated_e56cb252.starlight"
      ],
      "size": 13551144192,
      "md5": [
        "cd3db230ad3b35727bfc50172ee19ae1",
        "e56cb25255a440880e9b9440f66dea24"
      ],
      "min_cards": 1,
      "kv": [
        "q4_0",
        "int8",
        "bf16",
        "kvarn_k3v3"
      ],
      "spec": [
        "dflash2",
        "mtp",
        "off"
      ],
      "vendor": "nvidia",
      "note": "ABLITERATED. NVFP4-mini with the refusal direction taken out of the 70 tensors that write into the residual stream (layers 17-51: the down, GDN output and attention output projections), as in huihui-ai/Huihui-Qwen3.8-27B-abliterated (revision 739e3c5b), re-quantised to NVFP4 from that checkpoint the way this file's own writers are made. Every other byte is the NVFP4-mini file's. Same limits as NVFP4-mini: starts with DFlash2, the token table in system RAM and the hosted cache on (window 262,144); cache types and windows are NVFP4-mini's, not measured separately.",
      "default_spec": "dflash2",
      "embed_host": true,
      "window": 98304,
      "abl_of": "27b-c-nv",
      "drafters": [
        {
          "file": "qwen38_27b_dflash2_zlab_hyb3_pruned.sidecar.json",
          "recommended": true
        },
        {
          "file": "qwen38_27b_dflash2_zlab_q4g64_pruned.sidecar.json"
        }
      ]
    }
  ]
}
```

</details>

## Verify the download

```bash
echo "44456d056b28e0c64ec838860826501af5d8b0cede00b8c0c11a900062797b34  NVFP4-mini-abliterated_cd3db230.starlight" | sha256sum --check
```

The same line is in `SHA256SUMS`. The Starlight Launcher knows the file by its MD5, `cd3db230ad3b35727bfc50172ee19ae1`.

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
