---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
  - en
  - zh
tags:
  - megacapybara
  - qwen3.8
  - mxfp4
  - mxfp6
  - quantized
  - blackwell
  - rtx-5090
  - speculative-decoding
  - dflash2
  - mtp
---

# Qwen3.8 27B-MC

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Qwen's BF16 release) converted to **[MegaCapybara](https://github.com/perkel666/MegaCapybara)**'s MX weight
formats, in five sizes for one **NVIDIA RTX 5090**. The formats are MXFP4 and MXFP6, which Blackwell tensor cores read
at full rate.

## Run it with MegaCapybara

These files run in **[MegaCapybara](https://github.com/perkel666/MegaCapybara)**, the fastest inference engine for Qwen3.8-27B on the NVIDIA RTX 5090,
for Windows 11 and Linux: up to 500 tokens/s for one coding agent and up to 2,000 tokens/s for a team of them, with
OpenAI's and Anthropic's APIs (Claude Code runs on it).

1. Download MegaCapybara from [its releases](https://github.com/perkel666/MegaCapybara/releases/latest) and unpack it.
2. Run `MegaCapybaraLauncher` and press the arrow next to the model to download a size from this repository; the
   drafter and the vision tower come with it.
3. Pick a preset, press **Load server**, and point your client at `http://127.0.0.1:8080/v1`.

Without the launcher, put the files in the package's `model` folder and start a preset: the
[usage guide](https://github.com/perkel666/MegaCapybara/blob/main/docs/USAGE.md) has the details.

An uncensored version, converted the same way from an abliterated release, is
[perkel/Qwen3.8-27B-Uncensored-MC](https://huggingface.co/perkel/Qwen3.8-27B-Uncensored-MC).

## Files

| File | Launcher name | Weight formats | On disk | In VRAM | KL | Top-1 |
|---|---|---|---|---|---|---|
| `qwen3.8-27b-tiny.mcapy` | Qwen3.8 27B-MCTiny | all MXFP4 | 16.6 GB | 12.70 GiB | 0.0582 | 92.7% |
| `qwen3.8-27b-small.mcapy` | Qwen3.8 27B-MCSmall | MXFP4; MXFP6 for the head, the mixer outputs of 24 layers and the last 8 layers' MLPs | 17.6 GB | 13.67 GiB | 0.0478 | 93.8% |
| `qwen3.8-27b-medium.mcapy` | Qwen3.8 27B-MCMedium | MXFP4; MXFP6 for the head, every mixer output, half the mixer inputs and MLP outputs | 19.5 GB | 15.39 GiB | 0.0298 | 95.2% |
| `qwen3.8-27b-large.mcapy` | Qwen3.8 27B-MCLarge | all MXFP6 | 22.9 GB | 18.66 GiB | 0.0081 | 97.7% |
| `qwen3.8-27b-xxl.mcapy` | Qwen3.8 27B-MCXXL | MXFP6; the head and layers 24-39's MLP inputs at ~16 bits (two FP8 terms) | 28.2 GB | 23.58 GiB | 0.0051 | 98.3% |

You need one model file. These support files are shared by all five:

| File | What it is | On disk |
|---|---|---|
| `qwen3.8-27b-dflash2.mcapy` | DFlash2 speculative drafter, converted from [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (the default: the same answers, faster) | 1.3 GB |
| `qwen3.8-27b-mtp.mcapy` | Qwen's own MTP layer as a smaller, slower drafter | 0.28 GB |
| `qwen3.8-27b-vision-fp8.mcapy` | the vision tower in FP8 (images; nearly exact) | 0.57 GB |
| `qwen3.8-27b-vision-bf16.mcapy` | the original BF16 vision tower | 1.0 GB |

The launcher lists every model file in its `model\` folder by its name, smallest first, and projects each one's
accuracy, speed and memory for the chosen settings from scores stored in the file.

## Accuracy against the original

![Top-1 by size: our models and Unsloth's GGUFs](top1-vs-unsloth.png)

- **What KL and top-1 measure:** KL divergence and top-1 agreement (how often the most likely next token is the same)
  against Qwen's BF16 weights computed with FP32 math.
- **The test text:** 81,880 held-out tokens: reasoning traces, math, code, chat and wiki. It is disjoint from the
  calibration text. Tool calls are not scored: they differ even between BF16 and FP32 math.
- **Settings:** the engine at 16-bit activations and an unquantized KV cache. FP8 activations, the engine's default,
  add a little KL; the launcher shows each setting's cost.
- **The chart:**
  - Unsloth's points are their published top-1 for their Qwen3.8-27B GGUFs, measured on their own text.
  - Sizes are the bytes each puts on the GPU, read from the GGUF headers. The token embedding table and the MTP layer
    are not counted, because llama.cpp keeps the table in RAM as MegaCapybara does.
  - As a cross-check, their UD-Q4_K_XL scores 96.2% on our test, against the 96.4% they publish.

<details><summary>KL per kind of text</summary>

| Size | Reasoning | Math | Code | Chat | Wiki |
|---|---|---|---|---|---|
| Tiny | 0.0138 | 0.0156 | 0.0471 | 0.1750 | 0.0395 |
| Small | 0.0107 | 0.0138 | 0.0424 | 0.1456 | 0.0262 |
| Medium | 0.0058 | 0.0078 | 0.0290 | 0.0874 | 0.0190 |
| Large | 0.0009 | 0.0010 | 0.0075 | 0.0283 | 0.0026 |
| XXL | 0.0007 | 0.0008 | 0.0069 | 0.0151 | 0.0021 |

</details>

## Speed on an RTX 5090

Tokens per second; greedy, thinking off, answers up to 2,000 tokens, FP8 activations, DFlash2.

| Size | Coding, 1 conversation | Prose, 1 conversation | Coding, 8 at once (total) | Reading an 8K-token prompt |
|---|---|---|---|---|
| Tiny | 540 | 248 | 1,890 | 7,789 |
| Small | 495 | 232 | 1,754 | 7,619 |
| Medium | 440 | 208 | 1,638 | 7,377 |
| Large | 377 | 176 | 1,427 | 6,955 |
| XXL | 313 | 148 | 1,254 | 6,178 |

Measured with nothing else on the card. Its memory was overclocked to 16.8 GHz (+20%); decoding is bound by memory
bandwidth, so a stock card is slower.

## Which size

- **Tiny and Small:** the fastest, and the most room for context.
- **Medium:** the balance; the launcher's default.
- **Large and XXL:** the closest to the original, with less room for context.

## How the files were made

- **Source:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), revision `1d4bf0f`. Its BF16 weights are
  the reference for every score above.
- **Rounding:** layer by layer, GPTQ with MX block scales (32 weights share a power-of-two scale), on 514K calibration
  tokens. The calibration text is 2K and 16K-token sequences of reasoning, math, code, chat and wiki.
- **Storage changes for the engine:**
  - norm gains are folded into the following matrices;
  - the attention value rows are Hadamard-rotated;
  - the output head's rows are ordered by token frequency.

  None of these change the model's function.
- **XXL's 16-bit parts** are stored as two FP8 E4M3 terms (value and residual), about 8 significant bits.

## License

Apache 2.0, as the original [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and
[Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). The changes are listed in `NOTICE`.
Converted by Perkel's Software Corner.
