---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
library_name: gguf
tags:
  - gguf
  - imatrix
  - llama.cpp
  - quantized
quantized_by: Krasnopjorovs
---

# Qwen3.8-27B — imatrix GGUF quantizations

GGUF imatrix builds of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

Quantized with [llama.cpp](https://github.com/ggerganov/llama.cpp) `version: 10358 (030ebb558)` using importance-matrix calibration on a public multilingual + code + math corpus.

## Prompt format

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>
```

**Multimodal.** `mmproj-Qwen3.8-27B-f16.gguf` and `-bf16.gguf` are in this repo and
pair with any quant above — pass one with `--mmproj`. Use the f16 build on pre-Ampere
hardware.

**MTP speculative decoding — measured.** The MTP head is included in every quant.
Run it with `--spec-type draft-mtp`. The draft budget matters more than anything else
here; measured on a single 72 GB card, Q4_K_M, 400-token completions at temp 0:

| Config | acceptance | mean draft len | tok/s |
|---|---|---|---|
| `--spec-draft-n-max 3` | 0.52 | 2.56 | **83.7** |
| `--spec-draft-n-max 6` | 0.22 | 2.33 | 49.7 |

Going past 3 is a large net loss: the head generates twice the draft tokens for the
same number of accepted ones, so **keep `--spec-draft-n-max 3`**.

The MTP head sits at Q4_0 in the imatrix quants (Q8_0 quant excepted) since
calibration never exercises it. A Q8_0 head was built and measured for comparison:
acceptance rose to 0.55 and throughput to 84.8 tok/s — about 1% for 60 MB, inside
run-to-run noise. Q4_0 is the right default.

That comparison build ships here as `Qwen3.8-27B-Q4_K_M_MTP8.gguf` (15.83 GB) if you
want to verify the numbers yourself — identical to `Q4_K_M` except the MTP head is
Q8_0.

**Sampling** (per Qwen): thinking mode `temp 1.0, top_p 0.95, top_k 20,
presence_penalty 0`; non-thinking `temp 0.7, top_p 0.80, top_k 20,
presence_penalty 1.5`. Thinking is on by default.

`reasoning_effort` accepts `xhigh` (default), `medium`, `low`. Note Qwen's own
caveat: on multi-turn agentic work a lower effort is not reliably faster end to end —
shallower analysis means more retries.

## Available quants

| Filename | Quant | Size (GiB) | Description |
|---|---|---|---|
| [Qwen3.8-27B-Q8_0.gguf](https://huggingface.co/Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF/blob/main/Qwen3.8-27B-Q8_0.gguf) | `Q8_0` | 27.05 GB | Practically lossless. Closest to source with significant size cut. |
| [Qwen3.8-27B-Q6_K_L.gguf](https://huggingface.co/Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF/blob/main/Qwen3.8-27B-Q6_K_L.gguf) | `Q6_K` | 21.36 GB | Q6_K with Q8_0 embed/output tensors. Near-lossless top tier. |
| [Qwen3.8-27B-Q6_K.gguf](https://huggingface.co/Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF/blob/main/Qwen3.8-27B-Q6_K.gguf) | `Q6_K` | 20.79 GB | Near-lossless quality. Recommended for highest practical fidelity. |
| [Qwen3.8-27B-Q5_K_L.gguf](https://huggingface.co/Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF/blob/main/Qwen3.8-27B-Q5_K_L.gguf) | `Q5_K_M` | 18.86 GB | Q5_K_M with Q8_0 embed/output. High quality with small overhead. |
| [Qwen3.8-27B-Q5_K_M.gguf](https://huggingface.co/Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF/blob/main/Qwen3.8-27B-Q5_K_M.gguf) | `Q5_K_M` | 18.13 GB | High quality, balanced size. Recommended general-purpose. |
| [Qwen3.8-27B-Q5_K_S.gguf](https://huggingface.co/Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF/blob/main/Qwen3.8-27B-Q5_K_S.gguf) | `Q5_K_S` | 17.62 GB | Slightly smaller than Q5_K_M with similar quality. |
| [Qwen3.8-27B-Q4_K_L.gguf](https://huggingface.co/Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF/blob/main/Qwen3.8-27B-Q4_K_L.gguf) | `Q4_K_M` | 16.51 GB | Q4_K_M with Q8_0 embed/output. Sweet spot of quality and size. |
| [Qwen3.8-27B-Q4_K_M.gguf](https://huggingface.co/Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF/blob/main/Qwen3.8-27B-Q4_K_M.gguf) | `Q4_K_M` | 15.63 GB | Best size/quality tradeoff. Recommended default. |
| [Qwen3.8-27B-Q4_K_M_MTP8.gguf](https://huggingface.co/Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF/blob/main/Qwen3.8-27B-Q4_K_M_MTP8.gguf) | `Q4_K_M` | 15.83 GB | Q4_K_M with the MTP head at Q8_0 instead of Q4_0 — trades ~0.6 GB for a more accurate speculative draft. |
| [Qwen3.8-27B-Q4_K_S.gguf](https://huggingface.co/Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF/blob/main/Qwen3.8-27B-Q4_K_S.gguf) | `Q4_K_S` | 14.74 GB | Compact with minor quality loss versus Q4_K_M. |
| [Qwen3.8-27B-IQ4_NL.gguf](https://huggingface.co/Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF/blob/main/Qwen3.8-27B-IQ4_NL.gguf) | `IQ4_NL` | 14.94 GB | Slightly larger than IQ4_XS. Online repacking for ARM CPU inference. |
| [Qwen3.8-27B-IQ4_XS.gguf](https://huggingface.co/Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF/blob/main/Qwen3.8-27B-IQ4_XS.gguf) | `IQ4_XS` | 14.27 GB | Most efficient sub-Q4. Smaller than Q4_K_S with comparable quality. |

## Calibration

Imatrix generated from [reapmix (community calibration mix)](https://huggingface.co/datasets/eaddario/imatrix-calibration) — ~400K tokens — multilingual + code + math. This is the same class of public calibration data used by other community GGUF publishers; no claim of unique calibration is made for this release.

`*_L` and `*_XL` variants override the output tensor and/or token embedding to Q8_0 (versus the base type), at small extra disk for typically improved output stability at low bit-rates.

## Download

Single file:

```bash
hf download Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF --include "Qwen3.8-27B-Q4_K_M.gguf" --local-dir .
```

Whole repo:

```bash
hf download Krasnopjorovs/Qwen3.8-27B-Imatrix-GGUF --local-dir ./Qwen3.8-27B-gguf
```

## Run

```bash
./llama-server -m Qwen3.8-27B-Q4_K_M.gguf -c 32768 -ngl 99 --host 0.0.0.0 --port 8080
```

## Picking a quant

- **Q8_0 / Q6_K_L** — RAM headroom, want ceiling quality
- **Q5_K_M / Q4_K_L** — workstation default, very small quality loss
- **Q4_K_M** — best general size/quality tradeoff, the default choice
- **Q4_K_S / IQ4_NL** — tighter budgets; IQ4_NL repacks for ARM CPUs
- **IQ4_XS** — smallest here, fits a 16 GB card with context to spare

## Build info

- llama.cpp release: `version: 10358 (030ebb558)`
- Generated: 2026-08-14T22:28:22

## Credits

- Original model by [Qwen](https://huggingface.co/Qwen)
- Calibration: [reapmix (community calibration mix)](https://huggingface.co/datasets/eaddario/imatrix-calibration)
- llama.cpp by ggerganov and contributors