---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
license: apache-2.0
quantized_by: aj9o9
tags:
  - gguf
  - qwen
  - qwen3.8
  - llama.cpp
  - imatrix
---

# Qwen3.8-27B GGUF

GGUF of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for [llama.cpp](https://github.com/ggml-org/llama.cpp).
More files will land here. First one is a role mix I baked for my 3090, not a flat IQ3.

## A note from me

Same deal as my [Ling-3.0-flash](https://huggingface.co/aj9o9/Ling-3.0-flash-GGUF)
and [Nemotron-3.5-Lightning](https://huggingface.co/aj9o9/nvidia-nemotron-3.5-lightning-30b)
uploads: I make these to run on my own box, then share them.

**Please report anything you find.** Bad output, crashes, wrong metadata, a better
flag for a particular card — open a discussion here or reach me at
[@ItsmeAjayKV](https://x.com/ItsmeAjayKV).

## Files

| File | Size | What it is | Status |
|---|---|---|---|
| **`Qwen3.8-27B-gdn8-q6attn-iq3ffn.gguf`** | **13.6 GB (12.7 GiB) / 3.99 bpw** | Role mix. GDN state Q8, attn Q6, mid-FFN IQ3_XXS. See below. | **up** |
| `Qwen3.8-27B-Q4_K_M.gguf` | ~16–18 GB | Flat K-quant, if I bake one from the same BF16 | waiting |
| `Qwen3.8-27B-Q5_K_M.gguf` | ~19–20 GB | Same | waiting |
| `Qwen3.8-27B-imatrix-v6.gguf` | ~14 MB | Imatrix used for the mix. Reusable. | maybe |

This is **not** a flat `IQ3_XXS`. Only the mid-FFN tensors are IQ3_XXS. The filename is the recipe.

## Download (CLI)

```bash
hf download hf://aj9o9/Qwen3.8-27B-GGUF/Qwen3.8-27B-gdn8-q6attn-iq3ffn.gguf
```

That pulls just the mix, into the current directory. Same thing, older-style:

```bash
hf download aj9o9/Qwen3.8-27B-GGUF --include "Qwen3.8-27B-gdn8-q6attn-iq3ffn.gguf"
```

Imatrix (only if you want to requant, not needed to run):

```bash
hf download hf://aj9o9/Qwen3.8-27B-GGUF/Qwen3.8-27B-imatrix-v6.gguf
```

Rule of thumb I actually use: **pick the largest quant that fits in RAM/VRAM, not the largest one you can download.**

## Should you use the mix?

**Take this** if you want ~13G and you care more about keeping attention / GDN state fat than a uniform 3-bit file.

**Skip this** if you can hold [Unsloth UD-Q3_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (13.4G) or a Q4. Those beat it on every number I ran. I am not going to pretend otherwise.

## What the mix actually is

Official BF16 → my `my-mix.txt` + bartowski's Qwen3.8 calibration-v6.

| Role | Type |
|---|---|
| GDN state (`ssm_*` except out) | Q8_0 |
| Full attention Q/K/V/O, embed, output, MTP | Q6_K |
| GDN mixers (`attn_qkv`, `attn_gate`, `ssm_out`) | Q4_K |
| FFN edge (layers 0–3, 60–63) | IQ3_S |
| FFN mid (everything else `ffn_*`) | IQ3_XXS |
| norms | F32 |

Imatrix: [bartowski calibration-v6](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-calibration-v6.txt), rendered through this model's chat template. 583 chunks at `-c 512`, `--parse-special --process-output`. About 63% of that file is tool-call text. I did **not** use wiki-only calib.

Needs a recent llama.cpp with `qwen35`. Old trees will not load it.

## Numbers I actually measured

Same box, same llama.cpp, same prompts. Wiki KLD is vs official BF16 logits (`wiki.test.raw`, n_ctx=512). GLSL / hard is my locked suite, think **off**, t=0.

| model | size | wiki KLD ↓ | GLSL ↑ | hard |
|---|---:|---:|---:|---:|
| Unsloth Q4_K_M | 16G | 0.015 | 0.946 | 6/6 |
| Unsloth UD-Q3_K_XL | 13.4G | **0.031** | **0.922** | **6/6** |
| bartowski Q3_K_S | 13.7G | 0.070 | 0.892 | 5/6 |
| **this mix** | **13.6 GB** | 0.073 | 0.863 | 5/6 |
| Unsloth Q5_K_M | 19G | 0.006 | 0.855 | 5/6 |
| bartowski IQ3_M | 13.9G | 0.057 | 0.831 | 5/6 |

Read it like this:

- If you have 16G+, take **Q4**.
- If you want the best *13G-class* file I measured, take **Unsloth UD-Q3_K_XL**, not mine.
- This mix **beats bartowski IQ3_M on GLSL** (0.863 vs 0.831) at a slightly smaller size, and it **beats Unsloth Q5 on GLSL** at 6G less. That is the honest reason it exists.
- Hard: I fail `lfu_cache`. Q4 and UD-Q3 pass all six. I am not going to hide that.
- Tools (single-shot + short agent loops) were a tie. Everyone passed the easy set. I will not claim a tools win.

Wiki KLD is English Wikipedia. The imatrix is chat + tools. Those two will not rank the same, and they didn't.

## How I run it

```bash
llama-server \
  -m Qwen3.8-27B-gdn8-q6attn-iq3ffn.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 999 -fa on --jinja \
  -np 1 -t 12 \
  --alias qwen38-27b-gdn8 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-mtp \
  -c 24576
```

From the Hub:

```bash
llama-server \
  --hf-repo aj9o9/Qwen3.8-27B-GGUF \
  --hf-file Qwen3.8-27B-gdn8-q6attn-iq3ffn.gguf \
  -ngl 999 -fa on --jinja \
  --spec-type draft-mtp \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -c 24576
```

MTP tensors are in the file (Q6_K). `--spec-type draft-mtp` is optional; it drafts, it does not change quality.

Official sampling from the Qwen card:

- Thinking: `temperature=1.0`, `top_p=0.95`, `top_k=20`
- Instruct / no-think: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `presence_penalty=1.5`

I graded the table above at t=0, think off, so I could actually compare quants.

## Hardware this was made on

| | |
|---|---|
| GPU | RTX 3090 24 GB |
| System RAM | 64 GB |
| Runtime | llama.cpp `master`, arch `qwen35` |

The mix is meant to leave room for context on a 24 GB card. A 16 GB card can load the weights; keep `-c` honest.

Vision: this GGUF is text weights only. If you want the encoder, grab an `mmproj` from the official convert or from bartowski/unsloth and pass `--mmproj`. I have not tested that pairing.

## About the model

- 27B dense, hybrid 16 × (3 GDN + 1 full attn), 64 layers + MTP
- hidden 5120, native 262k context
- thinking on by default in the official template

See [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for the real model card. Their numbers are BF16, not this file.

## Links

- Official weights: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- Runtime: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- Calib used for the mix: [bartowski Qwen3.8-27B calibration-v6](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-calibration-v6.txt)

## License

Apache 2.0, same as [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). `LICENSE` is in this repo.

## Disclaimer

Not affiliated with Alibaba, Qwen, Unsloth, or bartowski. Provided as-is.
Use the official card for intended use, safety, and limitations.
