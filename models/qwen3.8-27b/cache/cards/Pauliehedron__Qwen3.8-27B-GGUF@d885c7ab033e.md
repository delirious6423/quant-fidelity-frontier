---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - quantized
  - llama.cpp
  - imatrix
  - vision
  - multimodal
language:
  - en
  - zh
  - ja
  - ko
---
# Pauliehedron "Meter Stick" Qwen3.8-27B

GGUF conversion of Qwen 3.8 27B (multimodal), quantized with imatrix
calibration. Every rung uses the same uniform container shape (identical
tensor-block structure, Q8_0 MTP head), so rungs differ only in quant
format. This repo is the measurement meter: on one host, one corpus, one
instrument, quant formats are compared like-for-like to show what each
quant grade costs.

## Files

| File | Kind |
|------|------|
| `Qwen3.8-27B-f16.gguf` | f16 source (the rebuild anchor) |
| `Qwen3.8-27B-IQ4_NL-uniform-imx-calib.gguf` | IQ4_NL imatrix quant |
| `Qwen3.8-27B-IQ4_XS-uniform-imx-calib.gguf` | IQ4_XS imatrix quant |
| `Qwen3.8-27B-iq3_s-uniform-imx-calib.gguf` | iq3_s imatrix quant |
| `Qwen3.8-27B-iq3_xxs-uniform-imx-calib.gguf` | iq3_xxs imatrix quant |
| `mmproj-Qwen3.8-27B-bf16.gguf` | vision projector |
| `imatrix-Qwen3.8-27B-calib.gguf` | calibration imatrix |
| `imatrix-Qwen3.8-27B-wiki.gguf` | WikiText-2 imatrix arm |
| `imatrix-Qwen3.8-27B-meta.gguf` | `llama-imatrix` metadata |

## Container

All rungs share one uniform container (`gmlx validate`): 866 tensors, 65
blocks, 360 F32 small tensors (norms/rope/freqs), and the Q8_0 MTP head
(blk.64, 8 tensors — identical on every rung). The per-rung codec mix
(imx = imatrix; plain = no imatrix):

| Rung | main body | tail |
|------|-----------|------|
| IQ4_NL (imx) | IQ4_NL ×433 | Q5_K ×64 + Q6_K ×1 |
| IQ4_XS (imx) | IQ4_XS ×433 | Q5_K ×64 + Q6_K ×1 |
| iq3_s (imx) | IQ3_S ×498 | — |
| iq3_xxs (imx) | IQ3_XXS ×498 | — |
| Q4_K_M (imx / plain) | Q4_K ×433 | Q6_K ×65 |
| Q5_K_M (imx / plain) | Q5_K ×433 | Q6_K ×65 |
| Q6_K (imx / plain) | Q6_K ×498 | — |

## Rebuild

Base safetensors → `convert_hf_to_gguf.py` (f16) → `llama-imatrix` on the
public WikiText-2 raw-train split → `llama-quantize --imatrix` per quant.
The f16 keeper and the imatrix bins are in this repo; the base safetensors
and the calibration corpus are fetched from their sources (linked below).

## Measured (our host)

M1 Studio (Apple M1 Max, 32-core GPU, 64 GB unified memory). Speeds are
local measurements with [llama.cpp](https://github.com/ggml-org/llama.cpp)
/ [gmlx (asher/gmlx)](https://github.com/asher/gmlx) on this host — your
results will differ by hardware, context depth, and server load.

KLD cells: `llama-perplexity --kl-divergence`, corpus
[wiki.test.raw](https://huggingface.co/datasets/Salesforce/wikitext) (WikiText `wikitext-2-raw-v1` test split), n_ctx 8192 · 8 chunks, reference = the repo's f16 logits.

| Quant | Result |
|------|--------|
| `IQ4_NL (imatrix)` | pp 142.45 t/s, tg 14.38 t/s, KLD 0.03686 (gmlx) |
| `IQ4_XS (imatrix)` | KLD 0.03784 (gmlx) |
| `IQ3_XXS (imatrix)` | pp 138.03 t/s, tg 16.0 t/s, KLD 0.144804 |

The Q4_K_M / Q5_K_M / Q6_K uniform-imx rungs and their plain twins are
measured (KLD 0.0294 / 0.0143 / 0.0082 imx; 0.0426 / 0.0225 / 0.0128 plain,
same instrument/corpus) and pending publication in this repo.

## License

Derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(Apache-2.0). Redistribution permitted with attribution; the base's license
terms require distributing a copy of the License with derivative works, so
the Apache-2.0 text is included in this repo as `LICENSE`. The calibration
and eval corpora are the public Salesforce/WikiText dataset — see the
[dataset card](https://huggingface.co/datasets/Salesforce/wikitext) for its terms; the
corpus itself is not redistributed here.
