---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- xefp4
- 4-bit
- intel-arc
- opencl
- mtp
---

# Qwen3.8-27B-XeFP4

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) quantized to **XeFP4**, an open 4-bit weight format for
Intel Arc GPUs. It runs on **XeQ Runtime**, the OpenCL engine for XeFP4: **https://github.com/kinoqdev/XeFP4**
(engine update of 2026-11-21 or later).

> These files do **not** load in transformers, llama.cpp or vLLM. They need XeQ Runtime.
> This repository contains the text model only.

## Files

| File | Size | SHA-256 (`.bin`) | Content |
|---|---|---|---|
| `q38_xe8g32e.bin` + `.json` | 14,402,528,256 B | `c312b6d9b2bfcf74a48ccc4f8d96b88e2510e0b69ae9e40c6d20058b7ad938a0` | main model |
| `q38_mtp_xe8.bin` + `.json` | 217,556,992 B | `6e77bafed849070920be7fc4dc197fc416644b901b1dbd9fa7acf6bc288a095b` | MTP head for speculative decoding (optional) |
| `vocab_order.npy` | 993,408 B | `b40ec3751835a586b201f86e4ef489d8e911c930a265dc66869d3beab5fb6314` | token order used by the MTP draft step |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | | | from the source model |
| `LICENSE`, `NOTICE` | | | |

`.json` header SHA-256: `q38_xe8g32e.json` `1ee930a425328987cfa5fe35d9a5965d17c82f767374adf2466ca3e921241129`, `q38_mtp_xe8.json` `daaadfd67e8d7b6dabb879ebb6d7be7b071220cb2a7a749f4f793ced1fffa1e9`.

## Usage

```bash
git clone https://github.com/kinoqdev/XeFP4 && cd XeFP4
pip install -r requirements.txt
python -m xeq selftest <path to this repository>
python -m xeq run <path to this repository> -p "Explain Rayleigh scattering in two sentences."
python -m xeq serve <path to this repository> --port 8080 --ctx 8192
```

MTP is used automatically when `q38_mtp_xe8.bin` is present (`--no-mtp` turns it off). Thinking mode is off by default
(`--think` turns it on; it writes long reasoning before the answer, which at these speeds takes minutes).

**Memory.** The weights are 13.4 GiB. Allocated GPU memory (FP16 KV cache, the default):

| context | MTP | allocated |
|---|---|---|
| 4K | off | 14.2 GiB |
| 8K | on | 15.1 GiB |
| 32K | on | 18.2 GiB |

- Arc A770 16 GB: up to an 8K context with MTP (the engine picks a 1024-token prefill chunk on a GPU under 20 GiB).
- Arc 140T with 32 GB RAM: a 32K context fits in the default shared GPU memory. If XeQ prints
  `[xeq] warning: ... GiB allocated`, lower `--ctx` or use `--no-mtp`.

## Quantization

| Part | Format |
|---|---|
| all linear layers | XeFP4-8: E2M1 values, block 32, 3-bit block factor, FP16 row scale (4.096 bits/weight) |
| embedding | XeFP4-8 |
| lm_head | INT8, per-row FP32 scale |
| MTP layer | XeFP4-8 (fc split by input half) |
| norms, gates, conv, A_log, dt_bias | FP32 |

Linear layers use an error-compensating (GPTQ-style) encoder: columns are encoded in order and each column's rounding
error is passed on to the columns not yet encoded, weighted by the layer's input statistics; block factors and row
scales are chosen with the same statistics. Calibration: 128 sequences × 512 tokens, none from the evaluation sets:
English web text, Korean (Wikipedia, news, chat), code in 12 languages and Markdown, and multilingual chat in 15
languages. Any XeFP4 decoder reads the file; only the choice of stored values differs. The format is specified in
[XEFP4_SPEC.md](https://github.com/kinoqdev/XeFP4/blob/main/spec/XEFP4_SPEC.md).

## Accuracy

Perplexity (lower is better), 16 FineWeb documents × 512 tokens and 8 sequences × 512 tokens for the other text types
(none in the calibration data):

| | size | English web | Korean Wikipedia | code | Markdown |
|---|---|---|---|---|---|
| BF16 | – | 9.1256 | 3.7758 | 2.3763 | 7.0056 |
| **XeFP4 (this model)** | **14.40 GB** | **9.2987 (+1.90%)** | **3.8085 (+0.87%)** | **2.3832 (+0.29%)** | **7.0943 (+1.27%)** |
| GGUF UD-IQ4_XS | 14.35 GB | 9.2566 (+1.44%) | 3.8423 (+1.76%) | 2.3860 (+0.41%) | 7.0819 (+1.09%) |

The XeFP4 row was measured with XeQ Runtime (prefill path); BF16 and GGUF rows come from a reference evaluation on the
same token sequences. Code covers Python, C++, JavaScript, Rust, Java, Go, TypeScript and Swift.

## Performance

Batch 1, greedy decoding, FP16 KV cache. MTP decoding gives the same output tokens as plain decoding.

**Intel Arc A770 16 GB** (driver 32.0.101.9150):

| | tok/s |
|---|---|
| decode, MTP off | 21.5 |
| decode, MTP on (2 drafts per step) | 35.0 (short prompt), 29.1 (150-word explanation) |
| prefill, 1,800-token prompt | 772 |

**Intel Arc 140T** (Core Ultra 7 235H, 32 GB DDR5-5600, Windows; the iGPU's speed varies with temperature):

| prompt / context | prefill tok/s | decode at that position, MTP off |
|---|---|---|
| 2K | about 215 | 4.7 |
| 8K | about 200 | 4.4 |
| 16K | about 190 | 4.0 |
| 26K | about 170 | |
| 32K | | 3.4 |

- Decode with MTP on the 140T: 8.6 tok/s on long answers (2.4 tokens per verification pass); it varies with the text.
- Dense 27B decoding reads all 13.7 GB of weights for every token, so it is bound by memory bandwidth. On an integrated
  GPU the Mixture-of-Experts models (for example Gemma 4 26B-A4B, about 20–24 tok/s on the same 140T) are much faster.

## Sampling

Decoding is greedy by default (same output every time; best for code, facts, translation and tool calls). For
open-ended chat and writing, turn on sampling with `--temperature`, `--top-p`, `--top-k`, `--min-p`,
`--presence-penalty`, `--repetition-penalty` and `--seed` (the server reads the same fields from the request). The
source model's generation config uses temperature 1.0, top_p 0.95, top_k 20. With sampling the engine decodes without
MTP (multi-draft MTP is greedy only).

## Limitations

- Tested on Intel Arc A770 (Xe-HPG) and Arc 140T (Xe-LPG+). Other Intel GPUs are untested.
- Text only. The source model's vision encoder is not included.
- Batch size 1.
- Prefill and token-by-token decoding differ slightly numerically, so greedy output can differ at near-tie positions.
- Accuracy is close to GGUF UD-IQ4_XS: better on Korean and code, slightly behind on English web text and Markdown.

## License and attribution

- Derived from Qwen3.8-27B by the Qwen team (Alibaba Cloud), licensed under Apache-2.0.
- The weights were modified by quantization to the XeFP4 format.
- See `LICENSE` and `NOTICE`.
- XeFP4 and XeQ Runtime are not affiliated with or endorsed by Alibaba Cloud or Intel.
