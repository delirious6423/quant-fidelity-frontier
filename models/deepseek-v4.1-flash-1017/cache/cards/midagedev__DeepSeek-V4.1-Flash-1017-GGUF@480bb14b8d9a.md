---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- bloomery
- deepseek
- imatrix
---

# DeepSeek-V4.1-Flash-1017 bq3km

A Q3_K_M-layout GGUF of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
made with an importance matrix that [bloomery](https://github.com/midagedev/bloomery)'s own engine collected. It runs on
one workstation: one 48 GB card and 256 GB of host RAM. The file is 347.4 GB in 9 shards.

Against a plain Q3_K_M of the same model, it is closer to the reference on English, Korean and code text, at the same
speed. The tables below have the numbers.

The base model's license is MIT. Its `LICENSE` file is included here unchanged.

## The files

| File | What |
|---|---|
| `bq3km/DeepSeek-V4.1-Flash-1017-bq3km-000NN-of-00009.gguf` | the model, 9 shards, 347.4 GB; GGUF architecture `deepseek41`, the chat template in the header |
| `DeepSeek-V4.1-Flash-1017.imatrix.gguf` | the importance matrix the file was quantized with (llama.cpp's imatrix GGUF form) |
| `LICENSE` | the base model's, unchanged |

## Quality

KLD is measured against `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` `UD-Q4_K_XL` (the source of this file), over 4
windows of 2,048 tokens of each text. Lower is closer. PPL is the text's own perplexity under each file; it does not
depend on the reference.

| Text | Q3_K_M (`vcruz305/DeepSeek-V4.1-Flash-GGUF`) | **bq3km (this file)** |
|---|---|---|
| English (wikitext-2 test) | KLD 0.259, PPL 2.295 | **KLD 0.215, PPL 2.183** |
| Korean (Korean Wikipedia articles not in the calibration text) | KLD 0.1345, PPL 3.510 | **KLD 0.1191, PPL 3.434** |
| Code (Rust source of mistral.rs) | KLD 0.0758, PPL 2.012 | **KLD 0.0651, PPL 1.976** |

The KLD standard errors are about ±0.010 (English), ±0.0046 (Korean) and ±0.0035 (code). On the same windows the
reference `UD-Q4_K_XL` has PPL 3.268 (Korean) and 1.897 (code).

## Speed

bloomery `generate_ds41`, RTX A6000 (300 W) with the routed experts that do not fit on the card served from host RAM
(`--place a`), AMD Threadripper PRO 5975WX, 3 rounds with the order rotated:

| | Q3_K_M (`vcruz305`) | bq3km |
|---|---|---|
| prompt, 4,096 tokens | 387.7 tok/s | 403.1 tok/s (+4.0 %) |
| prompt, 512 tokens | 245.2 tok/s | 238.0 tok/s (−2.9 %; one cold row in three) |
| decode at depth 512 | 45.72 tok/s | 45.34 tok/s |
| decode at depth 4,096 | 37.06 tok/s | 37.04 tok/s |

The speed rows were measured on an earlier write of this file with the same tensor types and sizes (its importance matrix lacked the routed downs of host-served experts; the rows here have them). The two files read the same bytes a token, so their speed is the same.

## The quantization

| Tensors | Type |
|---|---|
| routed and shared experts, gate and up | Q3_K (with the imatrix) |
| routed and shared experts, down | Q4_K; layers 0 and 1: Q5_K |
| attention (q_a, q_b, kv, output_a, output_b, compressor), indexer projections, hyper-connection fns, `token_embd`, engram | Q3_K |
| `output` (the head) | Q6_K |
| norms, router, biases | as in the source (F32) |

Totals: Q3_K 471 tensors, 243.2 GB; Q4_K 76 tensors, 97.1 GB; Q5_K 4 tensors, 6.2 GB; Q6_K 1 tensor, 0.5 GB.

- **Command.** `bloomery-quantize` requantized the `UD-Q4_K_XL` shards (`--allow-requantize --keep-split`) with a
  per-tensor rule file for the types above, on the host's 32 cores, in 2,291 s.
- **imatrix.** bloomery's own collector ran `UD-Q4_K_XL` over 40 chunks of 2,048 tokens of a mixed text: English and
  Korean Wikipedia and code (no wikitext-2 test article is in it). It counts every routed expert pick: gate, up and
  down, also for the experts the host serves. 27 of the 15,360 routed experts were never picked by that text and
  quantize without importance.

## Quick start

bloomery serves it (add `--dry-run` to print the start without downloading):

```
bloomery-serve --hf midagedev/DeepSeek-V4.1-Flash-1017-GGUF:bq3km
```

The file is a standard GGUF of the `deepseek41` architecture; any runtime that reads DeepSeek-V4.1-Flash GGUF files and
the K-quant types should read it. We have tested it only with bloomery.

### Optional: the r8 sidecar

bloomery can page the routed experts' gate and up from an r8 sidecar beside the file (155.7 GB). It is not needed: the
default path reads the file itself. To make it, build `r8conv` with plain cargo (`cargo build --release -p
bloomery-model --bin r8conv`, see bloomery's `docs/BUILD.md`) and run `r8conv convert <first shard>` and
`r8conv verify <first shard>`; it writes `bq3km-r8/` beside `bq3km/` in about two minutes from a page-cached file.

## How it was made

The quantization, the importance matrix, the measurements and this card were done with AI help (Claude) under the
maintainer's review. Every number above was measured on the machine named, with bloomery's own runners.
