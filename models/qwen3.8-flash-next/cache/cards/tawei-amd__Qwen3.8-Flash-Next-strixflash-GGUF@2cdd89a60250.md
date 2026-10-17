---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
- gguf
- halocpp
- strix-halo
- gfx1151
---

# Qwen3.8-Flash-Next, strix-flash mix A (GGUF)

Quantized weights of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for local serving
on AMD Strix Halo (Ryzen AI Max, gfx1151) with [Halocpp](https://github.com/tawei-amd/Halocpp). They are made
for, and tested with, the strix-flash engine that Halocpp runs for this model.

```
./halocpp pull qwen3.8-flash-next
./halocpp serve qwen3.8-flash-next          # OpenAI-compatible API on 127.0.0.1:52625
```

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 3,235 | `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08` |
| `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2,786,568,256 | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |
| `Qwen3.8-Flash-Next-strixflash-mixA-00001-of-00006.gguf` | 375,531,936 | `ac483373b2c194c2ed73e3d428a6c577dc294fc54e877972db033697ce10aa3e` |
| `Qwen3.8-Flash-Next-strixflash-mixA-00002-of-00006.gguf` | 28,800,138,432 | `0e9027aa7de4817c91b5954039b563484466392e29a6b051b096f9c47a71a1c6` |
| `Qwen3.8-Flash-Next-strixflash-mixA-00003-of-00006.gguf` | 19,790,357,632 | `5e1d95f7a3e6a60e9b7da64beea5b289228edc4cd97341509e93592b8ee9a65b` |
| `Qwen3.8-Flash-Next-strixflash-mixA-00004-of-00006.gguf` | 19,608,645,248 | `64f0284324de3f8e0fa15856cfc575b9d5a261ce399a6c9d2fd3ca596a36acf6` |
| `Qwen3.8-Flash-Next-strixflash-mixA-00005-of-00006.gguf` | 19,719,695,840 | `93a0d6ee5205363965d9965c8a5c072241fe6a5141fc9fff12fab74690531fc5` |
| `Qwen3.8-Flash-Next-strixflash-mixA-00006-of-00006.gguf` | 4,103,658,848 | `ce1ce2d8bb8e73f761694768d5dc5edef1270a8bb960dd6515f9aa1086b24c9d` |
| `tokenizer/chat_template.jinja` | 8,952 | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| `tokenizer/config.json` | 4,745 | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` |
| `tokenizer/generation_config.json` | 202 | `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` |
| `tokenizer/tokenizer_config.json` | 17,928 | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` |
| `tokenizer/tokenizer.json` | 12,809,320 | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |

`Qwen3.8-Flash-Next-strixflash-mixA-*.gguf` is one model split into GGUF shards with llama.cpp's
`llama-gguf-split` (load the first shard; the others are found by name). `SHA256SUMS` lists every file.

## How mix A was made

Starting from unsloth's `UD-IQ4_XS` GGUF of the model
([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), revision
`38bb39ee97821de2c9009abb7e93950eec396e66`), the dense trunk was re-quantized with GPTQ; the routed experts are kept as published:

| tensors | format |
|---|---|
| DeltaNet input projections, attention q / k / v | Q5_0, GPTQ |
| DeltaNet output, shared experts, LM head | IQ4_NL, GPTQ |
| hyper-connections, attention output | Q8_0 |
| routed experts | as published: IQ3_S gate / up, IQ4_NL down (Q8_0 in 5 layers) |

GPTQ used Hessians collected by the engine from 48 calibration sequences, alternating wikitext-2 test text
beyond its first 100,000 tokens and C/C++ source code, so the calibration text is disjoint from the
perplexity evaluation below. About 4.4 GB of weights are read per decoded token.

`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` is the model's multi-token-prediction block, used as the draft head
for speculative decoding; it is unsloth's file, unchanged. The `tokenizer/` files are the base model's, unchanged
(revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`).

## Accuracy

Perplexity on 32,704 wikitext-2 test tokens (64 sequences of 512, decode path): measurement pending.

## License

These weights are a derivative of Qwen3.8-Flash-Next and are distributed under the Qwen Community License 1.0,
included as [LICENSE](LICENSE). Its conditions apply to every use of these files.
