---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
- gguf
- qwen
- qwen3_5
- unsloth
- imatrix
- conversational
---

# Qwen3.8-27B-MTP-GGUF-SPLIT

Split GGUF shards of [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), using the current Unsloth Dynamic v3 `UD-Q3_K_XL` quantization.

The original quantized tensors are preserved; this is a split, not a requantization.

## Included quantization

| Quantization | Total size | Shards | llama.cpp type |
|---|---:|---:|---|
| `Qwen3.8-27B-UD-Q3_K_XL` | 13,146,393,824 bytes | 3 | `Q3_K - Large` |

| File | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3.8-27B-UD-Q3_K_XL-00001-of-00003.gguf` | 4,906,066,688 | `7503289f9bf53bd41e5c29bbf6a125eb5744b3ca5842b8224f5763dc07dda3db` |
| `Qwen3.8-27B-UD-Q3_K_XL-00002-of-00003.gguf` | 4,890,964,352 | `32a142816d057e24e3bc6a55e6523a081559d3828638805779eeca822d8b0c6a` |
| `Qwen3.8-27B-UD-Q3_K_XL-00003-of-00003.gguf` | 3,349,362,784 | `7112823753700a9dfb15289fbbe0006f7c1b301b409cedc0dd9f61549c061291` |

## Run with llama.cpp

Download all three shards into the same directory:

```bash
hf download Nielk38/Qwen3.8-27B-MTP-GGUF-SPLIT \
  --include "*.gguf" \
  --local-dir ./Qwen3.8-27B-MTP-GGUF-SPLIT
```

Load only the first shard; llama.cpp discovers the other two automatically:

```bash
llama-cli \
  -m ./Qwen3.8-27B-MTP-GGUF-SPLIT/Qwen3.8-27B-UD-Q3_K_XL-00001-of-00003.gguf \
  -ngl 999 \
  -c 8192
```

The Qwen3.8 checkpoint is MTP-trained. This repository contains the main Q3 GGUF only. The optional standalone MTP draft model and vision projectors can be downloaded from the [source GGUF repository](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main).

## Provenance and validation

- Source file: `Qwen3.8-27B-UD-Q3_K_XL.gguf`
- Source revision: `27af057ecb382ddfea5d12837360a8980560e3ed`
- Source SHA-256: `8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e`
- Splitter: `llama-gguf-split` from llama.cpp `b10502` (`0adcc3bb5`)
- Split command:

```bash
llama-gguf-split --split --split-max-size 4900M \
  Qwen3.8-27B-UD-Q3_K_XL.gguf \
  Qwen3.8-27B-UD-Q3_K_XL
```

Validated locally on 2026-08-19 with llama.cpp `b10502` on an AMD Radeon RX 7900 XTX by loading shard `00001` directly. The prompt `What is 2+2? Answer with only the numeral.` returned `4`.

The shards were also merged back with `llama-gguf-split --merge`. All 866 tensor descriptors and tensor bytes matched the source, with logical tensor SHA-256 `2b7efe87386ffa39812f47aac97d431fe1f7706a8733b8cff1833e324e0371f2`. The merged container retains llama.cpp's three `split.*` metadata keys, so its whole-file hash is intentionally different from the unsplit source container.

See [`SHA256SUMS`](./SHA256SUMS) for machine-readable shard checksums.

