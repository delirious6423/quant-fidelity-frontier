---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
- gguf
- qwen
- qwen4
- qwen4exp
- unsloth
- imatrix
- moe
- conversational
---

# Qwen3.8-Flash-Next-GGUF-SPLIT

Lossless GGUF re-sharding of [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), using the Unsloth Dynamic `UD-Q2_K_XL` quantization.

The original quantized tensors are preserved. This is a split, not a requantization.

## Included quantization

| Quantization | Total size | Shards | llama.cpp type |
|---|---:|---:|---|
| `Qwen3.8-Flash-Next-UD-Q2_K_XL` | 78,869,130,144 bytes | 13 | `Q2_K - Medium` |

| File | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00013.gguf` | 375,531,808 | `f478bdd42e2ae59f6e7eed41748b298cd4a82b6e6e3f5523b2998140db55a785` |
| `Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00013.gguf` | 28,800,138,432 | `682b0f9aa1b3a395d7c8260bd9743970a7d7ddb09a7ebe7e828f8c0ff9a7211b` |
| `Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00013.gguf` | 4,752,442,848 | `bd19fdd67776bac4982499574af1b167798683bb63a9c4a8deaf92d20c1913e3` |
| `Qwen3.8-Flash-Next-UD-Q2_K_XL-00004-of-00013.gguf` | 4,810,631,520 | `533813dd16ceb8d28ad2f2f2609bd698df994406038869402d480457a571b03e` |
| `Qwen3.8-Flash-Next-UD-Q2_K_XL-00005-of-00013.gguf` | 4,862,601,376 | `d1e37377a215b0c49f93d4522737149cd443ff6ac41cc922a2e53025997b246c` |
| `Qwen3.8-Flash-Next-UD-Q2_K_XL-00006-of-00013.gguf` | 4,637,864,256 | `96bf228245d9145f3adb17f1b496b5bd1d9ef8f6b35210c07737f8251c9aecfe` |
| `Qwen3.8-Flash-Next-UD-Q2_K_XL-00007-of-00013.gguf` | 4,810,631,616 | `3fc5534bffe752c241dbda43c7ec0c04d0325af262033cc7abf1f2b26535e159` |
| `Qwen3.8-Flash-Next-UD-Q2_K_XL-00008-of-00013.gguf` | 4,868,358,560 | `abbca5605a3f6040a33b94938086327f0c4f0ed594e8e33216b2a981b8b0b9b6` |
| `Qwen3.8-Flash-Next-UD-Q2_K_XL-00009-of-00013.gguf` | 4,627,091,776 | `de3cb71d033e74c64618b3f3de7bbb1144ab6a93534e90ccfa223ad9bf7dcbda` |
| `Qwen3.8-Flash-Next-UD-Q2_K_XL-00010-of-00013.gguf` | 4,810,631,616 | `bedc57cd1b1426403f42c84c88d9a330b53b4a08621da30151c0aefba8b9fea2` |
| `Qwen3.8-Flash-Next-UD-Q2_K_XL-00011-of-00013.gguf` | 4,862,601,408 | `09467d5ec1d05e5afa89bd8828f256fa0bccfade478523892a2d03145f935d0d` |
| `Qwen3.8-Flash-Next-UD-Q2_K_XL-00012-of-00013.gguf` | 4,637,864,256 | `e8c9fed9407bd14baed359a6d4d91324e9b3f9c2929e60b51c7c78151fb4b232` |
| `Qwen3.8-Flash-Next-UD-Q2_K_XL-00013-of-00013.gguf` | 2,012,740,672 | `9bfbfc1aa0b2f0e10ed7c88ce8a197665c36c6328e1bf5ec64bdbea777f43546` |

### Shard-size limitation

Shard `00002` contains one indivisible 28,800,138,240-byte PLE/n-gram embedding tensor. GGUF split files preserve whole tensors, so this shard cannot be reduced to 4.9 GB without changing the format and breaking standard llama.cpp compatibility. Every other data shard was produced with a `4900M` maximum.

## Required llama.cpp build

At publication time, this experimental `qwen4exp` architecture requires [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742), as linked by the source repository.

## Run with llama.cpp

Download all 13 shards into the same directory:

```bash
hf download Nielk38/Qwen3.8-Flash-Next-GGUF-SPLIT \
  --include "*.gguf" \
  --local-dir ./Qwen3.8-Flash-Next-GGUF-SPLIT
```

Load only the first shard; llama.cpp discovers the other 12 automatically:

```bash
llama-cli \
  -m ./Qwen3.8-Flash-Next-GGUF-SPLIT/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00013.gguf \
  -ngl auto \
  -lm mmap \
  -c 2048
```

For CPU-only execution, add `-dev none -ngl 0`. The model is larger than 32 GB RAM, so mmap-backed host/SSD paging is required on systems with similar memory capacity.

## Provenance and validation

- Source repository: [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-Q2_K_XL)
- Source revision: `8bdc666649440e9bdc97e16f3f75782c98478ff5`
- Source shard SHA-256:
  - `a4f3b21e77353999829f2f767e9ac21ce9c71d29a74f2cc9eda48c9bf23c8b86` (`00001-of-00003`)
  - `2e3bf1ee7d2a04e261e9f342a2d968f696cce5941d082b0e434deb9b1edc12c6` (`00002-of-00003`)
  - `ec8c106759fdf4f463039c34c0707718d7d8908d53d892bd4f002e71620803f9` (`00003-of-00003`)
- Splitter: `llama-gguf-split` from llama.cpp build `226`, commit `035e227`
- Split commands:

```bash
llama-gguf-split --merge \
  Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf

llama-gguf-split --split --split-max-size 4900M \
  Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf \
  Qwen3.8-Flash-Next-UD-Q2_K_XL
```

Validated locally on 2026-08-27 with llama.cpp build `226` (`035e227`) and an AMD Radeon RX 7900 XTX by loading shard `00001` directly. The prompt `What is 2+2? Answer with only the numeral.` returned `4`.

All 1,224 tensors and 78,858,104,320 logical tensor bytes were read back from the 13 output shards in order. Their logical tensor SHA-256 matches the merged source:

```text
7df6751fb6b8190f2ea6050219e98d157a530c94eb11f568324887e2348dc32b
```

See [`SHA256SUMS`](./SHA256SUMS) for machine-readable container checksums.
