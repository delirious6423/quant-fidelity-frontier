---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - ds4
  - dwarfstar
  - qwen4exp
  - moe
  - mixed-quantization
  - apple-silicon
---

# Qwen3.8-Flash-Next — DS4 Mixed-Q3-64GB and Mixed-Q3-96GB

Two per-layer mixes of the existing Q2 and Q4 GGUFs of Qwen3.8-Flash-Next for [DwarfStar (ds4)](https://github.com/antirez/ds4). Both sit between the Q2 and Q4 releases in size and accuracy. Measured on a Mac Studio M5 Ultra with 96 GB of unified memory.

> Unofficial community build. Not affiliated with or endorsed by Qwen, Alibaba, antirez or Ivan Fioravanti.
> These are quantized language-model weights, not lossless copies of the original model.
> **DS4-specific**: the files use the `qwen4exp` architecture with the BF16 n-gram table stored inside the file. They are not expected to load in other runtimes (not tested).

## Files

| File | Size | Resident weights | Notes |
|---|---:|---:|---|
| `Qwen3.8-Flash-Next-Mixed-Q3-64GB.gguf` | 153,915,375,616 bytes (143.3 GiB) | 47.96 GiB | Runs on a 64 GB Mac |
| `Qwen3.8-Flash-Next-Mixed-Q3-96GB.gguf` | 164,500,750,336 bytes (153.2 GiB) | 57.83 GiB | For 96 GB-class Macs (not tested on 64 GB) |

Each file contains the 95.37 GiB BF16 n-gram table, which ds4 reads directly from disk; keep it on a fast local SSD. Checksums are in `SHA256SUMS`.

### Memory use

From ds4's startup log, including KV cache and buffers, with `--prefill-chunk 1024`:

| File | ctx 8192 | ctx 32768 | ctx 131072 |
|---|---:|---:|---:|
| Mixed-Q3-64GB | ≈49 GiB | ≈50 GiB | ≈54 GiB |
| Mixed-Q3-96GB | ≈59 GiB | ≈60 GiB | ≈63 GiB |

Only the plain Q2 file was measured at every context length (KV + buffers: 1.47 GiB at 8192, 2.30 GiB at 32768, 5.61 GiB at 131072); the Mixed sizes are those values plus each file's resident weights. `--prefill-chunk 8192` adds about 6 GiB. On an M4 Max with 64 GB, Mixed-Q3-64GB measured 49.44 GiB at `--ctx 8192`, ≈54 tok/s decode with MTP and ≈604 tok/s prefill (6K prompt).

## Quantization layout

Only the routed-expert tensors (`ffn_*_exps`) differ. There are 49 blocks; block 48 is the embedded MTP block.

| File | gate / up | down |
|---|---|---|
| Mixed-Q3-64GB | Q4_K: 41, 42, 44–46, 48 · IQ2_XXS: the other 43 | MXFP4: 0, 3–4, 6–10, 14–23, 30–48 (37 blocks) · Q2_K: 1–2, 5, 11–13, 24–29 (12 blocks) |
| Mixed-Q3-96GB | IQ2_XXS: 0–24 · Q4_K: 25–48 | MXFP4: all 49 blocks |

All other tensors are identical across the files: attention / SSM / shared-expert projections and `output.weight` in Q8_0; `token_embd` and the n-gram table in BF16; HC tensors in F16; router and norms in F32.

## How they were made

Built with `gguf-tools/mixed/splice_mixed_expert_layers_gguf.py` from the ds4 repository, taking the Q2 GGUF as base and the Q4 GGUF as donor (both from [antirez/qwen3.8-flash-next-gguf](https://huggingface.co/antirez/qwen3.8-flash-next-gguf)). No re-quantization is performed; tensors are copied from the two source files.

```bash
# Mixed-Q3-64GB
python3 gguf-tools/mixed/splice_mixed_expert_layers_gguf.py \
  --base Qwen3.8-Flash-Next-Q2.gguf --donor Qwen3.8-Flash-Next-Q4.gguf \
  --q4-gate-up-layers 41,42,44-46 \
  --q4-down-layers 0,3,4,6-10,14-23,30-47 \
  --out Qwen3.8-Flash-Next-Mixed-Q3-64GB.gguf

# Mixed-Q3-96GB
python3 gguf-tools/mixed/splice_mixed_expert_layers_gguf.py \
  --base Qwen3.8-Flash-Next-Q2.gguf --donor Qwen3.8-Flash-Next-Q4.gguf \
  --q4-gate-up-layers 25-48 --q4-down-layers 0-48 \
  --out Qwen3.8-Flash-Next-Mixed-Q3-96GB.gguf
```

Block 48 (MTP) already uses Q4_K / MXFP4 in the Q2 base file, so the Mixed-Q3-64GB command does not need to list it. The tensor payload sizes the script reports for these commands (153,904,343,552 and 164,489,718,272 bytes) plus the 11,032,064-byte header equal the file sizes above. The `general.name` and `general.file_type` metadata fields were edited afterwards (same-length replacement; `file_type` set to 1024, "unspecified") because the Q2 base's values did not describe a mix. No tensor data was changed.

## Usage

```bash
git clone https://github.com/antirez/ds4.git && cd ds4 && make
./ds4 -m /path/to/FILE.gguf --ctx 8192 --prefill-chunk 1024
./ds4-server -m /path/to/FILE.gguf --ctx 8192 --prefill-chunk 1024 --mtp
```

Requires a DwarfStar build with the native BF16 n-gram reader (older builds expecting `--ple` cannot load these files).

### Vision

Not included. The vision encoder for Qwen3.8 Flash Next is a separate GGUF: get it with `./download_model.sh qwen38-vision` (ds4 repository) and pass it with `--vision FILE`. Image input has been confirmed to work with these files.

## Measured on Mac Studio M5 Ultra, 96 GB

ds4 commit 8db1d1d. Plain Q2 and Q4 are shown for reference.

| | Q2 | Mixed-Q3-64GB | Mixed-Q3-96GB | Q4 |
|---|---:|---:|---:|---:|
| Decode, `ds4-bench`, 8K ctx (tok/s) | 69.5 | 69.2 | 69.9 | 69.4 |
| Prefill, `ds4-bench`, 8K ctx (tok/s) | 1,941 | 1,918 | 1,938 | 1,956 |
| Decode, CLI, no MTP, prose / code (tok/s) | 73.5 / 73.5 | 73.2 / 73.4 | 73.2 / 73.9 | 73.7 / 73.8 |
| Decode, CLI, MTP, prose / code (tok/s) | 85.2 / 101.9 | 84.9 / 101.1 | 83.0 / 103.5 | 83.8 / 101.3 |

CLI runs: 256 generated tokens, greedy (`--temp 0 --nothink`), ctx 8192, `--prefill-chunk 1024`. MTP speed-up depends heavily on the content. Speed is nearly the same across all four files; the difference between them is accuracy and size.

### Agreement with a hosted reference

Average negative log-likelihood (NLL, lower is better) of continuations produced by Alibaba's hosted `qwen3.8-flash` (via OpenRouter), using `gguf-tools/quality-testing` with the bundled Qwen3.8 fixtures. Percentages are relative to the plain Q4 GGUF.

| Set | Q2 | Mixed-Q3-64GB | Mixed-Q3-96GB | Q4 |
|---|---:|---:|---:|---:|
| 100 short prompts (ctx 4096) | 0.3527 (+21.4%) | 0.3327 (+14.5%) | 0.3069 (+5.6%) | 0.2906 |
| 12 long prompts (ctx 32768) | 0.1528 (+21.9%) | 0.1409 (+12.4%) | 0.1352 (+7.8%) | 0.1253 |

Top-1 agreement (100 prompts / 12 long prompts): Mixed-Q3-64GB 89.7% / 95.3%.

The hosted model is described by the ds4 README as based on Flash Next, not as a byte-identical checkpoint, so this is an external check, not proof of exact native-weight behaviour. The 100 prompts have about 56 target tokens each and the long set has only 12 cases: small differences are not conclusive.

## Credits

- Original model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen)
- Q2 / Q4 main and MTP tensors: Ivan Fioravanti ([Q2](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2), [Q4](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4)), as repackaged in [antirez/qwen3.8-flash-next-gguf](https://huggingface.co/antirez/qwen3.8-flash-next-gguf)
- Native n-gram packaging, splice tool and ds4 engine: Salvatore Sanfilippo (antirez) / DwarfStar
- Layer mixing and this release: Krissaito470

## License

Distributed under the Qwen Community License 1.0 (Copyright (c) 2026 Qwen); the `LICENSE` file is included and its copyright and permission notice must be kept in all copies.
