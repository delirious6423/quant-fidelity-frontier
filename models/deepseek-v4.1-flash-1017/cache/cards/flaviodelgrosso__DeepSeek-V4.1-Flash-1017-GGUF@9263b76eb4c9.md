---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- gguf
- deepseek41
- text-generation
- speculative-decoding
---

# DeepSeek V4.1 Flash 1017 — Razor GGUF

Razor-native artifacts of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, packed directly from the official safetensors checkpoint by [Razor](https://github.com/flaviodelgrosso/razor). Nothing is derived from third-party GGUFs or importance matrices.

| File | Size | SHA-256 | What |
|---|---:|---|---|
| `DeepSeek-V4.1-Flash-1017-R2.75.gguf` | 396 GB | `6558432dc1e65f009267759e50d6132b75738894883c3a95449f752a3b7bab6a` | The model: calibrated routed experts at 2.75 bpw |
| `DeepSeek-V4.1-Flash-1017-DSpark.gguf` | 8.6 GB | `7a2fe312c8b5752bdcecbb4d2fb782efe5bbcd33cdc4f0568ce2bbe591f5ffd5` | Lossless DSpark drafter for speculative decoding, same revision |

> [!IMPORTANT]
> GGUF is only the container. Expert, FP8 and Engram payloads are opaque `I8`/`U8` tensors whose encoding is declared in `deepseek41.*` metadata, so **llama.cpp and other GGUF runtimes cannot load these files**. They run only in Razor on Apple Silicon.

## R2.75

- **Routed experts:** one tensor per (layer, expert, projection), `gate_up` and `down`, each at a tier chosen by a calibrated precision map: Razor `q2` (2-bit) or `q3` (2-bit base + 1-bit residual plane), with a fixed per-layer block Hadamard rotation. Average **2.75 bpw** over routed experts, counting every scale and block header; alignment padding (16 KiB) excluded.
- **Statics:** the checkpoint's FP8 E4M3 (E8M0 block scales), BF16 and F32 tensors verbatim, except nine classes re-encoded to Q4_K/Q5_K by measured sensitivity. Statics read per decoded token: 4.42 GiB (7.64 GiB verbatim).
- **Engram tables:** the checkpoint's FP8 rows, lossless (`e4m3_e8m0_32_row264`), read from disk on demand.
- **Tokenizer and Engram hashing:** generated from the checkpoint's `tokenizer.json` and config, stored in metadata. Single file.
- **Provenance in metadata:** `general.source.url` / `general.source.revision`, SHA-256 of every checkpoint shard, packer version, precision map name (`r2.75`) and SHA-256 (`eebc9b03206b1892403ce5c08f6f9972ccc86003b526c598c1f9cf5e548a7e43`).
- **Calibration:** routing traces, per-expert activation statistics and per-projection ΔKLD sensitivity measured on the lossless artifact (the checkpoint's own MXFP4 experts), then a byte-budgeted allocation constrained by a replay of Razor's expert cache. The calibration and evaluation corpora are disjoint.

**Reproducible:** `razor convert deepseek41 --precision-map razor-convert/src/deepseek41/r2.75.map` on the checkpoint at the revision above produces this file byte for byte (`--verify` checks every payload against the checkpoint). Without a precision map the same command builds the 501.5 GB lossless artifact.

## Quality and speed

Measured against the lossless artifact (the teacher), Apple M5 Max 128 GB:

| | R2.75 | Uniform 2.75 bpw baseline |
|---|---:|---:|
| Mean KLD vs teacher (calibration corpus) | **0.0960** | 0.1241 |
| Top-1 agreement | **0.918** | — |

- Same budget, formats, kernels and statics as the baseline; only the tier assignment differs: **−22.6% KLD**.
- Full evaluation-corpus scores are pending and will be added here.
- **Decode:** 21.8–22.4 tok/s with SSD expert streaming (`engine_bench --placement ssd:32 --context 16384`, 2,048-token prompt, 256 generated), peak footprint 43.4 GB, no swap.

## Use

```sh
razor download deepseek41 deepseek41-dspark
razor serve --model ~/.razor/models/DeepSeek-V4.1-Flash-1017-R2.75.gguf \
  --placement ssd --expert-cache-gib 32 --context 16384 \
  --speculation opportunistic --dspark ~/.razor/models/DeepSeek-V4.1-Flash-1017-DSpark.gguf
```

Experts stream from the internal SSD (explicit `pread`, no page-fault streaming) into a 32 GiB pinned cache; fits 128 GB Macs.

## License

MIT, as upstream. Model weights © DeepSeek ([license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE)).
