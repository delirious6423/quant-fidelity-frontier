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

Razor-native artifacts of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, packed directly from the official safetensors checkpoint by [Razor](https://github.com/flaviodelgrosso/razor). Nothing is derived from third-party GGUFs or quantized packs.

| File | Size | SHA-256 | What |
|---|---:|---|---|
| `DeepSeek-V4.1-Flash-1017-T1.75.gguf` | 328 GB | `2cb4e7444c12164b531a835bd82f115631d54cd5cd1259378368ef8bcb9fa760` | The model: trellis-coded routed experts at 1.75 bpw (not uploaded yet) |
| `DeepSeek-V4.1-Flash-1017-DSpark.gguf` | 8.6 GB | `7a2fe312c8b5752bdcecbb4d2fb782efe5bbcd33cdc4f0568ce2bbe591f5ffd5` | Lossless DSpark drafter for speculative decoding, same revision |

> [!IMPORTANT]
> GGUF is only the container. Expert, FP8 and Engram payloads are opaque `I8`/`U8` tensors whose encoding is declared in `deepseek41.*` metadata, so **llama.cpp and other GGUF runtimes cannot load these files**. They run only in Razor on Apple Silicon.

## T1.75

- **Routed experts:** every expert matrix (gate, up, down independently) is `razor_trellis_v1`: 16×16 tiles coded by a tail-biting Viterbi search over Razor's own 16-bit-window pair codebook, with random sign vectors and a 128-point Hadamard rotation. Rates 1.51–3.01 bpw (`t24`–`t48`), assigned per matrix by a calibrated allocation; average **1.75 bpw** over routed experts.
- **Encoding:** LDLQ against per-expert Hessians from calibration activations of the lossless artifact (gate/up with a layer-wide prior, down with diagonal damping 1.0).
- **Statics:** the checkpoint's FP8 E4M3, BF16 and F32 tensors verbatim, except nine classes re-encoded to Q4_K/Q5_K by measured sensitivity. Statics read per decoded token: 4.42 GiB (7.64 GiB verbatim).
- **Engram tables:** the checkpoint's FP8 rows, lossless, read from disk on demand.
- **Tokenizer and Engram hashing:** generated from the checkpoint's `tokenizer.json` and config, stored in metadata. Single file.
- **Provenance in metadata:** `general.source.url` / `general.source.revision`, SHA-256 of every checkpoint shard, packer version and precision map.

**Rebuild:** `make calib-trace` + `make convert-v41-trellis` in the Razor repository (see `docs/getting-started.md`); about 27 h on an M5 Max.

## Quality and speed

Against the lossless artifact (the teacher), 724 items / 97,713 teacher-forced positions, Apple M5 Max 128 GB:

| Scope | KLD mean | Top-1 agreement |
|---|---:|---:|
| chat | 0.139 | 0.875 |
| coding | 0.056 | 0.950 |
| long-context | 0.079 | 0.959 |
| multilingual | 0.455 | 0.743 |
| reasoning | 0.058 | 0.945 |
| **overall** | **0.164** | **0.891** |

Multilingual text is the weak spot (about 61% of the summed KLD).

**Decode** (context 16,384, 2,048-token prompt): 39–41 tok/s with `--placement auto`, 30.6–30.7 tok/s with SSD expert streaming and a 32 GiB cache (`ssd:32`); chat with thinking 35 and 21 tok/s. Prefill 231–250 tok/s.

## Use

```sh
razor download deepseek41 deepseek41-dspark
razor serve --model ~/.razor/models/DeepSeek-V4.1-Flash-1017-T1.75.gguf --context 16384
```

Experts stream from the internal SSD (explicit `pread`, no page-fault streaming) into a pinned cache; fits 128 GB Macs.

## License

MIT, as upstream. Model weights © DeepSeek ([license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE)).
