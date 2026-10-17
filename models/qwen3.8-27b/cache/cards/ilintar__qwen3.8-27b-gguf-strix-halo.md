---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - z-lab/Qwen3.8-27B-DFlash2
pipeline_tag: image-text-to-text
tags:
  - gguf
  - llama.cpp
  - qwen3.8
  - dflash2
  - speculative-decoding
  - rocm
  - strix-halo
  - gfx1151
---

# Qwen3.8-27B GGUF for Strix Halo

This repository contains the target and DFlash2 GGUFs selected during a ROCm optimization study on AMD Strix Halo (`gfx1151`). The files are ordinary GGUF models, but they were quantized and validated together with a llama.cpp branch containing the UMA scheduler ring and the ROCm TOP_K implementation.

## Files

| File | Size | Contents |
| --- | ---: | --- |
| `Qwen3.8-27B-IQ4_XS-ALL-IMATRIX-Q8-OUT-MTP.gguf` | 16,110,851,680 bytes | Main model: 496 IQ4_XS, 10 Q8_0, and 360 F32 tensors |
| `Qwen3.8-27B-DFlash2-IQ4_XS.gguf` | 1,038,313,376 bytes | DFlash2 draft: 44 IQ4_XS, 5 Q5_K, and 32 F32 tensors |

The compatible Q8_0 multimodal projector is not duplicated in this repository. Supply `mmproj-Qwen3.8-27B-Q8_0.gguf` separately when using image input.

## Main-model quantization

The main GGUF was quantized from the Bartowski BF16 GGUF, including its embedded MTP layer, using the published Bartowski importance matrix.

- All 496 active matrices covered by the importance matrix use IQ4_XS.
- `output.weight`, `token_embd.weight`, and the eight block-64/MTP matrices remain Q8_0 because the published matrix does not calibrate them.
- One-dimensional and convolutional structural tensors remain F32.
- An explicit tensor map avoids the Q5_K substitutions made by the ordinary `MOSTLY_IQ4_XS` preset.

On 16 held-out 512-token chunks, the measured perplexity was `15.3977 +/- 0.73357`, compared with `15.1721 +/- 0.72292` for the source BF16 GGUF. This is a small local evaluation, not a general model-quality benchmark.

## DFlash2 quantization

The DFlash2 source is [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2). The IQ4_XS draft was requantized from its Q8_0 GGUF.

Standalone DFlash importance-matrix collection currently requires the target context and cannot be run directly through `llama-imatrix`. A transferred target-model matrix was therefore used:

- 18 draft tensors received exact target-name and input-width matches.
- 23 received the mean importance vector for target tensors with the same input width.
- 8 received uniform weights where no compatible target vector was available.

The standard IQ4_XS preset retained the five attention-V matrices as Q5_K. The resulting draft is 49.51% smaller than the 2,056,414,816-byte Q8_0 source.

## Strix Halo results

The counterbalanced server test used the same target model, custom HIP/ROCr retained-PM4 runtime, deterministic 128-token workloads, two independently loaded servers per configuration, and fixed DFlash2 widths 3 and 6.

### Maximum draft width 3

| Workload | Q8_0 draft | IQ4_XS draft | Change | Q8 acceptance | IQ4 acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | 25.257 t/s | 25.742 t/s | +1.92% | 48.70% | 47.13% |
| Reasoning | 30.739 t/s | 32.481 t/s | +5.67% | 69.11% | 71.67% |
| JSON | 39.587 t/s | 41.032 t/s | +3.65% | 98.96% | 98.96% |

### Maximum draft width 6

| Workload | Q8_0 draft | IQ4_XS draft | Change | Q8 acceptance | IQ4 acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | 25.215 t/s | 25.666 t/s | +1.79% | 30.34% | 29.30% |
| Reasoning | 37.797 t/s | 39.143 t/s | +3.56% | 57.65% | 57.65% |
| JSON | 56.483 t/s | 58.529 t/s | +3.62% | 95.58% | 95.58% |

The deterministic target-output hash matched between the Q8_0 and IQ4_XS drafts for every workload. These figures characterize the tested prompts and runtime; speculative throughput depends strongly on prompt-dependent acceptance.

Without speculative decoding, the main model measured `14.0976 t/s` for `tg128` with retained PM4. The retained command-list path improved that result by about 3.2% over ordinary AQL in the matched test.

## Runtime

The tested software branches are:

- [pwilkin/llama.cpp `strix-halo`](https://github.com/pwilkin/llama.cpp/tree/strix-halo)
- [pwilkin/rocm-systems `ilintar-experiments`](https://github.com/pwilkin/rocm-systems/tree/ilintar-experiments)

Example `llama-server` invocation:

```bash
export LD_LIBRARY_PATH=/path/to/custom/hip/lib:/path/to/custom/rocr/lib:/opt/rocm/core-10.0/lib:/opt/rocm/core-10.0/lib/llvm/lib
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1
export ENABLE_RETAINED_PM4=1
export DEBUG_HIP_GRAPH_PM4=1

llama-server \
  -m Qwen3.8-27B-IQ4_XS-ALL-IMATRIX-Q8-OUT-MTP.gguf \
  --mmproj /path/to/mmproj-Qwen3.8-27B-Q8_0.gguf \
  --mmproj-device ROCm0 \
  -dev ROCm0 \
  -ngl 999 \
  -fa on \
  -fit off \
  --load-mode none \
  -c 65536 \
  -b 2048 \
  -ub 512 \
  --parallel 1 \
  --jinja \
  --spec-type draft-dflash \
  --spec-draft-model Qwen3.8-27B-DFlash2-IQ4_XS.gguf \
  --spec-draft-device ROCm0 \
  --spec-draft-ngl 99 \
  --spec-draft-n-min 0 \
  --spec-draft-n-max 6 \
  --spec-draft-p-min 0.10
```

Width 3 was marginally faster for the low-acceptance prose test. Width 6 was substantially faster for the higher-acceptance reasoning and structured-output tests.

## Checksums

```text
9e5f86c794b45b215a2768723c94819500c0c8f896094d628728e6fc94cd6324  Qwen3.8-27B-IQ4_XS-ALL-IMATRIX-Q8-OUT-MTP.gguf
11c7848014bd68040a42837b381bbefff5d0acc22cf20b6055a48d560c834445  Qwen3.8-27B-DFlash2-IQ4_XS.gguf
```

## License

The source model and DFlash2 metadata declare the Apache 2.0 license. Users remain responsible for reviewing the upstream model repositories and their terms.
