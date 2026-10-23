---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
  - gguf
  - deepseek
  - metal
---

# DeepSeek V4.1 Flash 1017 GGUF for DwarfStar

A 2.36 bpw quant (excluding engrams) of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
for [DwarfStar (ds4)](https://github.com/antirez/ds4), with 2.25bpw routed experts and 8-bit or higher
everything else. It uses the same recipe and file layout as
antirez's [`DeepSeek-V4.1-Flash-Q2.gguf`](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf),
built from the 1017 weights with an imatrix calibrated on 1017. Designed for SSD-streaming on 128GB systems, where decode
can reach up to 20tps with expert caching. Works in llama.cpp too, but I strongly recommend ds4.

| File | Size | Main weights |
| --- | ---: | ---: |
| `DeepSeek-V4.1-Flash-1017-Q2.gguf` | 340.60 GiB | 151.77 GiB |
| `DeepSeek-V4.1-Flash-1017-Vision.gguf` | 0.90 GiB | Separate vision encoder |

The language GGUF includes 188.83 GiB of native FP8 Engram tables. ds4 reads the rows it needs
directly from disk and never loads the whole table, so keep the file on a fast local SSD.

## Download and run

```sh
hf download nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF DeepSeek-V4.1-Flash-1017-Q2.gguf --local-dir gguf
./ds4 -m gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf --ssd-streaming --ctx 32768
```

The language model loads in current ds4 with no changes. Memory options are the same as for
antirez's Q2: SSD streaming on one 128 GB Mac, resident TP across two, or full residency on a
larger Mac. See the [ds4 model guide](https://github.com/antirez/ds4/blob/main/docs/MODELS.md#deepseek-v41-flash).

**Vision needs [ds4 PR #1225](https://github.com/antirez/ds4/pull/1225)** until it is merged.
Current ds4 only accepts a V4.1 vision file from the first V4.1 checkpoint, so it rejects this
one. Don't use the first checkpoint's `DeepSeek-V4.1-Flash-Vision.gguf` with this model instead:
ds4 will load it, but its vision tower was trained alongside different language weights. With
the PR applied:

```sh
hf download nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF DeepSeek-V4.1-Flash-1017-Vision.gguf --local-dir gguf
./ds4 -m gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf --vision gguf/DeepSeek-V4.1-Flash-1017-Vision.gguf \
  --ssd-streaming --ctx 32768
```

## Quantization

The stock `--quant q2` recipe from ds4's `deepseek41_quantize.py`: IQ2_XXS routed gate/up experts,
Q2_K down experts, Q8_0 attention projections, shared experts and output, F16/F32 elsewhere, and
the Engram tables copied byte for byte. Tensor names, types, offsets and the file size
(365,713,686,528 bytes) match antirez's Q2 exactly.

The one difference is the imatrix. antirez's imatrix isn't published and was collected on the
first checkpoint. This build uses Lucebox's 1017 imatrix,
[`imatrix/v41-1017-262144.dat`](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF/blob/bf3e05b8784d3806dd1d2dbd21dfd0fff1ab88dd/imatrix/v41-1017-262144.dat), collected on
the 1017 weights. 32 of the 15,360 routed experts never received a token during
calibration and are quantized without importance weighting.

Built with ds4 at `fc80bd6` plus PR #1225, which gives the converters the 1017 name, URL and
revision:

```sh
deepseek41_quantize.py --hf DeepSeek-V4.1-Flash-1017 --source-revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --imatrix v41-1017-262144.dat --out DeepSeek-V4.1-Flash-1017-Q2.gguf
deepseek41_vision.py --hf DeepSeek-V4.1-Flash-1017 --source-revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --out DeepSeek-V4.1-Flash-1017-Vision.gguf
```

Both files passed ds4's validators: `deepseek41_validate_gguf.py --payload` re-encodes a sample of
every tensor from the source and compares it byte for byte, and `deepseek41_vision.py --validate
--verify-payload` checks the vision file against the source.

## Checksums

```
f57e6c3e5e68c513249e1c472a2600bf59792ca78cacb7ac0531a6056376856e  DeepSeek-V4.1-Flash-1017-Q2.gguf
861f1097e8686111fa23dc4a6bf139fd0c4cc114db2f1b4090a267f7683c0dbf  DeepSeek-V4.1-Flash-1017-Vision.gguf
```

Source: `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e1088993b9b82de08f59de5be5a34e42170`.
The vision encoder comes from the same checkpoint. See the included upstream MIT license.
