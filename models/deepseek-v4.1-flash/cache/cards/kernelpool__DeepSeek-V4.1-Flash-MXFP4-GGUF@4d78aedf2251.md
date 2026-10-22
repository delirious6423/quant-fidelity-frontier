---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
- gguf
- deepseek
- metal
---

# DeepSeek-V4.1 Flash MXFP4 GGUF for DwarfStar

GGUF files for DeepSeek V4.1 Flash in [DwarfStar (DS4)](https://github.com/antirez/ds4). They work with
upstream `main` and with the [`kernelpool/ds4:ds41-perf`](https://github.com/kernelpool/ds4/tree/ds41-perf)
branch of [antirez/ds4#1073](https://github.com/antirez/ds4/pull/1073), which speeds up V4.1 decoding and
prefill on Metal. These are DS4 files, not llama.cpp GGUFs: llama.cpp does not load DS4's `deepseek41` layout.

The files come from the original September release, not from `DeepSeek-V4.1-Flash-1017`. Keep the main
model and the drafter from the same release: the 1017 checkpoint has different weights, including the
DSpark stages.

| File | Size | Content |
| --- | ---: | --- |
| `DeepSeek-V4.1-Flash-MXFP4.gguf.part1`, `.part2` | 467.2 GiB | Main model, `deepseek41` layout, in two parts: routed experts are the checkpoint's own MXFP4 blocks repacked without changing their values; Q8_0 attention, shared experts and output head; F16 embeddings; the two Engram tables in the checkpoint's FP8 encoding (188.8 GiB, read from disk) |
| `DeepSeek-V4.1-Flash-DSpark.gguf` | 8.0 GiB | DSpark drafter for speculative decoding, `deepseek41-dspark` layout: native MXFP4 experts, dense weights expanded losslessly to F16 or F32 |

The drafter replaces `DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf`, an older format that current DS4 does not
load; that file remains at revision `22a073e5`.

## Run

Join the two parts first (the parts can be deleted afterwards):

```sh
cat DeepSeek-V4.1-Flash-MXFP4.gguf.part1 DeepSeek-V4.1-Flash-MXFP4.gguf.part2 > DeepSeek-V4.1-Flash-MXFP4.gguf
```

Build either tree, then run:

```sh
git clone https://github.com/antirez/ds4 && cd ds4 && make                    # upstream
git clone -b ds41-perf https://github.com/kernelpool/ds4 && cd ds4 && make    # antirez/ds4#1073

./ds4 -m DeepSeek-V4.1-Flash-MXFP4.gguf
./ds4 -m DeepSeek-V4.1-Flash-MXFP4.gguf --dspark --mtp-model DeepSeek-V4.1-Flash-DSpark.gguf
./ds4-server -m DeepSeek-V4.1-Flash-MXFP4.gguf --dspark --mtp-model DeepSeek-V4.1-Flash-DSpark.gguf
```

The main model keeps 278 GiB of weights resident and reads the Engram tables from disk; the drafter adds
8 GiB. Tested on a 512 GB Mac Studio M3 Ultra. DSpark is optional and does not help every prompt; see
[speculative decoding](https://github.com/antirez/ds4/blob/main/docs/SPECULATIVE_DECODING.md).

## Conversion

Written from [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) at
revision `dba1be0a40aa45a94ad051997016db3960a90277`, which is recorded in each file's metadata. The MXFP4
mode of `gguf-tools/deepseek41_quantize.py` is in the `ds41-perf` branch (upstream's converter writes Q2
and Q4); `gguf-tools/deepseek41_dspark.py` is in both trees.

```sh
make -C gguf-tools libds4quants.dylib
python3 gguf-tools/deepseek41_quantize.py --hf models/DeepSeek-V4.1-Flash \
  --source-revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --quant mxfp4 --out DeepSeek-V4.1-Flash-MXFP4.gguf
python3 gguf-tools/deepseek41_dspark.py --hf models/DeepSeek-V4.1-Flash \
  --source-revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --out DeepSeek-V4.1-Flash-DSpark.gguf
```

## Quality

Scored with `gguf-tools/quality-testing/score_official` on the six tracked V4.1 Flash fixture sets
(`gguf-tools/quality-testing/deepseek-v4.1-flash-*`, 256 cases): upstream `main` and the `ds41-perf`
branch give identical results with this file.
