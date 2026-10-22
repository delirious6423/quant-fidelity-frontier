---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- gguf
- deepseek
- metal
---

# DeepSeek-V4.1 Flash 1017 MXFP4 GGUF for DwarfStar

GGUF files for DeepSeek V4.1 Flash 1017 in [DwarfStar (DS4)](https://github.com/antirez/ds4). They work with
upstream `main` and with the [`kernelpool/ds4:ds41-perf`](https://github.com/kernelpool/ds4/tree/ds41-perf)
branch of [antirez/ds4#1073](https://github.com/antirez/ds4/pull/1073), which speeds up V4.1 decoding and
prefill on Metal. These are DS4 files, not llama.cpp GGUFs: llama.cpp does not load DS4's `deepseek41` layout.

The files come from [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017);
files for the original September release are in
[`kernelpool/DeepSeek-V4.1-Flash-MXFP4-GGUF`](https://huggingface.co/kernelpool/DeepSeek-V4.1-Flash-MXFP4-GGUF).
Keep the main model, the drafter and the vision encoder from the same release: the two checkpoints have
different weights, including the DSpark stages and the vision tower.

| File | Size | Content |
| --- | ---: | --- |
| `DeepSeek-V4.1-Flash-1017-MXFP4.gguf.part1`, `.part2` | 467.2 GiB | Main model, `deepseek41` layout, in two parts: routed experts are the checkpoint's own MXFP4 blocks repacked without changing their values; Q8_0 attention, shared experts and output head; F16 embeddings; the two Engram tables in the checkpoint's FP8 encoding (188.8 GiB, read from disk) |
| `DeepSeek-V4.1-Flash-1017-DSpark.gguf` | 8.0 GiB | DSpark drafter for speculative decoding, `deepseek41-dspark` layout: native MXFP4 experts, dense weights expanded losslessly to F16 or F32 |
| `DeepSeek-V4.1-Flash-1017-Vision.gguf` | 0.9 GiB | Vision encoder for image input, `deepseek4-vision` layout: the checkpoint's vision tower and aligner in BF16 |

## Run

Join the two parts first (the parts can be deleted afterwards):

```sh
cat DeepSeek-V4.1-Flash-1017-MXFP4.gguf.part1 DeepSeek-V4.1-Flash-1017-MXFP4.gguf.part2 > DeepSeek-V4.1-Flash-1017-MXFP4.gguf
```

Build either tree, then run:

```sh
git clone https://github.com/antirez/ds4 && cd ds4 && make                    # upstream
git clone -b ds41-perf https://github.com/kernelpool/ds4 && cd ds4 && make    # antirez/ds4#1073

./ds4 -m DeepSeek-V4.1-Flash-1017-MXFP4.gguf
./ds4 -m DeepSeek-V4.1-Flash-1017-MXFP4.gguf --dspark --mtp-model DeepSeek-V4.1-Flash-1017-DSpark.gguf
./ds4 -m DeepSeek-V4.1-Flash-1017-MXFP4.gguf --vision DeepSeek-V4.1-Flash-1017-Vision.gguf
./ds4-server -m DeepSeek-V4.1-Flash-1017-MXFP4.gguf --dspark --mtp-model DeepSeek-V4.1-Flash-1017-DSpark.gguf
```

The main model keeps 278 GiB of weights resident and reads the Engram tables from disk; the drafter adds
8 GiB. Tested on a 512 GB Mac Studio M3 Ultra. DSpark is optional and does not help every prompt; see
[speculative decoding](https://github.com/antirez/ds4/blob/main/docs/SPECULATIVE_DECODING.md). With
`--vision`, `/read FILE` in `./ds4` submits a PNG or JPEG.

## Conversion

Written from [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, which the main model and drafter record in
`general.source.revision`. `gguf-tools/deepseek41_vision.py` always records the September revision
`df42c109`, so the vision encoder's metadata names that revision although it holds the 1017 weights. All
three converters write the original repo's URL into `general.source.url`. The MXFP4 mode of
`gguf-tools/deepseek41_quantize.py` is in the `ds41-perf` branch (upstream's converter writes Q2 and Q4);
`gguf-tools/deepseek41_dspark.py` and `gguf-tools/deepseek41_vision.py` are in both trees.

```sh
make -C gguf-tools libds4quants.dylib
python3 gguf-tools/deepseek41_quantize.py --hf models/DeepSeek-V4.1-Flash-1017 \
  --source-revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --quant mxfp4 --out DeepSeek-V4.1-Flash-1017-MXFP4.gguf
python3 gguf-tools/deepseek41_dspark.py --hf models/DeepSeek-V4.1-Flash-1017 \
  --source-revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --out DeepSeek-V4.1-Flash-1017-DSpark.gguf
python3 gguf-tools/deepseek41_vision.py --hf models/DeepSeek-V4.1-Flash-1017 \
  --out DeepSeek-V4.1-Flash-1017-Vision.gguf
```

## Quality

Scored with `gguf-tools/quality-testing/score_official` on the six tracked V4.1 Flash fixture sets
(`gguf-tools/quality-testing/deepseek-v4.1-flash-*`, 256 cases): upstream `main` and the `ds41-perf`
branch give identical results with this file. With the vision encoder, both trees also give identical
output on the image fixtures in `tests/vision-fixtures/glm53` (`text.png`, `earth.jpg`).
