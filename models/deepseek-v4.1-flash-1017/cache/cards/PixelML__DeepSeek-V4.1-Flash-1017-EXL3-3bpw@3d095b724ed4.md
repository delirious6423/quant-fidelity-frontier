---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
tags:
  - exl3
  - deepseek_v41
  - quantized
---

# DeepSeek-V4.1-Flash-1017 EXL3 3 bpw (routed experts)

Routed-expert EXL3 quant of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
@ [`1bae0e10`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170),
the October 17 2026 update of DeepSeek-V4.1-Flash (763B MoE). Made by Pixel ML for 4× CMP 170HX (64 GB) and 2× DGX Spark.

| | |
|---|---|
| Routed experts (40 layers × 384) | EXL3 K=3, MUL1 codebook, greedy beam 16 encode |
| Calibration | **none** (identity Hessian). Quality vs the FP8/MXFP4 source is **not yet measured** |
| Everything else | copied as stored by DeepSeek (FP8 attention / shared experts, BF16 embeddings, head, norms, vision, MTP) |
| Engram tables | **not included** (see below) |
| Size | 223.0 GB in 46 shards (46,080 EXL3 expert tensors), plus ~203 GB of Engram shards from the base repo |

## Engram tables

The Engram n-gram tables (`model-00047-of-00048.safetensors`, `model-00048-of-00048.safetensors`, ~189 GiB FP8) are
listed in `model.safetensors.index.json` but not uploaded. Download both from the **1017** base repo at the pinned
revision into the model folder:

```
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors --local-dir <model-folder>
```

The tables from the original DeepSeek-V4.1-Flash do not match 1017 (all 48 shards changed).

## Runtime

Same layout and `quantization_config` as other routed-expert V4.1 EXL3 packs: needs a vLLM build with an EXL3 plugin
that supports DeepSeek-V4.1 (for example the vllm-exl3 stacks used for V4.1-Flash EXL3 packs). Stock vLLM does not
load it. Prompt formatting: use `encoding/` (copied from the base repo); there is no Jinja chat template.

## How it was made

8× H100 (one shard group per GPU), about 6.6 GPU-hours:

```
python tools/quantize_experts_exl3.py --src DeepSeek-V4.1-Flash-1017 --dst out \
  --bits 3 --codebook mul1 --greedy --beam 16 --batch 8 --only-files <shard group>
```

The script refuses `--bits` other than 2 by default; we changed that one check to allow 3.

## Credits

- **Base model and weights:** DeepSeek-AI ([@deepseek-ai](https://huggingface.co/deepseek-ai)),
  [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, MIT.
- **Quantization script:** `tools/quantize_experts_exl3.py` and its helpers are authored by sfxnz
  ([@sfxnz](https://github.com/sfxnz)), [sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark @ 3de914682d0880c8e478712a60ada930339c22e4](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark/tree/3de914682d0880c8e478712a60ada930339c22e4), MIT.
- **3-bit settings and ExLlamaV3 pin:** the K=3 uncalibrated greedy/beam-16 run and ExLlamaV3 commit come from diffbot
  ([@diffbot](https://huggingface.co/diffbot)), [diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 @ 6062be5f](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/tree/6062be5f0f43c50ba562ef4e7087d4fc83eea36c) (`recipe/quantize/`), MIT.
- **EXL3 format and quantizer:** ExLlamaV3 authored by turboderp ([@turboderp](https://github.com/turboderp)),
  [turboderp-org/exllamav3 @ 5be886578ec80324c2c715269387be2058724b6e](https://github.com/turboderp-org/exllamav3/tree/5be886578ec80324c2c715269387be2058724b6e), MIT.

## Licence

MIT, same as the base model. Quantization adds no restrictions.
