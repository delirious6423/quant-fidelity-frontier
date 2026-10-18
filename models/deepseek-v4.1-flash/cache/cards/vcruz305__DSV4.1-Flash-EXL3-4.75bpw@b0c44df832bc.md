---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
tags:
  - exl3
  - exllamav3
  - deepseek
  - moe
  - quantization
  - mixed-precision
---

# DeepSeek-V4.1-Flash — EXL3 SAGE Mixed-K (4.75 bpw)

Loadable [ExLlamaV3](https://github.com/turboderp-org/exllamav3) pack of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

Quantized by [vcruz305](https://huggingface.co/vcruz305) with **SAGE**, our own
mixed-precision quantization method for EXL3. Tensors use different EXL3 `K`
values; the kernels read `K` from each trellis tensor at load.

**Status: complete.** 32 compiled `model-*-of-00032` shards + `model-00033-of-00033` (main-layer backbone) + index.

## Pack

| Item | Value |
| --- | --- |
| Format | EXL3 (trellis) + native tables where noted |
| Average bpw | **4.75** (453.07 GB over the 763B card: backbone + Engram + vision) |
| Shards | `model-00001-of-00032` … `model-00032-of-00032`, plus `model-00033-of-00033` |
| Index | `model.safetensors.index.json` |
| `quant_method` | `exl3` |
| Routed experts | EXL3, mixed K (SAGE, quality-first) |
| Engram | Native FP8 retained (`00031` / `00032`) |
| Protected non-expert tensors | Source FP8 (32x32 block, ue8m0 scales), not requantized: main-layer attn / router / shared experts / norms / hc in `00033`; DSpark, vision, head in `00001`–`00030` |

This is a compiled 8 GiB-class shard pack. It is not a dump of per-expert work files.

## About SAGE

SAGE is our own mixed-precision quantization method for EXL3. The method is not
published. If a tensor is EXL3, its `K` is self-describing on the trellis.

## Source model (DeepSeek, not this pack)

V4.1-Flash is DeepSeek’s MIT CED MoE: 40 layers (20 encoder + 20 decoder),
552B backbone + ~196B Engram, 384 routed experts + 1 shared, top-6.
Architecture, sampling defaults, and any benchmark rows on the
[upstream card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
are **DeepSeek’s**. They are not scores of this EXL3 pack. I have not claimed
them as mine.

Upstream sampling (start here, then tune): `temperature=1.0`, `top_p=0.95`.

## Load

ExLlamaV3, point at this repo (or a local snapshot):

```python
from exllamav3 import Config, Model, Tokenizer

cfg = Config.from_directory("vcruz305/DSV4.1-Flash-EXL3-4.75bpw")
model = Model.from_config(cfg)
model.load()
tokenizer = Tokenizer.from_config(cfg)
```

Needs a recent ExLlamaV3 with DeepSeek V4.1 / CED support. Multi-GPU via
whatever your ExLlamaV3 build exposes (`tp`, device map). This pack is sized
as a **TP4-class** weight ceiling (~422 GiB on disk), not a promise of a
specific serve topology.

## Files

- `model-00001-of-00032.safetensors` … `model-00030-of-00032.safetensors` — EXL3 body + copied protected tensors
- `model-00031-of-00032.safetensors`, `model-00032-of-00032.safetensors` — Engram
- `model-00033-of-00033.safetensors` — main-layer non-expert backbone (1,243 tensors, 6.88 GB), copied byte-for-byte from `deepseek-ai/DeepSeek-V4.1-Flash`. Added 2026-09-23; earlier downloads without it produce word salad.
- `model.safetensors.index.json`, `config.json`, `tokenizer.json`, `tokenizer_config.json`

## License

MIT, same as [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
Cite DeepSeek for the base model. This repository is the EXL3 pack only.

## Notes

- Quantization: vcruz305, SAGE mixed-K EXL3.
- Please do not file “missing experts/ work tree” issues. That is not this repo.
