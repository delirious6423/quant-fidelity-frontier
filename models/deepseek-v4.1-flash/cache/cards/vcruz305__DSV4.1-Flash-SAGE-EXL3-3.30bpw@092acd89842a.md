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

# DeepSeek-V4.1-Flash — SAGE EXL3 3.30 bpw

Loadable [ExLlamaV3](https://github.com/turboderp-org/exllamav3) pack of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

Quantized by [vcruz305](https://huggingface.co/vcruz305) with **SAGE**, our own
mixed-precision quantization method for EXL3. Tensors use different EXL3 `K`
values; the kernels read `K` from each trellis tensor at load.

This is the **TP2-class** sibling of [DSV4.1-Flash-EXL3-4.75bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw).
Same SAGE method, tighter expert budget.

**Status: compiled pack live** (40/40 routed-expert layers + native/Engram join).
31 `model-*-of-00031` shards on this repo.

## SAGE EXL3 3.30 bpw routed-expert quantization

The routed MoE expert bank averages **3.30 bpw**
(K2–K8). Engram remains native FP8 and protected tensors remain at
their native precision.

Final packed model size: **446.44 GB / 415.78 GiB** (`446440212472` bytes).

If total packed bytes are divided by all ~763B model parameters, the resulting
model-wide effective storage average is ~4.68 bits/parameter. **This is not
the EXL3 expert quantization bitrate**.

## Pack

| Item | Value |
| --- | --- |
| Format | EXL3 (trellis) + native tables where noted |
| Routed-expert bpw | **3.30** SAGE mixed-K (K2–K8, no K1) |
| Packed size | 446.44 GB / 415.78 GiB |
| Shards | 31 compiled `model-*-of-00031` (Engram last two) |
| Index | `model.safetensors.index.json` |
| `quant_method` | `exl3` |
| Routed experts | EXL3, mixed K (SAGE, TP2-class) |
| Engram | Native FP8 retained (same tables as the 4.75 pack) |
| Protected non-expert tensors | Copied (attn / shared / DSpark / vision / head), not wholesale-requantized |

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

cfg = Config.from_directory("vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw")
model = Model.from_config(cfg)
model.load()
tokenizer = Tokenizer.from_config(cfg)
```

Needs a recent ExLlamaV3 with DeepSeek V4.1 / CED support. Multi-GPU via
whatever your ExLlamaV3 build exposes (`tp`, device map). This pack is sized
as a **TP2-class** weight ceiling (~416 GiB on disk), not a promise of a
specific serve topology.

## Files

- `model-*-of-*.safetensors` — EXL3 body + copied protected tensors (after join)
- last two shards — Engram (native FP8, hardlinked from upstream `00047` / `00048`)
- `model.safetensors.index.json`, `config.json`, `tokenizer.json`, `tokenizer_config.json`

## License

MIT, same as [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
Cite DeepSeek for the base model. This repository is the EXL3 pack only.

## Notes

- Quantization: vcruz305, SAGE mixed-K EXL3, TP2, 3.30 expert-bpw.
- Please do not file “missing experts/ work tree” issues. That is not this repo.
