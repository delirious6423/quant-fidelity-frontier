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

# DeepSeek-V4.1-Flash — SAGE EXL3 1.59 bpw

Loadable [ExLlamaV3](https://github.com/turboderp-org/exllamav3) pack of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

Quantized by [vcruz305](https://huggingface.co/vcruz305) with **SAGE**, our own
mixed-precision quantization method for EXL3. Tensors use different EXL3 `K`
values; the kernels read `K` from each trellis tensor at load.

This is the **TP1-class** sibling of
[DSV4.1-Flash-SAGE-EXL3-3.30bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw) (TP2-class) and
[DSV4.1-Flash-EXL3-4.75bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw) (TP4-class).
Same SAGE method, the expert budget of a single 128 GB-class device.

**Status: compiled pack live** (40/40 routed-expert layers + native/Engram join).
17 `model-*-of-00017` shards on this repo.

## SAGE EXL3 1.59 bpw routed-expert quantization

The routed MoE expert bank averages **1.59 bpw**
(K1–K6). Engram remains native FP8 and protected tensors remain at
their native precision.

Final packed model size: **330.39 GB / 307.70 GiB** (`330385931208` bytes across
the 17 shards; `330366294360` bytes of tensor data in the index), of which the
routed experts are **108.86 GB / 101.4 GiB** (the 3.30 bpw pack carries 224.93 GB
of experts).

If total packed bytes are divided by all ~763B model parameters, the resulting
model-wide effective storage average is ~3.46 bits/parameter. **This is not
the EXL3 expert quantization bitrate**.

## Measured quality

Teacher-forced agreement with the **native FP4 release checkpoint** (DeepSeek's
own weights run through the reference PyTorch kernels), full-vocabulary KL
divergence per token, on 32 held-out sequences × 4096 tokens (8 each of general
text, code, math and reasoning), the same set for every pack in the table.
Top-1 / top-5 is how often the pack's most likely next token(s) match the
reference's. These are agreement metrics against the release checkpoint, not
benchmark scores and not generated-code pass rates.

| Pack | Expert bytes | KL vs FP4 (mean) | Top-1 | Top-5 | KL general | KL code | KL math | KL reasoning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **this pack, 1.59 bpw** | 108.9 GB | **0.1885** | **91.2%** | **77.5%** | 0.409 | 0.054 | 0.183 | 0.108 |
| 3.30 bpw sibling | 224.9 GB | 0.0910 | 94.0% | 83.0% | 0.159 | 0.035 | 0.097 | 0.073 |
| FP4 reference | native | 0 | 100% | 100% | 0 | 0 | 0 | 0 |

Top-1 by subset for this pack: general 82.0%, code 97.9%, math 91.7%,
reasoning 93.2% (3.30 bpw: 88.8 / 98.3 / 94.1 / 94.7). Mean KL is flat across
the context window (0.201 at tokens 0–512, 0.181 at 2048–4096), so the loss does
not grow with position. Halving the expert bytes of the 3.30 bpw pack roughly
doubles its KL to the reference; code and reasoning lose the least, open-ended
general text the most.

## Pack

| Item | Value |
| --- | --- |
| Format | EXL3 (trellis) + native tables where noted |
| Routed-expert bpw | **1.59** SAGE mixed-K (K1–K6) |
| Packed size | 330.39 GB / 307.70 GiB |
| Shards | 17 compiled `model-*-of-00017` (Engram last two) |
| Index | `model.safetensors.index.json` |
| `quant_method` | `exl3` |
| Routed experts | EXL3, mixed K (SAGE, TP1-class) |
| Engram | Native FP8 retained (same tables as the 3.30 and 4.75 packs) |
| Protected non-expert tensors | Copied (attn / shared / gate / MTP / DSpark / vision / head), not wholesale-requantized |

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

cfg = Config.from_directory("vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw")
model = Model.from_config(cfg)
model.load()
tokenizer = Tokenizer.from_config(cfg)
```

Needs a recent ExLlamaV3 with DeepSeek V4.1 / CED support and K1 trellis
kernels (routed experts go down to K1 in this pack). This pack is sized as a
**TP1-class** weight ceiling (~101 GiB of routed experts on device; the Engram
tables are read from disk), not a promise of a specific serve topology.

## Files

- `model-00001-of-00017.safetensors` … `model-00015-of-00017.safetensors` — EXL3 body + copied protected tensors
- `model-00016-of-00017.safetensors`, `model-00017-of-00017.safetensors` — Engram (native FP8, identical to upstream `00047` / `00048`)
- `model.safetensors.index.json`, `config.json`, `tokenizer.json`, `tokenizer_config.json`

## License

MIT, same as [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
Cite DeepSeek for the base model. This repository is the EXL3 pack only.

## Notes

- Quantization: vcruz305, SAGE mixed-K EXL3, TP1, 1.59 expert-bpw.
- Please do not file “missing experts/ work tree” issues. That is not this repo.
