---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- quantization
- imatrix
- moe
language:
- en
- ru
---

# Qwen3.8-Flash-Next — rcO4 (GGUF, 9 files, 109.6 GB)

Strata совместимый квант сделанный с использованием матрицы квантования сделанной преимущественно на русском "датасете"

Проектор для зрения используйте из иных широко известных репозиториев квантов 

## Storage by group

| Group | GiB | B params | bits/w | Types (share of bytes) |
| --- | ---: | ---: | ---: | --- |
| Routed experts | 69.53 | 120.80 | 4.94 | Q5_1 27%, Q4_K 25%, IQ4_XS 19%, Q5_K 17%, IQ4_NL 11% |
| PLE / n-gram table | 26.88 | 51.23 | 4.51 | IQ4_NL 100% |
| Attention / SSM | 2.67 | 2.68 | 8.54 | Q8_0 99%, BF16 1% |
| Hyper-connections | 1.20 | 0.64 | 16.02 | BF16 100% |
| Token embedding | 0.63 | 0.64 | 8.50 | Q8_0 100% |
| Output head | 0.63 | 0.64 | 8.50 | Q8_0 100% |
| Routers | 0.23 | 0.06 | 32.00 | F32 100% |
| Shared experts | 0.23 | 0.24 | 8.51 | Q8_0 100% |

К сожалению Strata не работает со слоями внимания и общими экспертами в bf16. Их пришлось квантовать в Q8_0

## The routed experts


| gate / up | down | Layers |
| --- | --- | --- |
| Q4_K | Q5_1 | 0, 3, 4, 7, 11, 15, 16, 17, 18, 22, 26, 29, 32, 33, 37, 38, 39, 42, 44, 46 |
| Q5_K | Q5_1 | 1, 19, 23, 25, 27, 28, 31, 35, 40, 43, 45, 47 |
| IQ4_XS | IQ4_NL | 2, 5, 6, 8, 9, 10, 12, 13, 14, 20, 21, 24, 30, 34, 36, 41 |

Веса важности для грубой карты квантования использовались из 
https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main/tensor-allocation
но, были подняты вверх, так что бы минимальным было 4-х битное квантование. Использовались  типы квантов штатно поддерживаемые Strata.


## Running it: the Strata pack (wibecoded by Qwen Flash Next)

These files are verified to load and serve in **Strata 0.1.40.4**. The engine does not read a GGUF of this model
on its own: it needs a **pack** next to these files. A pack holds no expert weights and no PLE table - those are read
from the GGUFs - so it is about 1.5 GB and builds in under a minute.

```
git clone --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp   # for gguf-py
pip install numpy
python tools/iq_pack.py --gguf <dir>/Qwen3.8-Flash-Next-rcO4-00001-of-00009.gguf --out <dir>/packs/rco4 --compat-bf16
```

Needs Python 3.10+, numpy, and llama.cpp's `gguf-py` importable (a clone at `third_party/llama.cpp`, or point
`STRATA_GGUF_PY` at its directory). All nine shards must sit beside the file you pass: the packer reads every one of
them and stops on a missing shard rather than packing half a model.

What comes out:

| File | Contents |
| --- | --- |
| `index.txt` | 1 079 tensors. 302 rows carry a shape and an offset only - the engine reads those bytes from the GGUF; the rest live in `dense.bin`. |
| `dense.bin` | ~1.54 GB of the small float tensors, stored in the form the engine's kernels read them (BF16 / F32 / F16). |
| `native_experts.txt` | v4, one row per layer: the expert types, the blob size, and which file each of gate / up / down is read from - five layers (9, 17, 24, 38, 45) have roles in two different shards. |
| `tokenizer/` | vocab, merges, token types and the chat template, exported from shard 1. |
| `conversions.json`, `compat-bf16.json` | every tensor whose stored form differs from the engine's, with the method and the largest absolute error. |

Then point the engine at the pack and at shard 1:

```
strata --pack <dir>/packs/rco4 --native <dir>/Qwen3.8-Flash-Next-rcO4-00001-of-00009.gguf \
       --expert-cache auto --prefill auto --spec 4 --max-context 262144 --kv int8
```

`--ple-gguf` needs nothing: the engine picks the shard that holds `per_layer_token_embd.weight` (shard 2 here).

Two things worth knowing:

- **`--compat-bf16` is for exactly one tensor.** `blk.1.ple_conv1d.weight` is BF16 in these files and the PLE conv
  kernel reads F16, so the flag rounds it (largest absolute error 3.0e-08); without it the packer refuses to finish.
  The other 96 conversions it records are exact - F32 tensors whose values are already BF16 - and nothing else in the
  model is touched.
- **Rebuild the pack after changing the GGUFs.** `index.txt` and `native_experts.txt` store byte offsets into them,
  and `native_experts.txt` names the shard files, so keep the pack beside the same nine files. (`compat-bf16.json`
  also records the path the pack was built on; that field is informational and nothing reads it back.)

Measured on this quant: 69.5 GiB of routed experts, which a 96 GB machine keeps fully resident, and the 26.9 GiB PLE
table read from SSD.