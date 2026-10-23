---
base_model: Qwen/Qwen3.8-Flash-Next
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - qwen
  - qwen3.8
  - moe
  - quantized
  - ombro
---
# Qwen3.8 Flash Next — Ombro V15 GGUF

Mixed-precision GGUF quantization of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

## What is Ombro?

Ombro starts from understanding each part of a model and spends precision only where it
actually makes a difference. It is named after my cat.

## Ombro V15

V15 stays well below [UD-Q5_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
in size while keeping far more precision than the Q4 variants.

Attention, the token embedding, the output and the other dense projections stay at
`BF16`, and every `down` projection keeps a higher precision than its `gate` and `up`.
Most expert layers match unsloth's UD-Q4_K_XL. To find the exceptions I compared the
original weights, the BF16 conversion, the current candidate and the existing GGUFs
layer by layer and looked for where the model starts to drift. A few layers steer the
output much more than the rest, and those get more bits.

V13c was already a big step, but V15 went further, so only V15 is released.

What changed along the way:

- **[V9](https://huggingface.co/Wladastic/Qwen3.8-Flash-Next-Ombro-v9-GGUF)** raises nine sensitive expert layers above [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).
- **[V13b](https://huggingface.co/Wladastic/Qwen3.8-Flash-Next-Ombro-v13b-GGUF)** keeps all dense projections at `BF16` and lifts layers 33, 38 and 46 to `Q6_K / Q6_K / Q8_0`.
- **V13c** (not released) does the same for layer 17, so all nine sensitive layers are `Q6_K / Q6_K / Q8_0`.
- **V15** also keeps the shared experts, token embedding and output at `BF16`.

In my testing that last change made the biggest difference (second table below).

Routed experts per layer (`gate` / `up` / `down`):

| Layer | UD-Q4_K_XL | V9 | V13b | V13c / V15 |
| --- | --- | --- | --- | --- |
| all other 39 | Q4_K / Q4_K / Q5_1 | Q4_K / Q4_K / Q5_1 | Q4_K / Q4_K / Q5_1 | Q4_K / Q4_K / Q5_1 |
| 2 | Q5_K / Q5_K / Q8_0 | Q6_K / Q6_K / Q8_0 | Q6_K / Q6_K / Q8_0 | Q6_K / Q6_K / Q8_0 |
| 4, 30, 47 | Q4_K / Q4_K / Q8_0 | Q6_K / Q6_K / Q8_0 | Q6_K / Q6_K / Q8_0 | Q6_K / Q6_K / Q8_0 |
| 17 | Q4_K / Q4_K / Q5_1 | Q6_K / Q6_K / Q5_1 | Q6_K / Q6_K / Q5_1 | **Q6_K / Q6_K / Q8_0** |
| 33 | Q4_K / Q4_K / Q5_1 | IQ4_XS / Q6_K / Q5_1 | **Q6_K / Q6_K / Q8_0** | Q6_K / Q6_K / Q8_0 |
| 35 | Q4_K / Q4_K / Q5_1 | Q6_K / Q6_K / Q8_0 | Q6_K / Q6_K / Q8_0 | Q6_K / Q6_K / Q8_0 |
| 38 | Q4_K / Q4_K / Q5_1 | Q4_K / Q6_K / Q5_1 | **Q6_K / Q6_K / Q8_0** | Q6_K / Q6_K / Q8_0 |
| 46 | Q4_K / Q4_K / Q8_0 | Q4_K / Q5_K / Q8_0 | **Q6_K / Q6_K / Q8_0** | Q6_K / Q6_K / Q8_0 |

| Part | UD-Q4_K_XL | V9 | V13b | V13c | V15 |
| --- | --- | --- | --- | --- | --- |
| Attention, SSM, hyper-connection and PLE projections | Q8_0 | Q8_0 | **BF16** | BF16 | BF16 |
| Shared experts | Q8_0 | Q8_0 | Q8_0 | Q8_0 | **BF16** |
| PLE table | IQ4_NL | IQ4_NL | IQ4_NL | IQ4_NL | IQ4_NL |
| Token embedding / output | Q8_0 | Q8_0 | Q8_0 | Q8_0 | **BF16** |
| **Size** | 111.3 GB | 114.5 GB | 118.9 GB | 119.2 GB | **120.6 GB** |
| **Effective BPW** | 5.03 | 5.18 | 5.38 | 5.39 | **5.45** |

V15 is about 9 GB larger than UD-Q4_K_XL, but it holds its context far better, drifts less
and often finishes thinking sooner. Decode speed drops slightly, from 21 to 20 tokens/s,
yet answers are done much earlier overall.

## Vision

`mmproj-BF16.gguf` is the vision projector (from [unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)). Add it with `--mmproj mmproj-BF16.gguf`.

## MTP heads

Draft heads for `--spec-type draft-mtp`:

| File | Size |
| --- | ---: |
| `mtp-Qwen3.8-Flash-Next-Q2_0exps-Q8_0.gguf` | 2.2 GB |
| `mtp-Qwen3.8-Flash-Next-Q2_0exps-Q4_0embd-Q8_0.gguf` | 1.5 GB |

```bash
llama-server -m Qwen3.8-Flash-Next-Ombro-v15-00001-of-00009.gguf \
  --spec-type draft-mtp \
  --spec-draft-model mtp-Qwen3.8-Flash-Next-Q2_0exps-Q8_0.gguf \
  --spec-draft-n-max 2
```
