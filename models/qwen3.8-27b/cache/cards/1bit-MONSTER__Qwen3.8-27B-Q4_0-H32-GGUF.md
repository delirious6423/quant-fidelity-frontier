---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
- 1bit-engine
- strix-halo
- rocm
- hadamard
- w4a4
pipeline_tag: text-generation
---

# Qwen3.8-27B Q4_0-H32 (Hadamard-rotated) for the 1bit engine

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) as a Q4_0 GGUF whose matmul weights
are rotated by a 32-point Walsh-Hadamard transform. It is made for the prompt-processing route of
the [1bit engine](https://github.com/1bit-MONSTER/engine) on AMD Strix Halo (Ryzen AI Max+ 395,
Radeon 8060S): 4-bit weights times 4-bit activations on the GPU's matrix units, with the rotation
keeping the 4-bit activations close to the full model.

**It only runs correctly on the 1bit engine's lean ROCm build.** Its weights are rotated, so the
activations must be rotated the same way, and only that build does it. Stock llama.cpp loads the
file but computes garbage. `1bit serve` recognises the file (it carries
`onebit.hadamard_q4_0 = 32`) and picks the right route by itself.

## Run it

```sh
# engine built with -DONEBIT_LEAN=ON -DONEBIT_LEAN_ROCM=ON (docs/lean.md)
1bit serve -m Qwen3.8-27B-Q4_0-H32.gguf
```

## Measured (Strix Halo, 2026-09-28)

KLD against BF16 over wikitext-2 (40 x 512); pp512 from llama-bench (ub 512):

| Qwen3.8-27B | pp512 | PPL | Mean KLD | Same top token |
|---|---|---|---|---|
| Q4_0, exact int8 | ~400 | 6.021 | 0.029 | 91.9% |
| Q4_0, W4A4 without rotation | 461-488 | 6.241 | 0.084 | 87.4% |
| **this file, W4A4** | **509** | **6.096** | **0.055** | **89.3%** |

Through `1bit serve`: a 1,838-token prompt at 440-470 tok/s. Decode on this route is about
11 tok/s; for decode-heavy work run Unsloth's Qwen3.8-27B GGUF on the engine's Vulkan route with
the DFlash2 drafter (`1bit serve --device vulkan --dflash ...`, 45.7 tok/s on code).

## How it was made

```sh
tools/hadamard_q4_0.py Qwen3.8-27B-Q8_0.gguf Qwen3.8-27B-Q4_0-H32.gguf --imatrix imatrix_unsloth.gguf
```

From [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)'s Q8_0 and its
imatrix. Rotated and Q4_0: the attention, FFN and delta-net alpha/beta projections (456 tensors).
`ssm_out` is Q5_K, the output head Q6_K, the MTP projection Q8_0, norms and convolutions F32.
15,182 MiB (4.66 bits per weight).

sha256 `fd61323dd0f78deeb65f7af5a5b2dcd8468af1edb531a9cd37d156bf49e37df4`

## Licence

Apache-2.0, as the base model ([LICENSE](LICENSE)). Qwen3.8 by the Qwen team. Quantization
source and imatrix by Unsloth.
