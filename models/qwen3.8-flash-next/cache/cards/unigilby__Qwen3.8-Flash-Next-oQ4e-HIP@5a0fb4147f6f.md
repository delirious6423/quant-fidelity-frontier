---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
library_name: mlx
tags:
- mlx
- qwen4_exp
- tensorfold
- hip
- oq
---

# Qwen3.8-Flash-Next oQ4e (HIP layout)

An oQ4e quantization of `Qwen/Qwen3.8-Flash-Next` (MLX affine, about 107 GB with the n-gram tables and MTP head) laid out so TensorFold's native HIP engine loads it: every fused stack has one bit width and group size. The dense mixers (attention, Gated DeltaNet input projections, hyper-connection mixers, n-gram key/value projections) are 8-bit group 64; the DeltaNet output projections are 5-bit group 64; the routed experts and the shared expert are 4-bit group 32.

It is the higher-quality choice of the two HIP-layout checkpoints; [oQ4e-mix5-HIP](https://huggingface.co/unigilby/Qwen3.8-Flash-Next-oQ4e-mix5-HIP) trades a little quality for about 22% faster decode on code.

## Quality against bf16

KL(bf16 || quant) over the full 248,320-entry vocabulary, teacher-forced, on WikiText-2 raw test (prose) and HumanEval prompt + canonical solution (code), 31,728 scored tokens per slice, main head only (mlx-vlm 0.7.10, mlx 0.32.4, the same logit path for every checkpoint; reference `Qwen/Qwen3.8-Flash-Next` in bf16).

| checkpoint | mean KL prose / code | top-1 agreement prose / code | perplexity prose / code |
|---|---|---|---|
| bf16 | | | 3.064 / 1.086 |
| oQ4e-HIP | 0.064 / 0.013 | 91.95% / 99.00% | 3.122 / 1.091 |
| oQ4e-mix5-HIP | 0.096 / 0.022 | 90.26% / 98.75% | 3.172 / 1.097 |
| `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` (uniform 4-bit) | 0.401 / 0.204 | 78.74% / 93.50% | 4.005 / 1.285 |

Task evaluation through TensorFold's native HIP server, greedy, thinking off, identical items per checkpoint (GSM8K 300 zero-shot exact match; HumanEval 164 pass@1; MMLU-Pro 700, 50 a category, letter within 16 tokens):

| checkpoint | GSM8K | HumanEval | MMLU-Pro |
|---|---|---|---|
| oQ4e-HIP | 94.7% | 93.9% | 69.1% |
| oQ4e-mix5-HIP | 94.3% | 93.3% | 67.6% |
| uniform 4-bit | 93.0% | 93.9% | 55.9% |

## Speed on a Strix Halo iGPU

AMD Ryzen AI MAX+ 395, Radeon 8060S `gfx1151`, 128 GB unified memory, TensorFold native HIP (`zig build native-hip -Dhip-arch=gfx1151`), MTP 3 drafts, `--parallel 1 --context 131072`, `tools/bench_openai.py` decode medians:

| checkpoint | bytes read per decoded token | code / chat tok/s | MTP tokens per verify round (code) |
|---|---|---|---|
| oQ4e-HIP | 5.44 GiB | 42.0 / 36.2 | 2.61 |
| oQ4e-mix5-HIP | 4.45 GiB | 51.3 / 41.4 | 3.05 |
| uniform 4-bit | 4.00 GiB | 50.0 / 44.0 | |

## How it was made

oMLX main (`e81d525`) oQe quantization (imatrix from oMLX's `oqe_code_multilingual` set, 128 samples of 512 tokens) of `Qwen/Qwen3.8-Flash-Next`, with the quantization predicate set so every tensor TensorFold's HIP loader fuses gets one width and group: base 4-bit group 32 (routed experts and the shared expert), the wider tensors at group 64, shared-expert gates 8-bit. The MTP head is stored under `language_model.mtp.*`. The n-gram tables are included.

## License

Derived from `Qwen/Qwen3.8-Flash-Next` under the Qwen community license 1.0; the license file is included unchanged.
