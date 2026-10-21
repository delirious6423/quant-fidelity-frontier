---
library_name: mlx
tags:
  - mlx
  - oq
  - quantized
license: apache-2.0
base_model:
  - google/gemma-4-26B-A4B-it
---

# gemma-4-26B-A4B-it-oQ3-fp16

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.3.6) mixed-precision quantization.
float16 gives ~20% faster prefill on M1/M2 Apple Silicon (native fp16). bfloat16 is safer on M3/M4 and for numerical stability.

## Quantization details

- **Model type**: gemma4
- **Bits**: 3
- **Group size**: 64
- **Format**: MLX safetensors

# Benchmark (on M1 max)


| **Model**                       | **Test Case** | **TTFT (ms)** | **TPOT (ms)** |  **pp TPS** | **tg TPS** | **E2E (s)** | **Throughput** | **Peak Mem** |
| :------------------------------ | :------------ | ------------: | ------------: | ----------: | ---------: | ----------: | -------------: | -----------: |
| **gemma-4-26B-A4B-it-oQ3**      | pp8192/tg128  |       17813.7 |         24.52 | 459.9 tok/s | 41.1 tok/s |      20.927 |    397.6 tok/s |     11.97 GB |
| **gemma-4-26B-A4B-it-oQ3**      | pp32768/tg128 |       91576.3 |         42.93 | 357.8 tok/s | 23.5 tok/s |      97.028 |    339.0 tok/s |     13.14 GB |
| **gemma-4-26B-A4B-it-oQ3-fp16** | pp8192/tg128  |       11592.0 |         21.59 | 706.7 tok/s | 46.7 tok/s |      14.334 |    580.4 tok/s |     11.96 GB |
| **gemma-4-26B-A4B-it-oQ3-fp16** | pp32768/tg128 |       60160.6 |         41.61 | 544.7 tok/s | 24.2 tok/s |      65.445 |    502.7 tok/s |     13.14 GB |

| Model                                                          | Benchmark     | Accuracy | Correct | Total | Time(s) |
| -------------------------------------------------------------- | ------------- | -------- | ------- | ----- | ------- |
| gemma-4-26B-A4B-it-oQ3                                         | ARC_CHALLENGE | 95.0%    | 285     | 300   | 244     |
| gemma-4-26B-A4B-it-oQ3                                         | GSM8K         | 93.3%    | 280     | 300   | 1582.7  |
| gemma-4-26B-A4B-it-oQ3                                         | HELLASWAG     | 86.0%    | 258     | 300   | 332.4   |
| gemma-4-26B-A4B-it-oQ3                                         | JMMLU         | 75.7%    | 227     | 300   | 256.4   |
| gemma-4-26B-A4B-it-oQ3                                         | MMLU          | 79.0%    | 237     | 300   | 612.7   |
| gemma-4-26B-A4B-it-oQ3-fp16                                    | ARC_CHALLENGE | 94.7%    | 284     | 300   | 183.5   |
| gemma-4-26B-A4B-it-oQ3-fp16                                    | GSM8K         | 93.3%    | 280     | 300   | 1154    |
| gemma-4-26B-A4B-it-oQ3-fp16                                    | HELLASWAG     | 86.3%    | 259     | 300   | 254.4   |
| gemma-4-26B-A4B-it-oQ3-fp16                                    | JMMLU         | 77.3%    | 232     | 300   | 210.7   |
| gemma-4-26B-A4B-it-oQ3-fp16                                    | MMLU          | 78.0%    | 234     | 300   | 486.7   |
