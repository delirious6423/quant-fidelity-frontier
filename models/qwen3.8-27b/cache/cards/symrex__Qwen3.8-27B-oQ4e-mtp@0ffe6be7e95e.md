---
library_name: mlx
tags:
- mlx
- oq
- quantized
base_model:
- Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-oQ4e-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.5.7) mixed-precision quantization.

## Quantization details

- **Model type**: qwen3_5
- **Bits**: 4
- **Group size**: 64
- **Format**: MLX safetensors

## Performance Benchmark
- Run on: Apple Mac Studio M4 Max 128GB
- https://omlx.ai/benchmarks/ys8jls4a

```
oMLX - LLM inference, optimized for your Mac
https://github.com/jundot/omlx
Benchmark Model: Qwen3.8-27B-oQ4e-mtp
Engine: Auto
Context: Code (Python)
================================================================================

Single Request Results
--------------------------------------------------------------------------------
Test                                TTFT(ms)    TPOT(ms)        pp TPS        tg TPS      E2E(s)    Throughput    Peak Mem
pp1024/tg128                          4060.8       36.46   252.2 tok/s    27.6 tok/s       8.703   132.4 tok/s    16.41 GB
pp4096/tg128                         16274.7       37.61   251.7 tok/s    26.8 tok/s      21.065   200.5 tok/s    17.97 GB
pp8192/tg128                         33035.2       38.41   248.0 tok/s    26.2 tok/s      37.925   219.4 tok/s    18.88 GB
pp16384/tg128                        68216.7       40.47   240.2 tok/s    24.9 tok/s      73.369   225.1 tok/s    20.71 GB
pp32768/tg128                       144868.1       42.34   226.2 tok/s    23.8 tok/s     150.260   218.9 tok/s    24.35 GB
pp65536/tg128                       326137.5       48.35   200.9 tok/s    20.8 tok/s     332.294   197.6 tok/s    31.68 GB
pp131072/tg128                      811615.2       59.03   161.5 tok/s    17.1 tok/s     819.132   160.2 tok/s    46.79 GB

Continuous Batching
pp1024 / tg128
--------------------------------------------------------------------------------
Batch           tg TPS   Speedup        pp TPS    pp TPS/req    TTFT(ms)      E2E(s)
1x          27.6 tok/s     1.00x   252.2 tok/s   252.2 tok/s      4060.8       8.703
2x          51.6 tok/s     1.87x   195.4 tok/s    97.7 tok/s     10478.6      15.444
4x          76.1 tok/s     2.76x   246.0 tok/s    61.5 tok/s     16488.5      23.377
8x          85.4 tok/s     3.09x   244.7 tok/s    30.6 tok/s     32866.9      45.474
```

## Intelligence Benchmark Comparison

```
Intelligence Benchmark Comparison

               Mode    Sampled         Qwen3.8-27B-oQ4e-mtp
-----------------------------------------------------------
MMLU           Sample  1000/14042                     89.7%
TRUTHFULQA     Full    817                            88.1%
HUMANEVAL      Full    164                            14.0%
LIVECODEBENCH  Sample  300/1055                        4.7%

--- Detail ---

Model: Qwen3.8-27B-oQ4e-mtp
Benchmark         Accuracy   Correct   Total   Time(s)   Think
--------------------------------------------------------------
MMLU                 89.7%       897    1000   18034.6     Yes
TRUTHFULQA           88.1%       720     817   16666.3     Yes
HUMANEVAL            14.0%        23     164    3856.5     Yes
LIVECODEBENCH         4.7%        14     300   37976.1     Yes
```