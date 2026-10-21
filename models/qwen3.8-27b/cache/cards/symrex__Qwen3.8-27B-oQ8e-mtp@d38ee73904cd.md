---
library_name: mlx
tags:
- mlx
- oq
- quantized
base_model:
- Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-oQ8e-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.5.7) mixed-precision quantization.

## Quantization details

- **Model type**: qwen3_5
- **Bits**: 8
- **Group size**: 64
- **Format**: MLX safetensors

## Performance Benchmark
- Run on: Apple Mac Studio M4 Max 128GB
- https://omlx.ai/benchmarks/py3zj5nw

```
oMLX - LLM inference, optimized for your Mac
https://github.com/jundot/omlx
Benchmark Model: Qwen3.8-27B-oQ8e-mtp
Engine: Auto
Context: Code (Python)
================================================================================

Single Request Results
--------------------------------------------------------------------------------
Test                                TTFT(ms)    TPOT(ms)        pp TPS        tg TPS      E2E(s)    Throughput    Peak Mem
pp1024/tg128                          4124.8       61.72   248.3 tok/s    16.3 tok/s      11.976    96.2 tok/s    28.34 GB
pp4096/tg128                         16375.8       62.23   250.1 tok/s    16.2 tok/s      24.293   173.9 tok/s    29.93 GB
pp8192/tg128                         33176.0       63.24   246.9 tok/s    15.9 tok/s      41.221   201.8 tok/s    30.84 GB
pp16384/tg128                        68445.6       64.80   239.4 tok/s    15.6 tok/s      76.689   215.3 tok/s    32.67 GB
pp32768/tg128                       145270.8       67.35   225.6 tok/s    15.0 tok/s     153.838   213.8 tok/s    36.32 GB
pp65536/tg128                       326834.7       73.46   200.5 tok/s    13.7 tok/s     336.176   195.3 tok/s    43.78 GB
pp131072/tg128                      813006.5       84.20   161.2 tok/s    12.0 tok/s     823.715   159.3 tok/s    58.75 GB

Continuous Batching
pp1024 / tg128
--------------------------------------------------------------------------------
Batch           tg TPS   Speedup        pp TPS    pp TPS/req    TTFT(ms)      E2E(s)
1x          16.3 tok/s     1.00x   248.3 tok/s   248.3 tok/s      4124.8      11.976
2x          31.6 tok/s     1.94x   190.3 tok/s    95.2 tok/s     10762.3      18.851
4x          59.9 tok/s     3.67x   242.7 tok/s    60.7 tok/s     16691.6      25.428
8x          75.9 tok/s     4.66x   241.6 tok/s    30.2 tok/s     33214.6      47.401
```

## Intelligence Benchmark Comparison

```
Intelligence Benchmark Comparison

               Mode    Sampled         Qwen3.8-27B-oQ4e-mtp  Qwen3.8-27B-oQ8e-mtp
---------------------------------------------------------------------------------
MMLU           Sample  1000/14042                     89.7%                 89.4%
TRUTHFULQA     Full    817                            88.1%                 89.0%
HUMANEVAL      Full    164                            14.0%                 17.1%
LIVECODEBENCH  Sample  300/1055                        4.7%                  7.7%

--- Detail ---

Model: Qwen3.8-27B-oQ4e-mtp
Benchmark         Accuracy   Correct   Total   Time(s)   Think
--------------------------------------------------------------
MMLU                 89.7%       897    1000   18034.6     Yes
TRUTHFULQA           88.1%       720     817   16666.3     Yes
HUMANEVAL            14.0%        23     164    3856.5     Yes
LIVECODEBENCH         4.7%        14     300   37976.1     Yes

Model: Qwen3.8-27B-oQ8e-mtp
Benchmark         Accuracy   Correct   Total   Time(s)   Think
--------------------------------------------------------------
MMLU                 89.4%       894    1000   20399.6     Yes
TRUTHFULQA           89.0%       727     817   21917.2     Yes
HUMANEVAL            17.1%        28     164      4482     Yes
LIVECODEBENCH         7.7%        23     300   36101.5     Yes
```