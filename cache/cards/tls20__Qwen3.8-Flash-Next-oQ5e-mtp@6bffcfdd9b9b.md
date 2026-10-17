---
library_name: mlx
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- mlx
- oq
- oqe
- quantized
- apple-silicon
- multimodal
- vision
- moe
- reasoning
- code
- tool-use
- agent
---

# Qwen3.8-Flash-Next-oQ5e-mtp

This [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.6.4) mixed-precision quantization.

## Quantization details

- **Model type**: qwen4_exp
- **Bits**: 5
- **Group size**: 64
- **Format**: MLX safetensors
- **oQe**: Enabled
- **MTP**: Preserved

## Performance

These results are obtained on a MacBook Pro running M5 Max (40c) with 128Gb RAM in oMLX 0.7.0 on macOS 27.0.1.

**Single Request Results**

Without Lightning MTP:

| Test | TTFT(ms) | TPOT(ms) | ppTPS | tgTPS | E2E(s) | Throughput | PeakMem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pp 1024 / tg 1024 | 657.6 | 14.2 | 1557.2 | 70.5 | 15.2 | 134.7 | 84.7 GB |
| pp 4096 / tg 1024 | 1894.8 | 15.1 | 2161.7 | 66.4 | 17.3 | 295.4 | 84.9 GB |
| pp 8192 / tg 1024 | 4089.2 | 15.1 | 2003.3 | 66.4 | 19.5 | 472.3 | 85.7 GB |
| pp 16384 / tg 1024 | 7806.7 | 15.5 | 2098.7 | 64.7 | 23.6 | 736.2 | 86.7 GB |
| pp 32768 / tg 1024 | 15499.6 | 15.7 | 2114.1 | 63.9 | 31.5 | 1071.9 | 87.4 GB |
| pp 65536 / tg 1024 | 30930.7 | 16.4 | 2118.8 | 61.2 | 47.7 | 1396.6 | 88.8 GB |
| pp 131072 / tg 1024 | 61668.4 | 16.7 | 2125.4 | 60.0 | 78.7 | 1677.7 | 92.0 GB |
| pp 200000 / tg 1024 | 99489.7 | 17.4 | 2010.3 | 57.5 | 117.3 | 1713.7 | 96.5 GB |

With Lightning MTP (adaptive max depth = 3):

| Test | TTFT(ms) | TPOT(ms) | ppTPS | tgTPS | E2E(s) | Throughput | PeakMem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pp 1024 / tg 1024 | 692.9 | 11.9 | 1477.8 | 84.0 | 12.9 | 159.0 | 86.6 GB |
| pp 4096 / tg 1024 | 1995.8 | 16.0 | 2052.3 | 62.4 | 18.4 | 278.2 | 86.7 GB |
| pp 8192 / tg 1024 | 4261.3 | 12.6 | 1922.4 | 79.6 | 17.1 | 538.0 | 87.8 GB |
| pp 16384 / tg 1024 | 8146.1 | 15.2 | 2011.3 | 65.8 | 23.7 | 733.9 | 89.0 GB |
| pp 32768 / tg 1024 | 16013.0 | 16.4 | 2046.3 | 61.2 | 32.8 | 1031.6 | 89.9 GB |
| pp 65536 / tg 1024 | 31855.0 | 15.4 | 2057.3 | 65.0 | 47.6 | 1398.0 | 91.4 GB |
| pp 131072 / tg 1024 | 63111.5 | 18.1 | 2076.8 | 55.4 | 81.6 | 1618.8 | 96.3 GB |
| pp 200000 / tg 1024 | 101087.0 | 17.1 | 1978.5 | 58.7 | 118.6 | 1695.6 | 99.6 GB |

Compared to without MTP, prefill is slightly slower while generation is slightly faster for smaller context. Peak memory is slightly higher.

**Continuous Batching Results**

Without Lightning MTP:

| Batch | tgTPS | ppTPS | avgTTFT(ms) | E2E(s) | Speedup |
| --- | --- | --- | --- | --- | --- |
| 1x baseline | 70.5 | 1557.2 | 657.6 | 15.2 | 1.00x |
| 2x | 87.2 | 1170.8 | 1726.5 | 25.2 | 1.24x |
| 4x | 120.2 | 1157.7 | 2226.7 | 37.6 | 1.70x |
| 8x | 148.3 | 1073.8 | 4114.8 | 62.9 | 2.10x |

Lightning MTP is not recommended for multiple sessions.

## Intelligence

| Test | Results |
| --- | --- | 
| MBPP | 88.6% | 
| HumanEval | 94.5% | 
| MMLU | 92% | 
| JMMLU | 90.6% | 
| TruthfulQA | 95.3% | 
| LiveCodeBench | 59% |


