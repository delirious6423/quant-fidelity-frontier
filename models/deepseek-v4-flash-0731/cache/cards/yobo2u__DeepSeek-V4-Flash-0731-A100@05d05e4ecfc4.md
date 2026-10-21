---
library_name: transformers
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - deepseek
  - deepseek-v4
  - text-generation
  - a100
  - a800
  - sm80
  - sglang
  - dspark
  - mxfp4
  - bfloat16
language:
  - en
  - zh
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731-A100

Converted DeepSeek-V4-Flash-0731 weights for deployment on **NVIDIA A100 / A800 (SM80)** GPUs.

> **For deployment configuration, installation steps, launch scripts and benchmarks, see:**  
> **[GitHub: yobo2u/DeepSeek-V4-Flash-0731-A100](https://github.com/yobo2u/DeepSeek-V4-Flash-0731-A100)**

> **SM90+ users (H100 / H800 / H20) should use the [official upstream weights](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) directly** — this repository is not needed.

## About This Conversion

Base model: [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

The SM80 architecture used by A100/A800 does not support the native FP8 execution path that the original model targets on newer GPUs. The weights in this repository were converted offline so they can run on SM80 GPUs together with a dedicated SGLang 0.5.16 monkeypatch:

- Non-expert weights: converted to **BF16**
- MoE expert weights: kept in **MXFP4**
- Shards: **48 safetensors files**
- Total size: approximately **173 GB** (decimal)
- Context length: up to **1,048,576 tokens**
- Recommended inference framework: **SGLang 0.5.16**
- Recommended hardware: **8× A100 80GB or 8× A800 80GB, TP=8**
- Speculative decoding: **DSpark** supported

This repository provides converted model weights only. The full runtime environment, pinned versions, monkeypatch source, launch arguments and benchmark scripts are maintained in the GitHub configuration repository.

## Deployment

Do not reuse launch commands written for stock vLLM or newer-architecture GPUs. SM80 deployment requires a pinned SGLang version plus the A100 monkeypatch:

- **Full deployment guide:** https://github.com/yobo2u/DeepSeek-V4-Flash-0731-A100
- **Launch script:** https://github.com/yobo2u/DeepSeek-V4-Flash-0731-A100/blob/main/launch.sh
- **Benchmark script:** https://github.com/yobo2u/DeepSeek-V4-Flash-0731-A100/blob/main/benchmark_dspark_full.py
- **Benchmark reports:** https://github.com/yobo2u/DeepSeek-V4-Flash-0731-A100/tree/main/benchmarks
- **A100 monkeypatch:** https://github.com/yaleyoou/deepseek-v4-a100-sglang-v0516
- **SGLang:** 0.5.16, pinned commit `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`

## Download

```bash
hf download yobo2u/DeepSeek-V4-Flash-0731-A100 \
  --local-dir /path/to/models/DeepSeek-V4-Flash-0731-A100
```

For a model of this size, use low download concurrency and resumable transfers, then verify that all 48 shards referenced by `model.safetensors.index.json` are present.

The same weights are also published on ModelScope: [yobo2u/DeepSeek-V4-Flash-0731-A100](https://modelscope.cn/models/yobo2u/DeepSeek-V4-Flash-0731-A100)

## Verified Configuration

| Item | Value |
|---|---|
| GPU | 8× NVIDIA A800-SXM4-80GB (SM80) |
| Tensor parallel | 8 |
| SGLang | 0.5.16 (commit `fdebc938`) |
| Dtype | BF16 + MXFP4 experts |
| Serving port | 8082 |
| Context length | 1,048,576 |
| `--mem-fraction-static` | **0.85** |
| `--chunked-prefill-size` | **32,768** |
| `--max-running-requests` | 16 |
| Speculative decoding | DSpark |

## Measured Performance

Measured on the verified configuration above (config C), covering 1K / 4K / 16K / 64K / 128K context × concurrency 1 / 8 / 16 × 2 full repeats = **30 groups, 900 requests, 0 errors and 0 timeouts**:

| Metric | Value |
|---|---|
| Single-stream decode (concurrency 1, 1000/TPOT) | ~**217 tok/s** |
| Single-stream aggregate throughput (concurrency 1) | ~**205 tok/s** (166–222) |
| Aggregate throughput (concurrency 16) | ~**1,232 tok/s** (peak 1,334) |
| DSpark accept rate | ~**60%** (0.40–0.86, increases with context) |
| DSpark accept length | ~**4.01** (3.01–5.30) |
| TTFT (concurrency 1) | ~**296 ms** |
| Peak VRAM | ~**51.9 GB/GPU** |

> Accept rate and accept length rise substantially with context length: about 0.40 / 3.0 at 1K, and about 0.82 / 5.1 at 128K.
> The table reports means across all 30 groups; per-group values and raw data are in the JSON files under the GitHub `benchmarks` directory.

### Tuning Findings

A three-way factor decomposition over `mem-fraction-static` and `chunked-prefill-size` (30 groups / 900 requests each):

| Config | mem-fraction | chunked-prefill | Mean throughput | Peak VRAM |
|---|---|---|---|---|
| A | 0.85 | 16,384 | 746.2 tok/s | 48.58 GiB |
| B | 0.90 | 32,768 | 759.8 tok/s | 53.80 GiB |
| **C (recommended)** | **0.85** | **32,768** | **766.4 tok/s** | 51.92 GiB |

- **Keep `mem-fraction-static` at `0.85`.** Raising it to `0.90` squeezes the KV cache headroom, dropping the 128K accept rate from 0.80 to 0.69 and costing roughly 11% throughput at long context.
- Raising `chunked-prefill-size` from 16,384 to 32,768 helps at short-to-medium context and higher concurrency, and is **not** the cause of the 128K regression.
- Full factor-decomposition report (HTML and PDF) is available in the [GitHub benchmarks directory](https://github.com/yobo2u/DeepSeek-V4-Flash-0731-A100/tree/main/benchmarks).

Performance varies with input length, concurrency, driver, CUDA version, interconnect topology and the specific A100/A800 SKU. The GitHub repository is the source of truth for methodology and results.

## Limitations and Notes

1. These are community weights converted for SM80 deployment — **not an official DeepSeek model release**.
2. Running them depends on a dedicated monkeypatch. Downloading the weights alone does not make them loadable by Transformers, vLLM, or unmodified SGLang.
3. `--quantization fp8` is part of how the current SGLang monkeypatch path is wired up; it does not mean every weight is stored in native FP8.
4. Follow the version matrix and launch arguments in the GitHub documentation to avoid SGLang API and kernel incompatibilities.
5. For model capabilities, recommended sampling parameters, chat template format and usage policy, refer to the original model repository.

## Credits

- Original model and architecture: [DeepSeek-AI](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- SM80 monkeypatch: [yaleyoou/deepseek-v4-a100-sglang-v0516](https://github.com/yaleyoou/deepseek-v4-a100-sglang-v0516)
- Original patch: [Qeeweew/deepseek-v4-a100-sglang](https://github.com/Qeeweew/deepseek-v4-a100-sglang)
- Inference framework: [SGLang](https://github.com/sgl-project/sglang)
- Deployment configuration: [yobo2u/DeepSeek-V4-Flash-0731-A100](https://github.com/yobo2u/DeepSeek-V4-Flash-0731-A100)

## License

Released under the **MIT License**, following the original repository. Users must also comply with the upstream model repository's license terms and applicable laws and regulations.
