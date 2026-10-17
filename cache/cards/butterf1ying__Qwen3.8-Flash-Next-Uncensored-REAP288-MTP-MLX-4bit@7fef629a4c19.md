---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
library_name: mlx
language:
- en
- zh
tags:
- mlx
- mlx-serve
- apple-silicon
- qwen
- qwen4_exp
- moe
- quantized
- 4-bit
- mtp
- reap
- uncensored
- vision-language
pipeline_tag: image-text-to-text
---

# Qwen3.8-Flash-Next-Uncensored-REAP288-MTP-MLX-4bit

**A practical, experimental Qwen3.8-Flash-Next build for 64 GB Apple Silicon systems.**

This checkpoint combines **REAP-288 expert pruning**, **MLX 4-bit quantization**, an **SSD-backed N-gram table**, and the model's **native MTP head** in a package tested with [`mlx-serve`](https://github.com/ddalcu/mlx-serve).

It is intended for local inference, coding-agent experiments, long-context research, and community evaluation of native MTP on Apple Silicon.

> [!IMPORTANT]
> This is an unofficial community derivative, not an official Qwen release. It is labeled “Uncensored” because it was assembled for reduced-refusal behavior; the degree of refusal reduction has not been formally benchmarked on this exact checkpoint.

## Highlights

- **Runs on a 64 GB Mac:** successfully loaded and generated text on an Apple M1 Max with 64 GB unified memory.
- **REAP-288 MoE:** keeps 288 routed experts per layer while preserving the trained top-10 routing width.
- **SSD-backed N-gram weights:** the approximately 29.8 GiB N-gram/PLE table is memory-mapped instead of being kept fully resident.
- **Native MTP included:** the one-layer Qwen MTP head is integrated into the checkpoint and its speculative path was confirmed active in `mlx-serve`.
- **Compact distribution:** approximately 71.67 GiB for the complete model package.
- **Vision weights retained:** multimodal weights are included, although vision behavior has not yet been validated on this build.
- **OpenAI-compatible serving:** works through a local `/v1/chat/completions` endpoint with the tested runtime.

## Why this build exists

Qwen3.8-Flash-Next combines a large sparse MoE trunk with a very large learned N-gram embedding table. Straight conversions can be difficult to run on memory-limited Macs.

This build takes a different route:

1. **REAP pruning** reduces each routed MoE layer from 512 experts to 288.
2. **Affine 4-bit MLX weights** reduce the resident model footprint.
3. The large **N-gram table is stored as a contiguous memory-mapped file**, allowing `mlx-serve` to fetch required rows from SSD on demand.
4. The model's **native MTP head is preserved** for optional speculative decoding.

The result is an experimental Flash-Next-class checkpoint that can run on hardware where a full conversion may not fit comfortably. Quality and speed will vary by workload, and independent evaluation is encouraged.

## Technical summary

| Item | Value |
|---|---|
| Architecture | `qwen4_exp` |
| Language layers | 48 |
| Hidden size | 2560 |
| Attention layout | Hybrid linear/full attention |
| Routed experts | 288 per layer |
| Active experts | Top-10 routed + shared expert |
| Main quantization | MLX affine 4-bit, group size 64 |
| N-gram quantization | 4-bit, group size 32 |
| N-gram table | `ngram_table.bin`, approximately 29.8 GiB, memory-mapped |
| MTP | Native one-layer head included |
| Configured context | 262,144 tokens |
| Vision | Weights included; not yet validated here |
| Total download | Approximately 71.67 GiB |
| Tested runtime | `mlx-serve 26.8.11` |

## Tested environment

The release was smoke-tested in the following environment:

- Apple M1 Max
- 64 GB unified memory
- `mlx-serve 26.8.11`
- Text-only mode
- Runtime-reported peak memory of approximately 40.3 GB during a short local test
- Native MTP path confirmed active through the runtime's `mode=mtp` statistics

These observations confirm that the package loads and that native MTP executes. They are **not** presented as a general performance benchmark.

## Download

Install the Hugging Face CLI if needed, then download the repository:

```bash
hf download butterf1ying/Qwen3.8-Flash-Next-Uncensored-REAP288-MTP-MLX-4bit \
  --local-dir ~/Models/Qwen3.8-Flash-Next-Uncensored-REAP288-MTP-MLX-4bit
```

Make sure the destination drive has enough free space for the approximately 71.67 GiB download and temporary filesystem overhead.

## Install the tested runtime

```bash
brew tap ddalcu/mlx-serve https://github.com/ddalcu/mlx-serve
brew install mlx-serve
```

This exact package layout targets `mlx-serve`. Older or architecture-unaware versions of `mlx-lm`, `mlx-vlm`, and Transformers may reject `model_type: qwen4_exp` or may not support the external N-gram table.

## Quick start

Start a local-only OpenAI-compatible server with native MTP enabled:

```bash
mlx-serve \
  --model ~/Models/Qwen3.8-Flash-Next-Uncensored-REAP288-MTP-MLX-4bit \
  --serve \
  --host 127.0.0.1 \
  --port 11234 \
  --no-vision \
  --mtp
```

The API will be available at:

```text
http://127.0.0.1:11234/v1
```

Example request:

```bash
curl http://127.0.0.1:11234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-Next-Uncensored-REAP288-MTP-MLX-4bit",
    "messages": [
      {"role": "user", "content": "Write a small Python LRU cache with tests."}
    ],
    "temperature": 0,
    "max_tokens": 512,
    "enable_mtp": true
  }'
```

### Runtime options

- `--host 127.0.0.1` keeps the server accessible only from the local machine.
- `--no-vision` is recommended until this build's vision path receives more testing.
- `--mtp` explicitly enables the native MTP path for this MoE checkpoint.
- `--mtp-depth 1`, `2`, or `3` can be tested manually; the best value depends on the workload and hardware.
- Send `"enable_mtp": false` per request, or use `--no-mtp`, to establish a serial-decoding baseline.

## MTP: what to expect

MTP performance is workload-dependent. Code completion and repetitive structured output may accept more draft tokens than short prose, while some prompts may see little benefit or even a slowdown.

A fair comparison should:

1. Load the model once.
2. Run at least one warm-up request.
3. Compare the same prompt with MTP enabled and disabled.
4. Generate enough tokens to reduce short-request noise.
5. Repeat each case at least three times and report the median.
6. Record context length, MTP depth, runtime version, hardware, memory size, and temperature.

The first request can be slower because kernels, model pages, and SSD-backed N-gram pages are still cold. Please do not treat a single short first response as representative throughput.

MTP drafts are accepted only after verification by the target model. The purpose is to improve decoding throughput when acceptance and verification cost are favorable—not to change the model's intended output behavior.

## Community benchmarks welcome

There is intentionally no headline tokens-per-second claim yet. If you test this checkpoint, please open a Hugging Face discussion and include:

```text
Hardware:
Unified memory:
macOS version:
mlx-serve version:
Text or vision:
Prompt/context tokens:
Generated tokens:
Temperature:
MTP enabled:
MTP depth:
Warm-up performed:
Median tok/s over 3+ runs:
MTP acceptance statistics:
Peak memory:
```

Results from M1, M2, M3, M4, and M5 Max/Ultra systems are especially useful. Reproducible reports may be added to this model card with attribution.

## N-gram table

`ngram_table.bin` is a static learned N-gram/PLE weight table. It is **not** chat history, an inference log, or a collection of user prompts.

The file remains contiguous because the tested runtime memory-maps it for direct random row access. Splitting it manually would break compatibility with the current configuration and loader.

## Provenance and acknowledgements

This build relies on work from the following projects and community contributors:

- [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — official base architecture and weights
- [`Sawfwair/Qwen3.8-Flash-Next-MLX-4bit`](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-4bit) — MLX 4-bit conversion work used by the REAP build
- [`sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit) — REAP-288 expert selection and pruning work
- [`ddalcu/mlx-serve`](https://github.com/ddalcu/mlx-serve) — inference, external N-gram mapping, and native MTP runtime used for this release

Thank you to Qwen and all community contributors who made these experiments possible.

## Limitations and responsible use

- Reduced-refusal behavior does not make the model more accurate, safe, or suitable for every task.
- The model may generate harmful, illegal, offensive, biased, or factually incorrect content.
- REAP pruning may affect rare knowledge, rare names, and workloads outside the pruning calibration distribution.
- Vision weights are present, but vision quality and memory usage have not been validated on this release.
- Do not expose the server publicly without authentication, rate limiting, moderation, and appropriate legal safeguards.
- Users are responsible for evaluating the model for their own hardware, domain, and risk requirements.

## Privacy

The published repository contains model artifacts and documentation only. It does not include personal conversations, inference logs, private datasets, account tokens, or local user files.

## License

This repository follows the **Qwen Community License 1.0** inherited from the official base model. See [`LICENSE`](LICENSE) for the complete terms.

Review the license before redistribution, hosted deployment, or commercial use. Downloading the checkpoint does not remove upstream license obligations or requirements under applicable law.

---

## 中文说明

这是一个为 Apple Silicon 本地运行设计的实验性 Qwen3.8-Flash-Next 社区衍生版本，主要特点是：

- 每层保留 288 个路由专家，保持 Top-10 激活；
- 主体采用 MLX 4bit 量化；
- 约 29.8 GiB 的 N-gram 表通过 SSD 内存映射按需读取；
- 模型内置原生 MTP 头，并已确认 MTP 推测解码链路可以运行；
- 已在 M1 Max 64 GB 上完成文本加载和生成测试；
- 完整下载约 71.67 GiB；
- 当前没有发布代表性的速度结论，欢迎不同设备用户预热后进行多轮对比实测。

本模型降低了拒答行为，不代表更安全或更准确。请在受控环境中使用，并遵守 Qwen Community License 1.0 及当地法律法规。
