---
language:
  - en
  - zh
license: other
license_name: gemma
license_link: https://ai.google.dev/gemma/terms
library_name: transformers
pipeline_tag: text-generation
tags:
  - gemma4
  - gemma-4
  - vllm
  - fp8
  - fp8-dynamic
  - compressed-tensors
  - quantization
  - h200
  - nvidia-h200
  - mixture-of-experts
  - moe
  - inference
  - production-ready
  - largitdata
quantized_by: largitdata-inc
base_model:
  - google/gemma-4-26B-A4B-it
model_type: gemma4
---

# Gemma 4 26B-A4B IT FP8 Dynamic Norouter

**Production-ready offline FP8 checkpoint for vLLM — 47% less VRAM, 80% more concurrency vs BF16.**

We searched for a usable offline FP8 checkpoint of Gemma 4 26B-A4B-it but couldn't find one that worked cleanly with vLLM. So we vibe-coded our own and are sharing it with the community.

This repository hosts an offline FP8 checkpoint derived from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) for vLLM serving. No on-the-fly quantization needed at startup.

Published by [**Largitdata Inc.**](https://www.largitdata.com/)

> **Note:** This is a derived operational checkpoint, not an official Google release. The original model's license terms, safety guidance, and documentation remain authoritative.

📖 [中文說明 / Chinese Version](#中文說明)

---

## Model Details

- **Base model:** [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **Derived format:** offline FP8 checkpoint for vLLM
- **Quantization tool:** [`llmcompressor`](https://github.com/vllm-project/llm-compressor)
- **Quantization method:** `FP8_DYNAMIC`
- **Calibration data:** None required (dynamic quantization)
- **Excluded weights:**
  - `norm`-class 1D tensors — excluded to avoid `expected 2D linear weight` validation errors during quantization
  - `re:.*router\.proj$` — MoE router weights excluded to maintain compatibility with the Gemma4 vLLM loading path
- **Output directory name:** `gemma-4-26B-A4B-it-FP8-DYNAMIC-NOROUTER`
- **Primary serving target:** `vllm/vllm-openai:gemma4`
- **Organization:** [Largitdata Inc.](https://www.largitdata.com/)

## Test Environment

- **GPU:** `NVIDIA H200 NVL` (`143 GB VRAM`)
- **Runtime:** `vllm/vllm-openai:gemma4`
- **KV cache dtype:** `fp8`
- **`max_model_len`:** `32768`
- **`gpu_memory_utilization`:** `0.55`

Observed vLLM startup characteristics:

- model weight loading: `15.76 s`
- model loading total: `16.88 s`
- `torch.compile`: `56.98 s`
- engine init: `102.17 s`
- total time to `/v1/models` ready: about `153 s`

Observed runtime capacity:

- `max_num_batched_tokens = 8192`
- available KV cache memory: `46.37 GiB`
- GPU KV cache size: `405,184 tokens`
- maximum concurrency at `32,768` tokens/request: `38.87x`

## Serving Capacity Comparison

| Metric | FP8 Dynamic Norouter | BF16 Baseline |
|---|---|---|
| Model loading memory | **25.75 GiB** | 48.5 GiB |
| GPU KV cache size | **405,184 tokens** | 225,376 tokens |
| Max concurrency @ 32K tokens/req | **38.87x** | 21.62x |
| VRAM savings | **47% less** | — |
| KV cache gain | **80% more** | — |

## Basic Benchmark

Single-request warm benchmark against the OpenAI-compatible vLLM endpoint:

- prompt tokens: `38`
- completion tokens: `256`
- temperature: `0`

| Metric | FP8 Dynamic Norouter | BF16 Baseline |
|---|---|---|
| Avg end-to-end latency | 1.629 s | **1.536 s** |
| Avg completion throughput | 157.19 tok/s | **166.62 tok/s** |
| Avg total throughput | 180.53 tok/s | **191.36 tok/s** |

These numbers are single-request warm-path measurements, not multi-client throughput tests. In production multi-client scenarios, the FP8 variant's larger KV cache is expected to provide superior aggregate throughput.

**BF16 is ~6% faster on single-request latency, but the FP8 variant uses 47% less VRAM and provides 80% more KV cache capacity.** For production environments serving multiple concurrent users, the FP8 variant offers a better trade-off.

### Accuracy Evaluation — MMLU-Pro

We ran MMLU-Pro (full set, 14 categories) comparing this FP8 checkpoint against its BF16 baseline on the same harness and prompt template.

| Model | MMLU-Pro Accuracy | Wall clock |
|---|---|---|
| 26B BF16 (baseline) | **81.59%** | 65 min |
| 26B FP8 Dynamic Norouter (this) | **81.33%** | 63 min |
| Δ (FP8 − BF16) | **−0.26 pp** | — |

**FP8 quantization cost is essentially zero** — the 0.26 pp gap sits inside benchmark noise. Our 26B BF16 number (81.59%) is ~1.0 pp below Google's reported 82.6%, consistent with harness/prompt differences rather than a quantization defect; both BF16 and FP8 shift together.

For context, we also evaluated the sibling 31B variant under identical conditions:

| Model | MMLU-Pro |
|---|---|
| 26B BF16 | 81.59% |
| 26B FP8 | 81.33% |
| 31B BF16 | 84.60% |
| 31B FP8 Dynamic | 84.55% |

See also: [`LargitData/gemma-4-31b-it-fp8`](https://huggingface.co/LargitData/gemma-4-31b-it-fp8).

## Usage

Example vLLM launch:

```bash
docker run -d \
  --name vllm-gemma4-26b-fp8-norouter \
  --restart unless-stopped \
  --ipc=host \
  --shm-size 16G \
  --gpus all \
  -v /models \
  -p 8001:8000 \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  vllm/vllm-openai:gemma4 \
  --model /models/gemma-4-26B-A4B-it-FP8-DYNAMIC-NOROUTER \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.55 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --host 0.0.0.0 \
  --port 8000
```

## Known Limitations

- Single-request latency is ~6% higher than BF16 due to FP8 dequantization overhead.
- MMLU-Pro measured (see above); MT-Bench and other benchmarks not yet run. Community contributions are welcome.
- Only tested on NVIDIA H200 NVL. Other GPUs (A100, H100) may require adjusting `gpu-memory-utilization`.
- MoE router weights (`router.proj`) and `norm`-class 1D tensors are excluded from quantization for vLLM compatibility. No routing degradation has been observed, but systematic evaluation has not been performed.

## Intended Use

This artifact is intended for:

- Operational vLLM deployment on H200-class hardware
- Reproducible offline FP8 serving experiments
- Environments where startup-time on-the-fly quantization is undesirable
- Production inference with higher concurrency requirements

This artifact is not intended to replace the original base model documentation, safety guidance, or license terms.

## License

This repository contains a derived checkpoint based on [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it). Usage is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Citation

If you use this artifact, please cite both the derived checkpoint and the upstream base model.

```bibtex
@misc{largitdata_gemma4_26b_a4b_it_fp8_dynamic_norouter_2026,
  title        = {Gemma 4 26B-A4B IT FP8 Dynamic Norouter},
  author       = {David Chiu},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/largitdata-inc/gemma-4-26b-a4b-it-fp8-dynamic-norouter}},
  note         = {Derived offline FP8 checkpoint from google/gemma-4-26B-A4B-it for vLLM serving, published by Largitdata Inc. \url{https://www.largitdata.com/}}
}

@misc{google_gemma4_26b_a4b_it,
  title        = {Gemma 4 26B-A4B IT},
  author       = {Google},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/google/gemma-4-26B-A4B-it}}
}
```

## Disclaimer

Users are responsible for verifying license compatibility, downstream serving behavior, numerical quality, and safety characteristics for their own environment.

---

## 中文說明

**給 vLLM 用的離線 FP8 checkpoint — 比 BF16 省 47% VRAM，平行處理能力多 80%。**

我們在網路上找了一輪，沒有找到堪用的 Gemma 4 26B 離線 FP8 版本，索性自己 vibe coding 做了一版，貢獻給社群。

這個 Repo 提供從 [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) 衍生出的離線 `FP8` checkpoint，讓 `vLLM` 可以直接載入服務，不需要在啟動時執行 on-the-fly 量化。

由 [**Largitdata Inc.**](https://www.largitdata.com/) 發佈。

> **注意：** 這是衍生的操作用 checkpoint，並非 Google 官方發佈。原始模型的授權條款、安全指引與文件仍以官方為準。

### 模型細節

- **基底模型：** [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **格式：** 離線 FP8 checkpoint，供 vLLM 使用
- **量化工具：** [`llmcompressor`](https://github.com/vllm-project/llm-compressor)
- **量化方式：** `FP8_DYNAMIC`
- **校準資料：** 不需要（動態量化）
- **排除的權重：**
  - `norm` 類一維 tensor — 避免量化驗證時產生 `expected 2D linear weight` 類錯誤
  - `re:.*router\.proj$` — MoE router 權重，維持與 Gemma4 vLLM 載入路徑的相容性
- **主要部署目標：** `vllm/vllm-openai:gemma4`

### 測試環境

- **GPU：** `NVIDIA H200 NVL`（`143 GB VRAM`）
- **Runtime：** `vllm/vllm-openai:gemma4`
- **KV cache dtype：** `fp8`
- **`max_model_len`：** `32768`
- **`gpu_memory_utilization`：** `0.55`

啟動實測數據：

- 模型權重載入：`15.76 s`
- 模型載入總計：`16.88 s`
- `torch.compile`：`56.98 s`
- 引擎初始化：`102.17 s`
- `/v1/models` 就緒總時間：約 `153 s`

執行期容量：

- `max_num_batched_tokens = 8192`
- 可用 KV cache 記憶體：`46.37 GiB`
- GPU KV cache 大小：`405,184 tokens`
- 最大平行處理量（`32,768` tokens/request）：`38.87x`

### 服務容量比較

| 指標 | FP8 Dynamic Norouter | BF16 原版 |
|---|---|---|
| 模型載入記憶體 | **25.75 GiB** | 48.5 GiB |
| GPU KV cache 大小 | **405,184 tokens** | 225,376 tokens |
| 最大平行處理量 @ 32K tokens/req | **38.87x** | 21.62x |
| VRAM 節省 | **47%** | — |
| KV cache 增加 | **80%** | — |

### 基礎效能測試

單請求暖機測試（OpenAI 相容 vLLM endpoint）：

- prompt tokens：`38`
- completion tokens：`256`
- temperature：`0`

| 指標 | FP8 Dynamic Norouter | BF16 原版 |
|---|---|---|
| 平均端到端延遲 | 1.629 s | **1.536 s** |
| 平均 completion 吞吐量 | 157.19 tok/s | **166.62 tok/s** |
| 平均總吞吐量 | 180.53 tok/s | **191.36 tok/s** |

以上為單請求暖機路徑測量值，非多用戶吞吐量測試。在生產環境多用戶場景下，FP8 版本更大的 KV cache 預期能提供更好的整體吞吐量。

**結論**：`BF16` 單請求略快（約 6%），但 FP8 版本 VRAM 用量減少 47%，可用 KV cache 增加 80%。需要同時服務多用戶的生產環境，FP8 版本更具優勢。

### 精度評估 — MMLU-Pro

使用相同 harness 與 prompt template，對 FP8 checkpoint 與 BF16 原版同時執行 MMLU-Pro（全集，14 類別）：

| 模型 | MMLU-Pro 準確率 | 總執行時間 |
|---|---|---|
| 26B BF16（基準） | **81.59%** | 65 min |
| 26B FP8 Dynamic Norouter（本版本） | **81.33%** | 63 min |
| Δ (FP8 − BF16) | **−0.26 pp** | — |

**FP8 量化的代價基本為零** — 0.26 pp 的差距落在 benchmark 噪音範圍內。我們的 26B BF16 成績（81.59%）比 Google 官方的 82.6% 低約 1.0 pp，這是 harness / prompt 差異造成，非量化問題；BF16 與 FP8 兩者一起偏移。

作為對照，同條件下 31B 版本結果：

| 模型 | MMLU-Pro |
|---|---|
| 26B BF16 | 81.59% |
| 26B FP8 | 81.33% |
| 31B BF16 | 84.60% |
| 31B FP8 Dynamic | 84.55% |

另見：[`LargitData/gemma-4-31b-it-fp8`](https://huggingface.co/LargitData/gemma-4-31b-it-fp8)。

### 使用方式

vLLM 啟動範例：

```bash
docker run -d \
  --name vllm-gemma4-26b-fp8-norouter \
  --restart unless-stopped \
  --ipc=host \
  --shm-size 16G \
  --gpus all \
  -v /models \
  -p 8001:8000 \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  vllm/vllm-openai:gemma4 \
  --model /models/gemma-4-26B-A4B-it-FP8-DYNAMIC-NOROUTER \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.55 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --host 0.0.0.0 \
  --port 8000
```

### 已知限制

- 單請求延遲比 BF16 高約 6%，主因為 FP8 dequantization 的額外開銷
- MMLU-Pro 已完成（見上），MT-Bench 等其他 benchmark 尚未執行，歡迎社群補充
- 僅在 H200 NVL 上實測，其他 GPU（如 A100、H100）可能需要調整 `gpu-memory-utilization`
- MoE router 權重（`router.proj`）與 `norm` 類一維 tensor 被排除在量化範圍外以維持 vLLM 相容性，目前未觀察到分流品質下降，但尚無系統性評估

### 使用場景

此 checkpoint 適用於：

- 在 H200 等級硬體上以 vLLM 進行生產部署
- 可重現的離線 FP8 服務實驗
- 不希望在啟動時執行 on-the-fly 量化的環境
- 需要更高平行處理能力的生產推論場景

此 checkpoint 不取代原始基底模型的文件、安全指引或授權條款。

### 授權

此 Repo 包含基於 [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) 的衍生 checkpoint，使用須遵守 [Gemma 使用條款](https://ai.google.dev/gemma/terms)。

### 免責聲明

使用者需自行驗證授權相容性、下游服務行為、數值品質與安全特性。
