---
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
language:
  - en
  - ja
base_model:
  - prism-ml/Ternary-Bonsai-2.5-27B-GGUF
  - Qwen/Qwen3.8-27B
  - higa2515/Qwen3.8-27B-QSA-BF16
tags:
  - ternary
  - ptq1_0
  - pq2_0
  - q2_0
  - 2-bit
  - qsa
  - sparse-attention
  - long-context
  - 256k
  - gguf
  - llama-cpp
  - cuda
  - mtp
  - speculative-decoding
  - agentic
  - tool-calling
  - bonsai
  - qwen35
---

# Ternary-Bonsai-2.5-27B-QSA (GGUF)

**English** | [日本語](README.ja.md)

[![arXiv](https://img.shields.io/badge/arXiv-2608.30320-b31b1b?style=flat-square)](https://arxiv.org/abs/2608.30320)
[![Base Model](https://img.shields.io/badge/Base-Ternary--Bonsai--2.5--27B-2b5c8f?style=flat-square)](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF)
[![Variants](https://img.shields.io/badge/GGUF-6%20Variants%20%7C%205.78--7.56%20GiB-4b5563?style=flat-square)](#available-gguf-files)
[![Context](https://img.shields.io/badge/Context-256K%20%7C%2015.85%20GiB%20VRAM-3c8d7d?style=flat-square)](#hardware-and-vram-guide)
[![License](https://img.shields.io/badge/License-Apache--2.0-4b5563?style=flat-square)](LICENSE)

`Ternary-Bonsai-2.5-27B-QSA` adds **Qwen Sparse Attention (QSA)** ([arXiv:2608.30320](https://arxiv.org/abs/2608.30320)) to PrismML's [`Ternary-Bonsai-2.5-27B-GGUF`](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF) family, enabling **256K context in 15.85 GiB VRAM** on a single 24 GB GPU (**2.36x decode / 1.55x prefill** vs dense) and **192K context in 13.23 GiB VRAM** on a single 16 GB GPU (**4.32x decode** at 128K).

All 6 upstream packing formats (`PTQ1_0`, `PQ2_0`, `Q2_0`, with and without the bundled `blk.64` MTP draft head) are provided with the exact upstream ternary weights preserved verbatim and 80 FP32 QSA indexer tensors distilled directly against `Ternary-Bonsai-2.5-27B`.

> [!IMPORTANT]
> **Engine patch required**: Stock `llama.cpp` does not execute QSA GGUFs. Apply [`patches/bonsai-qsa-engine.patch`](patches/bonsai-qsa-engine.patch) to `PrismML-Eng/llama.cpp` (`prism` branch) to enable the zero-mask block-sparse CUDA/CPU QSA attention kernel (`F16` and `Q8_0` Hadamard-rotated KV cache) and the `LLAMA_QSA_DENSE=1` dense A/B switch.

![Single-GPU throughput and peak VRAM across 8K to 256K context](./assets/throughput-vram.png)

## Available GGUF Files

All 6 variants contain the same ternary weights in different `llama.cpp` packing layouts plus the same 200 MiB FP32 QSA indexers (`top_k=2048`, `block_size=4`).

| File | Format | MTP Head | Size | Best For |
|---|---|---|---:|---|
| [`Ternary-Bonsai-2.5-27B-PTQ1_0-QSA-bonsai-v1-mtp.gguf`](Ternary-Bonsai-2.5-27B-PTQ1_0-QSA-bonsai-v1-mtp.gguf) | `PTQ1_0` (1.75 bpw) | Yes | **6.00 GiB** | **Default pick**: smallest size + fast MTP decode up to 128K |
| [`Ternary-Bonsai-2.5-27B-PTQ1_0-QSA-bonsai-v1.gguf`](Ternary-Bonsai-2.5-27B-PTQ1_0-QSA-bonsai-v1.gguf) | `PTQ1_0` (1.75 bpw) | No | **5.78 GiB** | Tightest disk/VRAM footprint (dedicated 256K without MTP) |
| [`Ternary-Bonsai-2.5-27B-PQ2_0-QSA-bonsai-v1-mtp.gguf`](Ternary-Bonsai-2.5-27B-PQ2_0-QSA-bonsai-v1-mtp.gguf) | `PQ2_0` (2.13 bpw) | Yes | **7.17 GiB** | Higher throughput on Prism 2-bit packed kernels + MTP |
| [`Ternary-Bonsai-2.5-27B-PQ2_0-QSA-bonsai-v1.gguf`](Ternary-Bonsai-2.5-27B-PQ2_0-QSA-bonsai-v1.gguf) | `PQ2_0` (2.13 bpw) | No | **6.95 GiB** | Prism 2-bit packed kernels without MTP |
| [`Ternary-Bonsai-2.5-27B-Q2_0-QSA-bonsai-v1-mtp.gguf`](Ternary-Bonsai-2.5-27B-Q2_0-QSA-bonsai-v1-mtp.gguf) | `Q2_0` | Yes | **7.56 GiB** | Standard `Q2_0` packing + MTP |
| [`Ternary-Bonsai-2.5-27B-Q2_0-QSA-bonsai-v1.gguf`](Ternary-Bonsai-2.5-27B-Q2_0-QSA-bonsai-v1.gguf) | `Q2_0` | No | **7.34 GiB** | Standard `Q2_0` packing without MTP |

Standalone indexers ([`bonsai25-qsa-indexers.safetensors`](bonsai25-qsa-indexers.safetensors), 200 MiB) and the zero-dependency grafter ([`scripts/bonsai_qsa_graft.py`](scripts/bonsai_qsa_graft.py)) are also included.

---

## Hardware and VRAM Guide

Measured single-GPU configurations using `PTQ1_0` (`-ngl 999 -fa on -np 1 -b 512 -ub 512`). Because 48 of 64 layers are `O(1)` Gated DeltaNet and `qsa-attn.cu` allocates no `[n_kv, n_tokens]` dense mask buffer, VRAM grows only with the 16 attention layers' KV cache.

| GPU Class | Practical Context | KV Cache | Peak VRAM (`PTQ1_0`) | MTP Recommendation | Measured Decode (`tok/s`) |
|---|---:|---|---:|---|---:|
| 12 GB (RTX 3060 / 4070) | 64K to 96K | `q8_0` | ~8.6 to 9.9 GiB | Enable up to 64K (`--spec-type draft-mtp --spec-draft-n-max 2`) | Scales with memory BW |
| 16 GB (Tesla T4, `sm_75`) | **128K to 192K** | `q8_0` | **10.77 GiB** (128K) / **13.23 GiB** (192K) | Enable for 8K to 64K (`36.6` to `44.3 tok/s`) | **23.18** (128K) / **21.51** (192K) |
| 24 GB (A10G / RTX 3090 / 4090) | **256K** | `q8_0` | **15.85 GiB** (256K) | Enable for 8K to 128K (`50.4` to `86.5 tok/s`); omit at 256K | **44.06** (256K on A10G) |
| 48 GB (L40S / A6000) | **256K** | `q8_0` or `f16` | **16.04 GiB** (`q8_0` at 256K) | Pass `--spec-draft-window` if combining MTP at 256K | **58.09** (256K, `2,224` PP) |

---

## Quick Start

### 1. Download and Verify

```bash
hf download sahenjp/Ternary-Bonsai-2.5-27B-QSA-GGUF \
  Ternary-Bonsai-2.5-27B-PTQ1_0-QSA-bonsai-v1-mtp.gguf \
  patches/bonsai-qsa-engine.patch \
  SHA256SUMS \
  --local-dir ./bonsai-2.5-qsa

(
  cd ./bonsai-2.5-qsa &&
  grep -E ' (Ternary-Bonsai-2\.5-27B-PTQ1_0-QSA-bonsai-v1-mtp\.gguf|patches/bonsai-qsa-engine\.patch)$' SHA256SUMS | sha256sum -c -
)
```

### 2. Build the Patched Engine (CUDA or CPU)

Requires CMake 3.20+, Ninja, Python 3.10+, and CUDA Toolkit 12+ for GPU inference (omit `-DGGML_CUDA=ON` for CPU builds).

```bash
cd ./bonsai-2.5-qsa
git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp.git bonsai-qsa-engine
git -C bonsai-qsa-engine apply "$PWD/patches/bonsai-qsa-engine.patch"

cmake -G Ninja -S bonsai-qsa-engine -B bonsai-qsa-engine/build-cuda \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF

cmake --build bonsai-qsa-engine/build-cuda --target llama-cli llama-server llama-perplexity -j "$(nproc)"
```

### 3. Run Inference

```bash
# Single 24 GB GPU: full 256K context (~15.85 GiB VRAM)
./bonsai-qsa-engine/build-cuda/bin/llama-server \
  -m ./Ternary-Bonsai-2.5-27B-PTQ1_0-QSA-bonsai-v1-mtp.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 262144 -np 1 -b 512 -ub 512 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0.0 --repeat-penalty 1.0

# Single 16 GB GPU: 64K context + MTP speculative decoding (+50% to +68% decode)
./bonsai-qsa-engine/build-cuda/bin/llama-server \
  -m ./Ternary-Bonsai-2.5-27B-PTQ1_0-QSA-bonsai-v1-mtp.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 65536 -np 1 -b 512 -ub 512 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0.0 --repeat-penalty 1.0

# Switch to exact Dense full attention on the same GGUF for A/B checks:
LLAMA_QSA_DENSE=1 ./bonsai-qsa-engine/build-cuda/bin/llama-server ...
```

Quick OpenAI-compatible API test:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Ternary-Bonsai-2.5-27B-QSA",
    "messages": [{"role": "user", "content": "Explain why block-sparse attention speeds up long-context decoding in two sentences."}],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "max_tokens": 256
  }'
```

<details>
<summary>Optional: Rebuild any QSA GGUF locally from PrismML's base file</summary>

```bash
python3 scripts/bonsai_qsa_graft.py \
  Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf \
  bonsai25-qsa-indexers.safetensors \
  Ternary-Bonsai-2.5-27B-PTQ1_0-QSA-bonsai-v1-mtp.gguf
```

</details>

---

## Recommended Sampling Parameters

| Mode | `temperature` | `top_p` | `top_k` | `min_p` | `presence_penalty` | `repetition_penalty` |
|---|---:|---:|---:|---:|---:|---:|
| **Thinking mode** (default) | `1.0` | `0.95` | `20` | **`0.0`** | `0.0` | `1.0` |
| **Non-thinking / Instruct** | `0.7` | `0.80` | `20` | **`0.0`** | `1.5` | `1.0` |
| **Deterministic evaluation** | `0.0` | `1.0` | `1` | `0.0` | `0.0` | `1.0` |

> [!TIP]
> Pass `--min-p 0` explicitly in `llama.cpp`. Its built-in default is `0.05`, which can truncate long reasoning chains.

---

## Benchmarks

All quality comparisons run on identical `Ternary-Bonsai-2.5-27B-PTQ1_0` weights, `Q8_0` KV cache (`-fa on -ctk q8_0 -ctv q8_0`), and greedy decoding (`temperature=0.0, top_k=1`), changing only the QSA indexers or `LLAMA_QSA_DENSE=1`.

![128K recall-grid by task and 16-layer QSA distillation curve](./assets/quality-benchmarks.png)

### 1. Throughput and Peak VRAM (8K to 256K, Single GPU)

| GPU | Configuration | Context (`-c`) | Prompt Tokens | Prefill (`tok/s`) | Decode (`tok/s`) | Peak VRAM (`MiB`) |
|---|---|---:|---:|---:|---:|---:|
| Tesla T4 (16 GB) | Dense (`F16` KV) | 8,192 | 5,834 | 310.54 | 30.05 | 6,667 |
| Tesla T4 (16 GB) | QSA (`F16` KV) | 8,192 | 5,834 | 298.72 | 27.49 | 6,675 |
| Tesla T4 (16 GB) | QSA (`F16` KV) + MTP | 8,192 | 5,834 | 287.82 | **44.31** | 7,351 |
| Tesla T4 (16 GB) | QSA (`Q8_0` KV) | 65,536 | 55,886 | 221.71 | 24.50 | 8,555 |
| Tesla T4 (16 GB) | QSA (`Q8_0` KV) + MTP | 65,536 | 55,886 | 213.26 | **36.63** | 9,457 |
| Tesla T4 (16 GB) | Dense (`Q8_0` KV) | 131,072 | 117,806 | 138.58 | 5.37 | 11,109 |
| Tesla T4 (16 GB) | QSA (`Q8_0` KV) | 131,072 | 117,806 | **214.49** | **23.18** | **11,031** |
| Tesla T4 (16 GB) | QSA (`Q8_0` KV) | 196,608 | 179,898 | 208.82 | 21.51 | 13,551 |
| NVIDIA A10G (24 GB) | Dense (`F16` KV) | 8,192 | 5,831 | 1,109.30 | 58.71 | 6,839 |
| NVIDIA A10G (24 GB) | QSA (`F16` KV) | 8,192 | 5,831 | 1,084.22 | 51.49 | 6,847 |
| NVIDIA A10G (24 GB) | QSA (`F16` KV) + MTP | 8,192 | 5,831 | 1,014.78 | **86.47** | 7,523 |
| NVIDIA A10G (24 GB) | Dense (`Q8_0` KV) | 131,072 | 117,803 | 752.32 | 30.08 | 11,281 |
| NVIDIA A10G (24 GB) | QSA (`Q8_0` KV) | 131,072 | 117,803 | **885.56** | **48.43** | **11,203** |
| NVIDIA A10G (24 GB) | QSA (`Q8_0` KV) + MTP | 131,072 | 117,803 | 833.23 | 50.41 | 12,425 |
| NVIDIA A10G (24 GB) | Dense (`Q8_0` KV) | 262,144 | 239,923 | 543.05 | 18.64 | 16,305 |
| NVIDIA A10G (24 GB) | QSA (`Q8_0` KV) | 262,144 | 239,923 | **841.19** | **44.06** | **16,227** |
| NVIDIA L40S (48 GB) | QSA (`Q8_0` KV) | 262,144 | 239,923 | **2,224.38** | **58.09** | 16,430 |

*At 256K (`262,144`), the `blk.64` MTP head attends densely over KV unless bounded with `--spec-draft-window`, making plain QSA faster and lower-VRAM (`44.06 tok/s`, `16,227 MiB` vs `37.65 tok/s`, `18,089 MiB` with unbounded MTP on A10G).*

### 2. Hard 128K Structured Recall (`recall-grid` 128K, 240 Questions)

8 task types x 10 depths x 3 on PG-19 validation (`seed=20261202`, 15% clutter, 1-character look-alike decoys, `thinking off`, `max_tokens=1024`, NVIDIA L40S).

| Arm | Total | Accuracy | `guid` | `pointer` | `ingredients` | `quantities` | `revision` | `join` | `code` | `binding` | Cut-off |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dense (upper ref)** | **225/240** | **93.75%** | 29/30 | 27/30 | 30/30 | 25/30 | 27/30 | 30/30 | 30/30 | 27/30 | 3 |
| `higa2515-v2` (zero-shot) | 202/240 | 84.17% | 26/30 | 26/30 | 27/30 | 21/30 | 23/30 | 28/30 | 26/30 | 25/30 | 11 |
| **`Bonsai-2.5 distilled v1`** | **201/240** | **83.75%** | 26/30 | 24/30 | 27/30 | 21/30 | 24/30 | 28/30 | 27/30 | 24/30 | 10 |

- Paired exact two-sided McNemar: `bonsai25_v1` vs `higa2515_v2` `6` vs `7` (`p = 1.0`); `bonsai25_v1` vs `Dense` `4` vs `28` (`p = 1.9e-5`).
- Interactive depth x task heatmap: [`benchmarks/recall_grid_128k.html`](benchmarks/recall_grid_128k.html).

<details>
<summary>128K recall-grid error breakdown and depth-bin table</summary>

| Error Class | `Bonsai-2.5 v1` | `higa2515-v2` | `Dense` |
|---|---:|---:|---:|
| Wrong ingredient set (`quantities` / `ingredients`) | 8 | 7 | 1 |
| Value of another record (mostly `pointer`) | 8 | 6 | 2 |
| Value instead of `null` (`revision` / `binding`) | 7 | 7 | 6 |
| 1-character look-alike decoy (`guid` / `code`) | 5 | 6 | 1 |
| Cut off at `max_tokens=1024` | 10 | 11 | 3 |
| `null` instead of value / wrong amounts | 1 | 1 | 2 |

| Depth Bin | `Dense` | `Bonsai-2.5 v1` | `higa2515-v2` |
|---|---:|---:|---:|
| 0-10% | 24/24 (100%) | 23/24 (96%) | 23/24 (96%) |
| 10-20% | 23/24 (96%) | 21/24 (88%) | 21/24 (88%) |
| 20-30% | 22/24 (92%) | 20/24 (83%) | 20/24 (83%) |
| 30-40% | 22/24 (92%) | 18/24 (75%) | 19/24 (79%) |
| 40-50% | 23/24 (96%) | 20/24 (83%) | 21/24 (88%) |
| 50-60% | 22/24 (92%) | 17/24 (71%) | 16/24 (67%) |
| 60-70% | 23/24 (96%) | 21/24 (88%) | 23/24 (96%) |
| 70-80% | 20/24 (83%) | 19/24 (79%) | 19/24 (79%) |
| 80-90% | 22/24 (92%) | 21/24 (88%) | 20/24 (83%) |
| 90-100% | 24/24 (100%) | 21/24 (88%) | 20/24 (83%) |

</details>

### 3. Long-Context Perplexity (`llama-perplexity`, Held-Out Splits)

| Corpus & Window | Sparsity | Dense (`PPL ± SE`) | `Bonsai-2.5 v1` (`Δ%`) | `higa2515-v2` (`Δ%`) | `pentacoxian-255k` (`Δ%`) |
|---|---:|---:|---:|---:|---:|
| English PG-19 (`8K`) | ~50% | `9.5160 ± 0.2106` | **`9.4951`** (`-0.220%`) | `9.4950` (`-0.221%`) | `9.4971` (`-0.199%`) |
| **English PG-19 (`64K`, 32K-64K tail)** | **93.8-96.9%** | `11.7533 ± 0.1891` | **`11.8891` (`+1.155%`)** | `11.9029` (`+1.273%`) | `11.9061` (`+1.300%`) |
| Japanese Wikipedia (`8K`) | ~50% | `12.0398 ± 0.2683` | `12.0319` (`-0.066%`) | `12.0283` (`-0.096%`) | `12.0305` (`-0.077%`) |
| **Japanese Wikipedia (`32K`, 16K-32K tail)** | **87.5-93.8%** | `11.9643 ± 0.2621` | **`12.0530` (`+0.741%`)** | `12.0596` (`+0.797%`) | `12.0676` (`+0.863%`) |
| Python Stdlib Code (`32K`, 16K-32K tail) | 87.5-93.8% | `1.7518 ± 0.0213` | `1.7540` (`+0.126%`) | `1.7539` (`+0.120%`) | `1.7535` (`+0.097%`) |

### 4. Retrieval, Semantic Paraphrase, Multi-Turn & 256K Needle Suite

| Test Suite | Context | `Dense` | `Bonsai-2.5 v1` | `higa2515-v2` | `pentacoxian-255k` |
|---|---:|---:|---:|---:|---:|
| Decoy Needle Recall (1-char decoys) | 64K (8 depths) | 8/8 (100%) | **8/8 (100%)** | 8/8 (100%) | 8/8 (100%) |
| Decoy Needle Recall (1-char decoys) | 128K (5 depths) | 1/1 | **5/5 (100%)** | 5/5 (100%) | 5/5 (100%) |
| Decoy Needle Recall (`240K`, 99.15% sparse) | **256K** (5 depths) | 1/1 | **5/5 (100%)** | 5/5 (100%) | 5/5 (100%) |
| NoLiMa Semantic Paraphrase (0 lexical overlap) | 64K / 128K | 8/8 / 8/8* | **8/8 / 8/8** | 8/8 / 8/8 | 8/8 / 8/8 |
| Multi-Update Early Memory (`v1` initial) | 64K / 128K | 8/8 / 8/8 | **8/8 / 8/8** | 8/8 / 8/8 | 8/8 / 8/8 |
| Multi-Update Latest State (`v3` final) | 64K / 128K | 8/8 / 8/8 | 8/8 / 7/8 | 8/8 / 7/8 | **8/8 / 8/8** |
| Japanese Document QA (Wiki-ja, 8 depths) | 64K | 8/8 (100%) | **8/8 (100%)** | 8/8 (100%) | 8/8 (100%) |
| Japanese 4-Turn Constraint + Turn-1 Memory | 64K (4 turns) | 8/8 (100%) | **8/8 (100%)** | 8/8 (100%) | 8/8 (100%) |

*\*Dense returned `"Fused silica"` instead of `"fused-silica"` on 1 item at 64K and 128K (exact match `7/8`, semantic `8/8`).*

### 5. General Capability Parity (`ctx=8192`)

| Configuration | GSM8K (`test[:30]`) | Python Coding Exec (20) | OpenAI `tool_calls` (12) | Total (62) | Exact Parity vs Dense |
|---|---:|---:|---:|---:|---:|
| `Dense` (`LLAMA_QSA_DENSE=1`) | 25/30 (83.3%) | 19/20 (95.0%) | 12/12 (100%) | 56/62 (90.3%) | baseline |
| **`Bonsai-2.5 distilled v1` (QSA)** | **26/30 (86.7%)** | **19/20 (95.0%)** | **12/12 (100%)** | **57/62 (91.9%)** | **61/62 (98.4%)** |
| **`Bonsai-2.5 distilled v1` (QSA + MTP)** | **25/30 (83.3%)** | **19/20 (95.0%)** | **12/12 (100%)** | **56/62 (90.3%)** | **62/62 (100.0%)** |

### 6. Indexer Distillation Summary (`v2` -> `bonsai25-v1`)

| Held-Out Validation Metric (`seq_len=12,288`) | `higa2515-v2` Zero-Shot | `Bonsai-2.5 distilled v1` | Change |
|---|---:|---:|---:|
| Mean Validation KL (all 16 layers) | `0.41017` | **`0.33094`** | **-19.3%** |
| Peak Drifted Layer KL (Layer 59) | `0.83821` | **`0.49865`** | **-40.5%** |
| Layer 55 / 51 / 63 KL | `0.635 / 0.505 / 0.508` | **`0.404 / 0.373 / 0.404`** | **-36.4% / -26.1% / -20.4%** |
| Single-Layer Answer > Decoy Win Rate | `43.43%` | **`49.06%`** | **+5.62 pp** |
| Top-64 / Top-512 Answer Block Recall | `93.05%` / `99.93%` | **`93.40%` / `99.96%`** | **+0.35 pp / +0.02 pp** |

<details>
<summary>Per-layer validation KL across all 16 QSA layers</summary>

| Layer | `higa2515-v2` | `Bonsai-2.5 v1` | Reduction |
|---:|---:|---:|---:|
| 3 | 0.25779 | 0.22018 | -14.6% |
| 7 | 0.29763 | 0.27563 | -7.4% |
| 11 | 0.32785 | 0.29977 | -8.6% |
| 15 | 0.34878 | 0.31775 | -8.9% |
| 19 | 0.31742 | 0.29727 | -6.3% |
| 23 | 0.31485 | 0.29990 | -4.7% |
| 27 | 0.32383 | 0.29966 | -7.5% |
| 31 | 0.30497 | 0.27806 | -8.8% |
| 35 | 0.49230 | 0.37172 | -24.5% |
| 39 | 0.34296 | 0.30800 | -10.2% |
| 43 | 0.41586 | 0.35054 | -15.7% |
| 47 | 0.33185 | 0.29662 | -10.6% |
| 51 | 0.50544 | 0.37346 | -26.1% |
| 55 | 0.63538 | 0.40404 | -36.4% |
| 59 | 0.83821 | 0.49865 | -40.5% |
| 63 | 0.50756 | 0.40388 | -20.4% |

</details>

---

## Architecture and Hadamard Compatibility

![Ternary-Bonsai-2.5-27B and QSA forward dataflow](./assets/architecture.png)

In `Ternary-Bonsai-2.5-27B`, Sylvester Walsh-Hadamard rotation (`prism.hadamard.axis = input-last-dimension`) is folded into the 401 ternary linear weights while `token_embd.weight` stores the inverse transform, keeping the residual stream and `attn_norm` in `Qwen3.8-27B`'s unrotated primal basis:

1. The 16 QSA indexers (layers `3, 7, ..., 63`, 4-token pooled blocks, `top_k=2048` = 512 blocks + own block) read `attn_norm` directly with no extra Hadamard transform.
2. With `-ctk q8_0 -ctv q8_0` KV rotation (`self_k_rot`, `self_v_rot`), `qsa-attn.cu` attends directly on rotated Q/K blocks and applies the inverse V-rotation to the attention output.
3. Skipping `[n_kv, n_tokens]` dense mask allocation keeps 256K peak VRAM at `16,227 MiB`.

---

## License and Acknowledgements

- **Weights, distilled QSA indexers, scripts, and docs**: Apache-2.0 ([`LICENSE`](LICENSE)).
- **`patches/bonsai-qsa-engine.patch`**: MIT (derived from `PrismML-Eng/llama.cpp`).
- **Notices and checksums**: [`NOTICE.md`](NOTICE.md), [`SHA256SUMS`](SHA256SUMS), and [`release-manifest.json`](release-manifest.json).

Built on the open-source work of:

- **Qwen Team**: Qwen Sparse Attention ([arXiv:2608.30320](https://arxiv.org/abs/2608.30320), *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability*) and [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
- **PrismML**: [`prism-ml/Ternary-Bonsai-2.5-27B-GGUF`](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF) and [`PrismML-Eng/llama.cpp`](https://github.com/PrismML-Eng/llama.cpp).
- **higa2515 / kuraneko1**: [`higa2515/Qwen3.8-27B-QSA-BF16`](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-BF16), [`higa2515/Qwen3.8-27B-QSA-GGUF`](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF), [`qsa-indexer-forge`](https://github.com/kuraneko1/qsa-indexer-forge), and [`Qwen3.8-27B-QSA-llama.cpp`](https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp).
- **pentaCoxian**: [`pentaCoxian/Qwen-QSA-llama.cpp`](https://github.com/pentaCoxian/Qwen-QSA-llama.cpp) (QSA CUDA block-sparse attention kernel reference).
