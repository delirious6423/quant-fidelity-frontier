---
license: other
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - qwen
  - qwen3.8
  - flash-next
  - llama.cpp
  - mtp
  - k-quant
---

# Qwen3.8-Flash-Next GGUF (vcruz305)

Real K-quant ladder from full HF BF16 weights with **MTP heads in-main** (`blk.48` / `nextn.*`) and a **shared BF16 PLE**.

## Layout

| path | content |
|---|---|
| `Q2_K/` … `Q6_K/` | Backbone shards **00001–00006** (MTP included in tensor set) |
| `PLE-BF16/Qwen3.8-Flash-Next-PLE-BF16.gguf` | Shared BF16 PLE (~95.4 GiB) — use as each quant's `00007` |
| `imatrix/` | Importance matrix used for Q2–Q6 |
| `scripts/link-ple.sh` | `ln`/`cp` PLE → `…-00007-of-00007.gguf` per quant |
| `MTP/` | Optional standalone MTP draft heads (if present) |

## Setup

```bash
bash scripts/link-ple.sh
```

## Approximate backbone sizes (GiB) + shared PLE

| quant | backbone | MTP tensors |
|---|---:|---:|
| Q2_K | ~49.5 | ≥20 |
| Q3_K_M | ~62.5 | ≥20 |
| Q4_K_M | ~80.2 | ≥20 |
| Q5_K_M | ~91.3 | ≥20 |
| Q6_K | ~107.8 | ≥20 |
| PLE BF16 (shared) | ~95.4 | — |

## Run (DGX Spark / unified memory)

Requires **qwen4exp**-capable llama.cpp (PR #27742 class).

```bash
llama-cli \
  -m Q4_K_M/Qwen3.8-Flash-Next-Q4_K_M-00001-of-00007.gguf \
  --load-mode mmap \
  -ngl 99 \
  -ot "per_layer_token_embd.weight=CPU" \
  -c 1024 -n 64 -st --temp 0 \
  -p "The capital of France is"
```

Do **not** mlock the PLE on 128G unified-memory boxes. PLE stays NVMe-backed via mmap.

### MTP speculative decode

Weights include MTP (`nextn_predict_layers` / `blk.48`). Runtime `--spec-type draft-mtp` needs a llama.cpp build with **qwen4exp graph_mtp** (not all mainline builds yet).

## Build notes

- Converter: llama.cpp qwen4exp with MTP export enabled
- Imatrix: AtomicChat-compatible matrix
- PLE left BF16 (not re-quantized); tok emb Q8_0; output Q6_K
