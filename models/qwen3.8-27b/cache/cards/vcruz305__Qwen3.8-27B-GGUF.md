---
language:
  - en
  - zh
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: vcruz305
tags:
  - gguf
  - qwen
  - qwen3.8
  - llama.cpp
  - imatrix
---

# Qwen3.8-27B GGUF

Standalone llama.cpp K-quants of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (official **BF16**).

This pack is for **small cards**. 12GB and 24GB come first.

## Why there is no MTP in these files

Official BF16 includes a 1-layer MTP / nextn head (`mtp.*` in shard 18). I left it out on purpose.

MTP is for speculative decode. It does **not** make the 27B smarter. It adds ~15 tensors (`blk.64` / `nextn_*`) and roughly 0.5–0.8 GB. On a 12GB 3060 the headline file is the **9.98 GB Q2_K**; that extra head steals the KV you barely have. Almost nobody on 12GB will turn on `--spec-type draft-mtp`.

These GGUFs are the **64-layer language trunk** (`qwen35`, blocks 0–63). Need vision? Use a separate `mmproj`. Need MTP/spec decode? Other packs keep that head.

## Chat template

Official 3.8 jinja wraps every assistant turn in `<think>…</think>` even when reasoning is empty, then opens another `<think>` on generate. That truncates multi-turn agents.

**These GGUFs have that template replaced.** Re-download so `--jinja` uses the fixed one. `chat_template.jinja` is also in the repo if you still have yesterday's bytes.

```bash
llama-server -m Qwen3.8-27B-Q2_K.gguf --jinja --reasoning-format deepseek
```

## Files

One file per quant. **Q2_K and Q3_K_M are imatrix** (300-chunk wiki / code / zh / math on the official-BF16 trunk). Q4 and up are plain K-quants — imatrix barely moves them.

| File | Quant | Size | Card |
|---|---|---:|---|
| `Qwen3.8-27B-Q2_K.gguf` | 2-bit **imatrix** | 9.98 GiB | 12GB start |
| `Qwen3.8-27B-Q3_K_M.gguf` | 3-bit **imatrix** | 12.39 GiB | 16GB |
| `Qwen3.8-27B-Q4_K_M.gguf` | 4-bit | 15.41 GiB | 24GB start |
| `Qwen3.8-27B-Q5_K_M.gguf` | 5-bit | 17.91 GiB | 24GB comfortable |
| `Qwen3.8-27B-Q6_K.gguf` | 6-bit | 20.57 GiB | 24GB tight / largest full-GPU on RTX 6000 |
| `Qwen3.8-27B-Q8_0.gguf` | 8-bit | 26.63 GiB | 32GB+ |

`qwen38-27b.imatrix` is the calib artifact if you want to requant. You do **not** pass it to `llama-server`.

## How to run

12GB:

```bash
hf download vcruz305/Qwen3.8-27B-GGUF \
  --local-dir Qwen3.8-27B-GGUF \
  --include "Qwen3.8-27B-Q2_K.gguf"

llama-server \
  -m Qwen3.8-27B-GGUF/Qwen3.8-27B-Q2_K.gguf \
  -a qwen38-27b \
  --host 127.0.0.1 --port 8085 \
  -ngl 99 -c 32768 -np 1 --jinja --reasoning-format deepseek
```

24GB (Q6_K measured on an RTX 6000 at native 262K with `-ctk q4_0 -ctv q4_0 -fa on`):

```bash
llama-server \
  -m Qwen3.8-27B-Q6_K.gguf \
  -a qwen38-27b \
  --host 127.0.0.1 --port 8085 \
  -ngl 99 -fa on -c 262144 \
  -ctk q4_0 -ctv q4_0 -np 1 --jinja
```

Needs llama.cpp new enough for `qwen35` (Gated DeltaNet hybrid).

## Source

- Base: https://huggingface.co/Qwen/Qwen3.8-27B
- Convert: `convert_hf_to_gguf.py --outtype bf16 --no-mtp` → `llama-quantize` (`--imatrix` on Q2/Q3)
- License follows the official Qwen card (Apache-2.0)

## Model Card Authors / Contact

Victor Cruz ([vcruz305](https://huggingface.co/vcruz305)).
