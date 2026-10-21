---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
language:
- en
- zh
library_name: llama.cpp
pipeline_tag: text-generation
inference: false
tags:
- gguf
- llama-cpp
- deepseek-v4
- quantized
- imatrix
- vulkan
- amd
- strix-halo
- speculative-decoding
- custom-kernels
- conversational
---

# DeepSeek-V4-Flash-IQ2XXS-TM64-Q4KAttention-v1

DeepSeek V4 Flash, served locally with a customized `llama.cpp` fork. The distinctive part of this package is the **TM64** layout.

- **Build / runtime fork:** https://github.com/erayyap/llama.cpp
- **Validated binary release:** https://github.com/erayyap/llama.cpp/releases/tag/strix-halo-tm64-b54f16e1-3cacee17
- **Runtime source commit:** `4fc4f230f6cc7735dc05cd403a8eb73ccd64813c`

> **Credit:** the base quants in this repo were **not** built by this uploader — they were downloaded from the upstream repos below. The uploader's contribution is the **TM64 relayout** of the IQ2_XXS expert tensors and the matching Vulkan kernels. The deployed main artifact additionally carries the locally requantized Q4_K attention tier (see deployment records).

| File | Role | Source repo | Size |
|---|---|---|---|
| `DeepSeek-V4-Flash-IQ2XXS-TM64-Q4KAttention-v1.gguf` | Main model (TM64-relayouted IQ2_XXS experts, Q4_K attention) | `antirez/deepseek-v4-gguf` | ~79 GB |
| `dspark-DeepSeek-V4-Flash-0731-MXFP4-Q4_K_M-dense.gguf` | DSpark draft (selective-Q4 dense), speculative decoding | `Ralii/DeepSeek-V4-Flash-0731-DSpark-Selective-Q4_K_M-GGUF` | ~10 GB |

### Upstream source repos

- **Main model:** https://huggingface.co/antirez/deepseek-v4-gguf
  - File: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` — an IQ2_XXS/Q2_K routed-expert, Q8 attention/shared/output quant built by **antirez**.
  - The uploaded file is that quant with the IQ2_XXS expert tensors re-laid-out into the TM64 tile-major layout.
- **Draft:** https://huggingface.co/Ralii/DeepSeek-V4-Flash-0731-DSpark-Selective-Q4_K_M-GGUF
  - File: `dspark-DeepSeek-V4-Flash-0731-MXFP4-Q4_K_M-dense.gguf` — routed experts kept native MXFP4, eligible dense tensors selective-Q4_K_M, by **Ralii**.

This repo holds the two GGUF files used together for a full speculative-decode setup:

| File | Role | Size |
|---|---|---|
| `DeepSeek-V4-Flash-IQ2XXS-TM64-Q4KAttention-v1.gguf` | Main model (TM64-relayouted IQ2_XXS experts, Q4_K attention) | ~79 GB |
| `dspark-DeepSeek-V4-Flash-0731-MXFP4-Q4_K_M-dense.gguf` | DSpark draft (selective-Q4 dense), used for speculative decoding | ~10 GB |

## Download and run

Install the Hugging Face CLI and download both files:

```bash
hf download erayyapagci/DeepSeek-V4-Flash-IQ2XXS-TM64-Q4KAttention-v1 \
  DeepSeek-V4-Flash-IQ2XXS-TM64-Q4KAttention-v1.gguf \
  dspark-DeepSeek-V4-Flash-0731-MXFP4-Q4_K_M-dense.gguf \
  --local-dir "$HOME/models/deepseek-v4-vulkan"
```

Download and extract the validated Strix Halo runtime from the [GitHub release](https://github.com/erayyap/llama.cpp/releases/tag/strix-halo-tm64-b54f16e1-3cacee17), then launch it with the packaged tuned configuration:

```bash
MODEL_PATH="$HOME/models/deepseek-v4-vulkan/DeepSeek-V4-Flash-IQ2XXS-TM64-Q4KAttention-v1.gguf" \
DRAFT_PATH="$HOME/models/deepseek-v4-vulkan/dspark-DeepSeek-V4-Flash-0731-MXFP4-Q4_K_M-dense.gguf" \
./bin/run-server.sh
```

The launcher defaults to F16 KV, F16 lightning-indexer cache, NP1, 409,600 context tokens per slot, batch/microbatch 3072, adaptive DSpark width up to five, and the validated Strix Halo Vulkan kernel controls. Use `NP=2` or `NP=3` for multiple slots; each NP count receives a separate state path.

## What TM64 is

TM64 ("Tile-major BM64") is an **alternate GGUF tensor layout** for the IQ2_XXS expert tensors, plus the Vulkan kernels that consume it. It is a private serialized type, `IQ2_XXS_TM64`, so standard IQ2 kernels cannot read the alternate layout.

- The 86 IQ2_XXS expert gate/up tensors are re-laid-out offline from
  `[expert][M row][K/256 block]` → `[expert][M/64 tile][K/256 block][row within tile]`.
- Complete 66-byte quantization blocks are permuted **without changing any bits** — only addressing changes.
- This gives Vulkan **direct BM64 (block-major, 64-row) addressing** for wide F16-B MMID and widths 2–5 vector MMID, instead of forcing production-width batches through the width-512 BN16 specialization.
- Adaptive small/medium tiles keep production-width batches out of the width-512 BN16 path.
- Kernel repair (accepted patch): retain `BM64/BN16` at width 512; `BM64/BN32` at width 1024; `BM128/BN64` at widths 2286/3072.

## Correctness and speed

- Exact standard↔TM64 Vulkan output hashes at widths 2, 3, 4, 5, and 512; the standard path passed CPU-reference comparison at all five widths, so the TM64 result is transitively exact.
- All 86 inverse permutations reproduce their source tensor SHA-256; all non-target bytes are identical.
- Cooled A/B/B/A kernel speedups: **1.24949×** (512), **1.19271×** (1024), **1.10918×** (2286), **1.07336×** (3072).
- Whole-model median timing (resident model, A/B/B/A/B/A/A/B): prompt throughput **+14.86%** (76.0 → 87.3 tok/s), decode throughput **+2.50%** (19.97 → 20.47 tok/s). Output hashes and draft acceptance identical across all phases.

## Platform caveat

TM64 is specific to **AMD/RADV/RDNA3 (gfx1151 / device `0x1586`)** and the exact `4096x2048x256`, top-6 expert tensors. It is an opt-in layout winner; on other devices standard IQ2_XXS kernels apply (or the file must be re-converted).

## Quantization notes (beyond TM64)

- Routed experts: IQ2_XXS / Q2_K
- Attention projection tier: Q4_K
- Dense/draft: MXFP4 + Q4_K_M (selective)
- Main model context: up to 409,600 tokens per slot (multi-slot variants up to 1,228,800)
- Draft spec: `--spec-type draft-dspark --spec-draft-n-max 5`

## License

Original model license applies. See upstream model card.
