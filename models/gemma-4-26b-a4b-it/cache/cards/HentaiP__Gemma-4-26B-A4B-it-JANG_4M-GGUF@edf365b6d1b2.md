---
license: gemma
language:
- en
tags:
- moe
- gguf
- gemma-4
- llama-cpp
- text-generation
base_model: google/gemma-4-26B-A4B-it
---

# Gemma-4-26B-A4B-it-JANG_4M-GGUF

Standard text-only GGUF collection of the **Gemma-4-26B-A4B-it** model. 

This repository is **forked from** [OsaurusAI/Gemma-4-26B-A4B-it-JANG_4M](https://huggingface.co/OsaurusAI/Gemma-4-26B-A4B-it-JANG_4M) and converted to GGUF format.

It provides correct, aligned GGUF files for local high-performance inference. It resolves critical architectural bugs, multi-modal index misalignment, and memory-mapped file bound issues commonly found when compiling Gemma-4 models with `llama.cpp`.

## ✦ Technical Highlights

### 1. The 15.09 MB Zero-Padding Alignment Fix
*   **The Issue**: During standard `llama.cpp` text-only compilation of multi-modal architectures (like Gemma-4), the index headers register the existence of several multi-modal vision projection tensors (e.g., `vision_projection.weight` / `vision_projection_bf16.weight`). However, the compiler omits the actual data payload of these tensors from the GGUF data segment. This discrepancy leaves the physical file **exactly 15,819,552 bytes (approx. 15.09 MB)** shorter than what is claimed by the index header, causing `llama_model_load` / memory-mapped (`mmap`) loaders to crash with:
    `tensor 'blk.xx.ffn_gate_exps.weight' data is not within the file bounds`
*   **The Fix**: A bit-level physical offset compensation algorithm is applied. All GGUF files in this repository have been precisely padded at the tail with exactly `15,819,552` bytes of standard `\x00` null sectors. This aligns the physical file size with the GGUF header specifications, enabling 100% stable loads.

### 2. Consolidated MoE Expert Layout
Mapped custom 8-bit/4-bit adaptive expert namespaces directly into standardized `experts.gate_proj` and `experts.up_proj` blocks, maintaining consolidated 3D expert layers for natural `convert_hf_to_gguf.py` adaptive handling.

### 3. Root Level `text_config` Flattening
Nested properties under `text_config` (including expert allocations, dimensions, and layer counts) were flattened and merged directly into the root level of `config.json` before conversion.

---

## ✦ File Spectrum & Sizes

All GGUF files are fully synced and zero-padded to match GGUF index configurations exactly:

| File Name | Precision / Quant | Physical Size |
| :--- | :---: | :---: |
| **`Gemma-4-26B-A4B-it.F16.gguf`** | F16 (16-bit Float) | **47.05 GiB** (50,520,952,320 bytes) |
| **`Gemma-4-26B-A4B-it.Q8_0.gguf`** | Q8_0 (8-bit Quant) | **25.03 GiB** (26,875,676,160 bytes) |
| **`Gemma-4-26B-A4B-it.Q4_K_M.gguf`** | Q4_K_M (4-bit Quant) | **15.65 GiB** (16,811,832,832 bytes) |
| **`Gemma-4-26B-A4B-it.Q3_K_M.gguf`** | Q3_K_M (3-bit Quant) | **12.38 GiB** (13,302,550,528 bytes) |

---

## ✦ Running the Models

### LM-Studio
1. Place the preferred GGUF file in your LM-Studio local model directory (e.g. `~/.lmstudio/models/...`).
2. Load the model.
3. Keep standard hardware acceleration enabled.

### CLI Inference (`llama-cli`)
```bash
llama-cli -m /path/to/Gemma-4-26B-A4B-it.Q4_K_M.gguf -p "A deep thought:" -n 128
```
