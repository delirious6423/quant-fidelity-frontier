---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
---
https://www.aihublocal.com/llm-runner-aio
# Qwen3.8-27B IQ3_S GGUF (v8, MTP-Free, Q2_K Minimum)

> **Quantized GGUF variant of Qwen3.8-27B-UD** — regenerated from the official BF16 checkpoints, optimized for speculative decoding with DFlash2 draft models.

## 📊 Model Overview

| Property | Value |
|----------|-------|
| **Base Model** | Qwen3.8-27B-UD |
| **Version** | v8 (BF16-source regeneration) |
| **Quantization** | IQ3_S hybrid K-mix (per-tensor overrides) |
| **MTP Support** | ❌ Removed (MTP-free) |
| **Minimum Quant Type** | Q2_K (2.96 BPW) |
| **Total Tensors** | 851 |
| **Average Precision** | 3.35 BPW |
| **Architecture** | qwen35 (hybrid SSM + Attention), block_count=64 |
| **File Size** | 11,282,637,536 B (~10.5 GB) |

## 🎯 Design Goals

This quantization was specifically designed for **speculative decoding with DFlash2 draft models**. Key design decisions:

1. **Regenerated from Official BF16 Checkpoints** — instead of requantizing a pre-quantized UD build, the target, draft, and multimodal projector were all converted from the official Hugging Face BF16 checkpoints. This removes cumulative rounding error from intermediate quantizations.
2. **MTP Layer Removed** — blk.64 transformer block (15 tensors) stripped, `block_count` set to 64, `nextn_predict_layers` set to 0. Verified byte-identical against the official MTP-free reference.
3. **Minimum Q2_K Floor** — No tensors below Q2_K (IQ2_XS, IQ2_S, IQ1_S excluded). Ultra-low precision tensors cause inconsistent token acceptance rates and hallucinations in DFlash2 speculative decoding.
4. **Imatrix Calibration** — Importance matrix generated from the 54.6 GB BF16 GGUF using a 48K+ token corpus, ensuring accurate activation measurements for the IQ3_XXS assignments.
5. **V7 Protection Map Preserved** — the per-tensor type assignment replicates the v7 protection map exactly (851/851 tensor types verified identical), including the elevated full-attention `attn_k` layers (3, 7, 11, 15, 19, 27 → Q4_K).
6. **Draft Distribution Replication** — the DFlash2 draft reproduces the Unsloth reference per-tensor distribution exactly (81/81 tensor types match, byte-for-byte identical file size).

## 📈 Tensor Distribution

### Target Model (851 tensors, MTP-free)

```
353 f32     —  Full precision (norms, small projections, SSM constants)
228 iq3_s   —  IQ3_S quantized (critical attention/FFN tensors)
171 iq3_xxs —  IQ3_XXS quantized (medium-risk tensors, imatrix-guided)
 33 q4_K    —  Q4_K quantized (residual critical tensors)
 66 q2_K    —  Q2_K quantized (safest, uniform tensors — minimum floor)
```

### Full-Attention Layer (attn_k) Map — 16 layers

Only 16 of the 64 blocks carry a separate `attn_k` weight (full-attention layers); the remaining blocks use fused `attn_qkv`.

| Layer | Type | | Layer | Type |
|-------|------|-|-------|------|
| 3  | Q4_K   | | 35 | IQ3_S  |
| 7  | Q4_K   | | 39 | IQ3_S  |
| 11 | Q4_K   | | 43 | IQ3_S  |
| 15 | Q4_K   | | 47 | IQ3_S  |
| 19 | Q4_K   | | 51 | IQ3_S  |
| 23 | IQ3_S  | | 55 | IQ3_S  |
| 27 | Q4_K   | | 59 | IQ3_S  |
| 31 | IQ3_S  | | 63 | IQ3_S  |

### Draft Model (Qwen3.8-27B-DFlash2-Q2K_L, 81 tensors)

```
32 f32   —  Full precision (norms, embeddings)
16 q4_K  —  Critical draft weights
23 q2_K  —  Standard floor
10 q3_K  —  Mid-precision
```

Per-tensor mapping replicates the Unsloth reference build exactly (verified 81/81).

## 🔧 Quantization Pipeline

### Sources
- **Target:** `Qwen/Qwen3.8-27B` official BF16 checkpoint (18 shards, 55.6 GB)
- **Protection Map:** v7 per-tensor type assignments extracted from `Qwen3.8-27B-UD-IQ3_S-q2k-v7-noMTP-REVIEWED.gguf`
- **Draft:** `z-lab/Qwen3.8-27B-DFlash2-GGUF` BF16 GGUF (3.6 GB)
- **MMProj:** `Qwen/Qwen3.8-27B` vision tower, converted via `convert_hf_to_gguf.py --mmproj`

### Steps
1. **BF16 GGUF Conversion** — `convert_hf_to_gguf.py --outtype bf16` (54.6 GB, 866 tensors).
2. **MTP Removal** — custom streaming binary surgery: strip blk.64 (15 tensors), patch `qwen35.block_count 65→64`, `qwen35.nextn_predict_layers 1→0`; verify sha256 of all 851 data regions.
3. **Imatrix Generation** — `llama-imatrix` over the BF16 GGUF with a 48K+ token corpus → importance matrix.
4. **Hybrid K-mix Quantization** — `llama-quantize --allow-requantize --imatrix ... --tensor-type-file ...` applying the v7 per-tensor protection map (Q2_K floor enforced, 851/851 types verified against v7).
5. **Draft Replication** — per-tensor override extracted from the Unsloth reference GGUF (16×q4_K, 23×q2_K, 10×q3_K) applied to the z-lab BF16 GGUF.

### Tools
- `llama.cpp` (CUDA 13 + Vulkan build): `llama-quantize`, `llama-imatrix`, `llama-gguf`, `llama-server`
- Custom Python scripts for GGUF binary surgery and tensor-override generation

## 🚀 Usage

### llama.cpp Server
```bash
llama-server \
  -m Qwen3.8-27B-IQ3_S-noMTP-.gguf \
  --mmproj Qwen3.8-27B-mmproj-BF16.gguf \
  -ngl 99 \
  -c 90000
```

### Speculative Decoding with DFlash2
```bash
llama-server \
  -m Qwen3.8-27B-IQ3_S-noMTP-.gguf \
  -md Qwen3.8-27B-DFlash2-Q2K_L.gguf \
  --spec-type draft-dflash \
  --mmproj Qwen3.8-27B-mmproj-BF16.gguf \
  -c 90000 \
  -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -ngl 99
```

### Python (llama-cpp-python)
```python
from llama_cpp import Llama

llm = Llama(
    model_path="Qwen3.8-27B-IQ3_S-noMTP-.gguf",
    n_gpu_layers=99,
    n_ctx=8192,
)

output = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

## 📁 Files

| File | Size (bytes) | Description |
|------|-------------|-------------|
| `Qwen3.8-27B-IQ3_S-noMTP-.gguf` | 11,282,637,536 | Target model — IQ3_S hybrid, MTP-free, Q2_K floor, v7 protection map |
| `Qwen3.8-27B-DFlash2-Q2K_L.gguf` | 766,871,136 | DFlash2 draft model — exact Unsloth reference distribution |
| `Qwen3.8-27B-mmproj-BF16.gguf` | 931,145,984 | Multimodal projector — BF16 (334 tensors) |

### SHA256 Verification

```
5fff67319fbf26d06d0ecf51d383173f243e71d23cc3db0947290f29d1cd578b  Qwen3.8-27B-IQ3_S-noMTP-.gguf
10f95d16bce27608041acfece1f1c3b52974a76be8abb6e4407b2b53ccfc75b9  Qwen3.8-27B-DFlash2-Q2K_L.gguf
1410233d4274e4e8bb78352f184d3dbfbbd5c5d01a448945a804d13d2cbee49a  Qwen3.8-27B-mmproj-BF16.gguf
```

## ⚠️ Known Limitations

1. **No MTP Support** — MTP layer removed. Not compatible with MTP-based speculative decoding.
2. **CPU-Only Quantization** — `llama-quantize` has no GPU offload flags; quantization runs on CPU multithreading.
3. **Q2_K Minimum** — tensors below Q2_K intentionally excluded due to DFlash2 instability.
4. **MMProj VRAM** — the BF16 projector adds ~1–2 GB VRAM; drop `--mmproj` if running text-only.

## 📄 License

Based on [Qwen3.8-27B-UD](https://huggingface.co/Qwen/Qwen3.8-27B-UD). Quantization derivatives follow the original model license (Apache-2.0).

## 🔗 Related

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [DFlash2 Draft Models](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

---

**Generated:** 2026-09-12  
**Pipeline:** BF16 → GGUF → MTP-strip → imatrix → hybrid K-mix (Q2_K floor)  
**Verified:** SHA256 hashes above match the uploaded artifacts
