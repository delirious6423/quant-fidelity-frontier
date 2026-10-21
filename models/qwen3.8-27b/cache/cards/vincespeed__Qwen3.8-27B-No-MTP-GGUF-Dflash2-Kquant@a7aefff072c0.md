---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
---
https://www.aihublocal.com/llm-runner-aio
# Qwen3.8-27B Q3-Kmix GGUF (MTP-Free, Pure K-Quant Family)

> Four **pure K-quant** GGUF variants of Qwen3.8-27B-UD, regenerated from official BF16 checkpoints. All variants are MTP-free, enforce a Q2_K minimum floor, and contain **no importance-matrix (IQ\*) types** — every weight is a standard K-quant (`Q2_K` / `Q3_K` / `Q4_K`) or `f32`.

## 📊 Variants

| Variant | File Size | Avg Precision | Distribution (851 tensors) |
|---------|-----------|---------------|----------------------------|
| **v1k** | 10,523,284,960 B (~9.8 GB) | 3.13 BPW | 238 Q3_K · 223 Q2_K · 37 Q4_K · 353 f32 |
| **v2k** | 11,370,751,840 B (~10.6 GB) | 3.38 BPW | 405 Q3_K · 56 Q2_K · 37 Q4_K · 353 f32 |
| **v3k** | 13,059,088,480 B (~12.2 GB) | 3.88 BPW | 275 Q4_K · 167 Q3_K · 56 Q2_K · 353 f32 |
| **v4k** | 14,167,314,400 B (~13.2 GB) | 4.21 BPW | 442 Q4_K · 56 Q2_K · 353 f32 |

All variants: 851 tensors, `qwen35` hybrid SSM + Attention architecture, `block_count=64`, `nextn_predict_layers=0`.

## 🎯 Design Goals

1. **Pure K-Quant** — no `IQ*` types, no imatrix dependency: deterministic, reproducible quantization with standard llama.cpp kernels.
2. **Regenerated from Official BF16 Checkpoints** — no cumulative requantization error; all weights quantized once, directly from BF16.
3. **MTP-Free** — the extra MTP head block (blk.64) is stripped; `block_count=64`, `nextn_predict_layers=0`.
4. **Q2_K Minimum Floor** — no tensor falls below Q2_K (2.96 BPW); ultra-low types (IQ2_XS/IQ2_S/IQ1_S) cause unstable speculative-decoding acceptance rates.
5. **Layer-Band Protection Map** — precision is assigned by layer band and tensor role (see map below), keeping critical early/mid attention & FFN paths at higher precision while late, more redundant blocks run lighter.

## 🗺️ Tensor Quantization Map

Architecture: 64 transformer blocks. 48 blocks are SSM blocks (tensors: `ssm_alpha`, `ssm_beta`, `attn_qkv`, `attn_gate`, `ssm_out`); 16 blocks are full-attention blocks (layers **3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63**, tensors: `attn_q`, `attn_k`, `attn_v`, `attn_output`). All 64 blocks carry `ffn_down`, `ffn_gate`, `ffn_up`. Norms, biases and SSM constants stay `f32` in every variant (353 tensors).

### SSM-block tensors (`ssm_alpha`, `ssm_beta`, `attn_qkv`, `attn_gate`, `ssm_out`)

| Layers (SSM blocks) | v1k | v2k | v3k | v4k |
|---------------------|-----|-----|-----|-----|
| 0–29 ¹ | Q3_K | Q3_K | Q4_K | Q4_K |
| 30–51 ¹ | Q2_K | Q3_K | Q4_K | Q4_K |
| 52, 53, 54, 56 ¹ | Q4_K | Q4_K | Q4_K | Q4_K |
| 57, 58, 60, 61, 62 ¹ | Q2_K | Q2_K | Q2_K | Q2_K |

### FFN tensors (`ffn_down`, `ffn_gate`, `ffn_up`)

| Layers | v1k | v2k | v3k | v4k |
|--------|-----|-----|-----|-----|
| 0–29   | Q3_K | Q3_K | Q4_K | Q4_K |
| 30–51  | Q2_K | Q3_K | Q4_K | Q4_K |
| 52–55  | Q4_K | Q4_K | Q4_K | Q4_K |
| 56–63  | Q2_K | Q2_K | Q2_K | Q2_K |

### Full-attention tensors

| Tensors | Layers | v1k | v2k | v3k | v4k |
|---------|--------|-----|-----|-----|-----|
| `attn_k` | 3, 7, 55 | Q4_K | Q4_K | Q4_K | Q4_K |
| `attn_k` | 11, 15, 19, 23, 27, 31, 35, 39, 51, 59, 63 | Q3_K | Q3_K | Q4_K | Q4_K |
| `attn_k` | 43, 47 | Q2_K | Q3_K | Q4_K | Q4_K |
| `attn_q` · `attn_v` · `attn_output` | 3, 7, 11, 15, 19, 23, 27 | Q3_K | Q3_K | Q4_K | Q4_K |
| `attn_q` · `attn_v` · `attn_output` | 31, 35, 39, 43, 47, 51 | Q2_K | Q3_K | Q4_K | Q4_K |
| `attn_q` · `attn_v` · `attn_output` | 55 | Q4_K | Q4_K | Q4_K | Q4_K |
| `attn_q` · `attn_v` · `attn_output` | 59, 63 | Q2_K | Q2_K | Q2_K | Q2_K |

### Special tensors

| Tensor | v1k | v2k | v3k | v4k |
|--------|-----|-----|-----|-----|
| `token_embd.weight` | Q3_K | Q3_K | Q4_K | Q4_K |
| `output.weight`     | Q2_K | Q2_K | Q2_K | Q2_K |
| norms / biases / SSM constants (353) | f32 | f32 | f32 | f32 |

¹ *only the SSM layers inside these ranges (full-attention layers 3, 7, 11, … excluded).* Exception: `blk.52.ssm_alpha` follows the 30–51 band (v1k: Q2_K, v2k: Q3_K, v3k/v4k: Q4_K).

## 🔧 Quantization Pipeline

### Sources
- **Target:** `Qwen/Qwen3.8-27B` official BF16 checkpoint (18 shards, 55.6 GB) → BF16 GGUF (54.6 GB, 866 tensors)
- **MTP Removal:** streaming binary surgery — blk.64 (15 tensors) stripped, KV patched (`block_count` 65→64, `nextn_predict_layers` 1→0), all 851 data regions verified
- **MMProj:** vision tower converted via `convert_hf_to_gguf.py --mmproj`, then F16
- **Draft:** `z-lab/Qwen3.8-27B-DFlash2-GGUF` BF16 → Q2K_L (Unsloth reference distribution, 81/81 match)

### Steps
1. `convert_hf_to_gguf.py --outtype bf16` on the official checkpoint
2. MTP strip + byte-level verification against the official MTP-free reference
3. Per-tensor type override files (one per variant) applied via `llama-quantize --tensor-type-file`
4. Output verified: tensor count, per-type counts, and per-tensor types checked against the intended maps

### Tools
- `llama.cpp` (CUDA 13 + Vulkan build): `llama-quantize`, `llama-gguf`, `llama-server`
- Custom Python scripts for GGUF binary surgery and override generation

## 🚀 Usage

### llama.cpp Server
```bash
llama-server \
  -m Qwen3.8-27B-Q3Kmix-v2k-noMTP.gguf \
  --mmproj Qwen3.8-27B-mmproj-F16.gguf \
  -ngl 99 \
  -c 90000
```

### Speculative Decoding with DFlash2
```bash
llama-server \
  -m Qwen3.8-27B-Q3Kmix-v2k-noMTP.gguf \
  -md Qwen3.8-27B-DFlash2-Q2K_L.gguf \
  --spec-type draft-dflash \
  --mmproj Qwen3.8-27B-mmproj-F16.gguf \
  -c 90000 \
  -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -ngl 99
```

### Python (llama-cpp-python)
```python
from llama_cpp import Llama

llm = Llama(
    model_path="Qwen3.8-27B-Q3Kmix-v2k-noMTP.gguf",
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
| `Qwen3.8-27B-Q3Kmix-v1k-noMTP.gguf` | 10,523,284,960 | Target model — lightest variant (3.13 BPW) |
| `Qwen3.8-27B-Q3Kmix-v2k-noMTP.gguf` | 11,370,751,840 | Target model — balanced Q3_K-centric (3.38 BPW) |
| `Qwen3.8-27B-Q3Kmix-v3k-noMTP.gguf` | 13,059,088,480 | Target model — Q4_K-heavy (3.88 BPW) |
| `Qwen3.8-27B-Q3Kmix-v4k-noMTP.gguf` | 14,167,314,400 | Target model — maximum quality (4.21 BPW) |
| `Qwen3.8-27B-DFlash2-Q2K_L.gguf` | 766,871,136 | DFlash2 draft model (81 tensors, Unsloth reference distribution) |
| `Qwen3.8-27B-mmproj-F16.gguf` | 931,145,984 | Multimodal projector — F16 (334 tensors) |

### SHA256 Verification

```
0a5b96db463dfc2f3e3861cadcfd79a272794e9215e5c8e616e83dba91efa034  Qwen3.8-27B-Q3Kmix-v1k-noMTP.gguf
14ce02bc68e823b3248fa80bdf0b35d3f436b505916592076ea3d2d6ada7bf9a  Qwen3.8-27B-Q3Kmix-v2k-noMTP.gguf
0ce2e37ac491e5d00cd9900cb4e9d31bc7f4a4575f17943d13b39c23ed409e58  Qwen3.8-27B-Q3Kmix-v3k-noMTP.gguf
2d0222974178e4ecae6ad66eb19b089672262cfb5b242fe93b547bfe5ba79f6f  Qwen3.8-27B-Q3Kmix-v4k-noMTP.gguf
10f95d16bce27608041acfece1f1c3b52974a76be8abb6e4407b2b53ccfc75b9  Qwen3.8-27B-DFlash2-Q2K_L.gguf
48853fd5605c7f8464146b14490f53d1968d3d5d9edfbdd95f8bf42263ca6312  Qwen3.8-27B-mmproj-F16.gguf
```

## ⚠️ Known Limitations

1. **No MTP Support** — MTP head removed; not compatible with native MTP-based speculation (use the included DFlash2 draft instead).
2. **CPU-Only Quantization** — `llama-quantize` has no GPU offload; builds run on CPU multithreading.
3. **Q2_K Floor** — tensors below Q2_K intentionally excluded for speculative-decoding stability.
4. **MMProj VRAM** — the F16 projector adds ~1–2 GB VRAM; drop `--mmproj` for text-only runs.

## 📄 License

Based on [Qwen3.8-27B-UD](https://huggingface.co/Qwen/Qwen3.8-27B-UD). Quantization derivatives follow the original model license (Apache-2.0).

## 🔗 Related

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [DFlash2 Draft Models](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

---

**Generated:** 2026-09-13  
**Pipeline:** BF16 → GGUF → MTP-strip → pure K-mix overrides (Q2_K floor)  
**Verified:** SHA256 hashes above match the uploaded artifacts
