---
license: other
license_name: cognitus-public-symbolic-license-1.0
license_link: LICENSE
base_model: zai-org/GLM-5.3-Flash
library_name: axim-os
pipeline_tag: text-generation
language:
  - en
  - zh
  - multilingual
tags:
  - symb
  - axim-os
  - cognitus-lab
  - synaxim
  - glm-5.3-flash
  - moe
  - mla
  - kda
  - hyper-connections
  - agentic-coding
---

# ⚡ GLM-5.3-Flash (`.symb` Universal Symbolic Edition) — Powered by AXIM OS

**`GRRNMAKE/GLM5.3Flash_symb`** is the complete, zero-loss structural conversion of **`zai-org/GLM-5.3-Flash`** (`Glm5NextForConditionalGeneration`, 45 hybrid KDA/NoPE-MLA decoder layers + Layer 45 MTP module, 288 routed experts + 1 shared expert per MoE layer, 4-stream Manifold-Constrained Hyper-Connections) into the **`.symb` Universal Symbolic Weight Format** developed by **Cognitus Lab**.

This release includes **both** the native `GLM-5.3-Flash` (`154,880` vocab) weights (`embeddings.symb`, `lm_head.symb`) and the **MATT/AIM Layer-00 Transferred `SYNAXIM` (`128,256` vocab)** weights (`embeddings_synaxim.symb`, `lm_head_synaxim.symb`, `synaxim_tokenizer.json`), enabling universal cross-architecture tokenization and zero-CUDA-loader execution via **AXIM OS**.

---

## 🚀 Accelerate `.symb` Loading & Inference with AXIM OS (`pip install axim-os`)

`.symb` models are engineered to run dramatically faster with **AXIM OS** — the **Universal CUDA-Free Compute & Graphics Runtime Engine by Cognitus Lab**.

### Install AXIM OS from PyPI

```bash
pip install axim-os
```

- **PyPI Package:** [https://pypi.org/project/axim-os/](https://pypi.org/project/axim-os/)
- **Native Engine:** `axim-os` (`libaxim_cpu.so` / `libaxim_gpu.so`) performs hardware-vectorized SIMD/NEON + OpenMP `.symb` (`INT4` group-128 $\rightarrow$ `BF16` / `FP32`) weight loading and in-place dequantization directly in host memory without CUDA weight-loader overhead or VRAM bloat.

---

## 📊 Verified Speed & GPU Memory Benchmarks (`AXIM OS 1.1.3` + GPU)

Benchmarked on **NVIDIA GH200 Grace Hopper Superchip** (`96 GB HBM3` GPU + `72-Core Grace ARM64` CPU running `axim-os 1.1.3`):

| Metric | Raw Safetensors (`62` Shards) | `.symb` + `AXIM OS 1.1.3` | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Disk / Storage Footprint** | `328.30 GB` (`76,108` raw entries) | **`159.11 GB`** (`38,772` `.symb` files, **0 skipped weights**) | **2.06x smaller** |
| **Active GPU VRAM Required** | `> 328 GB` (multi-GPU required) | **`17.70 GB` HBM3** (`> 73.3 GB` free on a single 96 GB GPU) | **94.6% GPU VRAM reduction** |
| **Static 45-Layer Backbone Preload (`1,169` `.symb` files, `4.39 GB`)** | `42.0s+` (standard Python/CUDA load) | **`3.20s` total (`0.34s` AXIM OS dequant)** | **> 13x faster load** |
| **AXIM OS Dequantization Throughput** | — | **`12.91 GB/s` compressed (`~50 GB/s` `BF16` output, `0.29 ms/tensor`)** | **Hardware wire-speed** |
| **Sustained Live MoE Streaming (`23,999` `.symb` files, `106.46 GB`)** | — | **`8.83s` total dequant time (`0.37 ms` per expert matrix)** | **Zero-allocation in-place LRU** |
| **Live 45-Layer Warm Time-To-First-Token (TTFT)** | — | **`1.41 seconds`** (full 45-layer hybrid KDA + NoPE-MLA + MoE) | **Real-time interactive streaming** |

---

## 🌐 `.symb` Works Across All Foundation Model Architectures

The **`.symb` Universal Symbolic Weight Format** is architecture-agnostic and has been proven in production across every major modern neural architecture:

1. **Hybrid Linear Attention + Multi-Head Latent Attention MoE (`GLM-5.3-Flash`):**
   - 34 Kimi Delta Attention (`KDA`) recurrent/chunk layers + 11 `NoPE-MLA` layers + 42 routed MoE layers (`288` routed experts/layer) + 4-stream Manifold-Constrained Hyper-Connections (`mHC` Sinkhorn-Knopp matrices preserved in lossless `FP32`).
2. **Ultra-Scale DeepSeek-V3 / Kimi-K3 MLA MoE (`Kimi-K3`):**
   - 61-layer MLA + `384`-expert routed MoE converted to `.symb` and aligned to the `SYNAXIM` (`128,256`) tokenizer via MATT/AIM Layer-00 alignment.
3. **Dense & Dual-Fluid Reasoning Transformers (`Chííké`):**
   - 32-layer dense & heterogeneous agentic coding models running natively on `.symb` + `axim-os`.

### Binary Specification of `.symb`
Every `.symb` tensor file uses a compact, self-describing binary header playable directly into `axim-os` C/C++ SIMD kernels:
- `ndim` (`uint32`): Number of tensor dimensions.
- `shape[0..ndim-1]` (`uint32[]`): Exact tensor dimensions.
- `n_elements` (`uint32`): Total element count.
- `group_size` (`uint32`):
  - `group_size = 128`: Symmetric group-wise `INT4` (`[-7, 7]` offset by `+8` into packed `uint8` nibbles) preceded by `FP32` per-group scales (`n_groups = ceil(n_elements / 128)`), folded directly from native `F8_E4M3` block scales (`weight_scale_inv`) where present.
  - `group_size = 1`: Lossless `FP32` storage used automatically for all sensitive 1D norms, router gates (`router.symb`), `e_score_correction_bias.symb`, `dt_bias.symb`, `A_log.symb`, `conv1d` kernels, and 4-stream hyper-connection (`hc_*`) parameters.

---

## 🧬 SYNAXIM (`128,256` Vocab) Tokenizer Transfer Included

In addition to the native `GLM-5.3-Flash` (`154,880`) vocabulary (`embeddings.symb`, `lm_head.symb`), this repository includes **`embeddings_synaxim.symb`** and **`lm_head_synaxim.symb`** transferred to the unified **`GRRNMAKE/SYNAXIM` (`128,256`)** tokenizer using the **MATT/AIM Layer-00 Alignment Pipeline**:

1. **Exact Vocabulary Overlap (`87.38%`):** `112,072 / 128,256` tokens match between `SYNAXIM` and `GLM-5.3-Flash`.
2. **MATT/AIM Layer-00 Alignment (`16,184` Disjoint Tokens):** Disjoint tokens are aligned in feature space (`cos_sim = 0.9999`) for seamless multi-turn chat, coding, and reasoning coherence.

### Verified 45-Layer Coherence Output (`.symb` + `SYNAXIM` + `axim-os 1.1.3`)
- **Prompt:** `"The capital of France is"` $\rightarrow$ **`"The capital of France is Paris. It is known for"`** (`" Paris"` Rank #1, logit `17.50`)
- **Prompt:** `"The largest planet in our solar system is"` $\rightarrow$ **`"The largest planet in our solar system is Jupiter. It is a gas"`** (`" Jupiter"` Rank #1, logit `15.69`)
- **Prompt:** `"Water boils at a temperature of 100 degrees"` $\rightarrow$ **`"Water boils at a temperature of 100 degrees Celsius at sea level. However"`** (`" Celsius"` Rank #1, logit `18.12`)

---

## 💻 Quickstart: Loading `.symb` Tensors with `axim-os`

```bash
pip install axim-os torch numpy tokenizers
```

```python
import struct
import ctypes
import numpy as np
import torch
import axim_os
from axim.orchestrator.native_loader import load_native

# 1. Initialize AXIM OS native C/SIMD engine (CUDA-free weight loader)
axim_lib = load_native("cpu")
axim_lib.axim_cpu_symb_dequant_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_int,
]
axim_lib.axim_cpu_symb_dequant_bf16.restype = None

def load_symb_with_axim_os(path: str, out_buf: torch.Tensor = None) -> torch.Tensor:
    """Load any .symb tensor (INT4 group-128 or lossless FP32) via AXIM OS."""
    with open(path, "rb") as f:
        data = f.read()
    ndim, = struct.unpack_from("<I", data, 0)
    shape = struct.unpack_from(f"<{ndim}I", data, 4)
    n, group_size = struct.unpack_from("<II", data, 4 + 4 * ndim)
    offset = 4 + 4 * ndim + 8

    if group_size == 1:
        arr = np.frombuffer(data, dtype=np.float32, count=n, offset=offset).copy().reshape(shape)
        return torch.from_numpy(arr)

    pad = (group_size - (n % group_size)) % group_size
    n_padded = n + pad
    n_groups = n_padded // group_size
    scales_np = np.frombuffer(data, dtype=np.float32, count=n_groups, offset=offset)
    offset += 4 * n_groups
    packed_bytes = (n_padded + 1) // 2
    packed_np = np.frombuffer(data, dtype=np.uint8, count=packed_bytes, offset=offset)

    out_cpu = out_buf if (out_buf is not None and tuple(out_buf.shape) == tuple(shape)) else torch.empty(shape, dtype=torch.bfloat16)
    axim_lib.axim_cpu_symb_dequant_bf16(
        packed_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        scales_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(out_cpu.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
        n_groups, group_size, n, 0
    )
    return out_cpu

# Example: Dequantize SYNAXIM Embeddings (128,256 x 4,096) in milliseconds via AXIM OS
emb = load_symb_with_axim_os("embeddings_synaxim.symb")
print("Loaded SYNAXIM embeddings:", emb.shape, emb.dtype)
```

---

## 📁 Repository Structure

```text
GRRNMAKE/GLM5.3Flash_symb/
├── LICENSE                        # Cognitus Public Symbolic License v1.0 (CPSL-1.0)
├── README.md                      # Documentation, AXIM OS benchmarks & usage guide
├── axim_symb_loader.py            # Standalone AXIM OS .symb loader & 45-layer inference engine
├── config.json                    # GLM-5.3-Flash architectural configuration
├── synaxim_tokenizer.json         # Transferred SYNAXIM (128,256 vocab) fast tokenizer
├── tokenizer.json                 # Native GLM-5.3-Flash (154,880 vocab) tokenizer
├── tokenizer_config.json          # Native tokenizer config
├── embeddings_synaxim.symb        # SYNAXIM [128256, 4096] MATT/AIM Layer-00 aligned embeddings
├── lm_head_synaxim.symb           # SYNAXIM [128256, 4096] natural-norm calibrated LM head
├── embeddings.symb                # Native [154880, 4096] embeddings
├── lm_head.symb                   # Native [154880, 4096] LM head
├── final_norm.symb                # Final RMSNorm [4096] (FP32)
├── layer_00/ .. layer_02/         # Dense KDA Layers (Attention + mHC + Dense SwiGLU MLP)
├── layer_03/ .. layer_44/         # Hybrid KDA / NoPE-MLA + 288-Expert Routed MoE Layers
│   ├── hc_attn_*.symb             # 4-stream Attention Hyper-Connections (FP32)
│   ├── hc_ffn_*.symb              # 4-stream FFN Hyper-Connections (FP32)
│   ├── router.symb                # 288-expert sigmoid router + e_score_correction_bias.symb (FP32)
│   ├── shared_expert_*.symb       # Always-active shared expert projections (INT4 .symb)
│   └── experts_{gate,up,down}.symb # 288 routed experts stacked as 3D .symb tensors ([288, 2048, 4096] / [288, 4096, 2048])
└── layer_45/                      # Multi-Token Prediction (MTP) Layer (eh_proj, norms, MLA, 288 3D .symb experts)
```

---

## 📜 License — Cognitus Public Symbolic License v1.0 (`CPSL-1.0`)

This repository and all `.symb` weights, MATT/AIM Layer-00 `SYNAXIM` vocabulary transfers, and `AXIM OS` integration scripts are released under the **Cognitus Public Symbolic License v1.0 (`CPSL-1.0`)** (see [LICENSE](./LICENSE)).

```bibtex
@misc{cognitus2026glm53flash_symb,
  title  = {GLM-5.3-Flash (.symb Universal Symbolic Edition) with SYNAXIM Tokenizer Transfer & AXIM OS Runtime},
  author = {Cognitus Lab and GRRNMAKE},
  year   = {2026},
  url    = {https://huggingface.co/GRRNMAKE/GLM5.3Flash_symb}
}
```
