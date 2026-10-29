---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
language:
  - en
  - ja
  - zh
  - multilingual
tags:
  - mlx
  - apple-silicon
  - qwen
  - qwen3.8
  - quantization
  - 2-bit
  - mixed-precision
  - pyrite
---

# Qwen3.8-27B-Pyrite-2bit-MLX

[🇯🇵 日本語版モデルカード (README_ja.md) はこちら](./README_ja.md)

## Model Summary

**`Qwen3.8-27B-Pyrite-2bit-MLX`** is a sensitivity-aware ultra-low-bit MLX quantization of [**`Qwen/Qwen3.8-27B`**](https://huggingface.co/Qwen/Qwen3.8-27B) (64 layers, hybrid Gated DeltaNet linear attention + full attention + SwiGLU MLP, `51.75 GiB` in BF16).

Built with the **Project Pyrite** multilingual activation-weighted quantization pipeline, this model compresses the 27B flagship down to **`10.11 GiB` (`10,854,616,576 bytes`)** so it runs entirely in Unified Memory on a **16GB Apple Silicon Mac (M1/M2/M3/M4)** at **`~12.2 tokens/sec`**.

### Key Highlights

* **Zero-Byte-Overhead Quality Recovery (`-22.29%` EN KL / `-22.42%` JA KL / `-27.08%` Multilingual Val KL)**:
  Maintains the **exact same tensor byte budget (`10,854,616,576 bytes` = `10.1091 GiB`)** as a uniform 2-bit (`group_size=64`) MLP baseline—down to the single byte—while reducing KL divergence against the BF16 teacher by:
  * **English (`eval_heldout_en`)**: **`-22.29%` KL** (`0.6230 -> 0.4841`), improving Perplexity from **`10.92` to `9.60`** (`-1.32 PPL`) and Top-1 token agreement from `67.21%` to **`70.81%`** (`+3.60 pt`).
  * **Japanese (`eval_heldout_ja`)**: **`-22.42%` KL** (`0.5856 -> 0.4543`), improving Perplexity from **`18.35` to `15.55`** (`-2.81 PPL`) and Top-1 token agreement from `66.00%` to **`68.65%`** (`+2.65 pt`).
* **Exact Byte-Neutral Sensitivity Allocation**:
  Widening the quantization group size of `mlp.gate_proj` and `mlp.up_proj` from `64` to `128` (`2.50 -> 2.25` effective bits) across 64 layers frees exactly **`356,515,840 bytes` (`340.00 MiB`)** of scale/bias metadata. Every single byte is reinvested to upgrade the **30 most outlier-sensitive late-layer MLP projections** (`22 down_proj` + `8 gate/up_proj` matrices) to **`3-bit, group_size=64`** (`3.50` effective bits).
* **100% Native `mlx-lm` Compatibility**:
  Uses standard `mlx-lm` `QuantizedLinear` layers (`qwen3_5` architecture)—no custom C++/Metal kernels or patched forks required.

---

## Quick Start

### 1. Standard `mlx-lm` Usage

```bash
pip install -U mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("masahiroid/Qwen3.8-27B-Pyrite-2bit-MLX")
prompt = "Explain why sensitivity-aware mixed-precision quantization works well for large language models:"
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

### 2. Recommended Setup for 16GB Apple Silicon Macs

On a 16GB Mac, setting `mx.set_wired_limit(int(11.0 * 1024**3))` and materializing layers sequentially keeps the entire `10.11 GiB` model pinned in Unified Memory without SSD swap thrashing:

```python
import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer, hf_repo_to_path
from mlx_lm import generate

mx.set_wired_limit(int(11.0 * 1024**3))

model_path = hf_repo_to_path("masahiroid/Qwen3.8-27B-Pyrite-2bit-MLX")
model, _ = load_model(model_path, lazy=True)
tokenizer = load_tokenizer(model_path)

for layer in model.language_model.model.layers:
    mx.eval(layer.parameters())
    mx.clear_cache()
mx.eval(model.parameters())

print(generate(model, tokenizer, prompt="Write a clean Python implementation of Dijkstra's algorithm:", max_tokens=250))
```

---

## Quantization Architecture & Byte Budget

In `Qwen/Qwen3.8-27B` (64 layers, `hidden_size=5120`, `intermediate_size=17408`), each SwiGLU MLP projection (`gate_proj, up_proj, down_proj`) contains `89,128,960` parameters (`17.11B` total MLP parameters across 64 layers).

* Changing `gate_proj` and `up_proj` from `2-bit/g64` (`2.50` eff bits) to `2-bit/g128` (`2.25` eff bits) saves **`2,785,280 bytes`** per matrix (`128 × 2,785,280 = 356,515,840 bytes` total across 64 layers).
* Upgrading one `down_proj` matrix from `2-bit/g64` (`2.50` eff bits) to `3-bit/g64` (`3.50` eff bits) costs **`11,141,120 bytes`** (`4` units of `2,785,280 bytes`).
* Upgrading one `gate_proj` or `up_proj` matrix from `2-bit/g128` (`2.25` eff bits) to `3-bit/g64` (`3.50` eff bits) costs **`13,926,400 bytes`** (`5` units of `2,785,280 bytes`).
* Thus, upgrading **22 `down_proj` matrices (`22 × 4 = 88` units) + 8 `gate/up_proj` matrices (`8 × 5 = 40` units) = `128` units = `356,515,840 bytes`**—matching the uniform 2-bit baseline to the single byte (`10,854,616,576 bytes`).

| Component | Precision | Group Size | Effective Bits | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `embed_tokens` & `lm_head` (`vocab=248,320`) | `4-bit` | `64` | `4.50` | Protected multilingual vocabulary |
| Hybrid Attention (`linear_attn` × 48, `self_attn` × 16) | `4-bit` | `64` | `4.50` | Gated DeltaNet & Full Attention projections |
| Norms, Conv1d (`BF16`) & SSM `A_log` (`FP32`) | `BF16 / FP32` | — | `16 / 32` | Unquantized normalization & state-space parameters |
| **Default MLP `gate_proj` & `up_proj` (120 matrices)** | **`2-bit`** | **`128`** | **`2.25`** | Low-sensitivity pre-SwiGLU projections |
| **Default MLP `down_proj` (42 matrices)** | **`2-bit`** | **`64`** | **`2.50`** | Early/mid-layer `down_proj` |
| **Upgraded High-Sensitivity MLP Slots (30 matrices)** | **`3-bit`** | **`64`** | **`3.50`** | `L35–L36, L44–L63 down_proj` (22) + `L58–L63 gate/up_proj` (8) |
| **Total Model Tensor Footprint** | **Mixed** | **Mixed** | **~2.97 avg** | **`10.1091 GiB` (`10,854,616,576 bytes`)** |

---

## Evaluation Benchmarks (Strict Equal-Byte-Budget Comparison)

All candidates were evaluated at the **exact same tensor byte budget (`10,854,616,576 bytes` = `10.1091 GiB`)** against exact BF16 teacher logits streamed across all 64 layers on held-out English (`eval_heldout_en`), held-out Japanese (`eval_heldout_ja`), and Multilingual Validation (`calib_val_multi`).

| Model Configuration | Tensor Bytes | `eval_heldout_en` KL ($\downarrow$) | `eval_heldout_en` PPL ($\downarrow$) | EN Top-1 ($\uparrow$) | `eval_heldout_ja` KL ($\downarrow$) | `eval_heldout_ja` PPL ($\downarrow$) | JA Top-1 ($\uparrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BF16 Teacher (`Qwen/Qwen3.8-27B`)** | `51.75 GiB` | `0.000000` | `6.1480` | `100.00%` | `0.000000` | `10.1519` | `100.00%` |
| **Uniform 2-bit Baseline (`2bit_g64`)** | `10,854,616,576` (`10.11 GiB`) | `0.622954` | `10.9244` | `67.21%` | `0.585637` | `18.3547` | `66.00%` |
| **Pyrite `down32` (32 `down_proj` at 3-bit)** | `10,854,616,576` (`10.11 GiB`) | `0.516065 (-17.16%)` | `9.8880 (-1.04)` | `70.15%` | `0.512605 (-12.47%)` | `16.5653 (-1.79)` | `67.87%` |
| **Pyrite `mixed_22down_8gateup` (This Repo)** | **`10,854,616,576` (`10.11 GiB`)** | **`0.484077 (-22.29%)`** | **`9.6029 (-1.32)`** | **`70.81%`** | **`0.454348 (-22.42%)`** | **`15.5453 (-2.81)`** | **`68.65%`** |

---

## License & Citation

* **Base Model**: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) by Qwen Team, licensed under **Apache-2.0**.
* **License**: Apache-2.0.

