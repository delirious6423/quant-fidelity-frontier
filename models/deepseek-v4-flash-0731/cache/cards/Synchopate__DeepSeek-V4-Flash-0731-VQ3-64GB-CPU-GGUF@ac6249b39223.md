---
license: mit
library_name: gguf
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
  - gguf
  - deepseek-v4
  - mixture-of-experts
  - cpu-inference
  - vector-quantization
  - llama-cpp
  - claude-code
  - avx512
  - local-llm
pipeline_tag: text-generation
quantized_by: Synchopate
---

# 🐋 DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF (284.33B Parameters in 56.45 GiB — Surgical Hybrid `Fractal-64 + LUT-VQ3-64`)

### Run the Full 284.33B DeepSeek-V4-Flash Frontier MoE (`13.17B` Active) in 64 GB RAM with Zero Double-Quantization (`100% Fractal-64 MLA Attention` + `3.00-bpw LUT-VQ3-64 Interior Shared Experts`)

🔗 **Engine, AVX-512 VBMI `VPERMI2B` Kernels, RouteCrack-KV HARP-Cache & One-Click Agent CLI (`./modchip claude`):** [**https://github.com/synchopate/llama-modchip**](https://github.com/synchopate/llama-modchip)

---

## 🔬 Surgical Hybrid Architecture (`Fractal-64` MLA + `LUT-VQ3-64` Interior Pre-`SiLU` Shared Experts)

Built via [`scripts/build_dsv4_vq3.py`](https://github.com/synchopate/llama-modchip/blob/main/scripts/build_dsv4_vq3.py), this **Surgical Hybrid (`v2.5.0`)** combines `Fractal-64` (`GGML_TYPE_Q2_0 = 40`, `3.75 bpw`) and `LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `3.00 bpw`) with **zero double-quantization**:

1. **100% of MLA Attention Matrices Preserved in `Fractal-64` (`3.75 bpw`, `30 B / 64 W`):**
   Every `attn_q_b`, `attn_output_a`, `attn_output_b`, `attn_kv`, and `attn_compressor_*` tensor remains untouched in `Fractal-64` (`17.75 dB` SQNR, `99.15%` Cosine Similarity) with zero re-quantization loss.
2. **Anchor Shared Experts (`blk.0..4`, `blk.39..42`) & All Residual Output Projections (`blk.*.ffn_down_shexp.weight`) Preserved in `Q5_K` / `Q6_K`:**
   Protects the first 5 layers, final 4 layers, and every shared-expert down-projection writing into the residual stream.
3. **68 Interior Shared-Expert Pre-`SiLU` Projections (`blk.5..38` `ffn_gate_shexp` & `ffn_up_shexp`) Encoded in `LUT-VQ3-64` (`3.00 bpw`, `24 B / 64 W`):**
   Quantized directly from `Q5_K` (`5.5 bpw`) into our 8-level Lloyd-Max non-uniform codebook (`GGML_TYPE_VQ3_0 = 43`), decoded in 1 clock cycle via AVX-512 VBMI `vpermb` / `vpermi2b` inside ZMM registers.
4. **256 Routed MoE Experts:** Analytical 2nd-order `SwiGLU-H2` `IQ1_S` / `IQ1_M` / `IQ2_XXS` (`Top-6` lossless routing).

---

## 🚀 Quickstart (Linux `x86_64` & Windows 11)

### Linux (`x86_64` Static Binary Included)
```bash
hf download Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF --local-dir ./dsv4-vq3
chmod +x ./dsv4-vq3/llama-server-modchip-linux-x86_64
./dsv4-vq3/llama-server-modchip-linux-x86_64 \
  -m ./dsv4-vq3/DeepSeek-V4-Flash-0731-VQ3-00001-of-00003.gguf \
  -c 32768 -t 16 -tb 16 -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 --port 8080
```

### Windows 10/11
Clone [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip), place the three `.gguf` shards into `windows\models\`, build or download the patched Windows `.exe` (`windows\build_patched_engine.ps1` or [`v2.4.0-windows-cpu`](https://github.com/MR1KRAKEN/llama-modchip/releases/tag/v2.4.0-windows-cpu)), and run `windows\AVVIA_DSV4.bat`.
