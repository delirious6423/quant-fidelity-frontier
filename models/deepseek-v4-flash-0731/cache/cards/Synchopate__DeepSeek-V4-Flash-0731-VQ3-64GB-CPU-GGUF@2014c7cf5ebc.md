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

# DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF (`284.33B` in `56.40 GiB` — Hybrid `Fractal-64 + LUT-VQ3-64`)

3-shard hybrid GGUF checkpoint (`60,558,856,608` bytes = **`56.40 GiB` / `60.56 GB`**) of **`DeepSeek-V4-Flash-0731`** (`284.33B` total / `13.17B` active parameters; `43` layers = `3` token-ID hash-routed layers `0..2` + `40` learned-router layers `3..42`, `256` routed + `1` shared expert/layer, `Top-6` active, Multi-head Latent Attention + `CSA/LID/HCA` Sparse Compressors + 4-stream Sinkhorn-20 Hyper-Connections) sized for 64 GB DDR5 hosts with zero swap.

- **Engine, AVX-512 VBMI `VPERMI2B` Kernels, Windows 11 `1M`-Context Port, & CLI (`./modchip claude`):** [https://github.com/synchopate/llama-modchip](https://github.com/synchopate/llama-modchip)

---

## Hybrid Quantization Layout (`Fractal-64` MLA + `LUT-VQ3-64` Interior Shared Experts)

Built via [`scripts/build_dsv4_vq3.py`](https://github.com/synchopate/llama-modchip/blob/main/scripts/build_dsv4_vq3.py), this `v2.5.0` checkpoint combines `Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `3.75 bpw`) and `LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `3.00 bpw`):

1. **100% of MLA & Sparse Compressor Projections Preserved in `Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `3.75 bpw`, `30 B / 64 W`):**
   Every `attn_q_b`, `attn_output_a`, `attn_output_b`, `attn_kv`, `attn_compressor_*`, and `indexer_compressor_*` projection stays directly in `Fractal-64` (`17.75 dB` SQNR, `99.15%` Cosine Similarity) with zero double-quantization.
2. **Boundary Shared Experts (`blk.0..4`, `blk.39..42`) & All Output Projections (`blk.*.ffn_down_shexp.weight`) Preserved in `Q5_K` / `Q6_K`:**
   Keeps the first 5 layers, final 4 layers, and every shared-expert down-projection writing into the 4-stream residual highway at `Q5_K` / `Q6_K`.
3. **68 Interior Shared-Expert Pre-`SiLU` Projections (`blk.5..38` `ffn_gate_shexp` & `ffn_up_shexp`) -> `LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `3.00 bpw`, `24 B / 64 W`):**
   Encoded into our 128-centroid 3D Lloyd-Max codebook (`2 B` `FP16` scale `d` + `3 B` `8 × 3-bit` sub-scales `{2..9}` + `19 B` `20` 7-bit 3D + `2` 6-bit 2D tail indices), decoded in 6 AVX-512 ZMM registers (`384 B` LUT) via `VPERMI2B` (`_mm512_permutex2var_epi8`) with AVX2 and scalar fallbacks.
4. **256 Routed MoE Experts (`blk.*.ffn_{gate,up,down}_exps.weight`):**
   Analytical second-order `SwiGLU-H2` `IQ1_S` / `IQ1_M` / `IQ2_XXS` (`Top-6` unpruned routing).

*(Passing `--aggressive` to `scripts/build_dsv4_vq3.py` additionally converts MLA & shared-expert tensors across 255 matrices (`56.07 GiB` disk / `58.31 GiB` `VmRSS`), lifting raw `N=1` `tg64` decode on 16C Zen 5 from `7.74 tok/s` to `9.05 tok/s` (`+16.9%`) and `pp128` prefill to `41.11 tok/s`.)*

---

## Quickstart (Linux `x86_64` & Windows 11)

> **Runtime Requirements (`GGML_TYPE_Q2_0 = 42`, `GGML_TYPE_VQ3_0 = 43`, & `deepseek4` MLA):**
> - Requires the `llama-modchip v2.5.0` engine (bundled below for Linux `x86_64`, or built from [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip) on Linux, Windows 11, or macOS).
> - **Matching MLA KV Cache Types:** Always pass matching K and V cache types (`-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`); DeepSeek-V4 MLA shares a single latent KV tensor (`576` dim/layer) and rejects asymmetric `q8_0/q4_0`.
> - **Native Top-6 Routing:** Do not set `LLAMA_MOE_TOP_K` or `DSV4_MOE_TOPK` environment variables at load time.

### Linux (`x86_64` Binary Included)
```bash
hf download Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF --local-dir ./dsv4-vq3
chmod +x ./dsv4-vq3/llama-server-modchip-linux-x86_64

./dsv4-vq3/llama-server-modchip-linux-x86_64 \
  -m ./dsv4-vq3/DeepSeek-V4-Flash-0731-VQ3-00001-of-00003.gguf \
  -c 32768 -t 16 -tb 16 -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 --port 8080 --jinja
```

### Windows 11 (`1M`-Context Verified on 64 GB DDR5)
Clone [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip), place all three `.gguf` shards in `windows\models\`, build the `v2.5.0` engine via MSYS2 UCRT64 (`powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1 -Target dsv4`), and launch `windows\AVVIA_DSV4.bat`.
