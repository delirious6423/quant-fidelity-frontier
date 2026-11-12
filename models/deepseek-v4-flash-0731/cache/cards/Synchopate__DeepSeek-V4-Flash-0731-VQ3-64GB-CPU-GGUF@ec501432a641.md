---
license: mit
library_name: llama-modchip
inference: false
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
  - gguf
  - llama-modchip
  - deepseek-v4
  - mixture-of-experts
  - cpu-inference
  - vector-quantization
  - claude-code
  - avx512
  - avx2
  - local-llm
pipeline_tag: text-generation
quantized_by: Synchopate
---

# DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF (`284.33B` in `56.40 GiB` — Hybrid `Fractal-64 + LUT-VQ3-64`)

> [!IMPORTANT]
> **Custom Tensor Types (`GGML_TYPE_Q2_0 = 42` & `GGML_TYPE_VQ3_0 = 43`) — Do NOT use vanilla `Ollama`, `LM Studio`, `vLLM`, or unpatched `llama.cpp`:**
> Upstream engines do not define custom block-64 `Fractal-64` (`type 42`), `LUT-VQ3-64` (`type 43`), or the `deepseek4` MLA / Hadamard / Sinkhorn-20 operators and will fail immediately with `unknown tensor type 42` / `43` (or `40`). Use the ISA-matched prebuilt binaries included in this repository (`llama-server-modchip-linux-x86_64-avx512` for Zen 4/5 & Ice Lake+, or `llama-server-modchip-linux-x86_64-avx2` for Haswell / Skylake / Zen 2/3 / Alder & Raptor Lake with zero `%zmm` instructions) or build from [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip).

3-shard hybrid GGUF checkpoint (`60,558,856,608` bytes = **`56.40 GiB` / `60.56 GB`**) of **`DeepSeek-V4-Flash-0731`** (`284.33B` total / `13.17B` active parameters; `43` layers = `3` token-ID hash-routed layers `0..2` + `40` learned-router layers `3..42`, `256` routed + `1` shared expert/layer, unpruned `Top-6` active, Multi-head Latent Attention + `CSA/LID/HCA` Sparse Compressors + 4-stream Sinkhorn-20 Hyper-Connections) sized for 64 GB DDR5 hosts without SSD swap.

- **Engine, AVX-512 VBMI `VPERMI2B` + Portable AVX2 Kernels, Windows 11 `1M`-Context Port, & CLI (`./modchip`):** [https://github.com/synchopate/llama-modchip](https://github.com/synchopate/llama-modchip)

---

## Memory Footprint, Operating Regimes, & Continuous Quality (`v2.5.0`)

| Metric / Operating Regime (16C AMD Zen 5, 64 GB DDR5) | `Fractal-64` (`56.57 GiB`) | **`LUT-VQ3-64` Hybrid (`56.40 GiB` Shipped / `56.07 GiB` `--aggressive`)** | Notes |
| :--- | :---: | :---: | :--- |
| **GGUF File Size (3 Shards)** | `56.57 GiB` | **`56.40 GiB`** (`60,558,856,608` B) / `56.07 GiB` | Fits in 64 GB DDR5 (`0` swap) |
| **Peak Resident RAM (`VmRSS`, `-c 32768`)** | `58.20 GiB` | **`58.31 GiB`** (`~4.2 GiB` OS headroom) | Safe default on 64 GB GUI desktops (`~62.5 GiB` usable) |
| **Peak Resident RAM (`VmRSS`, `-c 262144`)** | `58.80 – 59.40 GiB` | **`58.90 – 59.50 GiB`** | Dedicated headless 64 GB server or 96 GB+ host |
| **Cold Prompt Prefill (`pp128`, 0 cached)** | `37.79 tok/s` | **`41.11 tok/s`** (`+8.8%`) | `43.98 tok/s` at `pp580`; `30.7–33.1 tok/s` at `3,038` tok |
| **Warm Prefix Reuse Prefill (`cache_prompt=True`)** | `44.36 tok/s` | **`44.36 tok/s`** (`339` new / `403` total KV) | `38.44 tok/s` at `2,418` KV tokens |
| **Raw Pure Decode (`tg64`, `N=1`, Unpruned `Top-6`)** | `7.74 – 9.55 tok/s` | **`9.05 tok/s`** (`+16.9%` on full-VQ3 ablation) | Zero expert dropping by default (`10.74` w/ `Top-4 L2`) |
| **Speculative / Agentic Burst (`Stockfish-PVS`)** | `11.5 – 21.2 tok/s` | **`19.4 – 25.3 tok/s`** (`+20.1%`) | `Stockfish-PVS` + multi-turn tool-call bursts |
| **Continuous `llama-perplexity` (`4,096` tok)** | `1.8284 ± 0.07038` | **`1.8258 ± 0.06978`** (`1.7438` direct from FP8) | Reproducible via `benchmarks/eval_perplexity_and_kld.py` |
| **Agentic Tool-Calling & Needle Gate** | `64.0 / 80` (`24/30` Tools) | **`66.0 / 80`** (`30/30` Tools, `20/20` Needle, `40/60` Hard-60) | Full breakdown in `benchmarks/QUALITY_GATE_VQ3_64.md` |

> **Why `DeepSeek-V4-Flash-0731` (and not `V4.1-Flash-1017`) for 64 GB Resident CPU Inference:** `DeepSeek-V4.1-Flash-1017` (`DeepseekV41ForCausalLM`, `510.3 GB` source) adds **`188.83 GiB` of native `FP8` Engram lookup tables** on top of **`151.77 GiB`** of `Q2` weights (**`340.60 GiB` total even at 2-bit quantization**), requiring `128 GB+` RAM with NVMe SSD table streaming or `512 GB` RAM for resident execution. `DeepSeek-V4-Flash-0731` (`284.33B`, zero Engram tables) is the largest DeepSeek-V4 Flash checkpoint that executes **100% resident inside 64 GB RAM**.

---

## Hybrid Quantization Layout (`Fractal-64` MLA + `LUT-VQ3-64` Interior Shared Experts)

Built via [`scripts/build_dsv4_vq3.py`](https://github.com/synchopate/llama-modchip/blob/main/scripts/build_dsv4_vq3.py):

1. **100% of MLA & Sparse Compressor Projections Preserved in `Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `3.75 bpw`, `30 B / 64 W`):**
   Every `attn_q_b`, `attn_output_a`, `attn_output_b`, `attn_kv`, `attn_compressor_*`, and `indexer_compressor_*` projection stays in `Fractal-64` (`17.75 dB` SQNR, `99.15%` Cosine Similarity vs `BF16`) with zero double-quantization.
2. **Boundary Shared Experts (`blk.0..4`, `blk.39..42`) & All Output Projections (`blk.*.ffn_down_shexp.weight`) Preserved in `Q5_K` / `Q6_K`:**
   Keeps the first 5 layers, final 4 layers, and every shared-expert down-projection writing into the 4-stream residual stream at `Q5_K` / `Q6_K`.
3. **68 Interior Shared-Expert Pre-`SiLU` Projections (`blk.5..38` `ffn_gate_shexp` & `ffn_up_shexp`) -> `LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `3.00 bpw`, `24 B / 64 W`):**
   Encoded into our 128-centroid 3D Lloyd-Max codebook (`2 B` `FP16` scale `d` + `22 B` `qs[22]` packing `20` 7-bit 3D stride-22 indices, `2` 6-bit 2D tail indices, and `24` high bits of `8 × 3-bit` sub-scales `{2..9}`; `13.57 dB` SQNR / `+3.42 dB` over `Q2_K`), decoded in 6 AVX-512 ZMM registers (`384 B` LUT) via `VPERMI2B` (`_mm512_permutex2var_epi8`) with portable AVX2 + BMI2 and scalar fallbacks.
4. **256 Routed MoE Experts (`blk.*.ffn_{gate,up,down}_exps.weight`):**
   Analytical second-order `SwiGLU-H2` `IQ1_S` / `IQ1_M` / `IQ2_XXS` (`Top-6` unpruned routing).

---

## Quickstart (Linux `x86_64` & Windows 11)

- **Matching MLA KV Cache Types:** Always pass matching K and V cache types (`-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`); DeepSeek-V4 MLA shares a single latent KV tensor (`576` dim/layer) and rejects asymmetric `q8_0/q4_0`.
- **64 GB OS Headroom (`-c 32768` vs `-c 262144`):** Use `-c 32768` on 64 GB desktop systems (`~58.3 GiB` `VmRSS`, leaving `~4.2 GiB` OS/GUI headroom in `~62.5 GiB` usable Linux RAM). Use `-c 262144` (`256K` tokens) on dedicated headless 64 GB nodes (with `zram`/swap) or `96 GB+` workstations.
- **Unpruned Top-6 Routing:** Do not set `LLAMA_MOE_TOP_K` or `DSV4_MOE_TOPK` environment variables at load time.

### Linux (`-avx512` and `-avx2` Prebuilt Binaries Included)
```bash
hf download Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF --local-dir ./dsv4-vq3
cd ./dsv4-vq3

# Auto-selects -avx512 (Zen 4/5, Ice Lake+) or -avx2 (Haswell, Zen 2/3, Alder/Raptor Lake — zero SIGILL):
BIN="./llama-server-modchip-linux-x86_64-$(grep -q 'avx512_vbmi' /proc/cpuinfo && echo avx512 || echo avx2)"
chmod +x "$BIN"

# 64 GB Desktop safe default (-c 32768); change to -c 262144 on a dedicated headless 64GB / 96GB+ server:
"$BIN" \
  -m DeepSeek-V4-Flash-0731-VQ3-00001-of-00003.gguf \
  -c 32768 -t 16 -tb 16 -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 --port 8080 --jinja
```

### Windows 11 (`1M`-Context Verified on 64 GB DDR5)
Clone [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip), place all three `.gguf` shards in `windows\models\`, build the `v2.5.0` engine via MSYS2 UCRT64 (`powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1 -Target dsv4`), and launch `windows\AVVIA_DSV4.bat`.
