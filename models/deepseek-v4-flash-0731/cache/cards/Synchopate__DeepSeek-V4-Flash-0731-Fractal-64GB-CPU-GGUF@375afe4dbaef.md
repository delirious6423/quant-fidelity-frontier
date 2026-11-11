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
  - quantization
  - llama-cpp
  - claude-code
  - avx512
  - local-llm
pipeline_tag: text-generation
quantized_by: Synchopate
---

# DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF (`284.33B` in `56.57 GiB`)

3-shard `SwiGLU-H2 + Fractal-64` GGUF checkpoint (`60,739,342,752` bytes = **`56.57 GiB` / `60.74 GB`**) of **`DeepSeek-V4-Flash-0731`** (`284.33B` total / `13.17B` active parameters) engineered to keep 100% of weights and the `256K` MLA KV cache resident inside a single **64 GB DDR5** Linux, Windows 11, or Apple Silicon (`M5 Max 64 GB`) machine with zero swap.

- **Engine, AVX-512BW/AVX2 Kernels, Windows 11 `1M`-Context Port, & CLI (`./modchip claude`):** [https://github.com/synchopate/llama-modchip](https://github.com/synchopate/llama-modchip)

---

## Memory Footprint & Verified Performance on 64 GB Hosts

`DeepSeek-V4-Flash-0731` consists of **`43` layers** (`3` deterministic token-ID hash-routed layers `0..2` via `ffn_gate_tid2eid` + `40` learned-router layers `3..42`), `256` routed experts + `1` shared expert per layer with **`Top-6` active** (`13.17B` active parameters), Multi-head Latent Attention (`64` query heads, `1` latent KV head `512`, `wo_a` + `wo_b`), `CSA/LID/HCA` Sparse Compressors, and 4-stream Sinkhorn-20 Hyper-Connections (`hc=4`).

Standard 4-shard `IQ2_XXS` GGUF conversions occupy **`76.87 GiB` on disk**, exceeding physical memory on a 64 GB workstation (`~62.8 GiB` usable RAM) and forcing NVMe swap thrashing on every token. Transcoding via [`scripts/transcode_dsv4.c`](https://github.com/synchopate/llama-modchip/blob/main/scripts/transcode_dsv4.c) + [`scripts/transcode_dsv4_shard.py`](https://github.com/synchopate/llama-modchip/blob/main/scripts/transcode_dsv4_shard.py) reduces the checkpoint to **`56.57 GiB` (`-20.30 GiB` / `-26.4%`)**:

| Metric | 4-Shard `IQ2_XXS` (`76.87 GiB`) | **`DeepSeek-V4-Flash-Fractal-64GB` (`56.57 GiB`)** | Platform / Notes |
| :--- | :---: | :---: | :--- |
| **GGUF File Size (3 Shards)** | `76.87 GiB` *(OOM on 64 GB)* | **`56.57 GiB`** (`60,739,342,752` B) | `-20.30 GiB` (`-26.4%`) |
| **Resident Memory (`VmRSS`, `256K` Context)** | `> 64 GiB` *(SSD swap)* | **`57.6 – 58.9 GiB`** (`0` Swap) | 100% DDR5 / Unified Memory resident |
| **Pure CPU Prefill (`16C` Zen 5 Ryzen 9 9955HX)** | `25.05 tok/s` | **`43.16 – 44.36 tok/s`** | `+77.1%` (4-level cache tiling + `CSA/LID` skip) |
| **Pure CPU Decode (`16C` Zen 5 Ryzen 9 9955HX)** | `5.91 – 6.41 tok/s` | **`9.06 – 10.74 tok/s`** (`11.5–14.7` w/ spec) | Lossless `Top-6` (`9.06`) to `Top-4 L2` (`10.74`) |
| **Windows 11 `1M`-Context (`8C` Ryzen 7 9800X3D)** | OOM | **`21.3–22.6 tok/s` `pp512` / `6.35–8.04 tok/s` `tg128`** | Verified up to `1,048,576` tokens (`@MR1KRAKEN`) |
| **Apple M5 Max (`64 GB` Unified Memory)** | OOM | **`169.4 tok/s` `pp512` / `27.0 tok/s` `tg128`** | Metal `Q2_0` kernel, `0` swapouts |

---

## Quantization Architecture (`SwiGLU-H2` + `Fractal-64`)

1. **Analytical Second-Order `SwiGLU-H2` MoE Compression (`blk.*.ffn_{gate,up,down}_exps.weight` -> `IQ1_S` / `IQ1_M` / `IQ2_XXS`):**
   Weights super-block optimization directly by closed-form coupled SwiGLU row norms $I_{e,j} = \|W_{\text{gate}}^{(e)}[j,:]\|_2^2 \cdot \|W_{\text{up}}^{(e)}[j,:]\|_2^2$, preserving cross-projection fidelity across all 256 routed experts without external calibration data.
2. **Block-64 `Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `30 B / 64 W = 3.75 bpw`) MLA & Sparse Compressor Transcoding:**
   Transcodes dense `Q8_0` MLA and compressor projections (`attn_q_b`, `attn_output_a`, `attn_output_b`, `attn_compressor_gate`, `attn_compressor_kv`, `attn_kv`, `indexer_compressor_gate`, `indexer_compressor_kv`) into 64-weight `block_q2_0` blocks (`2 B` `FP16` scale `d` + `4 B` `8 × 4-bit` sub-scales `{4..19}` + `24 B` 3-bit indices into `{-14, -8, -4, -1, +1, +4, +8, +14}`, `17.75 dB` SQNR).
3. **Shared Experts, Routers, & Embeddings Preserved in `Q5_K` / `Q6_K` / `F32`:**
   All 43 shared experts (`ffn_{gate,up,down}_shexp`), router gates, RMSNorms, Sinkhorn-20 Hyper-Connection parameters, and embeddings remain at `Q5_K` / `Q6_K` / `F32`.

---

## Quickstart

> **Runtime Requirements (`GGML_TYPE_Q2_0 = 42` & `deepseek4` MLA):**
> - Requires the `llama-modchip` engine (`GGML_TYPE_Q2_0 = 42` + custom `deepseek4` MLA, Hadamard, and Sinkhorn-20 `HC_PRE/POST` operators). Unpatched `llama.cpp`, Ollama, or LM Studio will fail with `unknown tensor type 42`.
> - **Matching MLA KV Cache Types:** Always pass matching K and V cache types (`-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`).
> - **Native Top-6 Routing:** Do not set `LLAMA_MOE_TOP_K` or `DSV4_MOE_TOPK` environment variables at load time.

### Option A: Bundled Linux `x86_64` Binary
```bash
hf download Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF --local-dir ./DeepSeek-V4-Fractal
cd ./DeepSeek-V4-Fractal
chmod +x llama-server-modchip-linux-x86_64 llama-cli-modchip-linux-x86_64

./llama-server-modchip-linux-x86_64 \
  -m DeepSeek-V4-Flash-0731-Fractal-00001-of-00003.gguf \
  -c 32768 -t 16 -tb 16 -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --port 8080 --jinja
```

### Option B: Full `llama-modchip` Suite (Linux, Windows 11 `AVVIA_DSV4.bat`, Apple Silicon `run-dsv4.sh`, & Claude Code)
```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./modchip start /path/to/DeepSeek-V4-Flash-0731-Fractal-00001-of-00003.gguf
```
