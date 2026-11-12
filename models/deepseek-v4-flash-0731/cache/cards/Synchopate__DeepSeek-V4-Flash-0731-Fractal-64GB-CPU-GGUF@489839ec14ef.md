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
  - quantization
  - claude-code
  - avx512
  - avx2
  - local-llm
pipeline_tag: text-generation
quantized_by: Synchopate
---

# DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF (`284.33B` in `56.57 GiB`)

> [!IMPORTANT]
> **Custom Tensor Type (`GGML_TYPE_Q2_0 = 42`) — Do NOT use vanilla `Ollama`, `LM Studio`, `vLLM`, or unpatched `llama.cpp`:**
> Upstream engines do not define custom block-64 `Fractal-64` (`type 42`) or the `deepseek4` MLA / Hadamard / Sinkhorn-20 operators and will fail immediately with `unknown tensor type 42` (or `40`). Use the ISA-matched prebuilt binaries included in this repository (`llama-server-modchip-linux-x86_64-avx512` for Zen 4/5 & Ice Lake+, or `llama-server-modchip-linux-x86_64-avx2` for Haswell / Skylake / Zen 2/3 / Alder & Raptor Lake with zero `%zmm` instructions) or build from [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip).

3-shard `SwiGLU-H2 + Fractal-64` GGUF checkpoint (`60,739,342,752` bytes = **`56.57 GiB` / `60.74 GB`**) of **`DeepSeek-V4-Flash-0731`** (`284.33B` total / `13.17B` active parameters), sized to keep 100% of weights resident inside a single **64 GB DDR5** Linux, Windows 11, or Apple Silicon (`M5 Max 64 GB`) machine without SSD streaming.

- **Engine, AVX-512/AVX2/Metal Kernels, Windows 11 `1M`-Context Port, & CLI (`./modchip`):** [https://github.com/synchopate/llama-modchip](https://github.com/synchopate/llama-modchip)

---

## Memory Footprint, Operating Regimes, & Continuous Quality (`v2.5.0`)

`DeepSeek-V4-Flash-0731` consists of **`43` layers** (`3` deterministic token-ID hash-routed layers `0..2` via `ffn_gate_tid2eid` + `40` learned-router layers `3..42`), `256` routed experts + `1` shared expert per layer with **`Top-6` active** (`13.17B` active parameters), Multi-head Latent Attention (`64` query heads, `1` latent KV head `512`, `wo_a` + `wo_b`), `CSA/LID/HCA` Sparse Compressors, and 4-stream Sinkhorn-20 Hyper-Connections (`hc=4`).

| Metric / Operating Regime | 4-Shard `IQ2_XXS` (`76.87 GiB`) | **`DeepSeek-V4-Flash-Fractal-64GB` (`56.57 GiB`)** | Platform / Notes |
| :--- | :---: | :---: | :--- |
| **GGUF File Size (3 Shards)** | `76.87 GiB` *(OOM on 64 GB)* | **`56.57 GiB`** (`60,739,342,752` B) | `-20.30 GiB` (`-26.4%`) |
| **Peak Resident RAM (`VmRSS`, `-c 32768`)** | `> 64 GiB` *(SSD swap)* | **`58.20 GiB`** (`~4.3 GiB` OS headroom) | Safe default on 64 GB GUI desktops (`~62.5 GiB` usable) |
| **Peak Resident RAM (`VmRSS`, `-c 262144`)** | `> 64 GiB` | **`58.80 – 59.40 GiB`** | Dedicated headless 64 GB server or 96 GB+ host |
| **Cold Prompt Prefill (`pp128–pp580`, 0 cached)** | `24.73 – 25.19 tok/s` | **`37.79 – 43.98 tok/s`** | 4-level cache tiling (`30.7–33.1 tok/s` at `3,038` tok) |
| **Warm Prefix Reuse Prefill (`cache_prompt=True`)** | `25.12 tok/s` | **`44.36 tok/s`** (`339` new / `403` total KV) | `38.44 tok/s` at `2,418` KV tokens |
| **Raw Pure Decode (`N=1`, Unpruned `Top-6`)** | `5.91 tok/s` | **`9.06 – 9.55 tok/s`** (`10.74` w/ `Top-4 L2`) | Zero expert dropping by default (`1.00000` routing CosSim) |
| **Speculative / Agentic Burst (`Stockfish-PVS`)** | N/A | **`11.5 – 21.2 tok/s`** | Multi-token `Stockfish-PVS` + tool-call boilerplates |
| **Continuous `llama-perplexity` (`4,096` tok)** | — | **`1.8284 ± 0.07038`** | Reproducible via `benchmarks/eval_perplexity_and_kld.py` |
| **Windows 11 `1M`-Context (`8C` Ryzen 7 9800X3D)** | OOM | **`21.3–22.6 tok/s` `pp512` / `6.35–8.04 tok/s` `tg128`** | Verified up to `1,048,576` tokens (`@MR1KRAKEN`) |
| **Apple M5 Max (`64 GB` Unified Memory)** | OOM | **`169.4 tok/s` `pp512` / `27.0 tok/s` `tg128`** | Metal `Q2_0` kernel, `0` swapouts (`20.7–23.6` served) |

> **Why `DeepSeek-V4-Flash-0731` (and not `V4.1-Flash-1017`) for 64 GB Resident CPU Inference:** `DeepSeek-V4.1-Flash-1017` (`DeepseekV41ForCausalLM`, `510.3 GB` source) adds **`188.83 GiB` of native `FP8` Engram lookup tables** on top of **`151.77 GiB`** of `Q2` weights (**`340.60 GiB` total even at 2-bit quantization**), requiring `128 GB+` RAM with NVMe SSD table streaming or `512 GB` RAM for resident execution. `DeepSeek-V4-Flash-0731` (`284.33B`, zero Engram tables) is the largest DeepSeek-V4 Flash checkpoint that executes **100% resident inside 64 GB RAM**.

---

## Quantization Architecture (`SwiGLU-H2` + `Fractal-64`)

1. **Analytical Second-Order `SwiGLU-H2` MoE Compression (`blk.*.ffn_{gate,up,down}_exps.weight` -> `IQ1_S` / `IQ1_M` / `IQ2_XXS`):**
   Weights super-block optimization by coupled SwiGLU row norms $I_{e,j} = \|W_{\text{gate}}^{(e)}[j,:]\|_2^2 \cdot \|W_{\text{up}}^{(e)}[j,:]\|_2^2$ across all 256 routed experts.
2. **Block-64 `Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `30 B / 64 W = 3.75 bpw`) MLA & Sparse Compressor Transcoding:**
   Encodes MLA and compressor projections (`attn_q_b`, `attn_output_a`, `attn_output_b`, `attn_compressor_gate`, `attn_compressor_kv`, `attn_kv`, `indexer_compressor_gate`, `indexer_compressor_kv`) into 64-weight `block_q2_0` blocks (`2 B` `FP16` scale `d` + `4 B` `8 × 4-bit` sub-scales `{4..19}` + `24 B` 3-bit indices into `{-14, -8, -4, -1, +1, +4, +8, +14}`, `17.75 dB` SQNR / `99.15%` Cosine Similarity vs. `BF16`).
3. **Shared Experts, Routers, & Embeddings Preserved in `Q5_K` / `Q6_K` / `F32`:**
   All 43 shared experts (`ffn_{gate,up,down}_shexp`), router gates, RMSNorms, Sinkhorn-20 Hyper-Connection parameters, and embeddings remain at `Q5_K` / `Q6_K` / `F32`.

---

## Quickstart

- **Matching MLA KV Cache Types:** Always pass matching K and V cache types (`-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`).
- **64 GB OS Headroom (`-c 32768` vs `-c 262144`):** Use `-c 32768` on 64 GB desktop systems (`~58.2 GiB` `VmRSS`, preserving `~4.3 GiB` free RAM for GUI/OS to prevent Linux OOM kills). Use `-c 262144` (`256K` tokens) on dedicated headless 64 GB nodes (with `zram`/swap) or `96 GB+` workstations.
- **Unpruned Top-6 Routing:** Do not set `LLAMA_MOE_TOP_K` or `DSV4_MOE_TOPK` environment variables at load time.

### Option A: Bundled ISA-Matched Linux `x86_64` Binary (`-avx512` or `-avx2`)
```bash
hf download Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF --local-dir ./DeepSeek-V4-Fractal
cd ./DeepSeek-V4-Fractal

# Auto-selects -avx512 (Zen 4/5, Ice Lake+) or -avx2 (Haswell, Zen 2/3, Alder/Raptor Lake — zero SIGILL):
BIN="./llama-server-modchip-linux-x86_64-$(grep -q 'avx512_vbmi' /proc/cpuinfo && echo avx512 || echo avx2)"
chmod +x "$BIN"

# 64 GB Desktop safe default (-c 32768); change to -c 262144 on a dedicated headless 64GB / 96GB+ server:
"$BIN" \
  -m DeepSeek-V4-Flash-0731-Fractal-00001-of-00003.gguf \
  -c 32768 -t 16 -tb 16 -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --port 8080 --jinja
```

### Option B: Full `llama-modchip` Suite (Linux, Windows 11 `AVVIA_DSV4.bat`, Apple Silicon `run-dsv4.sh`, & Claude Code)
```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
python3 scripts/check_cpu_readiness.py --model 284b
./modchip start /path/to/DeepSeek-V4-Flash-0731-Fractal-00001-of-00003.gguf
```
