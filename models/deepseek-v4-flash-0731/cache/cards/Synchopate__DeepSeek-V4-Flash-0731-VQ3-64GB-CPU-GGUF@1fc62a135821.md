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
> **Patched `llama-modchip` Runtime Required (`GGML_TYPE_Q2_0 = 42`, `GGML_TYPE_VQ3_0 = 43`, + `deepseek4`):**
> Do not load this checkpoint in vanilla `Ollama`, `LM Studio`, `vLLM`, or unpatched `llama.cpp`. `Fractal-64` (`type 42`, `30 B / 64 W`) collides with deprecated `GGML_TYPE_IQ4_NL_4_4` (`18 B / 32 W`) and `LUT-VQ3-64` appends custom `type 43` (`24 B / 64 W`; ID `40` in `v2.4`), causing unpatched engines to fail with `unknown tensor type 43` or tensor size mismatch. Use the `GLIBC_2.30` portable Linux `x86_64` binaries included in this repository (`-avx512` or `-avx2`, verified via `SHA256SUMS`) or run `./modchip setup` from [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip).

Compact 3-shard hybrid GGUF checkpoint (`60,558,856,608` bytes = **`56.40 GiB` / `60.56 GB`**) of **`DeepSeek-V4-Flash-0731`** (`284.33B` total / `13.17B` active parameters; `43` layers, `256` routed + `1` shared expert/layer, unpruned `Top-6` active) combining `Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `3.75 bpw`) on MLA projections with `LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `3.00 bpw`) on 68 interior shared-expert pre-`SiLU` projections for lowest RAM footprint and fastest `AVX-512 VBMI` `VPERMI2B` throughput.

- **Maximum-Quality Aligned Alternative (`56.93 GiB`, Single-Pass `Q6_K`):** [`Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF) (`./modchip setup` default `dsv4-maxq`)
- **Engine & CLI (`v2.5.1`):** [https://github.com/synchopate/llama-modchip](https://github.com/synchopate/llama-modchip)

---

## Memory Footprint, Operating Regimes, & Continuous Quality (`v2.5.1`)

| Metric / Operating Regime (16C AMD Zen 5, 64 GB DDR5) | `Fractal-64` (`56.57 GiB`) | **`LUT-VQ3-64` (`56.40 GiB` Shipped)** | `MaxQ-Fractal` (`56.93 GiB` Single-Pass) |
| :--- | :---: | :---: | :---: |
| **GGUF File Size (3 Shards)** | `56.57 GiB` | **`56.40 GiB`** (`60,558,856,608` B) | `56.93 GiB` (`61,129,363,968` B) |
| **Anonymous-Heap `VmRSS` (`-c 32768 -ub 512`)** | `58.20 GiB` | **`58.31 – 58.61 GiB`** (`~4.0 GiB` margin) | `59.13 GiB` |
| **Cold Prompt Prefill (`pp128`, 0 cached)** | `37.79 tok/s` | **`41.11 tok/s`** (`43.98` at `pp580`) | `37.80 tok/s` |
| **Warm Prefix Reuse Prefill (`cache_prompt=True`)** | `44.36 tok/s` | **`44.36 tok/s`** (`339` new / `403` KV) | `44.00 tok/s` |
| **Raw Pure Decode (`tg64`, `N=1`, Unpruned `Top-6`)** | `9.55 tok/s` | **`9.05 – 9.15 tok/s`** (`10.74` w/ `PVS` on `56.07 GiB` `--aggressive`) | `8.75 tok/s` |
| **Speculative / Agentic Burst (`Stockfish-PVS`)** | `11.5 – 21.2 tok/s` | **`19.4 – 25.3 tok/s`** | `11.2 – 21.0 tok/s` |
| **`WikiText-2` (`wiki.test.raw`, `8,192` tok) PPL** | *(transcoded)* | **`12.2345 ± 0.59759`** | `9.3896 ± 0.40764` |
| **2-Pass Logit KLD vs. `MaxQ-Fractal` (`T=1.0`)** | — | **`0.8302 nats/tok (75.0% Top-1)`** | `0.0000 nats/tok` (`Ref`) |
| **80-Point Functional Gate (`Tools / Code / Needle`)** | `64.0 / 80` (`24/20/20`) | **`64.0 / 80`** (`24/20/20`; `70.0 / 80` on `--aggressive` `56.07 GiB`) | `70.0 / 80` *(single-pass `Q6_K`)* |
| **Extended `Hard-60` Coding/Math (`60` pts)** | `20.0 / 60` | **`30.0 / 60`** | `40.0 / 60` |

> **Why `DeepSeek-V4-Flash-0731` (and not `V4.1-Flash-1017`) for 64 GB Resident CPU Inference:** `DeepSeek-V4.1-Flash-1017` (`DeepseekV41ForCausalLM`, `510.3 GB` source) adds **`188.83 GiB` of native `FP8` Engram lookup tables** on top of **`151.77 GiB`** of `Q2` weights (**`340.60 GiB` total even at 2-bit quantization**), requiring `128 GB+` RAM with NVMe SSD table streaming or `512 GB` RAM for resident execution.

---

## Hybrid Quantization Layout (`scripts/build_dsv4_vq3.py`)

1. **MLA & Lightning Indexer Projections -> `Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `3.75 bpw`, `30 B / 64 W`):** `17.75 dB` SQNR (`99.15%` Cosine Similarity vs `BF16`).
2. **Boundary Shared Experts (`blk.0..4`, `blk.39..42`) & All `ffn_down_shexp` -> `Q5_K` / `Q6_K`.**
3. **68 Interior Shared-Expert Pre-`SiLU` Projections (`blk.5..38` `ffn_gate_shexp` & `ffn_up_shexp`) -> `LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `3.00 bpw`, `24 B / 64 W`):** `2 B` `FP16` scale `d` + `22 B` `qs[22]` (`20` 7-bit 3D stride-22 Lloyd-Max indices + `2` 6-bit 2D tail indices + `24` high bits of `8 × 3-bit` sub-scales `{2..9}`; `13.57 dB` SQNR), decoded via `AVX-512 VBMI` `VPERMI2B` with portable `AVX2` fallback.
4. **256 Routed MoE Experts:** `SwiGLU-H2` `IQ1_S` / `IQ1_M` / `IQ2_XXS` (`Top-6` unpruned routing).

---

## Quickstart

- **Matching MLA KV Cache Types:** Always pass matching `-ctk` and `-ctv` (`-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`).
- **64 GB Anonymous-Heap Headroom:** Use `-c 32768 -ub 512` on 64 GB desktop systems (`~58.3–58.6 GiB` anonymous-heap `VmRSS`). Reserve `-c 262144` for dedicated headless 64 GB nodes (with `zram`/swap) or `96 GB+` hosts.

```bash
hf download Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF --local-dir ./dsv4-vq3
cd ./dsv4-vq3
sha256sum -c SHA256SUMS --ignore-missing

BIN="./llama-server-modchip-linux-x86_64-$(grep -Eq 'avx512_?vbmi' /proc/cpuinfo && echo avx512 || echo avx2)"
chmod +x "$BIN"

"$BIN" \
  -m DeepSeek-V4-Flash-0731-VQ3-00001-of-00003.gguf \
  -c 32768 -t 16 -tb 16 -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --port 8080 --jinja
```
