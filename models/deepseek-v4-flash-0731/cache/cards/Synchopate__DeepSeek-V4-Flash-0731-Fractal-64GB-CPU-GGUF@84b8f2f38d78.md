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

# DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF (`284.33B` in `56.57 GiB` — Pure `Fractal-64` Cross-Platform Edition)

> [!IMPORTANT]
> **Patched `llama-modchip` Runtime Required (`GGML_TYPE_Q2_0 = 42` + `deepseek4` MLA/Sinkhorn-20):**
> Do not load this checkpoint in vanilla `Ollama`, `LM Studio`, `vLLM`, or unpatched `llama.cpp`. `Fractal-64` (`30 bytes / 64 weights`) repurposes upstream `GGML_TYPE_Q2_0 = 42` (`18 bytes / 32 weights`) in-place, so unpatched `llama.cpp` fails with a tensor bounds mismatch or `unknown model architecture: 'deepseek4'`. Use the `GLIBC_2.30` portable Linux `x86_64` binaries included in this repository (`-avx512` for Zen 4/5 & Ice Lake+, or `-avx2` for Haswell / Skylake / Zen 2/3 / Alder & Raptor Lake with zero `%zmm` or `AVX512_BF16` instructions; requires OpenMP `libgomp1`) or run `./modchip setup` from [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip).

3-shard `SwiGLU-H2 + Fractal-64` GGUF checkpoint (`60,739,342,752` bytes = **`56.57 GiB` / `60.74 GB`**) of **`DeepSeek-V4-Flash-0731`** (`284.33B` total / `13.17B` active parameters; `43` layers, `256` routed + `1` shared expert/layer, unpruned `Top-6` active) compatible across Linux `x86_64` (`AVX-512` / `AVX2`), Windows 11 (`1M` context verified), and Apple Silicon Metal (`M5 Max 64 GB`).

- **Single-Pass Upgraded Edition (`56.93 GiB`, `174` Dense Tensors in `Q6_K`):** [`Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF) (`./modchip setup` default `dsv4-maxq`)
- **Compact `LUT-VQ3-64` Hybrid (`56.40 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF) (`--preset dsv4-vq3`)
- **Opt-In Uncensored Variant (`56.93 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF) (`--preset dsv4-ablit`)
- **Engine & CLI (`v2.5.2`):** [https://github.com/synchopate/llama-modchip](https://github.com/synchopate/llama-modchip)

---

## Memory Footprint, Operating Regimes, & Platform Matrix (`v2.5.2`)

| Metric / Operating Regime | 4-Shard `UD-IQ1_S` (`76.87 GiB`) | **`Fractal-64` (`56.57 GiB`)** | `MaxQ-Fractal` (`56.93 GiB` Single-Pass) |
| :--- | :---: | :---: | :---: |
| **GGUF File Size (3 Shards)** | `76.87 GiB` *(OOM on 64 GB)* | **`56.57 GiB`** (`60,739,342,752` B) | `56.93 GiB` (`61,129,363,936` B) |
| **Anonymous-Heap `VmRSS` (`-c 32768 -ub 512`)** | `> 64 GiB` *(SSD swap)* | **`58.20 GiB`** (`~4.3 GiB` margin) | `57.54–59.13 GiB` |
| **Cold Prompt Prefill (`pp128–pp580`, 0 cached)** | `24.73 – 25.19 tok/s` | **`37.79 – 43.98 tok/s`** | `37.80 – 43.50 tok/s` |
| **Warm Prefix Reuse Prefill (`cache_prompt=True`)** | `25.12 tok/s` | **`44.36 tok/s`** (`339` new / `403` KV) | `44.00 tok/s` |
| **Raw Pure Decode (`N=1`, Unpruned `Top-6`)** | `5.91 tok/s` | **`9.06 – 9.55 tok/s`** | `8.75 tok/s` |
| **Speculative / Agentic Burst (`Stockfish-PVS`)** | N/A | **`11.5 – 21.2 tok/s`** | `11.2 – 21.0 tok/s` |
| **`AtomicChat` `wiki-alt.txt` (`ctx=5632`, `12` chunks) PPL** | `5.9734` *(51c)* | *(see `MaxQ-Fractal`)* | **`7.5133 ± 0.1054`** |
| **`WikiText-2` (`wiki.test.raw`, `8,192` tok) PPL** | — | *(see `MaxQ-Fractal`)* | **`11.7434 ± 0.57170`** |
| **80-Point Functional Gate (`Tools / Code / Needle`)** | — | **`64.0 / 80.0`** | **`64.0 / 80.0`** |
| **Windows 11 `1M`-Context (`8C` Ryzen 7 9800X3D)** | OOM | **`21.3–22.6 tok/s` `pp512` / `6.35–8.04 tok/s` `tg128`** | Compatible |
| **Apple M5 Max (`64 GB` Unified Memory)** | OOM | **`169.4 tok/s` `pp512` / `27.0 tok/s` `tg128`** (`0` swapouts) | Compatible (`Q2_0` Metal) |

---

## Quantization Architecture (`SwiGLU-H2` + `Fractal-64`)

1. **Routed MoE (`blk.*.ffn_{gate,up,down}_exps.weight` -> `IQ1_S` / `IQ1_M` / `IQ2_XXS`):** Second-order `SwiGLU-H2` coupled row-norm importance weighting across all 256 routed experts.
2. **MLA & Lightning Indexer (`GGML_TYPE_Q2_0 = 42`, `30 B / 64 W = 3.75 bpw`):** `2 B` `FP16` scale `d` + `4 B` `8 × 4-bit` sub-scales `{4..19}` + `24 B` 3-bit indices into `{-14, -8, -4, -1, +1, +4, +8, +14}` (`17.75 dB` SQNR vs `BF16`).
3. **Shared Experts, Routers, & Embeddings:** `Q5_K` / `Q6_K` / `F32` (for single-pass `Q6_K` across all 174 dense bottlenecks, see `MaxQ-Fractal`).

---

## Quickstart

- **Matching MLA KV Cache Types:** Always pass `-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`.
- **64 GB Anonymous-Heap Headroom:** Use `-c 32768 -ub 512` on 64 GB desktop systems (`~58.2 GiB` anonymous-heap `VmRSS`). Reserve `-c 262144` for headless 64 GB nodes with `zram`/swap or `96 GB+` hosts.

```bash
hf download Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF --local-dir ./DeepSeek-V4-Fractal
cd ./DeepSeek-V4-Fractal
sha256sum -c SHA256SUMS --ignore-missing

BIN="./llama-server-modchip-linux-x86_64-$(grep -Eq 'avx512_?vbmi' /proc/cpuinfo && echo avx512 || echo avx2)"
chmod +x "$BIN"

"$BIN" \
  -m DeepSeek-V4-Flash-0731-Fractal-00001-of-00003.gguf \
  -c 32768 -t 16 -tb 16 -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --port 8080 --jinja
```
