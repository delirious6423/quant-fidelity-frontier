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

# DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF (`284.33B` in `56.40 GiB` — Compact Hybrid `Fractal-64 + LUT-VQ3-64`)

> [!IMPORTANT]
> **Patched `llama-modchip` Runtime Required (`GGML_TYPE_Q2_0 = 42`, `GGML_TYPE_VQ3_0 = 43`, + `deepseek4`):**
> Do not load this checkpoint in vanilla `Ollama`, `LM Studio`, `vLLM`, or unpatched `llama.cpp`. `Fractal-64` (`type 42`, `30 B / 64 W`) repurposes upstream `GGML_TYPE_Q2_0 = 42` (`18 B / 32 W`) and `LUT-VQ3-64` appends custom `type 43` (`24 B / 64 W`; ID `40` in `v2.4`), causing unpatched engines to fail with `unknown tensor type 43` or tensor size mismatch. Use the IPC-hardened `GLIBC_2.30` portable Linux `x86_64` binaries (`v2.5.4`) included in this repository (`-avx512` or `-avx2`, verified via `SHA256SUMS`; requires OpenMP `libgomp1`) or run `./modchip setup` from [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip).

Compact 3-shard hybrid GGUF checkpoint (`60,558,856,960` bytes = **`56.40 GiB` / `60.56 GB`**) of **`DeepSeek-V4-Flash-0731`** (`284.33B` total / `13.17B` active parameters; `43` layers, `256` routed + `1` shared expert/layer, unpruned `Top-6` active) combining `Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `3.75 bpw`) on MLA projections with `LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `3.00 bpw`) on 68 interior shared-expert pre-`SiLU` projections for lowest RAM footprint and fastest `AVX-512 VBMI` `VPERMI2B` throughput.

- **Donor-Free Single-Pass `Q6_K` Aligned Default (`56.93 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF) (`./modchip setup` default `dsv4-maxq`)
- **Cross-Platform Pure `Fractal-64` (`56.57 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF) (`--preset dsv4-fractal`)
- **Opt-In Uncensored Variant (`56.93 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF) (`--preset dsv4-ablit`)
- **Engine & CLI (`v2.5.4`):** [https://github.com/synchopate/llama-modchip](https://github.com/synchopate/llama-modchip)

---

## Memory Footprint, Operating Regimes, & Continuous Quality (`v2.5.4`)

| Metric / Operating Regime (16C AMD Zen 5, 64 GB DDR5) | `Fractal-64` (`56.57 GiB`) | **`LUT-VQ3-64` (`56.40 GiB` Shipped)** | `MaxQ-Fractal` (`56.93 GiB` Single-Pass `v2.5.3`) |
| :--- | :---: | :---: | :---: |
| **GGUF File Size (3 Shards)** | `56.57 GiB` (`60,739,343,104` B) | **`56.40 GiB`** (`60,558,856,960` B) | `56.93 GiB` (`61,129,364,160` B) |
| **Anonymous-Heap `VmRSS` (`-c 32768 -ub 512`)** | `58.20 GiB` | **`58.31 – 58.61 GiB`** (`~4.0 GiB` margin) | `57.54 – 58.13 GiB` |
| **Cold Prompt Prefill (`0` cached)** | `37.79 tok/s` (`pp128`) | **`41.11 tok/s`** (`pp128`; `43.98` at `pp580`) | `36.11` (`pp67`) / `37.56` (Gate) |
| **Warm Prefix Reuse Prefill (`cache_prompt=True`)** | `44.36 tok/s` | **`44.36 tok/s`** (`339` new / `403` KV) | — |
| **Raw Pure Decode (`tg64`, `N=1`, Unpruned `Top-6`)** | `9.06 tok/s` | **`9.05 tok/s`** | `8.66 tok/s` |
| **Served Gate Decode (`Stockfish-PVS`) / Burst** | `9.55` *(`75.8%` acc)* / `11.5–21.2` | **`9.15 tok/s` *(`73.0%` acc; `10.74` `--aggressive`)* / `19.4–25.3`** | `8.80` *(`78.7%` acc)* / `11.2–21.0` |
| **`AtomicChat` `wiki-alt.txt` (`ctx=5632`, `12`c / `51`c) PPL** | — | **`7.6579 ± 0.1077`** *(12c; `4.4474` BF16)* | `7.3180 ± 0.1039` *(12c)* / `7.2977 ± 0.0504` *(51c)* |
| **`WikiText-2` (`wiki.test.raw`, `8,192` tok) PPL** | `11.5288 ± 0.55630` | **`12.2345 ± 0.59759`** | `11.6139 ± 0.58453` |
| **Logit KLD vs. `MaxQ v2.5.3` (32-Probe / Matched)** | `0.8437 / 0.4373 nats` | **`0.8409 / 0.4397 nats`** (`75.2%` Top-1, `94.3%` Top-5) | `0.0000 nats` (`Ref`) |
| **80-Point Functional Gate (`Tools / Code / Needle`)** | `64.0 / 80` (`24/20/20`) | **`64.0 / 80`** (`24/20/20`; `70.0 / 80` on `--aggressive` `56.07 GiB`) | `70.0 / 80.0` (`30/20/20`) |

---

## Hybrid Quantization Layout & Donor Provenance (`scripts/build_dsv4_vq3.py`)

1. **MLA & Lightning Indexer Projections -> `Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `3.75 bpw`, `30 B / 64 W`):** `17.75 dB` SQNR (`99.15%` Cosine Similarity vs `BF16`).
2. **Boundary Shared Experts (`blk.0..4`, `blk.39..42`) & All `ffn_down_shexp` -> `Q5_K` / `Q6_K`.**
3. **68 Interior Shared-Expert Pre-`SiLU` Projections (`blk.5..38` `ffn_gate_shexp` & `ffn_up_shexp`) -> `LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `3.00 bpw`, `24 B / 64 W`):** `2 B` `FP16` scale `d` + `22 B` `qs[22]` (`20` 7-bit 3D stride-22 Lloyd-Max indices + `2` 6-bit 2D tail indices + `24` high bits of `8 × 3-bit` sub-scales `{2..9}`; `13.57 dB` SQNR), decoded via `AVX-512 VBMI` `VPERMI2B` with portable `AVX2` fallback.
4. **256 Routed MoE Experts (`Top-6` unpruned routing):** Across the `86` `ffn_gate_exps` & `ffn_up_exps` tensors (`43` layers), copies `70` `blk.0..34` tensors (`60 × IQ1_S + 10 × IQ2_XXS` in anchor layers `{1,4,26,28}`) byte-for-byte and transcodes `16` `blk.35..42` tensors from **`Unsloth`** (`unsloth/DeepSeek-V4-Flash-0731-GGUF` `UD-IQ1_S @ fbbb5b93fb787c21338159b0af3318bb3f4d9768`) paired with second-order `SwiGLU-H2` coupled row-norm quantization on `ffn_down_exps` (for a 100% donor-free single-pass build directly from `0731` safetensors, use [`MaxQ-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF)).

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

---

## Credits & Donor Provenance

- **Unsloth ([`unsloth/DeepSeek-V4-Flash-0731-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)):** Upstream `UD-IQ1_S` (`@ fbbb5b93fb787c21338159b0af3318bb3f4d9768`, 50-chunk calibration) donor checkpoint for routed-MoE `gate_exps`/`up_exps` tensors (`general.quantized_by` in Shard-1 records exact donor provenance).
- **AtomicChat ([`AtomicChat/dsv4-eval-artifacts`](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)):** Open `wiki-alt.txt` (`-c 5632 -b 5632`) evaluation corpus and cross-vendor reference logs.
- **DeepSeek AI ([`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)):** Foundational `284.33B` `DeepSeek-V4-Flash-0731` model (`@ 7872f01b1d1f`, MIT License).
