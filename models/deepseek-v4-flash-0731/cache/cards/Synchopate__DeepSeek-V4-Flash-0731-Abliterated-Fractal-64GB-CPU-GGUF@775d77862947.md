---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
license: mit
library_name: llama-modchip
inference: false
tags:
  - gguf
  - llama-modchip
  - deepseek-v4
  - abliterated
  - uncensored
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

# DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF (`284.33B` in `56.93 GiB` — Opt-In Single-Pass `Q6_K` + `Fractal-64`)

> [!IMPORTANT]
> **Legacy Checkpoint — Superseded by MaxQ v3.0**
> This `v2.5.x` `Abliterated-Fractal` checkpoint (`80f9cc8c...`) remains permanently archived with all historical commit hashes intact for reproducibility, but is **superseded for opt-in uncensored / red-team deployments** by our 100% donor-free, zero-custom-type `MaxQ v3.0` subspace-abliterated editions:
> - **64 GB RAM Uncensored Flagship (`53.99 GiB`, built-in `NextN` MTP):** **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-GGUF)**
> - **32 GB RAM Uncensored Edition (`24.98 GiB` resident + lazy NVMe PLE):** **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated-GGUF)**
> - **Safety-Aligned 64 GB Default:** **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF)** · **[`Synchopate/DeepSeek-V4-Flash-0731-MaxQ-v4-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-v4-64GB-CPU-GGUF)**
> - **Runtime Compatibility (`v2.5.4` vs. `v3.0`):** Because these historical `v2.5.x` shards encode custom `Fractal-64` / `LUT-VQ3-64` tensors under legacy enum IDs `42`/`43`, run them with the **`llama-modchip v2.5.4` binaries bundled in this repository** (or git tag `v2.5.4`). Both **[`DeepSeek-V4-Flash-0731-MaxQ-v4`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-v4-64GB-CPU-GGUF)** and **[`Qwen3.8-Flash-Next-MaxQ-64GB`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF)** use **`100%` standard `ggml` tensor types (`<= 42`)** and run natively on **`llama-modchip v3.0`** and upstream `llama.cpp`.

> [!IMPORTANT]
> **Patched `llama-modchip` Runtime Required (`GGML_TYPE_Q2_0 = 42` + `deepseek4` MLA/Sinkhorn-20):**
> Do not load this checkpoint in vanilla `Ollama`, `LM Studio`, `vLLM`, or unpatched `llama.cpp`. `Fractal-64` (`30 bytes / 64 weights`) repurposes upstream `GGML_TYPE_Q2_0 = 42` (`18 bytes / 32 weights`) in-place, so unpatched `llama.cpp` fails with a tensor bounds mismatch or `unknown model architecture: 'deepseek4'`. Use the IPC-hardened `GLIBC_2.30` portable Linux `x86_64` binaries (`v2.5.4`) included in this repository (`-avx512` for Zen 4/5 & Ice Lake+, or `-avx2` for Haswell / Skylake / Zen 2/3 / Alder & Raptor Lake with zero `%zmm` or `AVX512_BF16` instructions; requires OpenMP `libgomp1`) or run `./modchip setup` from [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip).

> [!WARNING]
> **Opt-In Abliterated / Uncensored Checkpoint — Safety & Responsible Use Disclaimer:**
> - **Refusal Removal:** Quantized from [`prem-research/DeepSeek-V4-Flash-0731-abliterated`](https://huggingface.co/prem-research/DeepSeek-V4-Flash-0731-abliterated) (`@ 457210de3a9d27d8142407cd0dac7b6ed6ad9325`), which applies Activation Refusal Abliteration (ARA orthogonal projection across `o_b_proj`, shared `down_proj`, and top-8 routed experts in layers `17..36`) to `DeepSeek-V4-Flash-0731`. Standard upstream refusal filters are **disabled**.
> - **Intended Scope:** Authorized security research, CTF/vulnerability auditing, academic evaluation, and local offline development only. **Do not expose this checkpoint on public, multi-user, or unauthenticated endpoints without external access control and safety filtering.** Users are solely responsible for lawful compliance.
> - **Safety-Aligned Default Twin:** For standard upstream safety alignment with the exact same `56.93 GiB` single-pass tensor layout on `0731`, use [`Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF) (`./modchip setup` default `dsv4-maxq`).

64 GB CPU checkpoint (`61,129,364,224` bytes = **`56.93 GiB` / `61.13 GB`, `1.72 bpw`** across 3 shards) of **`DeepSeek-V4-Flash-0731-abliterated`** (`284.33B` total / `13.17B` active parameters; `43` layers, `256` routed + `1` shared expert/layer, unpruned `Top-6` routing). Quantized in a **single pass** across all `1,328` weight tensors directly from `BF16` / `FP8_E4M3` / `MXFP4` safetensors (`python3 scripts/build_dsv4_maxq_singlepass.py --mode abliterated --hf-repo prem-research/DeepSeek-V4-Flash-0731-abliterated --hf-rev 457210de3a9d27d8142407cd0dac7b6ed6ad9325 --out-prefix ./DeepSeek-V4-Flash-0731-Abliterated-Fractal`), preserving **174 dense 100%-active bottleneck tensors** (`173 × Q6_K + 1 × Q8_0`), **317 MLA & Indexer tensors** in `Fractal-64` (`3.75 bpw`), and all **129 routed MoE tensors** via `@antirez`'s `1.5M`-token `routed-moe-ds4-1p5m.dat` (`@ f71f23d552d6`, calibrated `2026-05-12` on April `DeepSeek-V4-Flash` `chat-v2` activations; Shard-1 GGUF header records `entries_count = 129` and nominal `chunks_count = 1`). Shards 2 & 3 (`6e3f499c...`, `0063f3a9...`) are the original `v2.5.1` single-pass build (with Shard-1 metadata cleaned in `v2.5.3`), verified **byte-identical** to `MaxQ-Fractal v2.5.3` across non-abliterated control experts (e.g., `blk.20.ffn_down_exps` experts `0, 1`) and differing on `prem-research`'s abliterated top-8 routed `down_proj` experts in layers `17..36` (`{53, 61, 70, 149, 156, 187, 210, 219}` in layer `20`).

---

## Unconfounded Same-Recipe Comparison vs. Safety-Aligned `MaxQ-Fractal-0731` (`v2.5.4`)

Measured on a 16-Core AMD Ryzen 9 9955HX (`64 GB` DDR5-5600, `0 GPUs`, Linux, unpruned `Top-6`):

| Metric / Suite | `MaxQ-Fractal` (`56.93 GiB`, Aligned `v2.5.3`) | **`Abliterated-Fractal` (`56.93 GiB`, Opt-In)** | `Fractal-64` (`56.57 GiB`) | `VQ3` (`56.40 GiB`) |
| :--- | :---: | :---: | :---: | :---: |
| **GGUF Size (3 Shards)** | `56.93 GiB` (`61,129,364,160` B) | **`56.93 GiB`** (`61,129,364,224` B) | `56.57 GiB` | `56.40 GiB` |
| **Anonymous-Heap `VmRSS` (`-c 32768 -ub 512`)** | `57.54–58.13 GiB` | **`59.13 GiB`** | `58.20 GiB` | `58.31 GiB` |
| **`AtomicChat` `wiki-alt.txt` (`ctx=5632`, `51`c / `12`c)** | `7.2977` / `7.3180` | — | — | — / `7.6579` |
| **`WikiText-2` (`wiki.test.raw`, `8,192` tok) PPL** | `11.6139 ± 0.58453` | **`11.3944 ± 0.56500`** | `11.5288 ± 0.55630` | `12.2345 ± 0.59759` |
| **32-Probe Step-0 KLD vs. `MaxQ v2.5.3` (`T=1.0`)** | `0.0000 nats` (`Ref`) | **`0.0959 nats` (`75.0%` Top-1, `96.9%` Top-5)** | `0.8437 nats` | `0.8409 nats` |
| **Prefix-Matched Multi-Step KLD vs. `MaxQ v2.5.3`** | `0.0000 nats` (`Ref`) | **`0.0512 nats` (`90.7%` Top-1, `99.4%` Top-5)** | `0.4373 nats` | `0.4397 nats` |
| **Functional Quality Gate (`80` pts)** | `70.0 / 80.0` (`30/30` Tools) | **`70.0 / 80.0`** (`30/30` Tools, `20/30` Code, `20/20` Needle) | `64.0 / 80.0` | `64.0 / 80.0` |
| **Technical Dual-Use Security Audit (`8` prompts)** | `8 / 8` | **`8 / 8`** | `8 / 8` | `8 / 8` |
| **Raw Pure Decode (`tg64`, `N=1`, Unpruned `Top-6`)** | `8.66 tok/s` | *(`~8.66 tok/s` via `MaxQ` layout)* | `9.06 tok/s` | `9.05 tok/s` |
| **Served Gate Decode (`Stockfish-PVS` E2E)** | `8.80 tok/s` *(`78.7%` acc)* | **`8.75 tok/s` *(`64.4%` acc)*** | `9.55 tok/s` *(`75.8%` acc)* | `9.15 tok/s` *(`73.0%` acc)* |

*(Note: Under a common `--target-tps 9.0` 100-point rubric, `MaxQ-Fractal` scores `89.56/100` and `Abliterated-Fractal` scores `89.44/100`; `Abliterated-Fractal`'s Gate JSON [`59.13 GiB` `VmRSS`, `8.75 tok/s`] was recorded in `v2.5.1` on the same Shard 2/3 weights. Prefix-matched rollout KLD truncates at the first greedy disagreement and biases downward relative to unconditional 32-probe step-0 KLD.)*

---

## Single-Pass Tensor Allocation

1. **174 Dense 100%-Active Bottleneck Matrices (`173 × Q6_K + 1 × Q8_0`):** `token_embd.weight`, `output.weight`, and all 43 layers of `attn_q_a`, `ffn_gate_shexp`, `ffn_up_shexp`, and `ffn_down_shexp`.
2. **317 MLA & Lightning Indexer Projections (`Fractal-64`, `GGML_TYPE_Q2_0 = 42`, `3.75 bpw`):** Direct from `FP8_E4M3 * scale` (`17.75 dB` SQNR, `99.15%` Cosine Similarity).
3. **129 3D Routed-MoE Tensors (`256` Experts/Layer):** Single-pass `IQ2_XXS` / `IQ1_S` / `IQ1_M` from `MXFP4` with `@antirez`'s `1.5M`-token `imatrix`, preserving `prem-research`'s abliterated routed experts in layers `17..36`.
4. **Norms, Router Gates, `tid2eid` Tables, & Sinkhorn-20 Hyper-Connections:** `F32` / `BF16` / `I32`.

---

## Quickstart

- **64 GB Anonymous-Heap Headroom:** At `-c 32768 -ub 512`, anonymous-heap `VmRSS` is **`59.13 GiB`** (`~3.4 GiB` OS margin in `~62.5 GiB` usable RAM; enable `zram` or close heavy GUI apps). Use `-c 262144` only on dedicated headless 64 GB nodes with `zram`/swap or `96 GB+` hosts.
- **Matching MLA KV Cache Types:** Always pass matching `-ctk` and `-ctv` (`-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`).

### Option A: `llama-modchip` Installer (`v2.5.4`, Explicit Opt-In)
```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./install.sh --preset dsv4-ablit     # Prompts for [y/N] confirmation (or pass --accept-abliterated with --yes)
./modchip chat
```

### Option B: Bundled Portable `GLIBC_2.30` Linux `x86_64` Binary (`-avx512` or `-avx2`)
```bash
hf download Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF --local-dir ./dsv4-ablit
cd ./dsv4-ablit
sha256sum -c SHA256SUMS --ignore-missing

BIN="./llama-server-modchip-linux-x86_64-$(grep -Eq 'avx512_?vbmi' /proc/cpuinfo && echo avx512 || echo avx2)"
chmod +x "$BIN"

"$BIN" \
  -m DeepSeek-V4-Flash-0731-Abliterated-Fractal-00001-of-00003.gguf \
  -c 32768 -t 16 -tb 16 -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --port 8080 --jinja
```

---

## Credits

- **Salvatore Sanfilippo ([`@antirez`](https://github.com/antirez)) — [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf):** Single-pass routed-MoE quantization and `1.5M`-token `imatrix` (`@ f71f23d552d6`).
- **Prem Research ([`prem-research/DeepSeek-V4-Flash-0731-abliterated`](https://huggingface.co/prem-research/DeepSeek-V4-Flash-0731-abliterated)):** Full-precision abliterated master checkpoint (`@ 457210de3a9d`).
- **DeepSeek AI ([`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)):** Foundational `284.33B` `DeepSeek-V4-Flash-0731` model (MIT License).
