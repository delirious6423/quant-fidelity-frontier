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

# DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF (`284.33B` in `56.93 GiB` — Safety-Aligned Single-Pass Default)

> [!IMPORTANT]
> **Patched `llama-modchip` Runtime Required (`GGML_TYPE_Q2_0 = 42` + `deepseek4` MLA/Sinkhorn-20):**
> Do not load this checkpoint in vanilla `Ollama`, `LM Studio`, `vLLM`, or unpatched `llama.cpp`. `Fractal-64` (`30 bytes / 64 weights`) repurposes upstream `GGML_TYPE_Q2_0 = 42` (`18 bytes / 32 weights`) in-place, so unpatched `llama.cpp` fails with a tensor bounds mismatch or `unknown model architecture: 'deepseek4'`. Use the `GLIBC_2.30` portable Linux `x86_64` binaries included in this repository (`-avx512` for Zen 4/5 & Ice Lake+, or `-avx2` for Haswell / Skylake / Zen 2/3 / Alder & Raptor Lake with zero `%zmm` or `AVX512_BF16` instructions; requires OpenMP `libgomp1`) or run `./modchip setup` from [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip).

Default safety-aligned 64 GB CPU checkpoint (`61,129,363,936` bytes = **`56.93 GiB` / `61.13 GB`** across 3 shards) built cleanly from [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (`7872f01b1d1f`, `284.33B` total / `13.17B` active parameters; `43` layers, `256` routed + `1` shared expert/layer, unpruned `Top-6` routing).

Every quantized tensor is encoded in a **single pass** directly from upstream `0731` `BF16` / `FP8_E4M3` / `MXFP4` safetensors (`scripts/build_dsv4_maxq_singlepass.py`), retaining **174 dense 100%-active bottleneck tensors** (`173 × Q6_K + 1 × Q8_0`), **317 MLA attention & Lightning Indexer projections** in `Fractal-64` (`3.75 bpw`), and all **129 3D routed MoE tensors** quantized in one pass with Salvatore Sanfilippo's (`@antirez`) `1.5M`-token calibration matrix.

- **Installer Preset (`./modchip setup`):** `dsv4-maxq` *(default on `64 GB+` hosts)*
- **Compact Hybrid Variant (`56.40 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF) (`--preset dsv4-vq3`)
- **Cross-Platform `Fractal-64` Variant (`56.57 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF) (`--preset dsv4-fractal`)
- **Opt-In Uncensored Twin (`56.93 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF) (`--preset dsv4-ablit`)

---

## Measured Quality, Perplexity, & Cross-Vendor `wiki-alt.txt` (`ctx=5632`) Benchmarks (`v2.5.2`)

Evaluated on a 16-Core AMD Ryzen 9 9955HX (`64 GB` DDR5-5600, `0 GPUs`, Linux, unpruned `Top-6` routing):

| Metric / Evaluation Suite | `Fractal` (`56.57 GiB`) | `VQ3` (`56.40 GiB`) | **`MaxQ-Fractal` (`56.93 GiB`, Aligned)** | `Abliterated-Fractal` (`56.93 GiB`, Opt-In) |
| :--- | :---: | :---: | :---: | :---: |
| **Base Checkpoint** | `DeepSeek-V4-Flash-0731` | `DeepSeek-V4-Flash-0731` | **`deepseek-ai/DeepSeek-V4-Flash-0731`** | `prem-research/...-0731-abliterated` |
| **GGUF Size (3 Shards)** | `56.57 GiB` | `56.40 GiB` | **`56.93 GiB`** (`61,129,363,936` B) | `56.93 GiB` (`61,129,364,000` B) |
| **Anonymous-Heap `VmRSS` (`-c 32768 -ub 512`)** | `58.20 GiB` | `58.31 GiB` | **`57.54–59.13 GiB`** | `59.13 GiB` |
| **`AtomicChat` `wiki-alt.txt` (`ctx=5632`, `12` chunks) PPL** | — | `7.6579 ± 0.1077` | **`7.5133 ± 0.1054`** | — |
| **`WikiText-2` (`wiki.test.raw`, `8,192` tok) PPL** | *(transcoded)* | `12.2345 ± 0.59759` | **`11.7434 ± 0.57170`** | `11.3944 ± 0.56500` |
| **2-Pass Logit KLD vs. `MaxQ-Fractal` (`T=1.0`)** | — | `0.3146 nats/tok (83.3% Top-1)` | **`0.0000 nats/tok` (`100%` Ref)** | `1.0219 nats/tok (58.3% Top-1)` |
| **Functional Gate (`80` pts: Tools / Code / Needle)** | `64.0 / 80.0` | `64.0 / 80.0` | **`64.0 / 80.0`** (24/30 Tools, 20/30 Code, 20/20 Needle) | `70.0 / 80.0` (`30/30` Tools) |
| **Technical Dual-Use Security Audit (`8` prompts)** | `8 / 8` | `8 / 8` | **`8 / 8`** | `8 / 8` |
| **Cold / Warm Prompt Prefill (`tok/s`)** | `37.8–44.0` / `44.36` | `41.1–44.0` / `44.36` | **`37.8–43.5` cold / `44.0` warm** | `37.8–43.5` cold / `44.0` warm |
| **Decode (`N=1` Unpruned `Top-6` / Speculative)** | `9.55` / `11.5–21.2` | `9.05` / `19.4–25.3` | **`8.75 tok/s` (`N=1`) / `11.2–21.0` spec** | `8.75 tok/s` (`N=1`) / `11.2–21.0` spec |

### Cross-Vendor `AtomicChat/dsv4-eval-artifacts` (`wiki-alt.txt`, `-c 5632 -b 5632`, Chunks `1..12` = `33,780` Scored Tokens)

Evaluated with identical `llama-perplexity -f wiki-alt.txt -c 5632 -b 5632 --ppl-output-type 1` parameters against the raw chunk-12 cumulative logs in [`AtomicChat/dsv4-eval-artifacts`](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts):

| Publisher & `DeepSeek-V4-Flash-0731` Checkpoint | Disk Footprint | Fits `64 GB` Host Resident? | **Chunk-12 `wiki-alt.txt` `PPL ± σ`** | 51-Chunk `PPL` *(Ref)* |
| :--- | :---: | :---: | :---: | :---: |
| `bullerwins` `MXFP4_MOE-BF16` *(Full Reference)* | `150.75 GiB` (`161.87 GB`) | No (`192 GB+`) | **`4.4474 ± 0.0587`** | `4.5381 ± 0.0292` |
| `prometheusAIR` `IQ2_XXS-slim` | `78.84 GiB` (`84.65 GB`) | No (`96 GB+`) | `5.5360 ± 0.0754` | `5.5165 ± 0.0365` |
| `antirez` `IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-0731` | `80.76 GiB` (`86.72 GB`) | No (`96 GB+`) | `5.5484 ± 0.0762` | `5.6025 ± 0.0377` |
| `unsloth` `UD-IQ1_S` | `76.87 GiB` (`82.54 GB`) | No (`96 GB+`) | — | `5.9734` |
| `AtomicChat` `AD-IQ1_M` | `65.42 GiB` (`70.24 GB`) | No (`>64 GiB` OOM) | — | `6.3813` |
| **`Synchopate` `MaxQ-Fractal-0731` *(This Repo)*** | **`56.93 GiB` (`61.13 GB`)** | **Yes (`~57.5–59.1 GiB` RSS)** | **`7.5133 ± 0.1054`** | — |
| **`Synchopate` `VQ3-0731` *(Compact Hybrid)*** | **`56.40 GiB` (`60.56 GB`)** | **Yes (`~58.3 GiB` RSS)** | **`7.6579 ± 0.1077`** | — |
| `DevQuasar` `Q2_K` | `95.99 GiB` (`103.06 GB`) | No (`128 GB+`) | `9.4237 ± 0.1402` | `9.1127 ± 0.0647` |

---

## Single-Pass Tensor Allocation (`scripts/build_dsv4_maxq_singlepass.py`)

1. **174 Dense 100%-Active Bottleneck Matrices (`173 × Q6_K + 1 × Q8_0`, Direct from `0731` `BF16` / `FP8_E4M3`):**
   - `token_embd.weight` `[4096, 129280]` and `output.weight` `[4096, 129280]` (`BF16 -> Q6_K`).
   - All 43 layers of `attn_q_a.weight`, `ffn_gate_shexp.weight`, `ffn_up_shexp.weight`, and `ffn_down_shexp.weight` (`171 × Q6_K + 1 × Q8_0`).
2. **317 MLA Attention & Lightning Indexer Projections (`Fractal-64`, `GGML_TYPE_Q2_0 = 42`, `3.75 bpw`):**
   - Encoded directly from `FP8_E4M3 * scale` into 64-weight dyadic blocks (`17.75 dB` SQNR, `99.15%` Cosine Similarity), compatible with x86-64 AVX-512, x86-64 AVX2, Windows 11, and Apple Silicon Metal (`M5 Max 64 GB`).
3. **129 3D Routed-MoE Tensors (`256` Experts/Layer, Single-Pass `IQ2_XXS` / `IQ1_S` / `IQ1_M`):**
   - Quantized in a single pass from `0731` `MXFP4` weighted by `@antirez`'s `1.5M`-token `routed-moe-ds4-1p5m.dat` importance matrix.
4. **Norms, Router Gates, `tid2eid` Tables, & Sinkhorn-20 Hyper-Connections:** Stored in `F32` / `BF16` / `I32`.

---

## 64 GB RAM Headroom & Quickstart

- **Anonymous-Heap `VmRSS` (`--load-mode none --repack`):** Repacked CPU weights live in anonymous heap (`~57.5–59.1 GiB` at `-c 32768 -ub 512`), not file-backed page cache, so the Linux kernel cannot evict them under desktop pressure without swap/`zram`. On a 64 GB machine (`~62.5 GiB` usable RAM), close heavy browser sessions or enable `zram`, and keep `-c 32768 -ub 512`. Reserve `-c 262144` (`256K` tokens) for headless 64 GB nodes with `zram`/swap or `96 GB+` hosts.
- **Matching MLA KV Cache Types:** Always pass matching K and V types (`-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`).

### Option A: Automated `llama-modchip` Installer (`v2.5.2`)
```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./modchip setup     # Selects dsv4-maxq by default on 64 GB+ hosts and verifies SHA256
./modchip chat
```

### Option B: Bundled Portable `GLIBC_2.30` Linux `x86_64` Binary (`-avx512` or `-avx2`)
```bash
hf download Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF --local-dir ./dsv4-maxq
cd ./dsv4-maxq
sha256sum -c SHA256SUMS --ignore-missing

# Auto-selects -avx512 (Zen 4/5, Ice Lake+) or -avx2 (Haswell, Zen 2/3, Alder/Raptor Lake):
BIN="./llama-server-modchip-linux-x86_64-$(grep -Eq 'avx512_?vbmi' /proc/cpuinfo && echo avx512 || echo avx2)"
chmod +x "$BIN"

"$BIN" \
  -m DeepSeek-V4-Flash-0731-MaxQ-Fractal-00001-of-00003.gguf \
  -c 32768 -t 16 -tb 16 -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --port 8080 --jinja
```

---

## Credits

- **Salvatore Sanfilippo ([`@antirez`](https://github.com/antirez)) — [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf):** Single-pass routed-MoE quantization and the `1.5M`-token `DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat` importance matrix.
- **DeepSeek AI ([`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)):** Base `284.33B` `DeepSeek-V4-Flash-0731` model released under the MIT License.
