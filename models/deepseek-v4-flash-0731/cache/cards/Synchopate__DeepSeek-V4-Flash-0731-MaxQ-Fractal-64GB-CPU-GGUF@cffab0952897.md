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

# DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF (`284.33B` in `56.93 GiB` — Donor-Free Single-Pass Safety-Aligned Default)

> [!IMPORTANT]
> **Patched `llama-modchip` Runtime Required (`GGML_TYPE_Q2_0 = 42` + `deepseek4` MLA/Sinkhorn-20):**
> Do not load this checkpoint in vanilla `Ollama`, `LM Studio`, `vLLM`, or unpatched `llama.cpp`. `Fractal-64` (`30 bytes / 64 weights`) repurposes upstream `GGML_TYPE_Q2_0 = 42` (`18 bytes / 32 weights`) in-place, so unpatched `llama.cpp` fails with a tensor bounds mismatch or `unknown model architecture: 'deepseek4'`. Use the IPC-hardened `GLIBC_2.30` portable Linux `x86_64` binaries (`v2.5.4`) included in this repository (`-avx512` for Zen 4/5 & Ice Lake+, or `-avx2` for Haswell / Skylake / Zen 2/3 / Alder & Raptor Lake with zero `%zmm` or `AVX512_BF16` instructions; requires OpenMP `libgomp1`) or run `./modchip setup` from [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip).

Default safety-aligned 64 GB CPU checkpoint (`61,129,364,160` bytes = **`56.93 GiB` / `61.13 GB`, `1.72 bpw`** across 3 shards) built in a **pure, donor-free single pass** across all `1,328 / 1,328` tensors directly from [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (`7872f01b1d1fe23eabc4c98b48bffcef5a386062`, `284.33B` total / `13.17B` active parameters; `43` layers, `256` routed + `1` shared expert/layer, unpruned `Top-6` routing).

Unlike `v2.5.2` (which rebuilt the `1,199` non-MoE tensors from `0731` while still copying donor routed-MoE tensors), `v2.5.3` quantizes **every dense, MLA, and 3D routed-MoE weight tensor in a single pass** directly from upstream `0731` `BF16` / `FP8_E4M3` / `MXFP4` safetensors (`python3 scripts/build_dsv4_maxq_singlepass.py --mode aligned --out-prefix ./DeepSeek-V4-Flash-0731-MaxQ-Fractal`), retaining **174 dense 100%-active bottleneck tensors** (`173 × Q6_K + 1 × Q8_0`), **317 MLA attention & Lightning Indexer projections** in `Fractal-64` (`3.75 bpw`), and all **129 3D routed-MoE tensors** weighted by Salvatore Sanfilippo's (`@antirez`) `1.5M`-token `DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat` importance matrix (`antirez/deepseek-v4-gguf @ f71f23d552d6`, calibrated `2026-05-12` on April `DeepSeek-V4-Flash` `chat-v2` activations; Shard-1 GGUF header records `entries_count = 129` and nominal `chunks_count = 1`).

- **Installer Preset (`./modchip setup`):** `dsv4-maxq` *(default on `60 GB+` / `64 GB` hosts)*
- **Compact Hybrid Variant (`56.40 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF) (`--preset dsv4-vq3`)
- **Cross-Platform `Fractal-64` Variant (`56.57 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF) (`--preset dsv4-fractal`)
- **Opt-In Uncensored Twin (`56.93 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF) (`--preset dsv4-ablit`)

---

## Measured Quality, Perplexity, Logit KLD, & CPU Telemetry (`v2.5.4`)

Evaluated on a 16-Core AMD Ryzen 9 9955HX (`64 GB` DDR5-5600, `0 GPUs`, Linux, unpruned `Top-6` routing; raw artifacts in [`benchmarks/results/`](https://github.com/synchopate/llama-modchip/tree/main/benchmarks/results)):

| Metric / Evaluation Suite | `Fractal-64` (`56.57 GiB`) | `VQ3` (`56.40 GiB`) | **`MaxQ-Fractal` (`56.93 GiB`, Aligned `v2.5.3`)** | `Abliterated-Fractal` (`56.93 GiB`, Opt-In) |
| :--- | :---: | :---: | :---: | :---: |
| **Base Checkpoint** | `DeepSeek-V4-Flash-0731` | `DeepSeek-V4-Flash-0731` | **`deepseek-ai/DeepSeek-V4-Flash-0731`** | `prem-research/...-0731-abliterated` |
| **GGUF Size (3 Shards)** | `56.57 GiB` (`60,739,343,104` B) | `56.40 GiB` (`60,558,856,960` B) | **`56.93 GiB`** (`61,129,364,160` B) | `56.93 GiB` (`61,129,364,224` B) |
| **Anonymous-Heap `VmRSS` (`-c 32768 -ub 512`)** | `58.20 GiB` | `58.31 GiB` | **`57.54 GiB` idle / `57.58 GiB` `N=1` / `58.13 GiB` `PVS`** | `59.13 GiB` |
| **`AtomicChat` `wiki-alt.txt` (`ctx=5632`, `51` chunks) PPL** | — | — | **`7.2977 ± 0.0504`** (`143,565` scored) | — |
| **`AtomicChat` `wiki-alt.txt` (`ctx=5632`, `12` chunks) PPL** | — | `7.6579 ± 0.1077` | **`7.3180 ± 0.1039`** (`33,780` scored) | — |
| **`WikiText-2` (`wiki.test.raw`, `8,192` tok) PPL** | `11.5288 ± 0.55630` | `12.2345 ± 0.59759` | **`11.6139 ± 0.58453`** | `11.3944 ± 0.56500` |
| **32-Probe Step-0 Lineage KLD vs. `MaxQ v2.5.3` (`T=1.0`)** | `0.8437 nats` (`56.2%` Top-1) | `0.8409 nats` (`59.4%` Top-1) | **`0.0000 nats` (`100%` Ref)** | **`0.0959 nats` (`75.0%` Top-1, `96.9%` Top-5)** |
| **Prefix-Matched Rollout KLD vs. `MaxQ v2.5.3`** | `0.4373 nats` (`74.3%` Top-1) | `0.4397 nats` (`75.2%` Top-1) | **`0.0000 nats` (`100%` Ref)** | **`0.0512 nats` (`90.7%` Top-1, `99.4%` Top-5)** |
| **Functional Gate (`80` pts: Tools / Code / Needle)** | `64.0 / 80.0` (`24/20/20`) | `64.0 / 80.0` (`24/20/20`) | **`70.0 / 80.0`** (`30/30` Tools, `20/30` Code, `20/20` Needle) | **`70.0 / 80.0`** (`30/30` Tools, `20/30` Code, `20/20` Needle) |
| **Technical Dual-Use Security Audit (`8` prompts)** | `8 / 8` | `8 / 8` | **`8 / 8`** | `8 / 8` |
| **Cold Prompt Prefill (`0` cached, `tok/s`)** | `37.79` (`pp128`) | `41.11` (`pp128`) | **`36.11` (`pp67`) / `37.56` (Gate)** | `35.65` (Gate, `v2.5.1`) |
| **Raw Pure Decode (`tg64`, `N=1`, Unpruned `Top-6`)** | `9.06 tok/s` | `9.05 tok/s` | **`8.66 tok/s`** *(`8.61` 3-run)* | *(`~8.66` via `MaxQ` layout)* |
| **Served Gate Decode (`Stockfish-PVS` E2E)** | `9.55 tok/s` *(`75.8%` acc)* | `9.15 tok/s` *(`73.0%` acc)* | **`8.80 tok/s` *(`78.7%` acc)*** | `8.75 tok/s` *(`64.4%` acc)* |

*(Notes: [1] Adding the `20`-pt speed/RAM Gate rubric under a common `--target-tps 9.0` target yields `89.56/100` for `MaxQ-Fractal`, `89.44/100` for `Abliterated-Fractal`, and `84.0/100` for `Fractal-64` & `VQ3`. [2] On `8,192`-token `WikiText-2`, `MaxQ-Fractal` [`11.6139 ± 0.5845`] is statistically tied within `< 0.15σ` with `Fractal-64` [`11.5288 ± 0.5563`], while improving Agentic Tool-Calling from `24/30` to `30/30`. [3] KLD against quantized `MaxQ` measures quantization-lineage divergence rather than ground-truth error, and prefix-matched rollout KLD truncates at the first greedy disagreement and biases downward relative to unconditional 32-probe step-0 KLD.)*

### Cross-Vendor `AtomicChat/dsv4-eval-artifacts` (`wiki-alt.txt`, `-c 5632 -b 5632`, Full 51 Chunks & Chunk-12 Prefix)

Evaluated with `llama-perplexity -f wiki-alt.txt -c 5632 -b 5632 --ppl-output-type 1` (`143,565` scored positions across all `51` chunks; `33,780` scored positions across chunks `1..12`) against [`AtomicChat/dsv4-eval-artifacts`](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts). **Hardware & aggregation notes:** (1) `AtomicChat` reference runs were collected on `8× RTX 5090` CUDA GPUs (`-DCMAKE_CUDA_ARCHITECTURES=120`), where `AtomicChat`'s README documents a `4–5%` hardware numerical gap (`4.5381` `RTX 5090` vs `4.3406` `H100` on the same GGUF), whereas `Synchopate` checkpoints were evaluated on a 16-core AMD Ryzen 9 9955HX AVX-512 CPU; and (2) `AtomicChat`'s summary table reports `Mean PPL(Q)` from `--kl-divergence` against `ref5632.kld`, whereas `Synchopate`'s `7.2977 ± 0.0504` is standalone `llama-perplexity`'s `Final estimate: PPL`. Fitting `284.33B` inside `< 60 GiB` resident RAM (`1.72 bpw`) trades **`+60.8%` `wiki-alt` perplexity vs. `150.75 GiB` `MXFP4_MOE-BF16`, `+22.2%` vs. `76.87 GiB` `Unsloth UD-IQ1_S`, and `+14.4%` vs. `65.42 GiB` `AtomicChat AD-IQ1_M`**, while being the only `284B` checkpoints below that run **100% resident in `64 GB` host RAM**:

| Publisher & `DeepSeek-V4-Flash-0731` Checkpoint | Disk Footprint | Fits `64 GB` Host Resident? | **Chunk-12 `wiki-alt.txt` `PPL ± σ`** | **Full 51-Chunk `wiki-alt.txt` `PPL`** |
| :--- | :---: | :---: | :---: | :---: |
| `bullerwins` `MXFP4_MOE-BF16` *(Full Reference)* | `150.75 GiB` (`161.87 GB`) | No (`192 GB+`) | `4.4474 ± 0.0587` | **`4.5381 ± 0.0292`** *(`Mean PPL(Q)` `8×5090`; `4.3406` `H100`)* |
| `prometheusAIR` `IQ2_XXS-slim` | `78.84 GiB` (`84.65 GB`) | No (`96 GB+`) | `5.5360 ± 0.0754` | `5.5165 ± 0.0365` *(`Mean PPL(Q)`)* |
| `antirez` `IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-0731` | `80.76 GiB` (`86.72 GB`) | No (`96 GB+`) | `5.5484 ± 0.0762` | `5.6025 ± 0.0377` *(`Mean PPL(Q)`)* |
| `unsloth` `UD-IQ1_S` | `76.87 GiB` (`82.54 GB`) | No (`96 GB+`) | — | **`5.9734`** *(`Mean PPL(Q)`)* |
| `AtomicChat` `AD-IQ1_M` | `65.42 GiB` (`70.24 GB`) | No (`>64 GiB` OOM) | — | **`6.3813`** *(`Mean PPL(Q)`)* |
| **`Synchopate` `MaxQ-Fractal-0731` (`v2.5.3`, *This Repo*)** | **`56.93 GiB` (`61.13 GB`)** | **Yes (`57.54–58.13 GiB` RSS)** | **`7.3180 ± 0.1039`** *(`+64.5%` vs BF16)* | **`7.2977 ± 0.0504`** *(`Final estimate` CPU: `+60.8%` vs BF16, `+22.2%` vs `UD-IQ1_S`, `+14.4%` vs `AD-IQ1_M`)* |
| **`Synchopate` `VQ3-0731` *(Compact Hybrid)*** | **`56.40 GiB` (`60.56 GB`)** | **Yes (`58.31 GiB` RSS)** | `7.6579 ± 0.1077` *(`Final estimate` `12`c)* | — |
| `DevQuasar` `Q2_K` *(uncalibrated naive `Q2_K` outlier)* | `95.99 GiB` (`103.06 GB`) | No (`128 GB+`) | `9.4237 ± 0.1402` | `9.1127 ± 0.0647` *(`Mean PPL(Q)`)* |

---

## Single-Pass Tensor Allocation (`scripts/build_dsv4_maxq_singlepass.py --mode aligned`)

1. **174 Dense 100%-Active Bottleneck Matrices (`173 × Q6_K + 1 × Q8_0`, Direct from `0731` `BF16` / `FP8_E4M3`):**
   - `token_embd.weight` `[4096, 129280]` and `output.weight` `[4096, 129280]` (`BF16 -> Q6_K`).
   - All 43 layers of `attn_q_a.weight`, `ffn_gate_shexp.weight`, `ffn_up_shexp.weight`, and `ffn_down_shexp.weight` (`171 × Q6_K + 1 × Q8_0`).
2. **317 MLA Attention & Lightning Indexer Projections (`Fractal-64`, `GGML_TYPE_Q2_0 = 42`, `3.75 bpw`):**
   - Encoded directly from `FP8_E4M3 * scale` into 64-weight dyadic blocks (`17.75 dB` SQNR, `99.15%` Cosine Similarity), compatible with x86-64 AVX-512, x86-64 AVX2, Windows 11, and Apple Silicon Metal (`M5 Max 64 GB`).
3. **129 3D Routed-MoE Tensors (`256` Experts/Layer, Single-Pass `IQ2_XXS` / `IQ1_S` / `IQ1_M`):**
   - Quantized in a single pass directly from upstream `0731` `MXFP4` safetensors weighted by `@antirez`'s `1.5M`-token `DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat` (`SHA256 = 02a7c78c29875e46...`), with zero intermediate donor GGUF transcoding.
4. **Norms, Router Gates, `tid2eid` Tables, & Sinkhorn-20 Hyper-Connections:** Stored in `F32` / `BF16` / `I32`.

---

## 64 GB RAM Headroom & Quickstart

- **Anonymous-Heap `VmRSS` (`--load-mode none --repack`):** Repacked CPU weights live in anonymous heap (`57.54–58.13 GiB` at `-c 32768 -ub 512`), not file-backed page cache, so the Linux kernel cannot evict them under desktop pressure without swap/`zram`. On a 64 GB machine (`~62.5 GiB` usable RAM), close heavy browser sessions or enable `zram`, and keep `-c 32768 -ub 512`. Reserve `-c 262144` (`256K` tokens) for headless 64 GB nodes with `zram`/swap or `96 GB+` hosts.
- **Matching MLA KV Cache Types:** Always pass matching K and V types (`-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`).

### Option A: Automated `llama-modchip` Installer (`v2.5.4`)
```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./modchip setup     # Selects dsv4-maxq by default on 60 GB+ / 64 GB hosts and verifies SHA256
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

- **Salvatore Sanfilippo ([`@antirez`](https://github.com/antirez)) — [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf):** Single-pass routed-MoE quantization research and the `1.5M`-token `DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat` (`@ f71f23d552d6`, committed `2026-05-12` from April `DeepSeek-V4-Flash` `chat-v2` activations) importance matrix.
- **AtomicChat ([`AtomicChat/dsv4-eval-artifacts`](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)):** Open `wiki-alt.txt` (`-c 5632 -b 5632`) long-context perplexity benchmark suite and cross-vendor reference logs.
- **DeepSeek AI ([`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)):** Base `284.33B` `DeepSeek-V4-Flash-0731` model (`@ 7872f01b1d1f`) released under the MIT License.
