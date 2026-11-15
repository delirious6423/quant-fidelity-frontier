---
license: mit
library_name: gguf
inference: false
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
  - gguf
  - llama-modchip
  - deepseek-v4
  - mixture-of-experts
  - cpu-inference
  - conecollapse
  - maxq-v4
  - metal
  - cuda
  - vulkan
  - quantization
  - claude-code
  - avx512
  - avx2
  - local-llm
pipeline_tag: text-generation
quantized_by: Synchopate
---

# DeepSeek-V4-Flash-0731-MaxQ-v4-64GB-CPU-GGUF (`284.33B` in `57.99 GiB` — 100% Donor-Free ConeCollapse + DualGram Default)

> [!IMPORTANT]
> **Zero Custom Tensor Types (`100%` Standard `ggml` Formats) — Requires `llama-modchip v2.5.5+` `deepseek4` Loader:**
> Unlike `MaxQ-Fractal` (`GGML_TYPE_Q2_0 = 42`), `MaxQ-v4` uses **zero custom tensor types** (`F32`, `BF16`, `I32`, `Q8_0`, `Q6_K`, `Q5_K`, `Q4_K`, `MXFP4`, `IQ2_XXS`, `IQ1_M`, `IQ1_S`), making it **100% natively compatible with Apple Metal GPU (`M1–M5`), NVIDIA CUDA MMQ (with automatic `MUL_MAT_ID + ADD_ID` fusion), Vulkan, ARM64 NEON, and x86-64 AVX2/AVX-512**.
> Because `34 / 43` MoE layers use collapsed expert intermediate width `K = 1792` (`7 × 256`) + `blk.*.ffn_down_exps.bias`, load it with the IPC-hardened **`v2.5.5+` `GLIBC_2.30` portable Linux `x86_64` binaries** included in this repository (`-avx512` for Zen 4/5 & Ice Lake+, or `-avx2` for Haswell / Skylake / Zen 2/3 / Alder & Raptor Lake with zero `%zmm` or `vdpbf16ps` instructions; requires OpenMP `libgomp1`) or build via `./modchip setup`, `apple-silicon/build-dsv4-engine.sh`, or `windows/build_patched_engine.ps1` from [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip).

Default safety-aligned `< 58.00 GiB` checkpoint (`62,270,037,248` bytes = **`57.993491 GiB` / `62.27 GB`** across 3 shards, `1,362` tensors) built **100% donor-free** directly from [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (`@ 7872f01b1d1fe23eabc4c98b48bffcef5a386062`, `284.33B` total / `13.17B` active parameters; `43` layers, `256` routed + `1` shared expert/layer, unpruned `Top-6` routing) with **zero weights copied from external quantized donors**.

- **Installer Preset (`./modchip setup`):** `dsv4-maxq` *(default on `60 GB+` / `64 GB` hosts in `v2.5.5+`)*
- **Single-Pass `Fractal-64` Variant (`56.93 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF) (`--preset dsv4-maxq-fractal`)
- **Compact Hybrid Variant (`56.40 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF) (`--preset dsv4-vq3`)
- **Cross-Platform `Fractal-64` Variant (`56.57 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF) (`--preset dsv4-fractal`)
- **Opt-In Uncensored Twin (`56.93 GiB`):** [`Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF) (`--preset dsv4-ablit`)

---

## Full 51-Chunk & Blind 49-Chunk `AtomicChat ref5632.kld` Evaluation (`v2.5.5`)

Evaluated with `llama-perplexity -f wiki-alt.txt -c 5632 -b 5632 --kl-divergence-base ref5632.kld` (`143,565` scored positions across all `51` chunks against `Mean PPL(base) = 4.331017 ± 0.027402`; raw log in [`benchmarks/results/atomicchat_wikialt_ctx5632_51c_dsv4_maxq_v4.log`](https://github.com/synchopate/llama-modchip/blob/main/benchmarks/results/atomicchat_wikialt_ctx5632_51c_dsv4_maxq_v4.log)):

| Checkpoint & Provenance | Footprint (`GiB`) | Partition (`Scored Tok`) | **`Mean PPL(Q)`** | **`Mean KLD` (nats)** | **`Median KLD`** | **`90% KLD`** | **`Max KLD`** | **`RMS Δp`** | **`Same top p` (Top-1)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MaxQ-v4` (`v2.5.5`, 100% Donor-Free `0731`)** | **`57.99 GiB`** | **All `1..51` (`143,565`)** | **`6.8592 ± 0.0482`** | **`0.64851 ± 0.00354`** | **`0.18382`** | **`1.70610`** | **`20.5424`** | **`27.562%`** | **`73.283%`** (`105,209`) |
| `MaxQ-Fractal` (`v2.5.3`, 100% Donor-Free `0731`) | `56.93 GiB` | All `1..51` (`143,565`) | `7.2977 ± 0.0504` | `0.74000 ± 0.00383` | `0.27039` | `1.97159` | `21.3412` | `29.857%` | `70.397%` (`101,065`) |
| **`MaxQ-v4` (`v2.5.5`, 100% Donor-Free `0731`)** | **`57.99 GiB`** | **Blind `3..51` (`137,935`)** | **`6.8471`** | **`0.62574`** | — | — | — | **`26.726%`** | **`73.595%`** (`101,513`) |
| `MaxQ-Fractal` (`v2.5.3`, 100% Donor-Free `0731`) | `56.93 GiB` | Blind `3..51` (`137,935`) | `7.2938` | `0.71791` | — | — | — | `29.054%` | `70.701%` (`97,521`) |

*(Methodology & statistical notes: [1] Compared to single-pass `MaxQ-Fractal` [`56.93 GiB`], `MaxQ-v4` [`57.99 GiB`] reduces `Mean PPL(Q)` by **`-6.01%`**, `Mean KLD` by **`-12.36%`**, `Median KLD` by **`-32.02%`**, and improves `Same top p` argmax agreement by **`+2.886` percentage points** [`+4,144` exact top-1 tokens] while eliminating all custom tensor types. [2] Because chunks `1..2` [`5,630` tokens] served as a fast diagnostic gate during `K=2048` uncentered DualGram & `output_norm` `tau=1.0` validation, the **Blind `3..51` partition** [`49` chunks, `137,935` tokens] eliminates gate selection bias and confirms `-6.12%` PPL and `-12.84%` Mean KLD vs. `MaxQ-Fractal` — and against an internal `AtomicChat AD-BF16 + AD-IQ1_M` donor hybrid ablation [`MaxQ-v3`], donor-free `MaxQ-v4` significantly wins on blind `3..51` `Mean KLD` [`0.62574` vs `0.64872`, `-3.54%`, paired $p = 6.58 \times 10^{-5}$], `RMS Δp` [$p = 8.84 \times 10^{-10}$], and `Same top p` [`+1.152` pp, $p = 1.26 \times 10^{-13}$] while matching blind PPL [$p = 0.840$]. [3] Vendor reference points on `wiki-alt.txt` that exceed `64 GB` host RAM: `bullerwins MXFP4_MOE-BF16` [`150.75 GiB`, `4.5381` `8×5090` / `4.3406` `H100`], `Unsloth UD-IQ1_S` [`76.87 GiB`, `5.9734`], `AtomicChat AD-IQ1_M` [`65.42 GiB`, `6.3813`].)*

---

## Donor-Free `MaxQ-v4` Architecture & Tensor Allocation (`1,362` Tensors)

Built reproducibly via [`scripts/build_dsv4_maxq_v4.py`](https://github.com/synchopate/llama-modchip/blob/main/scripts/build_dsv4_maxq_v4.py) and [`scripts/maxq_v4_collapse.c`](https://github.com/synchopate/llama-modchip/blob/main/scripts/maxq_v4_collapse.c):

1. **`446` `0731`-Imatrix-Weighted 2D `K`-Quant Projections + `44 × Q8_0` (`0` Custom Tensor Types):**
   - `43 × Q8_0` `blk.{0..42}.attn_kv.weight` + `1 × Q8_0` `blk.26.ffn_down_shexp.weight`.
   - `321 × Q6_K` (`token_embd.weight`, `output.weight`, `43` `attn_q_a`, `128` `ffn_*_shexp`, `145` KV/Indexer compressors & `indexer.attn_q_b`, `3` `L42` MLA projections).
   - `3 × Q5_K` (`blk.0.{attn_q_b, attn_output_a, attn_output_b}.weight`) and `123 × Q4_K` (`blk.1..41` MLA projections, including exact 3D group-aligned `attn_output_a` imatrix weighting).
   - Weighted across `446 / 446` 2D tensors (`0` unweighted fallbacks) by `imatrix_v4_clean_24k.dat` (`769` entries, `24,576` clean UTF-8 tokens from [`scripts/make_calib_v4_clean_utf8.py`](https://github.com/synchopate/llama-modchip/blob/main/scripts/make_calib_v4_clean_utf8.py), verified `0.000%` $\ge 8$-gram overlap with `wiki-alt.txt`, `wiki.test.raw`, and `eval_quality_and_tools.py`; shares encyclopedic English Wikipedia register with `wiki-alt.txt` alongside ~22% C++/Python/JSON/math/multilingual text).
2. **ConeCollapse (`34` Layers `K = 1792`, `9` Anchor Layers `K = 2048`) + FP64 Woodbury DualGram (`NC = 384`):**
   - **Anchor Layers (`K = 2048`, uncentered DualGram, `0` bias):** `{0, 30, 34, 36, 37, 39, 40, 41, 42}`, including MSE-optimal `MXFP4` (`4.25 bpw`) on `blk.{41,42}.ffn_down_exps.weight` and `IQ2_XXS` (`2.06 bpw`) on `blk.42.ffn_{gate,up}_exps.weight`.
   - **Collapsed Interior Layers (`K = 1792 = 7 × 256`, `-12.5%` MoE intermediate channels):** `{1..29, 31..33, 35, 38}`, solved per expert via **Deterministic Routed-Stride (`NC = 384`) + Router `sqrt(softplus)` Top-Score Fill** in an unconditionally well-conditioned ($\kappa_2 \le 86.33$) FP64 Cholesky Woodbury dual system $(H_c H_c^\top + \lambda I_{384}) Z = H_c$, followed by `IQ2_XXS` quantization of `ffn_down_exps.weight` and shrunk residual bias absorption (`0.60 * mu`) into `34 × F32` `blk.*.ffn_down_exps.bias` (`[4096, 256]`).
   - **Routed-MoE `gate/up` Allocation & Imatrix Lineage:** `25` layers in `IQ1_S` (`23` single-pass layers `{0, 2, 3, 5..19, 21..25}` weighted by `@antirez`'s public `1.5M`-token `DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat` [`@ f71f23d552d6`, pre-`0731` `chat-v2` activations] reordered by `0731` `||Wd[:,j]||^2`, plus `2` fresh `0731`-blended layers `{1, 4}`), `17` sensitive layers (`{20, 26..41}`) in fresh `IQ1_M` (`BETA_0731 = 0.50`), and `L42` in fresh `IQ2_XXS`.

### Shard Manifest (`62,270,037,248` Bytes = `57.993491 GiB`)

| Shard Filename | Exact Size (Bytes) | Size (`GiB`) | `SHA-256` |
| :--- | ---: | ---: | :--- |
| `DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf` | `5,258,336` | `0.0049` | `6112e887b21d1fdf7b7196232b528698a6ec2d5a59760a040cb4231f8e8d691b` |
| `DeepSeek-V4-Flash-0731-MaxQ-v4-00002-of-00003.gguf` | `35,543,586,816` | `33.1025` | `5d85c740992b70395da5c3e4a856b3fad6ef725271722c8d837b35befe28330d` |
| `DeepSeek-V4-Flash-0731-MaxQ-v4-00003-of-00003.gguf` | `26,721,192,096` | `24.8860` | `a8654d4d5fedf748d6f2b6404a8b7f9e19435a8aa8e2bb97ce9a66bc012c0f3a` |

---

## 64 GB RAM Headroom & Quickstart (`v2.5.5`)

- **Anonymous-Heap `VmRSS` (`--load-mode none --repack`):** Repacked CPU weights live in anonymous heap (`~58.6–59.13 GiB` at `-c 32768 -ub 512`), so the Linux kernel cannot evict them under desktop pressure without swap/`zram`. On a 64 GB machine (`~62.5 GiB` usable RAM), close heavy browser sessions or enable `zram`, and keep `-c 32768 -ub 512`. Reserve `-c 262144` (`256K` tokens) for headless 64 GB nodes with `zram`/swap or `96 GB+` hosts.
- **Matching MLA KV Cache Types:** Always pass matching K and V types (`-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`).

### Option A: Automated `llama-modchip` Installer (`Linux` & `macOS Apple Silicon`)
```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./modchip setup     # Selects dsv4-maxq (MaxQ-v4) by default on 60 GB+ / 64 GB hosts and verifies SHA256
./modchip chat
```

### Option B: Bundled Portable `GLIBC_2.30` Linux `x86_64` Binary (`-avx512` or `-avx2`)
```bash
hf download Synchopate/DeepSeek-V4-Flash-0731-MaxQ-v4-64GB-CPU-GGUF --local-dir ./dsv4-maxq-v4
cd ./dsv4-maxq-v4
sha256sum -c SHA256SUMS --ignore-missing

# Auto-selects -avx512 (Zen 4/5, Ice Lake+) or -avx2 (Haswell, Zen 2/3, Alder/Raptor Lake):
BIN="./llama-server-modchip-linux-x86_64-$(grep -Eq 'avx512_?vbmi' /proc/cpuinfo && echo avx512 || echo avx2)"
chmod +x "$BIN"

"$BIN" \
  -m DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf \
  -c 32768 -t 16 -tb 16 -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --port 8080 --jinja
```

---

## Credits

- **Salvatore Sanfilippo ([`@antirez`](https://github.com/antirez)) — [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf):** Single-pass routed-MoE quantization research and the public `1.5M`-token `DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat` (`@ f71f23d552d6`, committed `2026-05-12` from April `DeepSeek-V4-Flash` `chat-v2` activations) importance matrix.
- **AtomicChat ([`AtomicChat/dsv4-eval-artifacts`](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)):** Open `wiki-alt.txt` + `ref5632.kld` (`-c 5632 -b 5632`) long-context KLD/perplexity evaluation suite and cross-vendor reference logs.
- **DeepSeek AI ([`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)):** Base `284.33B` `DeepSeek-V4-Flash-0731` model (`@ 7872f01b1d1f`) released under the MIT License.
