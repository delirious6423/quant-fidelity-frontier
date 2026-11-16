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
  - maxq
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

# DeepSeek-V4-Flash-0731-MaxQ-v4-64GB-CPU-GGUF (`284.33B` in `57.99 GiB` — 100% Donor-Free ConeCollapse + DualGram · `llama-modchip v3.0`)

> [!IMPORTANT]
> **Zero Custom Tensor Types (`100%` Standard `ggml` Formats) — Upgraded with `llama-modchip v3.0` Portable Binaries:**
> `MaxQ-v4` uses **zero custom tensor types** (`F32`, `BF16`, `I32`, `Q8_0`, `Q6_K`, `Q5_K`, `Q4_K`, `MXFP4`, `IQ2_XXS`, `IQ1_M`, `IQ1_S`; `max_type = 39`), making it **100% natively compatible with Apple Metal GPU (`M1–M5`), NVIDIA CUDA MMQ (with automatic `MUL_MAT_ID + ADD_ID` fusion), Vulkan, ARM64 NEON, and x86-64 AVX2/AVX-512**.
> Because `34 / 43` MoE layers use collapsed expert intermediate width `K = 1792` (`7 × 256`) + `blk.*.ffn_down_exps.bias`, load it with the unified **`llama-modchip v3.0` `GLIBC_2.30` portable Linux `x86_64` binaries** included in this repository (`-avx512` for Zen 4/5 & Ice Lake+, or `-avx2` for Haswell / Skylake / Zen 2/3 / Alder & Raptor Lake; statically links `libstdc++` & `libgomp`, requiring zero external OpenMP `.so`) or install via `./install.sh --preset dsv4-maxq` from [`synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip).

Safety-aligned `< 58.00 GiB` **284B-A13B** 64 GB tier checkpoint (`62,270,037,248` bytes = **`57.993491 GiB` / `62.27 GB`** across 3 shards, `1,362` tensors) built **100% donor-free** directly from [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (`@ 7872f01b1d1fe23eabc4c98b48bffcef5a386062`, `284.33B` total / `13.17B` active parameters; `43` layers, `256` routed + `1` shared expert/layer, unpruned `Top-6` routing) with **zero weights copied from external quantized donors**.

- **Installer Preset (`llama-modchip v3.0`):** `./install.sh --preset dsv4-maxq`
- **64 GB Tier Primary Flagship (`3.43` `wiki-alt` PPL, `12.66 tok/s` built-in `NextN` MTP, `45.1 GiB` RSS):** [`Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF) (`./install.sh --preset qwen38-64gb`)
- **32 GB & 16 GB RAM Siblings (`MaxQ v3.0`):** [`Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF) · [`Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64-CPU-GGUF`](https://huggingface.co/Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64-CPU-GGUF) · [`Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-MaxQ-16GB-GGUF`](https://huggingface.co/Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-MaxQ-16GB-GGUF)
- **Archived Historical `v2.5.x` `DeepSeek-V4` Checkpoints:** [`MaxQ-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF) · [`VQ3`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF) · [`Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF) · [`Abliterated-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF)

---

## Full 51-Chunk & Blind 49-Chunk `AtomicChat ref5632.kld` Evaluation

Evaluated with `llama-perplexity -f wiki-alt.txt -c 5632 -b 5632 --kl-divergence-base ref5632.kld` (`143,565` scored positions across all `51` chunks against `Mean PPL(base) = 4.331017 ± 0.027402`; raw log in [`benchmarks/results/atomicchat_wikialt_ctx5632_51c_dsv4_maxq_v4.log`](https://github.com/synchopate/llama-modchip/blob/main/benchmarks/results/atomicchat_wikialt_ctx5632_51c_dsv4_maxq_v4.log)):

| Checkpoint & Provenance | Footprint (`GiB`) | Partition (`Scored Tok`) | **`Mean PPL(Q)`** | **`Mean KLD` (nats)** | **`Median KLD`** | **`90% KLD`** | **`Max KLD`** | **`RMS Δp`** | **`Same top p` (Top-1)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MaxQ-v4` (100% Donor-Free `0731`, Std `ggml`)** | **`57.99 GiB`** | **All `1..51` (`143,565`)** | **`6.8592 ± 0.0482`** | **`0.64851 ± 0.00354`** | **`0.18382`** | **`1.70610`** | **`20.5424`** | **`27.562%`** | **`73.283%`** (`105,209`) |
| `MaxQ-Fractal` (`v2.5.3` Legacy, 100% Donor-Free `0731`) | `56.93 GiB` | All `1..51` (`143,565`) | `7.2977 ± 0.0504` | `0.74000 ± 0.00383` | `0.27039` | `1.97159` | `21.3412` | `29.857%` | `70.397%` (`101,065`) |
| **`MaxQ-v4` (100% Donor-Free `0731`, Std `ggml`)** | **`57.99 GiB`** | **Blind `3..51` (`137,935`)** | **`6.8471`** | **`0.62574`** | — | — | — | **`26.726%`** | **`73.595%`** (`101,513`) |
| `MaxQ-Fractal` (`v2.5.3` Legacy, 100% Donor-Free `0731`) | `56.93 GiB` | Blind `3..51` (`137,935`) | `7.2938` | `0.71791` | — | — | — | `29.054%` | `70.701%` (`97,521`) |

*(Compared to `MaxQ-Fractal` [`56.93 GiB`], `MaxQ-v4` [`57.99 GiB`] reduces `Mean PPL(Q)` by **`-6.01%`**, `Mean KLD` by **`-12.36%`**, `Median KLD` by **`-32.02%`**, and improves `Same top p` argmax agreement by **`+2.886` percentage points** [`+4,144` exact top-1 tokens] while eliminating all custom tensor types. Full architectural details are documented in [`docs/DEEPSEEK_V4_OPTIMIZATIONS.md`](https://github.com/synchopate/llama-modchip/blob/main/docs/DEEPSEEK_V4_OPTIMIZATIONS.md).)*

---

## Donor-Free `MaxQ-v4` Architecture & Shard Manifest (`1,362` Tensors)

1. **`446` `0731`-Imatrix-Weighted 2D `K`-Quant Projections + `44 × Q8_0` (`0` Custom Tensor Types):**
   - `43 × Q8_0` `blk.{0..42}.attn_kv.weight` + `1 × Q8_0` `blk.26.ffn_down_shexp.weight`.
   - `321 × Q6_K` (`token_embd.weight`, `output.weight`, `43` `attn_q_a`, `128` `ffn_*_shexp`, `145` KV/Indexer compressors & `indexer.attn_q_b`, `3` `L42` MLA projections), `3 × Q5_K` (`L00` MLA), and `123 × Q4_K` (`blk.1..41` MLA projections).
   - Weighted across `446 / 446` 2D tensors (`0` unweighted fallbacks) by `24,576` clean UTF-8 calibration tokens (`0.000%` $\ge 8$-gram overlap with `wiki-alt.txt`, `wiki.test.raw`, and `eval_quality_and_tools.py`).
2. **ConeCollapse (`34` Layers `K = 1792`, `9` Anchor Layers `K = 2048`) + FP64 Woodbury DualGram (`NC = 384`):**
   - **Anchor Layers (`K = 2048`, uncentered DualGram, `0` bias):** `{0, 30, 34, 36, 37, 39, 40, 41, 42}`, including MSE-optimal `MXFP4` (`4.25 bpw`) on `blk.{41,42}.ffn_down_exps.weight` and `IQ2_XXS` (`2.06 bpw`) on `blk.42.ffn_{gate,up}_exps.weight`.
   - **Collapsed Interior Layers (`K = 1792 = 7 × 256`, `-12.5%` MoE intermediate channels):** `{1..29, 31..33, 35, 38}`, solved per expert via an unconditionally well-conditioned ($\kappa_2 \le 86.33$) FP64 Cholesky Woodbury dual system, followed by `IQ2_XXS` quantization of `ffn_down_exps.weight` and shrunk residual bias absorption (`0.60 * mu`) into `34 × F32` `blk.*.ffn_down_exps.bias` (`[4096, 256]`).
   - **Routed-MoE `gate/up` Allocation:** `25` layers in `IQ1_S` (using `@antirez`'s public `1.5M`-token `routed-moe-ds4-1p5m.dat` + `0731` activations), `17` sensitive layers (`{20, 26..41}`) in `IQ1_M`, and `L42` in `IQ2_XXS`.

| Shard Filename | Exact Size (Bytes) | Size (`GiB`) | `SHA-256` |
| :--- | ---: | ---: | :--- |
| `DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf` | `5,258,336` | `0.0049` | `6112e887b21d1fdf7b7196232b528698a6ec2d5a59760a040cb4231f8e8d691b` |
| `DeepSeek-V4-Flash-0731-MaxQ-v4-00002-of-00003.gguf` | `35,543,586,816` | `33.1025` | `5d85c740992b70395da5c3e4a856b3fad6ef725271722c8d837b35befe28330d` |
| `DeepSeek-V4-Flash-0731-MaxQ-v4-00003-of-00003.gguf` | `26,721,192,096` | `24.8860` | `a8654d4d5fedf748d6f2b6404a8b7f9e19435a8aa8e2bb97ce9a66bc012c0f3a` |

---

## 64 GB RAM Headroom & Quickstart (`llama-modchip v3.0`)

- **Anonymous-Heap `VmRSS` & `64 GB` Headroom:** At `-c 32768 -ub 512`, `VmHWM` is `~58.6–60.15 GiB` (`RssAnon = 2.16 GiB` under `v3.0` default zero-copy 3D MoE `--repack` with `mmap` enabled). On a 64 GB desktop (`~62.5 GiB` usable RAM), keep `-c 32768 -ub 512` (`./modchip` sets this automatically).
- **Matching MLA KV Cache Types & Vocabulary Isolation:** Always pass matching K and V types (`-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`) and set `RC_TABLES_DIR=/nonexistent` when running from a directory containing Nemotron RouteCrack tables.

### Option A: Automated `llama-modchip v3.0` Installer (`Linux`, `macOS Metal`, `Windows`)
```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./install.sh --preset dsv4-maxq
./modchip chat
```

### Option B: Bundled Portable `v3.0` `GLIBC_2.30` Linux `x86_64` Binary (`-avx512` or `-avx2`)
```bash
hf download Synchopate/DeepSeek-V4-Flash-0731-MaxQ-v4-64GB-CPU-GGUF --local-dir ./dsv4-maxq-v4
cd ./dsv4-maxq-v4
sha256sum -c SHA256SUMS --ignore-missing

BIN="./llama-server-modchip-linux-x86_64-$(grep -q 'avx512f' /proc/cpuinfo && grep -Eq 'avx512_?vbmi' /proc/cpuinfo && grep -Eq 'avx512_?vnni' /proc/cpuinfo && echo avx512 || echo avx2)"
chmod +x "$BIN"
N_CORES=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
THREADS=$(( N_CORES > 1 ? N_CORES - 1 : 1 ))

env RC_TABLES_DIR=/nonexistent "$BIN"   -m DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf   -c 32768 -np 1 -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --repack --jinja   --spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 7 --spec-ngram-simple-min-hits 1   -t "$THREADS" -tb "$THREADS" --poll 0 --poll-batch 0 --port 8080
```

---

## Credits

- **Salvatore Sanfilippo ([`@antirez`](https://github.com/antirez)) — [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf):** Single-pass routed-MoE quantization research and the public `1.5M`-token `DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat` (`@ f71f23d552d6`) importance matrix.
- **AtomicChat ([`AtomicChat/dsv4-eval-artifacts`](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)):** Open `wiki-alt.txt` + `ref5632.kld` (`-c 5632 -b 5632`) long-context KLD/perplexity evaluation suite.
- **DeepSeek AI ([`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)):** Base `284.33B` `DeepSeek-V4-Flash-0731` model (`@ 7872f01b1d1f`, MIT License).
