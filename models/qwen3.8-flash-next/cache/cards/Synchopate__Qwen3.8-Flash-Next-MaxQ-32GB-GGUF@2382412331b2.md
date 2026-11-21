---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
quantized_by: Synchopate
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - qwen3.8
  - qwen4exp
  - moe
  - gated-deltanet
  - maxq
  - 32gb-ram
  - cpu-inference
  - llama-cpp
  - llama-modchip
---

# Qwen3.8-Flash-Next-MaxQ-32GB-GGUF (`24.98 GiB` Resident Trunk + `10.73 GiB` Lazy NVMe PLE · Built-In `NextN` MTP)

**`Qwen3.8-Flash-Next-MaxQ-32GB`** is the **32 GB consumer RAM tier** edition of `Qwen3.8-Flash-Next` in the `MaxQ v3.1.1` lineup, built 100% donor-free from [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`rev 812d96e10a0dd82b69d749e68693698cf6c36daa`).

Across all `18` published `Unsloth` and `ISTA-DASLab` `Qwen3.8-Flash-Next` GGUF checkpoints (`54.40 GiB` to `329.72 GiB`), **`0 / 18` fit inside a 32 GB RAM machine (`<= 28.5 GiB`)** because the uncompressed `IQ4_NL` Per-Layer Embedding (`PLE`) table alone consumes `26.82 GiB`. `MaxQ-32GB` solves this via a two-shard Hybrid-RAM architecture:
1. **Shard 1 (`24.98 GiB` / `26,821,627,808 B`, resident in DDR5 RAM):** Retains **all `512 / 512` routed experts** (`42` ConeCollapsed `K = 512` interior layers with `42 × F32` `ffn_down_exps.bias` + `7` full-width `K = 640` boundary layers, `FP64 Woodbury DualGram`) **plus the built-in `blk.48` `NextN` Multi-Token Prediction (`MTP`) module**.
2. **Shard 2 (`10.73 GiB` / `11,520,055,616 B`, demand-paged on NVMe via `mmap`):** Compresses `320,001,536` PLE rows into a `D = 256` `PCA-Wiener + RaBitQ Q1_0 (41)` subspace (`2 MiB` inward-aligned `MADV_HUGEPAGE` ensures cold PLE pages are never pre-faulted into RAM).

Certified under strict cold-cache **Linux Cgroup v2 (`memory.max = 28.5 GiB`, `MemorySwapMax = 0`, `drop_caches=3`, `pgsteal = 0`, `workingset_refault = 0`, `oom_kill = 0`, `n = 3` cold restarts in [`qwen38_32gb_cgroup28g_v311.json`](https://github.com/synchopate/llama-modchip/blob/main/benchmarks/results/qwen38_32gb_cgroup28g_v311.json))**, `MaxQ-32GB` peaks at **`28.08 GiB` VmHWM / `28.26 GiB` cgroup peak (`<= 28.5 GiB`, tight `0.24 GiB` headroom at `-c 32768`, `cg.file = 25.14 GiB` of `35.71 GiB` on disk — proving `>10.5 GiB` PLE stays lazily unfaulted)** and delivers **`14.58 ± 0.32 tok/s` `RouteCrack 2.0`** (`10.00 ± 0.01` `N=1`; earlier warm-cache `v3.0` gate: `26.91–27.44 GiB`, `11.45–14.29 tok/s` `tg128`, `74.55 tok/s` `pp512`) on a 16-core Zen 5 CPU (`0` GPUs), coming within `0.077` Mean KLD (`1.0069` vs `0.9302`) of `54.40 GiB` pruned competitors despite a **`-18.69 GiB` smaller total file size**.

> [!IMPORTANT]
> **Mandatory `MaxQ-32GB` Sampling Configuration (`1.63 bpw` Trunk):**
> Because `MaxQ-32GB` compresses the 48-layer + MTP trunk into `24.98 GiB` (`1.63 bpw`), pure greedy decoding (`temperature = 0`) can enter repetition loops on open-ended reasoning (`2 / 8` strict pass at `temp = 0`, with `3 / 8` greedy repetition loops without sampling penalties). **Always configure the following sampling parameters for general coding, chat, and chain-of-thought** (reserving `temperature = 0` strictly for short structured JSON/SQL extraction):
> - `temperature = 0.6`
> - `top_p = 0.95`, `top_k = 20`
> - `presence_penalty = 0.2` – `1.0` (default `0.2`)
> - `repeat_penalty = 1.05`
>
> *(Note: `./modchip` and `install.sh --preset qwen38-32gb` pass `--temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.2 --repeat-penalty 1.05` automatically.)*

---

## Architecture & Parameter Summary (`general.size_label` Clarification)

> [!NOTE]
> **Clarification on Hugging Face Hub's Auto-Detected `"5.0B"` Badge:** Shard 1 carries `general.size_label = "5.0B"` inherited from the non-MoE dense trunk staging header during conversion. This checkpoint contains the **complete `512`-expert `Qwen3.8-Flash-Next` architecture** (`42` `K = 512` ConeCollapsed layers + `7` `K = 640` full-width layers + `320M`-row PLE + `blk.48` NextN MTP):

| Architectural Component | Specification |
| :--- | :--- |
| **Upstream Base Model** | `Qwen/Qwen3.8-Flash-Next` (`rev 812d96e10a0dd82b69d749e68693698cf6c36daa`, `329.72 GiB` BF16) |
| **Nominal / Collapsed Params** | **`~176.5B` nominal** (`~152.3B` post-ConeCollapse + PLE; `92.28%` mean retained spectral energy across 49 layers) |
| **Active Transformer Params / Token** | **`~4.6B–5.0B`** (`10` of `512` routed experts + `1` shared expert + dense GDN/GQA; reflected in `general.size_label`) |
| **Backbone Topology (`qwen4exp`)** | `48` Trunk Layers (`36` Gated DeltaNet `[48×128×128]` + `12` GQA Full Attention, `compress_ratio = 4`) + **`1` Built-In `NextN` MTP Layer (`blk.48`)** |
| **MoE Width & Routing** | **`512 / 512` Routed Experts** (`Top-10` active; `42 × K = 512` ConeCollapsed layers with `42 × F32` `ffn_down_exps.bias` + `7 × K = 640` boundary layers) |
| **Per-Layer Embedding (`PLE`)** | `320,001,536` rows (`D = 256` `PCA-Wiener` subspace in `Q1_0 = 41`, lazy demand-paged via NVMe `mmap`) |
| **GGUF Tensor Types (`1,128` LLM)** | `{F32: 393, Q8_0: 291, Q4_K: 99, Q5_K: 98, IQ1_S: 62, IQ1_M: 51, BF16: 49, MXFP4(39): 49, IQ2_XXS: 17, IQ2_XS: 10, Q2_0(42): 7, Q6_K: 1, Q1_0(41): 1}` (**`max_type = 42`, `0` custom types**) |

---

## File Manifest & Exact Byte / SHA-256 Verification

| File | Exact Size (`Bytes`) | Size (`GiB` / `GB`) | Tensors | Role & Notes | `SHA-256` |
| :--- | ---: | :---: | ---: | :--- | :--- |
| `Qwen3.8-Flash-Next-MaxQ-32GB-00001-of-00002.gguf` | **`26,821,627,808`** | `24.9796 GiB` (`26.822 GB`) | `1,127` | Resident Trunk (`48L` + `42` bias) + Built-In `blk.48` `NextN` MTP | `06cb65f9cf6e510cd16500ca1719701a0ffa13dcaa4c4d0cc135e4f2071a8b4d` |
| `Qwen3.8-Flash-Next-MaxQ-32GB-00002-of-00002.gguf` | **`11,520,055,616`** | `10.7289 GiB` (`11.520 GB`) | `1` | Lazy NVMe `mmap` `per_layer_token_embd.weight [256, 320001536]` (`Q1_0 = 41`) | `86b79714867743ddf5911ee7dff2717ae275f870551d86469cbdbf026c6b552e` |
| **LLM Total (Shards 1 + 2)** | **`38,341,683,424`** | **`35.7085 GiB` (`38.342 GB`)** | **`1,128`** | **Complete 32 GB Hybrid-RAM LLM (`28.26 GiB` cold cgroup peak `<= 28.5 GiB`, `0.24 GiB` headroom; `26.91–27.44 GiB` warm-cache gate)** | — |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` *(Optional)* | **`907,542,880`** | `0.8452 GiB` (`0.908 GB`) | `334` | Multimodal Vision Projector (`224 F32 + 110 BF16`) | `ef2af1d6aa8a22740c7252bc7ce40c5c2f1249aa8d9c8499a4d4572c3cfdf9df` |

---

## Quickstart (`llama-modchip v3.1.1` on 32 GB RAM)

### Option A — Automated 1-Command Setup (`synchopate/llama-modchip`)

```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./install.sh --preset qwen38-32gb
./modchip chat
```

### Option B — Direct `llama-server` Launch (Portable `GLIBC_2.30` Linux `x86_64` Binary)

```bash
# 1. Download portable v3.1.1 server binary (static libstdc++ & libgomp, GLIBC <= 2.30) and verify SHA-256:
BIN_ISA=$(grep -q "avx512f" /proc/cpuinfo && grep -Eq "avx512_?vbmi" /proc/cpuinfo && grep -Eq "avx512_?vnni" /proc/cpuinfo && echo "avx512" || echo "avx2")
curl -fLO "https://github.com/synchopate/llama-modchip/releases/download/v3.1.1/llama-server-modchip-linux-x86_64-${BIN_ISA}"
curl -fLO "https://github.com/synchopate/llama-modchip/releases/download/v3.1.1/SHA256SUMS"
sha256sum -c SHA256SUMS --ignore-missing
chmod +x "llama-server-modchip-linux-x86_64-${BIN_ISA}"

# 2. Download Shards 1 & 2 onto an NVMe SSD:
hf download Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF \
  --include "Qwen3.8-Flash-Next-MaxQ-32GB-*.gguf" --local-dir ./qwen38-32gb

# 3. Launch with mmap enabled (DEFAULT — do NOT pass --no-mmap!), non-convoy threading, and recommended 32GB sampling:
N_CORES=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
THREADS=$(( N_CORES > 1 ? N_CORES - 1 : 1 ))

env RC_TABLES_DIR=/nonexistent "./llama-server-modchip-linux-x86_64-${BIN_ISA}" \
  -m ./qwen38-32gb/Qwen3.8-Flash-Next-MaxQ-32GB-00001-of-00002.gguf \
  -c 32768 -np 1 -fa on -ctk q8_0 -ctv q8_0 --spec-type routecrack2 --jinja --ctx-checkpoints 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.2 --repeat-penalty 1.05 \
  -t "$THREADS" -tb "$THREADS" --poll 0 --poll-batch 0 --slot-save-path /dev/shm --port 8080
```

### Important 32 GB Deployment & Upstream `llama.cpp` Notes
1. **Keep Default `mmap` Enabled (`NEVER --no-mmap` on 32 GB Hosts):** `MaxQ-32GB` relies on OS `mmap` demand-paging for Shard 2 (`10.73 GiB` PLE) while keeping Shard 1 (`24.98 GiB`) resident in RAM (`28.08 GiB` VmHWM / `28.26 GiB` cold cgroup peak `<= 28.5 GiB`, tight `0.24 GiB` headroom at `-c 32768`; `26.91–27.44 GiB` warm-cache gate). On 32 GB desktop Linux hosts, keep default `mmap` enabled (and optionally wrap with `systemd-run --user --scope -p MemoryMax=28.5G`) so Linux automatically reclaims cold PLE pages.
2. **`--ctx-checkpoints 0` for Tight-RAM `/dev/shm` Snapshots:** Passing `--ctx-checkpoints 0` halves `Qwen3.8` Slot-0 snapshot size from `233.14 MiB` (`244,461,872 B`) down to `~120.6 MiB` and halves `/dev/shm` restore latency from **`38.49 ms` avg (`38.16 ms` min, `6.11 GiB/s`, `162×` faster than `6.18s` prefill, bit-exact `SHA256 = 8e5a6444...`)** down to **`~19 ms`**.
3. **`<think>` Token Budget (`max_tokens >= 16384`):** Set **`max_tokens >= 16384`** when `enable_thinking=True` (in `Step 3` `4,096`-token evals, `7/8` prompts triggered deterministic `enable_thinking=False` fallback when `<think>` exceeded `4,096` tokens), or set `enable_thinking=False` for `<= 4,096` token caps.
4. **Loader Requirement (`ConeCollapse` `ffn_down_exps.bias`):** Although `MaxQ-32GB` uses zero custom tensor types (`max_type = 42`), its `42` ConeCollapsed layers (`K = 640 -> 512`) include `42 × F32` `ffn_down_exps.bias` tensors (`GGML_OP_ADD_ID`), **requiring `llama-modchip v3.1.1+`** (do not expect unmodified stock Ollama/LM Studio to load `ffn_down_exps.bias`). `llama-modchip v3.1.1` runs cleanly with default `--repack` ON (`RssAnon = 0.69 GiB` `none` / `1.92 GiB` `draft-mtp`, `100%` bit-exact greedy SHA-256 `8b9bd8c4...`).
5. **Opt-In `LLAMA_MOE_TOP_K=8` Speed Knob:** Default routing uses all `10` active experts (`TOP_K=10`). Setting `env LLAMA_MOE_TOP_K=8` boosts `draft-mtp` `tg128` by `+5.74%` (`11.45 -> 12.11 t/s`) and `pp512` by `+8.50%` (`74.55 -> 80.89 t/s`) at the cost of `+0.4131` 40-chunk PPL (`5.8041 -> 6.2171`); keep `TOP_K=10` default unless extra throughput is prioritized.

---

## Empirical Benchmarks & Hardware Cgroup v2 Certification

### 1. Blind `wiki-alt.txt` (`40 × 512` Chunks) PPL + KLD vs. Donor-Free `BF16` Reference (`2.776624 ± 0.054868`)

| Checkpoint | Exact LLM Bytes (`GiB`) | Trunk + PLE (`GiB`) | `Mean PPL(Q) ± SE` | `Mean KLD ± SE` | `Median KLD` | `Same top p (%)` | Fits `32 GB` (`<= 28.5 GiB`)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MaxQ-32GB`** | **`38,341,683,424 B`** (`35.7085 GiB`) | **`24.980 + 10.729`** (`512/512` exps + MTP) | **`5.812001 ± 0.140197`** *(`5.8041` repack)* | **`1.006916 ± 0.014944`** | **`0.398583`** | **`66.275 ± 0.468%`** | **YES (`28.26 GiB <= 28.5 GiB` cold cg; `26.91–27.44` warm-cache)** |
| `ISTA-DASLab GSQ-RCO-Coder-IQ1_M` | `58,408,584,928 B` (`54.3970 GiB`) | `27.575 + 26.822` (`256/512` pruned, no MTP) | `5.091640 ± 0.121418` | `0.930241 ± 0.015625` | `0.296442` | `69.382 ± 0.456%` | No (`38.55 GiB` `VmHWM`) |
| **`MaxQ-64GB`** *(64 GB Sibling)* | `57,969,748,352 B` (`53.9885 GiB`) | `43.260 + 10.729` (`512/512` exps + MTP) | `3.432500 ± 0.070829` | `0.497117 ± 0.009892` | `0.105952` | `78.314 ± 0.408%` | No (`48.35 GiB` cold cg peak / `45.08 GiB` warm-cache) |

- **Blind Deterministic Suite Disclosure (`temp = 0`):** On the 8-prompt blind deterministic suite at greedy `temp = 0`, `MaxQ-32GB` passed `2 / 8` (`03_sql_window_grace_islands` `4/4` exact SQLite rows at `10.44 t/s`, `91.8%` MTP accept; `06_json_charterparty_settlement` exact `$15,600.0` JSON at `10.27 t/s`, `93.0%` MTP accept), hit the `4,096`-token length cap on `01_cpp` & `04_math` (`0` loops), and exhibited greedy `temp = 0` repetition loops on `05/07/08` without sampling penalties — motivating the mandatory `temp = 0.6, presence_penalty = 0.2, repeat_penalty = 1.05` defaults above.

### 2. Strict Cgroup v2 `28.5 GiB` Verification, `v3.0` Speedup & `248K` Speculative Shootout

Measured on 16 Zen 5 cores (`mmap` enabled, default `--repack` ON, `RC_TABLES_DIR=/nonexistent`):

| Mode & Workload (`-c 4096` / `-c 32768`, `-fa on -ctk q8_0 -ctv q8_0`) | Stage 4C (`-t 16 --poll 50 -nr`) | **`llama-modchip v3.1.1` (`-t 15 -tb 15 --poll 0`, `--repack` ON)¹** | Combined Gain¹ | MTP / Draft Accept | Peak `VmHWM`² (`RssAnon`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure `N=1` Decode (`--spec-type none`, `p512`)** | `68.66 pp` / `6.08 t/s tg` | **`82.11 t/s pp512`** / **`6.67 t/s tg128`** *(peak `86.28 pp`)* | **`+19.6% pp` / `+9.7% tg`** | — | `26.3 GiB` (`0.69 GiB`) |
| **Built-In `draft-mtp` (Default, `p512`)** | `61.39 pp` / `9.46 t/s tg` *(`8.39` pre-`v3.0` repack)* | **`74.55 t/s pp512`** / **`11.45 t/s tg128`** *(peak `11.69`)* | **`+21.5% pp` / `+21.1% tg`** *(`1.72×` vs `none`)* | **`73.73%`** (`261/354`) | **`26.91–27.44 GiB` warm-cache** (`1.92 GiB`) |
| **Built-In `draft-mtp` (`inplace_rust_code_edit`, cold `rep 0`)** | — | **`70.28 t/s pp`** / **`13.76 t/s tg128`** *(`14.29` 3-rep avg, `14.77` peak)* | **`2.06×` vs `none`** | **`100.00%`** (`285/285`) | **`26.91–27.44 GiB` warm-cache** |
| **Zero-Weight `Stockfish-PVS v2` (`inplace_edit`, cold `rep 0`)³** | — | **`78.03 t/s pp`** / **`12.82 t/s tg128`** | **`1.87×` vs `none`** *(`+11.0%` faster `pp`)* | **`91.25%`** (cold) | **`26.38 GiB`** (`-0.53 GiB`) |
| **Cold-Cache Cgroup v2 `32K` (`memory.max = 28.5 GiB`, `Swap = 0`, `n=3` cold restarts)** | **`10.00 ± 0.01`** (`N=1`) | **`14.58 ± 0.32 agg / 15.31 W1`** (`RC2`, `pgsteal=0`, `oom_kill=0`) | **`+45.8%`** | **`28.26 GiB` cg.peak** | **`28.08 GiB`** (`qwen38_32gb_cgroup28g_v311.json`) |

1. **Transparent Speedup Attribution:** Stage 4C ran oversubscribed `-t 16 --poll 50 --no-repack` on 16 logical CPUs, whereas Stage 4D-1 ran `-t 15 -tb 15 --poll 0 --poll-batch 0` with `v3.0` default `--repack` ON. The `+21.1%` (`+36.5%` vs pre-`v3.0` repack) decode gain reflects the **combined** `llama-modchip v3.1.1` SIMD/GDN/repack optimizations plus non-convoy `-t $((N_CORES - 1)) --poll 0` scheduling (`step0` isolated `-t 15 --poll 0` at `+25.9% pp / +31.3% tg` on `32GB none`).
2. **Cold-Cache Cgroup v2 Proof (`qwen38_32gb_cgroup28g_v311.json`):** After hypervisor `drop_caches=3` + `POSIX_FADV_DONTNEED` (`Cached = 0.3 GiB`, `MemoryMax = 28.5 GiB`, `MemorySwapMax = 0`, `-c 32768`, `n = 3` independent cold restarts), `MaxQ-32GB` records **`28.08 GiB` VmHWM / `28.26 GiB` cgroup peak** (`26.11` / `26.79 GiB` `N=1`, tight `0.24 GiB` headroom at `-c 32768`, `pgsteal = 0`, `workingset_refault = 0`, `cg.file = 25.14 GiB` out of `35.71 GiB` on disk — proving **`> 10.5 GiB` PLE stays lazily unfaulted**) at **`14.58 ± 0.32 tok/s`** `RouteCrack 2.0` vs **`10.00 ± 0.01`** `N=1` (`swap.peak = 0 B`, `oom_kill = 0`).
3. **Speculative Shootout Note:** Built-in `draft-mtp` won cold single-turn `rep 0` `tg128` on **all `6 / 6` (`100%`)** prompts (`11.69 vs 6.97–7.55 t/s` on `p512`, `10.25 vs 6.50–7.45 t/s` on `blind07`, `13.76 vs 12.82 t/s` on `inplace_edit`). Zero-weight `Stockfish-PVS v2` (`--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 3 --spec-ngram-simple-min-hits 1`) achieved `12.82 t/s` (`1.87×`) cold `rep 0` on source-in-context refactoring with `+11.0%` faster prefill and `-0.53 GiB` lower `VmHWM`, making it an ideal option for repetitive code-edit loops or MTP-less checkpoints.

---

## Portable `llama-modchip v3.1.1` Linux `x86_64` Binaries (`GLIBC <= 2.30`)

Download statically linked `v3.1.1` binaries (`libstdc++` + `libgomp` static, zero external OpenMP `.so`) from the [GitHub `v3.1.1` Release](https://github.com/synchopate/llama-modchip/releases/tag/v3.1.1) and verify against `SHA256SUMS`:

```text
b216b71a2f6939ff9396465a7ba2e8d220473d7a27050da0b87abd4a7bc430a4  llama-server-modchip-linux-x86_64-avx512
e1b90dc9d289aa91a2802db2b2694695663ea862205ab999fb22424d49a3cc47  llama-server-modchip-linux-x86_64-avx2
492b8084cf6cc11eea8317a07cb2f99b7323292036c108b02069a0bae81589ba  llama-cli-modchip-linux-x86_64-avx512
dee9ef216ef9ce106738b4ac78dcee45b1330982bf98eefd02f9258b017f3bfa  llama-cli-modchip-linux-x86_64-avx2
abc54c7960ca1cff00bcd9e0fd9b78f7400cc1f86629492127f8d481245d81b2  llama-modchip-v3.1.patch
```

- **64 GB RAM Hosts:** See **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF)** (`53.99 GiB`, `Mean PPL = 3.4325`, `15.34 agg / 21.38 W1 tok/s` `RouteCrack 2.0`).
- **Engine & CLI Repository:** **[`github.com/synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip)** (`Apache-2.0`; model weights governed by [`qwen-community-1.0`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)).
