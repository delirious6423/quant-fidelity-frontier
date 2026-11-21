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
  - cpu-inference
  - llama-cpp
  - llama-modchip
---

# Qwen3.8-Flash-Next-MaxQ-64GB-GGUF (`53.99 GiB` · 100% Donor-Free · Built-In `NextN` MTP)

**`Qwen3.8-Flash-Next-MaxQ-64GB`** is the flagship **64 GB RAM tier** checkpoint of the `MaxQ v3.1.1` lineup, built 100% donor-free from [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`rev 812d96e10a0dd82b69d749e68693698cf6c36daa`).

By compressing the `320,001,536`-row Per-Layer Embedding (`PLE`) table from `26.82 GiB` (`IQ4_NL`) down to **`10.73 GiB`** (`PCA-Wiener D=256 + RaBitQ Q1_0`), **`+16.09 GiB` of footprint is reinvested directly into keeping all `512 / 512` routed MoE experts unpruned (`K = 640`), `FP64 Woodbury DualGram` signed-scale `Q2_0`, and the built-in `blk.48` `NextN` Multi-Token Prediction (`MTP`) module (`0.81 GiB`)** — all using **100% standard `ggml` tensor types (`max_type = 42`, `0` custom `>= 43` types)**.

On held-out `wiki-alt.txt` (`40 × 512` chunks, `0.000000%` 8-gram and 13-gram overlap with calibration), **`MaxQ-64GB` (`53.99 GiB`) achieves `Mean PPL = 3.4325 ± 0.0708` and `Mean KLD = 0.4971 ± 0.0099`**, outperforming `ISTA-DASLab GSQ-RCO-Coder-IQ1_M` (`54.40 GiB`, `256/512` pruned experts, no MTP: `PPL = 5.0916`, `Mean KLD = 0.9302`) by **`-1.6591` PPL (`3.53×` smaller degradation over `BF16`)** and **`-46.6%` Mean KLD (`-64.3%` Median KLD)** *(note: `GSQ-RCO-Coder-IQ1_M` is the only published third-party `<= 55.5 GiB` `Qwen3.8` checkpoint and is coder-tuned with `256/512` experts pruned, evaluated here on general Wikipedia prose)* while running at **`15.34 tok/s` 4-workload aggregate / `21.38 tok/s` `W1` refactor (`+77.0%` vs `N=1` `8.67`, `+10.1%` vs `v3.0 draft-mtp` `13.94`, `79.95%` accept)** under **`RouteCrack 2.0` (`v3.1.1`)** (`71.97 tok/s` `pp512`, cold-cache `60.0 GiB` Cgroup v2 certified at **`48.09 GiB` VmHWM / `48.35 GiB` cgroup peak** (`pgsteal=0`, `workingset_refault=0`, `Swap=0`, `15.64 ± 0.05 agg / 22.00 W1 tok/s` across 3 cold restarts; `45.08 GiB` in the earlier warm-cache gate) on a `74.0 GiB` 16-core Zen 5 container with `0` GPUs).

---

## Architecture & Parameter Summary (`general.size_label` Clarification)

> [!NOTE]
> **Clarification on Hugging Face Hub's Auto-Detected `"5.0B"` Badge:** Shard 1 carries `general.size_label = "5.0B"` inherited from the non-MoE dense trunk staging header during conversion. This checkpoint contains the **complete, unpruned `176.5B` `Qwen3.8-Flash-Next` architecture** (`512/512` routed experts per layer + `320M`-row PLE table + `blk.48` NextN MTP), as detailed below:

| Architectural Component | Specification |
| :--- | :--- |
| **Upstream Base Model** | `Qwen/Qwen3.8-Flash-Next` (`rev 812d96e10a0dd82b69d749e68693698cf6c36daa`, `329.72 GiB` BF16) |
| **Total Parameters** | **`~176.5B`** (`~125.3B` Transformer + `512`-Expert MoE + `NextN` MTP + `51.2B` 320M-row PLE) |
| **Active Transformer Params / Token** | **`~5.0B`** (`10` of `512` routed experts + `1` shared expert + dense GDN/GQA; reflected in `general.size_label`) |
| **Backbone Topology (`qwen4exp`)** | `48` Trunk Layers (`36` Gated DeltaNet `[48×128×128]` + `12` GQA Full Attention, `compress_ratio = 4`) + **`1` Built-In `NextN` MTP Layer (`blk.48`)** |
| **MoE Width & Routing** | **`512 / 512` Unpruned Routed Experts** (`Top-10` active, `K = 640` full width) + `1` Shared Expert (`K = 640`), `0` `ffn_down_exps.bias` |
| **Per-Layer Embedding (`PLE`)** | `320,001,536` rows (`D = 256` `PCA-Wiener` subspace in `Q1_0 = 41`, `qwen4exp.embedding_length_per_layer_input = 256`) |
| **Vocabulary & Multimodal** | `248,320` tokens + optional `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (`0.85 GiB` BF16 vision projector) |
| **GGUF Tensor Types (`1,086` LLM)** | `{F32: 351, Q8_0: 339, Q6_K: 198, IQ2_S: 60, BF16: 49, Q2_0(42): 39, IQ3_XXS: 36, MXFP4(39): 7, IQ4_NL: 4, Q2_K: 2, Q1_0(41): 1}` (**`max_type = 42`, `0` custom types**) |

---

## File Manifest & Exact Byte / SHA-256 Verification

| File | Exact Size (`Bytes`) | Size (`GiB` / `GB`) | Tensors | Role & Notes | `SHA-256` |
| :--- | ---: | :---: | ---: | :--- | :--- |
| `Qwen3.8-Flash-Next-MaxQ-64GB-00001-of-00002.gguf` | **`46,449,692,736`** | `43.2596 GiB` (`46.450 GB`) | `1,085` | Trunk (`48L`, `512/512` exps) + Built-In `blk.48` `NextN` MTP | `f2b8bc79acb4b4d79033acd91c9c360ba96da92cd5841512befc450e15f4786f` |
| `Qwen3.8-Flash-Next-MaxQ-64GB-00002-of-00002.gguf` | **`11,520,055,616`** | `10.7289 GiB` (`11.520 GB`) | `1` | `per_layer_token_embd.weight [256, 320001536]` (`Q1_0 = 41`) | `86b79714867743ddf5911ee7dff2717ae275f870551d86469cbdbf026c6b552e` |
| **LLM Total (Shards 1 + 2)** | **`57,969,748,352`** | **`53.9885 GiB` (`57.970 GB`)** | **`1,086`** | **Complete 64 GB RAM LLM (`< 55.5 GiB` ceiling)** | — |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` *(Optional)* | **`907,542,880`** | `0.8452 GiB` (`0.908 GB`) | `334` | Multimodal Vision Projector (`224 F32 + 110 BF16`) | `ef2af1d6aa8a22740c7252bc7ce40c5c2f1249aa8d9c8499a4d4572c3cfdf9df` |

---

## Quickstart (`llama-modchip v3.1.1`)

### Option A — Automated 1-Command Setup (`synchopate/llama-modchip`)

```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./install.sh --preset qwen38-64gb
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

# 2. Download Shards 1 & 2 (and optional mmproj for vision):
hf download Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF \
  --include "Qwen3.8-Flash-Next-MaxQ-64GB-*.gguf" --local-dir ./qwen38-64gb

# 3. Launch with non-convoy threading (-t $((N_CORES - 1)) --poll 0), default --repack ON, and RouteCrack 2.0:
N_CORES=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
THREADS=$(( N_CORES > 1 ? N_CORES - 1 : 1 ))

env RC_TABLES_DIR=/nonexistent "./llama-server-modchip-linux-x86_64-${BIN_ISA}" \
  -m ./qwen38-64gb/Qwen3.8-Flash-Next-MaxQ-64GB-00001-of-00002.gguf \
  -c 32768 -np 1 -fa on -ctk q8_0 -ctv q8_0 --spec-type routecrack2 --jinja \
  -t "$THREADS" -tb "$THREADS" --poll 0 --poll-batch 0 --slot-save-path /dev/shm --port 8080
```

### Important Runtime Notes & Upstream `llama.cpp` Compatibility
1. **`--repack` on `llama-modchip v3.1.1` vs. Unpatched Upstream `llama.cpp`:**
   - **`llama-modchip v3.1.1`** fixes upstream `CPU_REPACK` `q8_cache` / `tiled.cpp` / `repack.cpp` bugs and keeps 3D MoE tensors zero-copy `mmap`-backed (`RssAnon = 1.81 GiB`, `93.0%` MTP acceptance) **out of the box with default `--repack` ON**.
   - If loading on **unpatched upstream `llama.cpp`** (requires `qwen4exp` `ple_head_dim * ple_n_heads` support), **pass `--no-repack` (`-nr`)** to avoid upstream `CPU_REPACK` `q8_cache` corruption and anonymous heap bloat.
2. **`<think>` Reasoning Token Budget (`max_tokens >= 16384`):**
   - `Qwen3.8-Flash-Next` generates detailed chain-of-thought reasoning inside `<think>...</think>` blocks that routinely exceed `4,096` tokens on multi-step coding, math, and physics problems. Set **`max_tokens >= 16384`** when `enable_thinking=True`, or pass `chat_template_kwargs={"enable_thinking": false}` when operating under `<= 4,096` output token caps.
3. **RouteCrack Vocabulary Isolation (`RC_TABLES_DIR=/nonexistent`):**
   - When running `llama-modchip` from a directory that contains `./workspace/routecrack` Nemotron tables, always set `RC_TABLES_DIR=/nonexistent` (done automatically by `./modchip`) so `Stockfish-PVS v2` runs in clean 32-bit `248K`-vocab online session mode.

---

## Empirical Benchmarks & Blind Certification

### 1. Blind `wiki-alt.txt` (`40 × 512` Chunks) PPL + KLD vs. Donor-Free `BF16` Reference & Matched Competitor

Evaluated on `40 × 512` held-out `wiki-alt.txt` chunks (`0.000000%` whitespace and normalized alphanumeric 8-gram and 13-gram overlap across calibration, blind prompts, and `wiki-alt.txt`) against the donor-free `BF16` reference logit base (`Mean PPL(base) = 2.776624 ± 0.054868`):

| Checkpoint | Exact LLM Bytes (`GiB`) | Trunk + PLE (`GiB`) | Built-In MTP | `Mean PPL(Q) ± SE` | `ln(PPL(Q)/PPL(base))` | `Mean KLD ± SE` | `Median KLD` | `90% / 99% / Max KLD` | `Same top p (%)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MaxQ-64GB`** | **`57,969,748,352 B`** (`53.9885 GiB`) | `43.260 + 10.729` (`512/512` exps) | **Yes (`blk.48`)** | **`3.432500 ± 0.070829`** | **`0.212053 ± 0.012342`** | **`0.497117 ± 0.009892`** | **`0.105952`** | **`1.4579 / 5.1231 / 11.9591`** | **`78.314 ± 0.408%`** |
| `ISTA-DASLab GSQ-RCO-Coder-IQ1_M` | `58,408,584,928 B` (`54.3970 GiB`) | `27.575 + 26.822` (`256/512` pruned) | No | `5.091640 ± 0.121418` | `0.606364 ± 0.018576` | `0.930241 ± 0.015625` | `0.296442` | `2.6974 / 7.8384 / 15.3380` | `69.382 ± 0.456%` |
| **`MaxQ-32GB`** *(32 GB Sibling)* | `38,341,683,424 B` (`35.7085 GiB`) | `24.980 + 10.729` (`512/512` exps) | **Yes (`blk.48`)** | `5.812001 ± 0.140197` | `0.738689 ± 0.018091` | `1.006916 ± 0.014944` | `0.398583` | `2.8893 / 7.0255 / 13.8770` | `66.275 ± 0.468%` |

- **18-Checkpoint Hugging Face Landscape Audit:** Across all `18` published `Unsloth` and `ISTA-DASLab` `Qwen3.8-Flash-Next` GGUFs (`54.40 GiB` to `329.72 GiB`), **`0 / 18` fit 32 GB RAM (`<= 28.5 GiB`)** and only **`1 / 18` fits 64 GB RAM (`<= 55.5 GiB`)** (`ISTA-DASLab GSQ-RCO-Coder-IQ1_M` at `54.397 GiB`, which `MaxQ-64GB` beats by `-1.6591` PPL and `-46.56%` Mean KLD while `0.41 GiB` smaller and including built-in `blk.48` MTP; Unsloth `UD-IQ1_S` starts at `61.820 GiB` with `26.822 GiB` uncompressed `IQ4_NL` PLE).
- **Numerical Equivalence (`v3.0` Default `--repack` vs. `--no-repack`):** Full 40-chunk `v3.0` default `--repack` achieves `Mean PPL = 3.43825 ± 0.07119`, `Mean KLD = 0.49803`, `Median KLD = 0.10561`, `Same top p = 78.314%`; `batch=1` (`32 × 248,320` logits `--repack` vs `--no-repack`) yields **`KLD = 0.00000 ± 0.00000`** (`Same top p = 100.000%`), and 128-token greedy generation is `100%` bit-exact (`SHA256 = 7d330f8d...`) across `--no-repack none`, `--repack none`, and `--repack draft-mtp`.

### 2. Blind 8-Prompt Coding & Reasoning Verification (`temp=0`, `max_tokens=4096`)

- **`5 / 8` strict deterministic pass (`5 / 5 = 100%` on all prompts completing within the `4,096`-token cap; `8 / 8 = 100%` zero repetition loops; `73.1%–93.5%` MTP acceptance):**
  - **Passed (`5/5` completed):** `01_cpp20_lockfree_ring` (`g++ -std=c++20 -O2 -Wall -Wextra -Werror -pthread` `rc=0`, `0` warnings, `alignas(64)` + `std::byte` + `acquire`/`release`, 500K-item 2-thread SPSC FIFO `All tests passed`; *note: unprompted `std::launder` metadata string omitted while functional compile+run passed 100%*), `02_python_dp_relay_line` (`4/4` exact DP + lexicographic tie-breaking), `03_sql_window_grace_islands` (`4/4` exact SQLite rows, `0` recursive CTEs), `05_italian_embedded_telemetry_arch` (`<= 48 MiB` ECC SRAM + `CRC32C`/`PTP`/`TCXO`), and `06_json_charterparty_settlement` (raw JSON, exact `$15,600.0` demurrage, `93.5%` MTP accept).
  - **Token-Cap Truncated at `4,096` (`3/8`, `0` loops):** `04_olympiad_sophie_germain_gcd`, `07_rust_borrow_checker_refactor`, and `08_physics_inductive_rail_ode` (`3/4` exact physical constants derived with `< 1.2e-7` relative error right at the `4,096`-token cutoff). *(Note: Under the `4,096`-token harness cap, `6/8` prompts triggered deterministic `enable_thinking=False` fallback when `<think>` exceeded `4,096` tokens; set `max_tokens >= 16384` in production when `<think>` is enabled.)*

### 3. `llama-modchip v3.1.1` `RouteCrack 2.0` Throughput, `60 GiB` Cgroup v2 Memory & Historical `v3.0` Baseline

- **`v3.1.1` `RouteCrack 2.0` (`--spec-type routecrack2`, 16C Zen 5, `0` GPU, `74.0 GiB` container; raw log [`benchmarks/results/qwen38_routecrack2_v31.json`](https://github.com/synchopate/llama-modchip/blob/main/benchmarks/results/qwen38_routecrack2_v31.json)):** **`15.34 tok/s` 4-workload aggregate** (`+77.0%` vs `N=1` `8.67`, `+27.9%` vs `MTP-1` `12.00`, `+10.1%` vs `v3.0 draft-mtp` `13.94`; **`W1` refactor `21.38`**, `W2` JSON `17.38`, `W3` C99 `12.63`, `W4` prose `12.78`; `79.95%` accept, `3.66` mean len; `254/254` greedy bit-exact on `W1+W2`, with batch-GEMM FP rounding drift on high-entropy `W3`/`W4` identical to upstream `MTP-1`/`draft-mtp`). Strict cold-cache `60.0 GiB` Cgroup v2 (`MemorySwapMax=0`, hypervisor `drop_caches=3` + `POSIX_FADV_DONTNEED`, guest `Cached = 0.3 GiB` before each load, `-c 32768`, `n = 3` independent cold server restarts × 4 workloads, `swap.peak = 0 B`, `oom_kill = 0`, `pgsteal = 0`, `workingset_refault = 0`; [`benchmarks/results/qwen38_64gb_cgroup60g_v311.json`](https://github.com/synchopate/llama-modchip/blob/main/benchmarks/results/qwen38_64gb_cgroup60g_v311.json)): Pure `N=1` averages **`8.78 ± 0.02 tok/s`** (`95% CI ±0.05`, `45.46 GiB` VmHWM / `46.53 GiB` cg.peak), while `RouteCrack 2.0` reaches **`15.64 ± 0.05 tok/s`** (`95% CI ±0.11`, `+78.1%`, `W1 22.00 ± 0.19`, **`48.09 GiB` VmHWM / `48.35 GiB` cg.peak `<= 60.0 GiB`**, `cg.file = 43.40 GiB`).

Historical `v3.0` (`draft-mtp` / `Stockfish-PVS v2`) single-turn baselines:

Measured on 16 Zen 5 cores (`mmap` enabled, default `--repack` ON, `RC_TABLES_DIR=/nonexistent`):

| Mode & Workload (`-c 4096` / `-c 32768`, `-fa on -ctk q8_0 -ctv q8_0`) | Stage 4C (`-t 16 --poll 50 -nr`) | **`llama-modchip v3.1.1` (`-t 15 -tb 15 --poll 0`, `--repack` ON)¹** | Combined Gain¹ | MTP / Draft Accept | Peak `VmHWM`² (`RssAnon`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure `N=1` Decode (`--spec-type none`, `p512`)** | `42.17 pp` / `4.69 t/s tg` | **`72.18 t/s pp512`** / **`6.37 t/s tg128`** | **`+71.2% pp` / `+35.7% tg`** | — | `44.9 GiB` (`0.55 GiB`) |
| **Built-In `draft-mtp` (Default, `p512`)** | `39.24 pp` / `7.33 t/s tg` | **`71.97 t/s pp512`** / **`12.66 t/s tg128`** *(peak `12.82`)* | **`+83.4% pp` / `+72.8% tg`** *(`1.99×` vs `none`)* | **`93.00%`** (`279/300`) | **`45.08 GiB`** (`1.81 GiB`) |
| **Built-In `draft-mtp` (`inplace_rust_code_edit`, cold `rep 0`)** | — | **`70.8 t/s pp`** / **`12.99 t/s tg128`** | **`2.04×` vs `none`** | **`93%+`** | `45.08 GiB` (`1.81 GiB`) |
| **Zero-Weight `Stockfish-PVS v2` (`inplace_edit`, cold `rep 0`)³** | — | **`75.2 t/s pp`** / **`10.58 t/s tg128`** | **`1.67×` vs `none`** | **`91.25%`** (cold) | `44.23 GiB` (`-0.85 GiB`) |
| **Cold-Cache Cgroup v2 `32K` (`memory.max = 60.0 GiB`, `Swap = 0`, `n=3` cold restarts)** | **`8.78 ± 0.02`** (`N=1`) | **`15.64 ± 0.05 agg / 22.00 W1`** (`RC2`, `pgsteal=0`, `oom_kill=0`) | **`+78.1%`** | **`48.35 GiB` cg.peak** | **`48.09 GiB`** (`qwen38_64gb_cgroup60g_v311.json`) |

1. **Transparent Speedup Attribution:** Stage 4C ran oversubscribed `-t 16 --poll 50 --no-repack` on 16 logical CPUs, whereas Stage 4D-1 ran `-t 15 -tb 15 --poll 0 --poll-batch 0` with `v3.0` default `--repack` ON. The `+72.8%` decode (`7.33 -> 12.66 t/s`) and `+83.4%` prefill (`39.24 -> 71.97 t/s`) gains represent the **combined** `llama-modchip v3.1.1` SIMD (`Q1_0`/`Q2_0`/`MXFP4` AVX-512), multi-threaded 8-`zmm` `GATED_DELTA_NET`, and zero-copy 3D MoE `--repack` improvements plus non-convoy `-t $((N_CORES - 1)) --poll 0` scheduling.
2. **Cold-Cache Cgroup v2 Proof (`qwen38_64gb_cgroup60g_v311.json`):** After hypervisor `drop_caches=3` + `POSIX_FADV_DONTNEED` (`Cached = 0.3 GiB`, `MemoryMax = 60.0 GiB`, `MemorySwapMax = 0`, `-c 32768`, `n = 3` independent cold restarts), `MaxQ-64GB` records **`48.09 GiB` VmHWM / `48.35 GiB` cgroup peak** (`45.46` / `46.53 GiB` `N=1`, `pgsteal = 0`, `cg.file = 43.40 GiB` out of `53.99 GiB` on disk) at **`15.64 ± 0.05 tok/s`** `RouteCrack 2.0` (`W1 22.00 ± 0.19`) vs **`8.78 ± 0.02`** `N=1` (`swap.peak = 0 B`, `oom_kill = 0`).
3. **Speculative Shootout & Tie-Breaking Note:** Built-in `draft-mtp` won cold single-turn `rep 0` `tg128` on **all `6 / 6` (`100%`)** benchmark prompts and is the recommended default. Zero-weight `Stockfish-PVS v2` (`--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 3 --spec-ngram-simple-min-hits 1`) achieved `10.58 t/s` (`1.67×`) on cold `rep 0` source-in-context refactoring with `-0.85 GiB` lower `VmHWM`, making it an effective fallback for MTP-less checkpoints or repetitive edit loops. At `temp=0`, `N=1` (`none`) is `3/3` bit-exact everywhere while `N=2..4` speculative batch verification can flip benign 1-ULP FP32 ties on rare tokens without affecting output validity.

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

- **32 GB RAM Hosts:** See **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF)** (`24.98 GiB` Shard 1 + lazy NVMe PLE, cold-cache `28.5 GiB` Cgroup v2 certified at **`28.08 GiB` VmHWM / `28.26 GiB` cg.peak** (`0.24 GiB` headroom at 32K, `pgsteal=0`, `>10.5 GiB` PLE unfaulted, `14.58 ± 0.32 tok/s` `RC2`; `26.91–27.44 GiB` warm-cache gate)).
- **Engine & CLI Repository:** **[`github.com/synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip)** (`Apache-2.0`; model weights governed by [`qwen-community-1.0`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)).
