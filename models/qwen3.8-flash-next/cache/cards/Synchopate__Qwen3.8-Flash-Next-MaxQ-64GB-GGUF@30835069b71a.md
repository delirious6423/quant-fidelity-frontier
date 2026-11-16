---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
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

**`Qwen3.8-Flash-Next-MaxQ-64GB`** is the flagship **64 GB RAM tier** checkpoint of the `MaxQ v3.0` lineup, built 100% donor-free from [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`rev 812d96e10a0dd82b69d749e68693698cf6c36daa`).

By compressing the `320,001,536`-row Per-Layer Embedding (`PLE`) table from `26.82 GiB` (`IQ4_NL`) down to **`10.73 GiB`** (`PCA-Wiener D=256 + RaBitQ Q1_0`), **`+16.09 GiB` of footprint is reinvested directly into keeping all `512 / 512` routed MoE experts unpruned (`K = 640`), `FP64 Woodbury DualGram` signed-scale `Q2_0`, and the built-in `blk.48` `NextN` Multi-Token Prediction (`MTP`) module (`0.81 GiB`)** — all using **100% standard `ggml` tensor types (`max_type = 42`, `0` custom `>= 43` types)**.

On held-out `wiki-alt.txt` (`40 × 512` chunks, `0.000000%` 8-gram and 13-gram overlap with calibration), **`MaxQ-64GB` (`53.99 GiB`) achieves `Mean PPL = 3.4325 ± 0.0708` and `Mean KLD = 0.4971 ± 0.0099`**, outperforming `ISTA-DASLab GSQ-RCO-Coder-IQ1_M` (`54.40 GiB`, `256/512` pruned experts, no MTP: `PPL = 5.0916`, `Mean KLD = 0.9302`) by **`-1.6591` PPL (`3.53×` smaller degradation over `BF16`)** and **`-46.6%` Mean KLD (`-64.3%` Median KLD)** while running at **`12.66 tok/s` `tg128` (`93.0%` MTP acceptance, `71.97 tok/s` `pp512`, `VmHWM = 45.08 GiB`)** on a 16-core Zen 5 CPU with zero GPUs.

---

## Architecture & Parameter Summary (`general.size_label` Clarification)

> [!NOTE]
> **Clarification on Hugging Face Hub's Auto-Detected `"5.0B"` Badge:** Shard 1 carries `general.size_label = "5.0B"` inherited from the non-MoE dense trunk staging header during conversion. This checkpoint contains the **complete, unpruned `176.5B` `Qwen3.8-Flash-Next` architecture** (`512/512` routed experts per layer + `320M`-row PLE table + `blk.48` NextN MTP), as detailed below:

| Architectural Component | Specification |
| :--- | :--- |
| **Upstream Base Model** | `Qwen/Qwen3.8-Flash-Next` (`rev 812d96e10a0dd82b69d749e68693698cf6c36daa`, `329.72 GiB` BF16) |
| **Total Parameters** | **`~176.5B`** (`~125.3B` Transformer + `512`-Expert MoE + `NextN` MTP + `51.2B` 320M-row PLE) |
| **Active Transformer Params / Token** | **`~5.0B`** (`10` of `512` routed experts + `1` shared expert + dense GDN/QSA; reflected in `general.size_label`) |
| **Backbone Topology (`qwen4exp`)** | `48` Trunk Layers (`36` Gated DeltaNet `[48×128×128]` + `12` QSA Full Attention, `compress_ratio = 4`) + **`1` Built-In `NextN` MTP Layer (`blk.48`)** |
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

## Quickstart (`llama-modchip v3.0`)

### Option A — Automated 1-Command Setup (`synchopate/llama-modchip`)

```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./install.sh --preset qwen38-64gb
./modchip chat
```

### Option B — Direct `llama-server` Launch (Portable `GLIBC_2.30` Linux `x86_64` Binary)

```bash
# 1. Download portable v3.0 server binary (static libstdc++ & libgomp, GLIBC <= 2.30) and verify SHA-256:
BIN_ISA=$(grep -q "avx512f" /proc/cpuinfo && grep -Eq "avx512_?vbmi" /proc/cpuinfo && grep -Eq "avx512_?vnni" /proc/cpuinfo && echo "avx512" || echo "avx2")
curl -fLO "https://github.com/synchopate/llama-modchip/releases/download/v3.0/llama-server-modchip-linux-x86_64-${BIN_ISA}"
curl -fLO "https://github.com/synchopate/llama-modchip/releases/download/v3.0/SHA256SUMS"
sha256sum -c SHA256SUMS --ignore-missing
chmod +x "llama-server-modchip-linux-x86_64-${BIN_ISA}"

# 2. Download Shards 1 & 2 (and optional mmproj for vision):
hf download Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF \
  --include "Qwen3.8-Flash-Next-MaxQ-64GB-*.gguf" --local-dir ./qwen38-64gb

# 3. Launch with non-convoy threading (-t $((N_CORES - 1)) --poll 0), default --repack ON, and NextN draft-mtp:
N_CORES=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
THREADS=$(( N_CORES > 1 ? N_CORES - 1 : 1 ))

env RC_TABLES_DIR=/nonexistent "./llama-server-modchip-linux-x86_64-${BIN_ISA}" \
  -m ./qwen38-64gb/Qwen3.8-Flash-Next-MaxQ-64GB-00001-of-00002.gguf \
  -c 32768 -np 1 -fa on -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --jinja \
  -t "$THREADS" -tb "$THREADS" --poll 0 --poll-batch 0 --slot-save-path /dev/shm --port 8080
```

### Important Runtime Notes & Upstream `llama.cpp` Compatibility
1. **`--repack` on `llama-modchip v3.0` vs. Unpatched Upstream `llama.cpp`:**
   - **`llama-modchip v3.0`** fixes upstream `CPU_REPACK` `q8_cache` / `tiled.cpp` / `repack.cpp` bugs and keeps 3D MoE tensors zero-copy `mmap`-backed (`RssAnon = 1.81 GiB`, `93.0%` MTP acceptance) **out of the box with default `--repack` ON**.
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

### 3. `llama-modchip v3.0` Throughput, Cgroup v2 Memory & `248K` Speculative Shootout

Measured on 16 Zen 5 cores (`mmap` enabled, default `--repack` ON, `RC_TABLES_DIR=/nonexistent`):

| Mode & Workload (`-c 4096` / `-c 32768`, `-fa on -ctk q8_0 -ctv q8_0`) | Stage 4C (`-t 16 --poll 50 -nr`) | **`llama-modchip v3.0` (`-t 15 -tb 15 --poll 0`, `--repack` ON)¹** | Combined Gain¹ | MTP / Draft Accept | Peak `VmHWM`² (`RssAnon`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure `N=1` Decode (`--spec-type none`, `p512`)** | `42.17 pp` / `4.69 t/s tg` | **`72.18 t/s pp512`** / **`6.37 t/s tg128`** | **`+71.2% pp` / `+35.7% tg`** | — | `44.9 GiB` (`0.55 GiB`) |
| **Built-In `draft-mtp` (Default, `p512`)** | `39.24 pp` / `7.33 t/s tg` | **`71.97 t/s pp512`** / **`12.66 t/s tg128`** *(peak `12.82`)* | **`+83.4% pp` / `+72.8% tg`** *(`1.99×` vs `none`)* | **`93.00%`** (`279/300`) | **`45.08 GiB`** (`1.81 GiB`) |
| **Built-In `draft-mtp` (`inplace_rust_code_edit`, cold `rep 0`)** | — | **`70.8 t/s pp`** / **`12.99 t/s tg128`** | **`2.04×` vs `none`** | **`93%+`** | `45.08 GiB` (`1.81 GiB`) |
| **Zero-Weight `Stockfish-PVS v2` (`inplace_edit`, cold `rep 0`)³** | — | **`75.2 t/s pp`** / **`10.58 t/s tg128`** | **`1.67×` vs `none`** | **`91.25%`** (cold) | `44.23 GiB` (`-0.85 GiB`) |
| **Strict Cgroup v2 `32K` Context (`memory.max = 60.0 GiB`, `Swap = 0`)** | `23.0 pp` / `8.1 t/s tg` (`-nr`) | **`rc = 0`, `oom = 0`, `oom_kill = 0`** | **60 GiB Ceiling PASS** | `draft-mtp` | **`45.64 GiB < 60.0 GiB`²** |

1. **Transparent Speedup Attribution:** Stage 4C ran oversubscribed `-t 16 --poll 50 --no-repack` on 16 logical CPUs, whereas Stage 4D-1 ran `-t 15 -tb 15 --poll 0 --poll-batch 0` with `v3.0` default `--repack` ON. The `+72.8%` decode (`7.33 -> 12.66 t/s`) and `+83.4%` prefill (`39.24 -> 71.97 t/s`) gains represent the **combined** `llama-modchip v3.0` SIMD (`Q1_0`/`Q2_0`/`MXFP4` AVX-512), multi-threaded 8-`zmm` `GATED_DELTA_NET`, and zero-copy 3D MoE `--repack` improvements plus non-convoy `-t $((N_CORES - 1)) --poll 0` scheduling.
2. **Authoritative `VmHWM` Note:** Process `VmHWM` (`45.08–45.64 GiB < 60.0 GiB`) is the authoritative total resident RAM metric (`RssAnon + RssFile`), as warm Linux page-cache runs record only newly faulted delta pages in cgroup `memory.peak`.
3. **Speculative Shootout & Tie-Breaking Note:** Built-in `draft-mtp` won cold single-turn `rep 0` `tg128` on **all `6 / 6` (`100%`)** benchmark prompts and is the recommended default. Zero-weight `Stockfish-PVS v2` (`--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 3 --spec-ngram-simple-min-hits 1`) achieved `10.58 t/s` (`1.67×`) on cold `rep 0` source-in-context refactoring with `-0.85 GiB` lower `VmHWM`, making it an effective fallback for MTP-less checkpoints or repetitive edit loops. At `temp=0`, `N=1` (`none`) is `3/3` bit-exact everywhere while `N=2..4` speculative batch verification can flip benign 1-ULP FP32 ties on rare tokens without affecting output validity.

---

## Portable `v3.0` Binary Checksums (`GLIBC <= 2.30`)

```text
54af3232375ede820f91143c106a594c83b815a3dca66fe98ab08bcb8b63d4f0  llama-server-modchip-linux-x86_64-avx512
0f0c07f97a98d8aeb9a012264fdeff5a787d2104a2afb122052bfe0a21c2d1a9  llama-server-modchip-linux-x86_64-avx2
531d870988dcc580322341d2bbd7f0cf6b6cc3b3376a2e86d092a021f7cddba2  llama-cli-modchip-linux-x86_64-avx512
49a918206afe371d16ab14fd66d9cc87e35c7f8c0471c2cc13c759f2e0350d03  llama-cli-modchip-linux-x86_64-avx2
```

- **32 GB RAM Hosts:** See **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF)** (`24.98 GiB` Shard 1 + lazy NVMe PLE, `VmHWM = 26.91 GiB <= 28.5 GiB`, `11.45–14.29 tok/s`).
- **Engine & CLI Repository:** **[`github.com/synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip)** (`Apache-2.0`; model weights governed by [`qwen-community-1.0`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)).
