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
  - 32gb-ram
  - cpu-inference
  - llama-cpp
  - llama-modchip
---

# Qwen3.8-Flash-Next-MaxQ-32GB-GGUF (`24.98 GiB` Resident Trunk + `10.73 GiB` Lazy NVMe PLE · Built-In `NextN` MTP)

**`Qwen3.8-Flash-Next-MaxQ-32GB`** is the **32 GB consumer RAM tier** edition of `Qwen3.8-Flash-Next` in the `MaxQ v3.0` lineup, built 100% donor-free from [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`rev 812d96e10a0dd82b69d749e68693698cf6c36daa`).

Across all `18` published `Unsloth` and `ISTA-DASLab` `Qwen3.8-Flash-Next` GGUF checkpoints (`54.40 GiB` to `329.72 GiB`), **`0 / 18` fit inside a 32 GB RAM machine (`<= 28.5 GiB`)** because the uncompressed `IQ4_NL` Per-Layer Embedding (`PLE`) table alone consumes `26.82 GiB`. `MaxQ-32GB` solves this via a two-shard Hybrid-RAM architecture:
1. **Shard 1 (`24.98 GiB` / `26,821,627,808 B`, resident in DDR5 RAM):** Retains **all `512 / 512` routed experts** (`42` ConeCollapsed `K = 512` interior layers with `42 × F32` `ffn_down_exps.bias` + `7` full-width `K = 640` boundary layers, `FP64 Woodbury DualGram`) **plus the built-in `blk.48` `NextN` Multi-Token Prediction (`MTP`) module**.
2. **Shard 2 (`10.73 GiB` / `11,520,055,616 B`, demand-paged on NVMe via `mmap`):** Compresses `320,001,536` PLE rows into a `D = 256` `PCA-Wiener + RaBitQ Q1_0 (41)` subspace (`2 MiB` inward-aligned `MADV_HUGEPAGE` ensures cold PLE pages are never pre-faulted into RAM).

Verified under hardware-enforced **Linux Cgroup v2 (`memory.max = 28.5 GiB`, `MemorySwapMax = 0`, `oom_kill = 0`)**, `MaxQ-32GB` runs at **`VmHWM = 26.91 GiB <= 28.5 GiB`** and delivers **`11.45–14.29 tok/s` `tg128` (`14.77 tok/s` peak, `74.55 tok/s` `pp512`)** on a 16-core Zen 5 CPU (`0` GPUs), coming within `0.077` Mean KLD (`1.0069` vs `0.9302`) of `54.40 GiB` pruned competitors despite a **`-18.69 GiB` smaller total file size**.

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
| **Active Transformer Params / Token** | **`~4.6B–5.0B`** (`10` of `512` routed experts + `1` shared expert + dense GDN/QSA; reflected in `general.size_label`) |
| **Backbone Topology (`qwen4exp`)** | `48` Trunk Layers (`36` Gated DeltaNet `[48×128×128]` + `12` QSA Full Attention, `compress_ratio = 4`) + **`1` Built-In `NextN` MTP Layer (`blk.48`)** |
| **MoE Width & Routing** | **`512 / 512` Routed Experts** (`Top-10` active; `42 × K = 512` ConeCollapsed layers with `42 × F32` `ffn_down_exps.bias` + `7 × K = 640` boundary layers) |
| **Per-Layer Embedding (`PLE`)** | `320,001,536` rows (`D = 256` `PCA-Wiener` subspace in `Q1_0 = 41`, lazy demand-paged via NVMe `mmap`) |
| **GGUF Tensor Types (`1,128` LLM)** | `{F32: 393, Q8_0: 291, Q4_K: 99, Q5_K: 98, IQ1_S: 62, IQ1_M: 51, BF16: 49, MXFP4(39): 49, IQ2_XXS: 17, IQ2_XS: 10, Q2_0(42): 7, Q6_K: 1, Q1_0(41): 1}` (**`max_type = 42`, `0` custom types**) |

---

## File Manifest & Exact Byte / SHA-256 Verification

| File | Exact Size (`Bytes`) | Size (`GiB` / `GB`) | Tensors | Role & Notes | `SHA-256` |
| :--- | ---: | :---: | ---: | :--- | :--- |
| `Qwen3.8-Flash-Next-MaxQ-32GB-00001-of-00002.gguf` | **`26,821,627,808`** | `24.9796 GiB` (`26.822 GB`) | `1,127` | Resident Trunk (`48L` + `42` bias) + Built-In `blk.48` `NextN` MTP | `06cb65f9cf6e510cd16500ca1719701a0ffa13dcaa4c4d0cc135e4f2071a8b4d` |
| `Qwen3.8-Flash-Next-MaxQ-32GB-00002-of-00002.gguf` | **`11,520,055,616`** | `10.7289 GiB` (`11.520 GB`) | `1` | Lazy NVMe `mmap` `per_layer_token_embd.weight [256, 320001536]` (`Q1_0 = 41`) | `86b79714867743ddf5911ee7dff2717ae275f870551d86469cbdbf026c6b552e` |
| **LLM Total (Shards 1 + 2)** | **`38,341,683,424`** | **`35.7085 GiB` (`38.342 GB`)** | **`1,128`** | **Complete 32 GB Hybrid-RAM LLM (`VmHWM = 26.91 GiB <= 28.5 GiB`)** | — |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` *(Optional)* | **`907,542,880`** | `0.8452 GiB` (`0.908 GB`) | `334` | Multimodal Vision Projector (`224 F32 + 110 BF16`) | `ef2af1d6aa8a22740c7252bc7ce40c5c2f1249aa8d9c8499a4d4572c3cfdf9df` |

---

## Quickstart (`llama-modchip v3.0` on 32 GB RAM)

### Option A — Automated 1-Command Setup (`synchopate/llama-modchip`)

```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./install.sh --preset qwen38-32gb
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

# 2. Download Shards 1 & 2 onto an NVMe SSD:
hf download Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF \
  --include "Qwen3.8-Flash-Next-MaxQ-32GB-*.gguf" --local-dir ./qwen38-32gb

# 3. Launch with mmap enabled (DEFAULT — do NOT pass --no-mmap!), non-convoy threading, and recommended 32GB sampling:
N_CORES=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
THREADS=$(( N_CORES > 1 ? N_CORES - 1 : 1 ))

env RC_TABLES_DIR=/nonexistent "./llama-server-modchip-linux-x86_64-${BIN_ISA}" \
  -m ./qwen38-32gb/Qwen3.8-Flash-Next-MaxQ-32GB-00001-of-00002.gguf \
  -c 32768 -np 1 -fa on -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --jinja --ctx-checkpoints 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.2 --repeat-penalty 1.05 \
  -t "$THREADS" -tb "$THREADS" --poll 0 --poll-batch 0 --slot-save-path /dev/shm --port 8080
```

### Important 32 GB Deployment & Upstream `llama.cpp` Notes
1. **Keep Default `mmap` Enabled (`NEVER --no-mmap` on 32 GB Hosts):** `MaxQ-32GB` relies on OS `mmap` demand-paging for Shard 2 (`10.73 GiB` PLE) while keeping Shard 1 (`24.98 GiB`) resident in RAM (`VmHWM = 26.91 GiB`). On 32 GB desktop Linux hosts, keep default `mmap` enabled (and optionally wrap with `systemd-run --user --scope -p MemoryMax=28.5G`) so Linux automatically reclaims cold PLE pages.
2. **`--ctx-checkpoints 0` for Tight-RAM `/dev/shm` Snapshots:** Passing `--ctx-checkpoints 0` halves `Qwen3.8` Slot-0 snapshot size from `233.14 MiB` (`244,461,872 B`) down to `~120.6 MiB` and halves `/dev/shm` restore latency from **`38.49 ms` avg (`38.16 ms` min, `6.11 GiB/s`, `162×` faster than `6.18s` prefill, bit-exact `SHA256 = 8e5a6444...`)** down to **`~19 ms`**.
3. **`<think>` Token Budget (`max_tokens >= 16384`):** Set **`max_tokens >= 16384`** when `enable_thinking=True` (in `Step 3` `4,096`-token evals, `7/8` prompts triggered deterministic `enable_thinking=False` fallback when `<think>` exceeded `4,096` tokens), or set `enable_thinking=False` for `<= 4,096` token caps.
4. **Unpatched Upstream `llama.cpp` Fallback (`--no-repack`):** `llama-modchip v3.0` runs cleanly with default `--repack` ON (`RssAnon = 0.69 GiB` `none` / `1.92 GiB` `draft-mtp`, `100%` bit-exact greedy SHA-256 `8b9bd8c4...`). Users on unpatched upstream `llama.cpp` should pass `--no-repack` (`-nr`).
5. **Opt-In `LLAMA_MOE_TOP_K=8` Speed Knob:** Default routing uses all `10` active experts (`TOP_K=10`). Setting `env LLAMA_MOE_TOP_K=8` boosts `draft-mtp` `tg128` by `+5.74%` (`11.45 -> 12.11 t/s`) and `pp512` by `+8.50%` (`74.55 -> 80.89 t/s`) at the cost of `+0.4131` 40-chunk PPL (`5.8041 -> 6.2171`); keep `TOP_K=10` default unless extra throughput is prioritized.

---

## Empirical Benchmarks & Hardware Cgroup v2 Certification

### 1. Blind `wiki-alt.txt` (`40 × 512` Chunks) PPL + KLD vs. Donor-Free `BF16` Reference (`2.776624 ± 0.054868`)

| Checkpoint | Exact LLM Bytes (`GiB`) | Trunk + PLE (`GiB`) | `Mean PPL(Q) ± SE` | `Mean KLD ± SE` | `Median KLD` | `Same top p (%)` | Fits `32 GB` (`<= 28.5 GiB`)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MaxQ-32GB`** | **`38,341,683,424 B`** (`35.7085 GiB`) | **`24.980 + 10.729`** (`512/512` exps + MTP) | **`5.812001 ± 0.140197`** *(`5.8041` repack)* | **`1.006916 ± 0.014944`** | **`0.398583`** | **`66.275 ± 0.468%`** | **YES (`VmHWM = 26.91 GiB`)** |
| `ISTA-DASLab GSQ-RCO-Coder-IQ1_M` | `58,408,584,928 B` (`54.3970 GiB`) | `27.575 + 26.822` (`256/512` pruned, no MTP) | `5.091640 ± 0.121418` | `0.930241 ± 0.015625` | `0.296442` | `69.382 ± 0.456%` | No (`38.55 GiB` `VmHWM`) |
| **`MaxQ-64GB`** *(64 GB Sibling)* | `57,969,748,352 B` (`53.9885 GiB`) | `43.260 + 10.729` (`512/512` exps + MTP) | `3.432500 ± 0.070829` | `0.497117 ± 0.009892` | `0.105952` | `78.314 ± 0.408%` | No (`45.08 GiB` `VmHWM`) |

- **Blind Deterministic Suite Disclosure (`temp = 0`):** On the 8-prompt blind deterministic suite at greedy `temp = 0`, `MaxQ-32GB` passed `2 / 8` (`03_sql_window_grace_islands` `4/4` exact SQLite rows at `10.44 t/s`, `91.8%` MTP accept; `06_json_charterparty_settlement` exact `$15,600.0` JSON at `10.27 t/s`, `93.0%` MTP accept), hit the `4,096`-token length cap on `01_cpp` & `04_math` (`0` loops), and exhibited greedy `temp = 0` repetition loops on `05/07/08` without sampling penalties — motivating the mandatory `temp = 0.6, presence_penalty = 0.2, repeat_penalty = 1.05` defaults above.

### 2. Strict Cgroup v2 `28.5 GiB` Verification, `v3.0` Speedup & `248K` Speculative Shootout

Measured on 16 Zen 5 cores (`mmap` enabled, default `--repack` ON, `RC_TABLES_DIR=/nonexistent`):

| Mode & Workload (`-c 4096` / `-c 32768`, `-fa on -ctk q8_0 -ctv q8_0`) | Stage 4C (`-t 16 --poll 50 -nr`) | **`llama-modchip v3.0` (`-t 15 -tb 15 --poll 0`, `--repack` ON)¹** | Combined Gain¹ | MTP / Draft Accept | Peak `VmHWM`² (`RssAnon`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure `N=1` Decode (`--spec-type none`, `p512`)** | `68.66 pp` / `6.08 t/s tg` | **`82.11 t/s pp512`** / **`6.67 t/s tg128`** *(peak `86.28 pp`)* | **`+19.6% pp` / `+9.7% tg`** | — | `26.3 GiB` (`0.69 GiB`) |
| **Built-In `draft-mtp` (Default, `p512`)** | `61.39 pp` / `9.46 t/s tg` *(`8.39` pre-`v3.0` repack)* | **`74.55 t/s pp512`** / **`11.45 t/s tg128`** *(peak `11.69`)* | **`+21.5% pp` / `+21.1% tg`** *(`1.72×` vs `none`)* | **`73.73%`** (`261/354`) | **`26.91 GiB <= 28.5 GiB`** (`1.92 GiB`) |
| **Built-In `draft-mtp` (`inplace_rust_code_edit`, cold `rep 0`)** | — | **`70.28 t/s pp`** / **`13.76 t/s tg128`** *(`14.29` 3-rep avg, `14.77` peak)* | **`2.06×` vs `none`** | **`100.00%`** (`285/285`) | **`26.91 GiB <= 28.5 GiB`** |
| **Zero-Weight `Stockfish-PVS v2` (`inplace_edit`, cold `rep 0`)³** | — | **`78.03 t/s pp`** / **`12.82 t/s tg128`** | **`1.87×` vs `none`** *(`+11.0%` faster `pp`)* | **`91.25%`** (cold) | **`26.38 GiB`** (`-0.53 GiB`) |
| **Strict Cgroup v2 `32K` Context (`memory.max = 28.5 GiB`, `Swap = 0`)** | `28.1 pp` / `10.4 t/s tg` (`-nr`) | **`29.0 t/s pp` / `11.1 t/s tg` (`rc=0, oom=0`)** | **28.5 GiB Ceiling PASS** | `draft-mtp` | **`27.39–27.44 GiB <= 28.5 GiB`²** |

1. **Transparent Speedup Attribution:** Stage 4C ran oversubscribed `-t 16 --poll 50 --no-repack` on 16 logical CPUs, whereas Stage 4D-1 ran `-t 15 -tb 15 --poll 0 --poll-batch 0` with `v3.0` default `--repack` ON. The `+21.1%` (`+36.5%` vs pre-`v3.0` repack) decode gain reflects the **combined** `llama-modchip v3.0` SIMD/GDN/repack optimizations plus non-convoy `-t $((N_CORES - 1)) --poll 0` scheduling (`step0` isolated `-t 15 --poll 0` at `+25.9% pp / +31.3% tg` on `32GB none`).
2. **Authoritative `VmHWM` Note:** Process `VmHWM` (`26.91–27.44 GiB <= 28.5 GiB`) is the authoritative total resident RAM metric (`RssAnon + RssFile`), as warm page-cache runs record delta page faults in cgroup `memory.peak`.
3. **Speculative Shootout Note:** Built-in `draft-mtp` won cold single-turn `rep 0` `tg128` on **all `6 / 6` (`100%`)** prompts (`11.69 vs 6.97–7.55 t/s` on `p512`, `10.25 vs 6.50–7.45 t/s` on `blind07`, `13.76 vs 12.82 t/s` on `inplace_edit`). Zero-weight `Stockfish-PVS v2` (`--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 3 --spec-ngram-simple-min-hits 1`) achieved `12.82 t/s` (`1.87×`) cold `rep 0` on source-in-context refactoring with `+11.0%` faster prefill and `-0.53 GiB` lower `VmHWM`, making it an ideal option for repetitive code-edit loops or MTP-less checkpoints.

---

## Portable `v3.0` Binary Checksums (`GLIBC <= 2.30`)

```text
54af3232375ede820f91143c106a594c83b815a3dca66fe98ab08bcb8b63d4f0  llama-server-modchip-linux-x86_64-avx512
0f0c07f97a98d8aeb9a012264fdeff5a787d2104a2afb122052bfe0a21c2d1a9  llama-server-modchip-linux-x86_64-avx2
531d870988dcc580322341d2bbd7f0cf6b6cc3b3376a2e86d092a021f7cddba2  llama-cli-modchip-linux-x86_64-avx512
49a918206afe371d16ab14fd66d9cc87e35c7f8c0471c2cc13c759f2e0350d03  llama-cli-modchip-linux-x86_64-avx2
```

- **64 GB RAM Hosts:** See **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF)** (`53.99 GiB`, `Mean PPL = 3.4325`, `12.66 tok/s` `draft-mtp`).
- **Engine & CLI Repository:** **[`github.com/synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip)** (`Apache-2.0`; model weights governed by [`qwen-community-1.0`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)).
