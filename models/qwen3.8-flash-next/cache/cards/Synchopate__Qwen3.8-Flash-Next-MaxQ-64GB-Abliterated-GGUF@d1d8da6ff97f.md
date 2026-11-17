---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - trohrbaugh/Qwen3.8-Flash-Next-heretic-2
base_model_relation: quantized
quantized_by: Synchopate
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - qwen3.8
  - qwen4exp
  - moe
  - abliterated
  - heretic
  - uncensored
  - maxq
  - cpu-inference
  - llama-cpp
  - llama-modchip
---

# Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-GGUF (`53.99 GiB` · Gated-Residual Over-Projected `Heretic-2` + Built-In `NextN` MTP)

> [!CAUTION]
> **Responsible-Use Disclaimer & Edition Positioning (Opt-In Uncensored Checkpoint):**
> **`Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated`** (`v3.0-r2`) applies TPE-optimized signed multi-layer over-projection (`beta = 1.83 - 2.39` across `blk.31..47` from [`trohrbaugh/Qwen3.8-Flash-Next-heretic-2`](https://huggingface.co/trohrbaugh/Qwen3.8-Flash-Next-heretic-2), `0 / 100` refusals at `0.0818` KLD — compensating for `Qwen3.8`'s widened multi-stream Gated Residual branches — plus signed `beta = 2.2508` alignment on `blk.48` `NextN` MTP; `blk.0..30` and `token_embd` `100%` untouched). Upstream safety refusal filters are removed and cannot be relied upon for content moderation.
> - **Intended Scope:** Authorized security research, reverse engineering, red-team auditing, CTF competitions, binary/exploit analysis, code auditing, and uncensored legal/policy analysis only.
> - **Deployment:** Do **not** expose unmoderated endpoints of this checkpoint to public networks without external access control and safety filtering.
> - **Default Pure Coding Flagship:** For general software engineering and standard upstream alignment, use **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF)** (`./install.sh --preset qwen38-64gb`).

**`Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated`** is the opt-in uncensored **64 GB RAM tier** twin of [`Qwen3.8-Flash-Next-MaxQ-64GB`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF):
- **Gated-Residual Signed Over-Projection (`27` Edited Tensors) + `blk.48` `NextN` MTP Synchronization:** Because `Qwen3.8-Flash-Next` routes activations through widened multi-stream Gated Residual Hyper-Connections (`hc_attn_*`, `hc_ffn_*`), standard `beta <= 1.0` single-branch orthogonalization leaves parallel residual streams carrying refusal state at late layers (`blk.32..47`). `v3.0-r2` encodes the `26` TPE-optimized trunk tensors from [`trohrbaugh/Qwen3.8-Flash-Next-heretic-2`](https://huggingface.co/trohrbaugh/Qwen3.8-Flash-Next-heretic-2) (`16` `blk.32..47` `ssm_out`/`attn_output` with `beta = 2.15..2.39`, `5` `blk.31..35` `ffn_down_shexp`, and `5` `blk.31..35` `512`-expert `ffn_down_exps` with exact stored channel-order preservation) plus **`1` signed `beta = 2.2508` `blk.48.attn_output.weight` `NextN` MTP projection** so speculative drafting stays aligned on uncensored outputs.
- **Bit-Identical Unedited Tensors (`1,058 / 1,085`) & Shared `10.73 GiB` PLE:** Preserves `1,058` Shard 1 tensors (`blk.0..30`, `ffn_gate_exps`/`ffn_up_exps`, `token_embd`, `output.weight`, routers, norms) **100% bit-identical** to aligned `MaxQ-64GB`, plus the exact shared `11,520,055,616 B` (`10.7289 GiB`) `PCA-Wiener D=256 + RaBitQ Q1_0` Shard 2 PLE table, using **100% standard `ggml` tensor types (`max_type = 42`, `0` custom types)**.

---

## Architecture & Parameter Summary (`general.size_label` Clarification)

> [!NOTE]
> **Clarification on Hugging Face Hub's Auto-Detected `"5.0B"` Badge:** Shard 1 carries `general.size_label = "5.0B"` inherited from the non-MoE dense trunk staging header during conversion. This checkpoint contains the **complete, unpruned `176.5B` `Qwen3.8-Flash-Next` architecture** (`512/512` routed experts per layer + `320M`-row PLE table + `blk.48` NextN MTP):

| Architectural Component | Specification |
| :--- | :--- |
| **Upstream Base Models** | `Qwen/Qwen3.8-Flash-Next` + `trohrbaugh/Qwen3.8-Flash-Next-heretic-2` (`27` over-projected `blk.31..48` tensors onto virgin `MaxQ-64GB`) |
| **Total Parameters** | **`~176.5B`** (`~125.3B` Transformer + `512`-Expert MoE + `NextN` MTP + `51.2B` 320M-row PLE) |
| **Active Transformer Params / Token** | **`~5.0B`** (`10` of `512` routed experts + `1` shared expert + dense GDN/QSA; reflected in `general.size_label`) |
| **Backbone Topology (`qwen4exp`)** | `48` Trunk Layers (`36` Gated DeltaNet + `12` QSA Full Attention, `compress_ratio = 4`) + **`1` Subspace-Aligned `NextN` MTP Layer (`blk.48`)** |
| **MoE Width & Routing** | **`512 / 512` Unpruned Routed Experts** (`Top-10` active, `K = 640` full width) + `1` Shared Expert (`K = 640`), `0` `ffn_down_exps.bias` |
| **GGUF Tensor Types (`1,086` LLM)** | `{F32: 351, Q8_0: 339, Q6_K: 198, IQ2_S: 60, BF16: 49, Q2_0(42): 39, IQ3_XXS: 36, MXFP4(39): 7, IQ4_NL: 4, Q2_K: 2, Q1_0(41): 1}` (**`max_type = 42`, `0` custom types**) |

---

## File Manifest & Exact Byte / SHA-256 Verification

| File | Exact Size (`Bytes`) | Size (`GiB` / `GB`) | Tensors | Role & Notes | `SHA-256` |
| :--- | ---: | :---: | ---: | :--- | :--- |
| `Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-00001-of-00002.gguf` | **`46,449,692,736`** | `43.2596 GiB` (`46.450 GB`) | `1,085` | Over-Projected Trunk (`48L`, `27` edited / `1,058` bit-identical) + `blk.48` `NextN` MTP | `1dac27e81a3a73dcec21edea78f224eb48393a726e3f9ee7676c603d3174e82f` |
| `Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-00002-of-00002.gguf` | **`11,520,055,616`** | `10.7289 GiB` (`11.520 GB`) | `1` | Shared `per_layer_token_embd.weight [256, 320001536]` (`Q1_0 = 41`) | `86b79714867743ddf5911ee7dff2717ae275f870551d86469cbdbf026c6b552e` |
| **LLM Total (Shards 1 + 2)** | **`57,969,748,352`** | **`53.9885 GiB` (`57.970 GB`)** | **`1,086`** | **Complete 64 GB RAM LLM (`< 55.5 GiB` ceiling)** | — |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` *(Optional)* | **`907,542,880`** | `0.8452 GiB` (`0.908 GB`) | `334` | Multimodal Vision Projector (`224 F32 + 110 BF16`) | `ef2af1d6aa8a22740c7252bc7ce40c5c2f1249aa8d9c8499a4d4572c3cfdf9df` |

---

## Quickstart (`llama-modchip v3.0`)

### Option A — Automated Setup (`synchopate/llama-modchip`)

```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./install.sh --preset qwen38-64gb-ablit --accept-abliterated
./modchip chat
```

### Option B — Direct `llama-server` Launch (Portable `GLIBC_2.30` Linux `x86_64` Binary)

```bash
BIN_ISA=$(grep -q "avx512f" /proc/cpuinfo && grep -Eq "avx512_?vbmi" /proc/cpuinfo && grep -Eq "avx512_?vnni" /proc/cpuinfo && echo "avx512" || echo "avx2")
curl -fLO "https://github.com/synchopate/llama-modchip/releases/download/v3.0/llama-server-modchip-linux-x86_64-${BIN_ISA}"
curl -fLO "https://github.com/synchopate/llama-modchip/releases/download/v3.0/SHA256SUMS"
sha256sum -c SHA256SUMS --ignore-missing
chmod +x "llama-server-modchip-linux-x86_64-${BIN_ISA}"

hf download Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-GGUF \
  --include "Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-*.gguf" --local-dir ./qwen38-64gb-ablit

N_CORES=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
THREADS=$(( N_CORES > 1 ? N_CORES - 1 : 1 ))

env RC_TABLES_DIR=/nonexistent "./llama-server-modchip-linux-x86_64-${BIN_ISA}" \
  -m ./qwen38-64gb-ablit/Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-00001-of-00002.gguf \
  -c 32768 -np 1 -fa on -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --jinja \
  -t "$THREADS" -tb "$THREADS" --poll 0 --poll-batch 0 --slot-save-path /dev/shm --port 8080
```

- **Upstream `llama.cpp` Fallback:** `llama-modchip v3.0` runs cleanly with default `--repack` ON (`VmHWM = 46.34 GiB`, `RssAnon = 2.31 GiB`, zero-copy 3D MoE `mmap` intact); users on unpatched upstream `llama.cpp` should pass `--no-repack` (`-nr`).
- **Mandatory `<think>` Token Budget (`max_tokens >= 16384`) & Anti-Repetition Sampling:** Always set **`max_tokens >= 16384`** when `enable_thinking=True`, or pass `enable_thinking=False` for `<= 4,096` token caps. Pass `--temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.2 --repeat-penalty 1.05` on open-ended completions to avoid deterministic `temp = 0` tail loops.
- **Vocabulary Isolation:** Always set `RC_TABLES_DIR=/nonexistent` when serving `Qwen3.8` from a directory containing Nemotron RouteCrack tables (`./modchip` sets this automatically).

---

## Architecture, Signal Fidelity & `NextN` MTP Synchronization (`v3.0-r2`)

### 1. Why Widened Gated-Residual Models Require Signed Over-Projection (`beta > 1.0`)

`Qwen3.8-Flash-Next` replaces a single residual stream with widened multi-stream **Gated Residual Hyper-Connections** (`hc_attn_*`, `hc_ffn_*`). Standard single-branch orthogonal projection (`beta <= 1.0`) leaves parallel residual streams carrying refusal state through layers `32..47`. **`MaxQ-64GB-Abliterated` (`v3.0-r2`)** encodes the `26` TPE-optimized trunk tensors from [`trohrbaugh/Qwen3.8-Flash-Next-heretic-2`](https://huggingface.co/trohrbaugh/Qwen3.8-Flash-Next-heretic-2) (`0 / 100` refusals vs. `99 / 100` upstream at `0.0818` KLD) plus `1` signed `beta = 2.2508` `blk.48.attn_output.weight` `NextN` MTP projection directly onto virgin [`MaxQ-64GB`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF):

| Checkpoint | Exact LLM Bytes | Trunk + PLE (`GiB`) | Edited / Unedited Shard 1 Tensors | Base `wiki-alt.txt` `Mean PPL(Q) ± SE` | Base `Mean KLD` vs. `BF16` | Abliteration Refusal / KLD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aligned `MaxQ-64GB`** *(Virgin Base)* | `57,969,748,352 B` | `43.26 + 10.73` | `0 / 1,085` (Ref) | **`3.43825 ± 0.07119`** *(`3.4325` `-nr`)* | **`0.49803`** *(`-46.6%` vs `ISTA`)* | `99 / 100` upstream safety aligned |
| **`MaxQ-64GB-Abliterated` (`v3.0-r2`)** | **`57,969,748,352 B`** | **`43.26 + 10.73`** | **`27` edited / `1,058` bit-identical** | *Inherits `MaxQ-64GB` across `1,058` tensors* | *`blk.0..30` `100%` untouched* | **`0 / 100` refusals, `0.0818` KLD** *(`Heretic-2`)* |
| `ISTA-DASLab GSQ-RCO-Coder-IQ1_M` | `58,408,584,928 B` | `27.58 + 26.82` | `256 / 512` experts pruned, no MTP | `5.09164 ± 0.12142` | `0.93024` | Safety aligned |

- **Exact Tensor Edit Map (`27` of `1,085` Shard 1 Tensors):** `16` `blk.32..47` `ssm_out`/`attn_output` (`Q6_K`, `beta = 2.15..2.39` peaking at `blk.41`), `5` `blk.31..35` `ffn_down_shexp` (`Q8_0`), `5` `blk.31..35` `512`-expert `ffn_down_exps` (`MXFP4` on `blk.31`, `Q2_0` on `blk.32..35` with exact stored channel-order preservation, `mean_col_cos = 0.933–1.000`, `beta` peaking at `1.83` on `blk.33`), and `1` `blk.48.attn_output` (`Q6_K`, signed `beta = 2.2508`). All `1,058` remaining Shard 1 tensors and the `10.73 GiB` `Q1_0` Shard 2 PLE table are bit-for-bit identical to Aligned `MaxQ-64GB`.
- **Edition Positioning:** Use **Aligned [`MaxQ-64GB`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF)** as the default pure software-engineering flagship, and **`MaxQ-64GB-Abliterated`** strictly for opt-in security research, reverse engineering, red-team auditing, and uncensored legal/policy analysis.

### 2. `llama-modchip v3.0` Throughput & `blk.48` `NextN` MTP Speed (`16` Zen 5 Cores, `0` GPUs)

- **Aligned `MaxQ-64GB` Baseline (`-t 15 -tb 15 --poll 0`, default `--repack` ON):** `71.97 tok/s` `pp512`, **`12.66 tok/s` `tg128`** (`+72.8%` combined `v3.0` SIMD/GDN/repack + non-convoy `-t <N-1> --poll 0` gain over Stage 4C's oversubscribed `-t 16 --poll 50 --no-repack` `7.33 tok/s`; `93.00%` MTP acceptance, `VmHWM = 45.08 GiB`).
- **`MaxQ-64GB-Abliterated` (`v3.0-r2`, `--spec-type draft-mtp`, `VmHWM = 46.34 GiB`):** Signed `beta = 2.2508` alignment on `blk.48` keeps `NextN` MTP synchronized with `blk.47` on uncensored outputs, delivering **`85.7–104.0 tok/s` prefill** and **`13.2–14.8 tok/s` decode (`81.3%–84.6%` MTP draft acceptance)** on multi-turn tool-calling and reasoning sessions on 16 physical Zen 5 cores.
- **Zero-Weight `Stockfish-PVS v2` Fallback:** Zero-weight `Stockfish-PVS v2` (`--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 3 --spec-ngram-simple-min-hits 1`) delivers `10.58 tok/s` (`1.67×` over `none`) cold `rep 0` on source-in-context refactoring with `-0.85 GiB` lower `VmHWM`.

---

## Portable `v3.0` Binary Checksums (`GLIBC <= 2.30`)

```text
54af3232375ede820f91143c106a594c83b815a3dca66fe98ab08bcb8b63d4f0  llama-server-modchip-linux-x86_64-avx512
0f0c07f97a98d8aeb9a012264fdeff5a787d2104a2afb122052bfe0a21c2d1a9  llama-server-modchip-linux-x86_64-avx2
531d870988dcc580322341d2bbd7f0cf6b6cc3b3376a2e86d092a021f7cddba2  llama-cli-modchip-linux-x86_64-avx512
49a918206afe371d16ab14fd66d9cc87e35c7f8c0471c2cc13c759f2e0350d03  llama-cli-modchip-linux-x86_64-avx2
```

- **Safety-Aligned 64 GB Edition:** **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF)**
- **32 GB Uncensored Sibling:** **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated-GGUF)**
- **Engine & CLI Repository:** **[`github.com/synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip)** (`Apache-2.0`; model weights governed by [`qwen-community-1.0`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)).
