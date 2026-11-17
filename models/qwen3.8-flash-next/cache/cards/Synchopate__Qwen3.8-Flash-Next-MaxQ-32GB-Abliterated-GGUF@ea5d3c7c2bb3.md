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
  - 32gb-ram
  - maxq
  - cpu-inference
  - llama-cpp
  - llama-modchip
---

# Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated-GGUF (`24.98 GiB` Resident Trunk + `10.73 GiB` Lazy NVMe PLE · Gated-Residual Over-Projected `Heretic-2`)

> [!CAUTION]
> **Responsible-Use Disclaimer & Edition Positioning (Opt-In Uncensored Checkpoint):**
> **`Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated`** (`v3.0-r2`) applies TPE-optimized signed multi-layer over-projection (`beta = 1.83 - 2.39` across `blk.31..47` from [`trohrbaugh/Qwen3.8-Flash-Next-heretic-2`](https://huggingface.co/trohrbaugh/Qwen3.8-Flash-Next-heretic-2), `0 / 100` refusals at `0.0818` KLD — compensating for `Qwen3.8`'s widened multi-stream Gated Residual branches — plus signed `beta_m = 0.449..1.710` ConeCollapse `ffn_down_exps.bias` alignment on `blk.31..35` and signed `beta = 2.2508` `blk.48` `NextN` MTP synchronization; `blk.0..30` and `token_embd` `100%` untouched). Upstream safety refusal filters are removed and cannot be relied upon for content moderation.
> - **Intended Scope:** Authorized security research, reverse engineering, red-team auditing, CTF competitions, binary/exploit analysis, code auditing, and uncensored legal/policy analysis only.
> - **Deployment:** Do **not** expose unmoderated endpoints of this checkpoint to public networks without external access control and safety filtering.
> - **Safety-Aligned Defaults:** For standard upstream alignment on 32 GB RAM, use **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF)** (`./install.sh --preset qwen38-32gb`); on 64 GB+ RAM hosts, **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF)** is the recommended default pure coding flagship.

**`Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated`** is the opt-in uncensored **32 GB consumer RAM tier** twin of [`Qwen3.8-Flash-Next-MaxQ-32GB`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF), encoding `32` over-projected Shard 1 tensors (`16` `blk.32..47` `ssm_out`/`attn_output`, `5` `blk.31..35` `ffn_down_shexp`, `5` `blk.31..35` `K = 512` ConeCollapsed `ffn_down_exps` with exact stored channel-order recovery, `5` `blk.31..35` `F32` `ffn_down_exps.bias`, and `1` `blk.48.attn_output` `NextN` MTP writer; `1,095 / 1,127` Shard 1 tensors bit-identical to `MaxQ-32GB`). It keeps the `24.98 GiB` Shard 1 trunk + `blk.48` `NextN` MTP resident in DDR5 RAM while demand-paging the shared `10.73 GiB` `PCA-Wiener D=256 + RaBitQ Q1_0` Shard 2 PLE table via NVMe `mmap` (`VmHWM = 28.40 GiB <= 28.5 GiB`, `RssAnon = 2.43 GiB`, `max_type = 42`, **`0` custom tensor types**).

> [!IMPORTANT]
> **Mandatory `MaxQ-32GB-Abliterated` Anti-Repetition Sampling & `<think>` Token Budget (`1.63 bpw` Trunk):**
> At `1.63 bpw`, pure greedy `temperature = 0` decoding without sampling penalties can exhibit tail repetition on open-ended completions (`2 / 4` loop-free on Blind-4 at `temp = 0` vs. **`4 / 4 = 100%` loop-free** with the sampling flags below; **always enable these flags for coding, security analysis, and chain-of-thought**, reserving `temp = 0` strictly for short JSON/SQL extraction):
> - `temperature = 0.6`
> - `top_p = 0.95`, `top_k = 20`
> - `presence_penalty = 0.2` – `1.0` (default `0.2`)
> - `repeat_penalty = 1.05`
> - **`<think>` Token Budget (`max_tokens >= 16384`):** Mandate **`max_tokens >= 16384`** whenever `enable_thinking=True`, or pass `enable_thinking=False` under `<= 4,096` output caps.
>
> *(Note: `./modchip` and `install.sh --preset qwen38-32gb-ablit` configure these sampling flags automatically.)*

---

## Architecture & Parameter Summary (`general.size_label` Clarification)

> [!NOTE]
> **Clarification on Hugging Face Hub's Auto-Detected `"5.0B"` Badge:** Shard 1 carries `general.size_label = "5.0B"` inherited from the non-MoE dense trunk staging header during conversion. This checkpoint contains the **complete `512`-expert `Qwen3.8-Flash-Next` architecture** (`42` `K = 512` ConeCollapsed layers + `7` `K = 640` boundary layers + `320M`-row PLE + `blk.48` NextN MTP):

| Architectural Component | Specification |
| :--- | :--- |
| **Upstream Base Models** | `Qwen/Qwen3.8-Flash-Next` + `trohrbaugh/Qwen3.8-Flash-Next-heretic-2` (`32` over-projected `blk.31..48` tensors onto virgin `MaxQ-32GB`) |
| **Nominal / Collapsed Params** | **`~176.5B` nominal** (`~152.3B` post-ConeCollapse + PLE; `92.28%` mean retained spectral energy) |
| **Active Transformer Params / Token** | **`~4.6B–5.0B`** (`10` of `512` routed experts + `1` shared expert + dense GDN/QSA; reflected in `general.size_label`) |
| **Backbone Topology (`qwen4exp`)** | `48` Trunk Layers (`36` Gated DeltaNet + `12` QSA Full Attention, `compress_ratio = 4`) + **`1` Subspace-Aligned `NextN` MTP Layer (`blk.48`)** |
| **MoE Width & Bias Alignment** | **`512 / 512` Routed Experts** (`42 × K = 512` ConeCollapsed layers with `42 × F32` `ffn_down_exps.bias` [`5` `blk.31..35` biases aligned with signed `beta_m = 0.449..1.710`, `37` bit-identical to `MaxQ-32GB`] + `7 × K = 640` boundary layers) |
| **GGUF Tensor Types (`1,128` LLM)** | `{F32: 393, Q8_0: 291, Q4_K: 99, Q5_K: 98, IQ1_S: 62, IQ1_M: 51, BF16: 49, MXFP4(39): 49, IQ2_XXS: 17, IQ2_XS: 10, Q2_0(42): 7, Q6_K: 1, Q1_0(41): 1}` (**`max_type = 42`, `0` custom types**) |

---

## File Manifest & Exact Byte / SHA-256 Verification

| File | Exact Size (`Bytes`) | Size (`GiB` / `GB`) | Tensors | Role & Notes | `SHA-256` |
| :--- | ---: | :---: | ---: | :--- | :--- |
| `Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated-00001-of-00002.gguf` | **`26,821,627,808`** | `24.9796 GiB` (`26.822 GB`) | `1,127` | Over-Projected Resident Trunk (`32` edited incl. `5` bias / `1,095` bit-identical) + `blk.48` `NextN` MTP | `09b54c335505f9f609415dbc774868ad5ae188f17e873dead3ec743ec84e1009` |
| `Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated-00002-of-00002.gguf` | **`11,520,055,616`** | `10.7289 GiB` (`11.520 GB`) | `1` | Shared Lazy NVMe `mmap` `per_layer_token_embd.weight` (`Q1_0 = 41`) | `86b79714867743ddf5911ee7dff2717ae275f870551d86469cbdbf026c6b552e` |
| **LLM Total (Shards 1 + 2)** | **`38,341,683,424`** | **`35.7085 GiB` (`38.342 GB`)** | **`1,128`** | **Complete 32 GB Hybrid-RAM LLM (`VmHWM = 28.40 GiB <= 28.5 GiB`)** | — |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` *(Optional)* | **`907,542,880`** | `0.8452 GiB` (`0.908 GB`) | `334` | Multimodal Vision Projector (`224 F32 + 110 BF16`) | `ef2af1d6aa8a22740c7252bc7ce40c5c2f1249aa8d9c8499a4d4572c3cfdf9df` |

---

## Quickstart (`llama-modchip v3.0` on 32 GB RAM)

```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./install.sh --preset qwen38-32gb-ablit --accept-abliterated
./modchip chat
```

Direct `llama-server` invocation (`mmap` enabled — **do NOT pass `--no-mmap`** on 32 GB hosts!):

```bash
N_CORES=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
THREADS=$(( N_CORES > 1 ? N_CORES - 1 : 1 ))

env RC_TABLES_DIR=/nonexistent ./llama-server-modchip-linux-x86_64-avx512 \
  -m ./qwen38-32gb-ablit/Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated-00001-of-00002.gguf \
  -c 32768 -np 1 -fa on -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --jinja --ctx-checkpoints 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.2 --repeat-penalty 1.05 \
  -t "$THREADS" -tb "$THREADS" --poll 0 --poll-batch 0 --slot-save-path /dev/shm --port 8080
```

- **32 GB `mmap` & `/dev/shm` Guidance:** Keep default `mmap` enabled (and optional `MemoryMax=28.5G`) so Linux reclaims cold PLE pages automatically; `--ctx-checkpoints 0` halves `/dev/shm` Slot-0 snapshot size from `233.1 MiB` to `~120.6 MiB` (`~19 ms` restore vs `38.5 ms` default).
- **Upstream `llama.cpp` & `<think>` Notes:** Pass `--no-repack` (`-nr`) only on unpatched upstream `llama.cpp` (`llama-modchip v3.0` runs default `--repack` ON cleanly); set `max_tokens >= 16384` when `enable_thinking=True` (or `enable_thinking=False` for `<= 4,096` caps).

---

## Architecture, Signal Fidelity & `NextN` MTP Synchronization (`v3.0-r2`)

`Qwen3.8-Flash-Next` routes activations through widened multi-stream **Gated Residual Hyper-Connections** (`hc_attn_*`, `hc_ffn_*`), where single-branch `beta <= 1.0` orthogonal projection leaves parallel residual branches carrying refusal state across layers `32..47`. **`MaxQ-32GB-Abliterated` (`v3.0-r2`)** encodes `32` over-projected Shard 1 tensors from [`trohrbaugh/Qwen3.8-Flash-Next-heretic-2`](https://huggingface.co/trohrbaugh/Qwen3.8-Flash-Next-heretic-2) (`0 / 100` refusals vs. `99 / 100` upstream at `0.0818` KLD) directly onto virgin [`MaxQ-32GB`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF):

| Checkpoint | Exact LLM Bytes | Trunk + PLE (`GiB`) | Edited / Unedited Shard 1 Tensors | Base `wiki-alt.txt` `Mean PPL(Q) ± SE` | Base `Mean KLD` vs. `BF16` | Peak `VmHWM` (`-c 32768`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aligned `MaxQ-32GB`** *(Virgin Base)* | `38,341,683,424 B` | `24.98 + 10.73` | `0 / 1,127` (Ref) | **`5.80406 ± 0.13982`** *(`5.8120` `-nr`)* | **`1.00399`** | **`26.91–27.44 GiB <= 28.5 GiB`** |
| **`MaxQ-32GB-Abliterated` (`v3.0-r2`)** | **`38,341,683,424 B`** | **`24.98 + 10.73`** | **`32` edited / `1,095` bit-identical** | *Inherits `MaxQ-32GB` across `1,095` tensors* | *`blk.0..30` `100%` untouched (`0.0818` `Heretic-2` KLD)* | **`28.40 GiB <= 28.5 GiB`** |

- **Exact Tensor Edit Map (`32` of `1,127` Shard 1 Tensors):** `16` `blk.32..47` `ssm_out`/`attn_output` (`Q5_K`, `beta = 2.15..2.39` peaking at `blk.41`), `5` `blk.31..35` `ffn_down_shexp` (`MXFP4`), `5` `blk.31..35` `K = 512` ConeCollapsed `ffn_down_exps` (`IQ2_XXS` on `blk.31, 35`, `IQ1_M` on `blk.32..34`, `mean_cos = 0.898–0.954`), `5` `blk.31..35` `F32` `ffn_down_exps.bias` (`signed beta_m = 0.449..1.710`), and `1` `blk.48.attn_output` (`Q5_K`, signed `beta = 2.2508`). All `1,095` remaining Shard 1 tensors and the `10.73 GiB` `Q1_0` Shard 2 PLE table are bit-for-bit identical to Aligned `MaxQ-32GB`.
- **`llama-modchip v3.0` Throughput & `28.5 GiB` Memory (`16` Zen 5 Cores, `0` GPUs):** Aligned `MaxQ-32GB` achieves `74.55 tok/s` `pp512` and **`11.45–14.29 tok/s` `tg128`** (`VmHWM = 26.91 GiB <= 28.5 GiB` Cgroup v2 `Swap=0` PASS; cold `rep 0` `Stockfish-PVS v2` refactoring fallback `12.82 tok/s`), with signed `beta = 2.2508` `blk.48` `NextN` MTP alignment keeping `draft-mtp` synchronized on uncensored outputs.
- **Edition Positioning:** For pure software engineering on 64 GB+ hosts, use Aligned **[`MaxQ-64GB`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF)**; on 32 GB hosts use Aligned **[`MaxQ-32GB`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF)** by default and `MaxQ-32GB-Abliterated` strictly for opt-in security research, reverse engineering, red-team auditing, and uncensored legal/policy analysis.

---

## Portable `v3.0` Binary Checksums (`GLIBC <= 2.30`)

```text
54af3232375ede820f91143c106a594c83b815a3dca66fe98ab08bcb8b63d4f0  llama-server-modchip-linux-x86_64-avx512
0f0c07f97a98d8aeb9a012264fdeff5a787d2104a2afb122052bfe0a21c2d1a9  llama-server-modchip-linux-x86_64-avx2
531d870988dcc580322341d2bbd7f0cf6b6cc3b3376a2e86d092a021f7cddba2  llama-cli-modchip-linux-x86_64-avx512
49a918206afe371d16ab14fd66d9cc87e35c7f8c0471c2cc13c759f2e0350d03  llama-cli-modchip-linux-x86_64-avx2
```

- **Safety-Aligned 32 GB Edition:** **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF)**
- **64 GB Uncensored Sibling:** **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-GGUF)**
- **Engine & CLI Repository:** **[`github.com/synchopate/llama-modchip`](https://github.com/synchopate/llama-modchip)** (`Apache-2.0`; model weights governed by [`qwen-community-1.0`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)).
