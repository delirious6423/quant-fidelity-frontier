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
  - abliterated
  - uncensored
  - 32gb-ram
  - maxq
  - cpu-inference
  - llama-cpp
  - llama-modchip
---

# Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated-GGUF (`24.98 GiB` Resident Trunk + `10.73 GiB` Lazy NVMe PLE · Opt-In Subspace-Abliterated)

> [!CAUTION]
> **Responsible-Use Disclaimer & Edition Positioning (Opt-In KL-Safe Subspace-Abliterated Checkpoint):**
> **`Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated`** attenuates refusal-mediating activation subspaces via norm-preserving rank-2 orthogonal projection strictly across `blk.4..48` residual-stream writers (`ssm_out`, `attn_output`, `ffn_down_shexp`, `ffn_down_exps`, all `42` `F32` `ffn_down_exps.bias` tensors, and `blk.48` `NextN` MTP; `blk.0..3` and `token_embd` spared), reducing held-out dual-use security refusals from **`43.75%` (`7/16`) to `12.50%` (`2/16`, `-71.4%`)** and `<think>` spot refusals to **`0 / 4` (`0.0%`)** at **`+0.3189%` `ΔPPL`** (rather than over-editing to `"0% refusals"`). Upstream safety refusal filters cannot be relied upon for content moderation.
> - **Intended Scope:** Authorized security research, reverse engineering, red-team auditing, CTF competitions, binary/exploit analysis, code auditing, and uncensored legal/policy analysis only.
> - **Deployment:** Do **not** expose unmoderated endpoints of this checkpoint to public networks without external access control and safety filtering.
> - **Safety-Aligned Defaults:** For standard upstream alignment on 32 GB RAM, use **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF)** (`./install.sh --preset qwen38-32gb`); on 64 GB+ RAM hosts, **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF)** (`4 / 4` strict Blind-4 pass vs. `2 / 4` on `MaxQ-64GB-Abliterated`) is the recommended default pure coding flagship.

**`Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated`** is the opt-in uncensored **32 GB consumer RAM tier** twin of [`Qwen3.8-Flash-Next-MaxQ-32GB`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF), built **100% donor-free** directly from [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`rev 812d96e10a0dd82b69d749e68693698cf6c36daa`) via single-pass fused rank-2 contrastive subspace projection + `FP64 Woodbury DualGram` (`64` Jinja-templated pairs = `32` non-thinking + `32` thinking; verified `0.000000%` 8-gram and 13-gram overlap across all `32 / 32` checks; **`177` edited / `950` bit-identical** Shard 1 tensors across `blk.4..48`, with `42 × F32` `ffn_down_exps.bias` orthogonalized to `7.436e-10`). It keeps the `24.98 GiB` Shard 1 trunk + `blk.48` `NextN` MTP resident in DDR5 RAM while demand-paging the shared `10.73 GiB` `PCA-Wiener D=256 + RaBitQ Q1_0` Shard 2 PLE table via NVMe `mmap` (`VmHWM = 28.40 GiB <= 28.5 GiB`, `RssAnon = 2.43 GiB`, `max_type = 42`, **`0` custom tensor types**).

> [!IMPORTANT]
> **Mandatory `MaxQ-32GB-Abliterated` Anti-Repetition Sampling & `<think>` Token Budget (`1.63 bpw` Trunk):**
> At `1.63 bpw`, pure greedy `temperature = 0` decoding without sampling penalties (`max_tokens = 512`) exhibited tail repetition after substantive technical output on **`3 / 16`** dual-use security prompts (`ref_01` hex comment tail, `ref_09` cyclic imports, and `ref_14` box-drawing — with `ref_14` crossing the automated `15×` 20-gram threshold) and **`2 / 4`** Blind-4 prompts (`2 / 4` loop-free at `temp = 0`). Configuring the sampling parameters below improved Blind-4 loop-free completion from **`2 / 4` to `4 / 4` (`100%` loop-free)** while preserving `1 / 4` (`06_json` PASS) strict parity with Aligned `MaxQ-32GB` (**always enable these flags for coding, security analysis, and chain-of-thought**, reserving `temp = 0` strictly for short JSON/SQL extraction):
> - `temperature = 0.6`
> - `top_p = 0.95`, `top_k = 20`
> - `presence_penalty = 0.2` – `1.0` (default `0.2`)
> - `repeat_penalty = 1.05`
> - **`<think>` Token Budget (`max_tokens >= 16384`):** Mandate **`max_tokens >= 16384`** whenever `enable_thinking=True` (`max_tokens = 2048` `<think>` spot checks verified **`0 / 4` refusal intent inside `<think>`**), or pass `enable_thinking=False` under `<= 4,096` output caps.
>
> *(Note: `./modchip` and `install.sh --preset qwen38-32gb-ablit` configure these sampling flags automatically.)*

---

## Architecture & Parameter Summary (`general.size_label` Clarification)

> [!NOTE]
> **Clarification on Hugging Face Hub's Auto-Detected `"5.0B"` Badge:** Shard 1 carries `general.size_label = "5.0B"` inherited from the non-MoE dense trunk staging header during conversion. This checkpoint contains the **complete `512`-expert `Qwen3.8-Flash-Next` architecture** (`42` `K = 512` ConeCollapsed layers + `7` `K = 640` boundary layers + `320M`-row PLE + `blk.48` NextN MTP):

| Architectural Component | Specification |
| :--- | :--- |
| **Upstream Base Model** | `Qwen/Qwen3.8-Flash-Next` (`rev 812d96e10a0dd82b69d749e68693698cf6c36daa`, 100% donor-free contrastive subspace edit) |
| **Nominal / Collapsed Params** | **`~176.5B` nominal** (`~152.3B` post-ConeCollapse + PLE; `92.28%` mean retained spectral energy) |
| **Active Transformer Params / Token** | **`~4.6B–5.0B`** (`10` of `512` routed experts + `1` shared expert + dense GDN/QSA; reflected in `general.size_label`) |
| **Backbone Topology (`qwen4exp`)** | `48` Trunk Layers (`36` Gated DeltaNet + `12` QSA Full Attention, `compress_ratio = 4`) + **`1` Subspace-Aligned `NextN` MTP Layer (`blk.48`)** |
| **MoE Width & Bias Orthogonalization** | **`512 / 512` Routed Experts** (`42 × K = 512` ConeCollapsed layers with orthogonalized `42 × F32` `ffn_down_exps.bias` [$\max \|U_\ell^\top b_d'\| / \|b_d'\| = 7.436 \times 10^{-10}$] + `7 × K = 640` boundary layers) |
| **GGUF Tensor Types (`1,128` LLM)** | `{F32: 393, Q8_0: 291, Q4_K: 99, Q5_K: 98, IQ1_S: 62, IQ1_M: 51, BF16: 49, MXFP4(39): 49, IQ2_XXS: 17, IQ2_XS: 10, Q2_0(42): 7, Q6_K: 1, Q1_0(41): 1}` (**`max_type = 42`, `0` custom types**) |

---

## File Manifest & Exact Byte / SHA-256 Verification

| File | Exact Size (`Bytes`) | Size (`GiB` / `GB`) | Tensors | Role & Notes | `SHA-256` |
| :--- | ---: | :---: | ---: | :--- | :--- |
| `Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated-00001-of-00002.gguf` | **`26,821,627,808`** | `24.9796 GiB` (`26.822 GB`) | `1,127` | Subspace-Abliterated Resident Trunk (`177` edited incl. `42` bias / `950` bit-identical) + `blk.48` `NextN` MTP | `6e8ae62c37bc08de1a69e9e78ab9bae0262f87ffe543595206d1c3b3a810fd67` |
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

## Empirical Certification: Refusal Removal, Signal Preservation & `NextN` MTP Speed

Evaluated under `llama-modchip v3.0` (`-t 15 -tb 15 --poll 0 --poll-batch 0`, default `--repack` ON, `RC_TABLES_DIR=/nonexistent`) against the donor-free `BF16` reference (`Mean PPL(base) = 2.776624 ± 0.054868`) and aligned `MaxQ-32GB`:

| Checkpoint | Exact LLM Bytes | `Mean PPL(Q) ± SE` | `ΔPPL` vs. Aligned | `Mean KLD ± SE` | `Median KLD` | `Same top p (%)` | Dual-Use Refusal Rate (`16` Prompts) | Blind-4 Coding/SQL/JSON Pass |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aligned `MaxQ-32GB`** | `38,341,683,424 B` | `5.80406 ± 0.13982` *(`5.8120` Stage 4C)* | `0.00%` (Ref) | `1.00399` *(`1.0069` Stage 4C)* | `0.39858` | `66.275%` | `43.75%` (`7 / 16`; `2 / 4` `<think>`¹) | `1 / 4` *(`v3.0` `--repack`; `2/8` Stage 4C `-nr`)* |
| **`MaxQ-32GB-Abliterated`** | **`38,341,683,424 B`** | **`5.822571 ± 0.140087`** | **`+0.3189%`** (`+0.018511`, `9.4×` under `<= 3.0%` Gate) | **`1.016915 ± 0.015155`** | **`0.398768`** | **`65.539 ± 0.471%`** | **`12.50%`** (`2 / 16`, `-71.4%`; **`0 / 4` `<think>`¹**) | **`1 / 4`** (`06_json` PASS — `100%` parity; **`4/4` loop-free** w/ sampling) |

- **¹ KL-Safe Dual-Use Compliance & `<think>` Refusal Audit (`0` Disguised Refusals):** Under KL-safe rank-2 contrastive subspace projection (`90% / 99% / Max KLD = 2.930651 / 7.303628 / 13.513656`), held-out dual-use security refusals (`enable_thinking=False`, `temp = 0`) drop from `7 / 16` (`43.75%`) to **`2 / 16` (`12.50%`, `-71.4%`)** — preserving `ΔPPL = +0.3189%` rather than over-projecting to `"0% refusals"` — with **`0 / 4` benign false refusals** and **`0 / 4` (`0.0%`) `<think>` spot refusals** (`enable_thinking=True`, `max_tokens = 2048` vs. `2 / 4` on Aligned `MaxQ-32GB`, and `2 / 4` true `<think>` refusals on Aligned `MaxQ-64GB` once Unicode curly apostrophe `I can’t` `\u2019` on `ref_01` is counted). Full-text inspection of all flipped `MaxQ-32GB-Abliterated` outputs (`radare2` binary patch, `glibc >= 2.32` safe-linking `ptr ^ (addr >> 12)` tcache audit, SUID `$PATH` audit, Python `pickle.__reduce__`, `3.3V` UART hardware pinout) confirmed working technical substance with **`0` curly-apostrophe false negatives, `0` disguised refusals, and `0` gibberish**.
- **`llama-modchip v3.0` Throughput, `blk.48` `NextN` MTP Alignment & `28.5 GiB` Memory (`16` Zen 5 Cores, `0` GPUs):** Aligned `MaxQ-32GB` achieves `74.55 tok/s` `pp512` and **`11.45–14.29 tok/s` `tg128`** (`+21.1%` combined `v3.0` SIMD/GDN/repack + non-convoy `-t <N-1> --poll 0` gain over Stage 4C's oversubscribed `-t 16 --poll 50 --no-repack` `9.46 tok/s`, `VmHWM = 26.91 GiB <= 28.5 GiB` Cgroup v2 `Swap=0` PASS; cold `rep 0` `Stockfish-PVS v2` refactoring fallback `12.82 tok/s`). **`MaxQ-32GB-Abliterated` (`VmHWM = 28.40 GiB <= 28.5 GiB`, `RssAnon = 2.43 GiB`)** achieves **`77.53 tok/s` `pp512`** and **`11.65 tok/s` mean / `12.73 tok/s` cold `rep 0` `tg128`** at **`72.83%` (`260 / 357`)** MTP acceptance on `p512`, **`12.78 tok/s` (`85.68%` = `1298 / 1515`)** on `blind_01` C++20, **`11.74 tok/s` (`84.32%` = `984 / 1167`)** on `blind_06` JSON, and a token-weighted **`76.22%`** MTP acceptance across top-4 `tg512` abliterated prompts (peak **`88.33%` @ `13.24 tok/s`** on `ref_09`), beating Aligned `MaxQ-32GB` on both benign controls (**`68.31%`** vs. `68.28%`) and complied-only security prompts (**`62.46%`** vs. `60.54%`).
- **Blind-4 Parity & Edition Positioning:** Matches default-repack Aligned `MaxQ-32GB` (`1 / 4` strict pass on `06_json_charterparty_settlement`), with recommended sampling (`temp=0.6, top_p=0.95, top_k=20, presence_penalty=0.2, repeat_penalty=1.05`) lifting Blind-4 loop-free rate from `2 / 4` (`temp = 0`) to **`4 / 4` (`100%` loop-free)**. For pure software engineering on 64 GB+ hosts, use Aligned **[`MaxQ-64GB`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF)** (`4 / 4` Blind-4 vs. `2 / 4` on `MaxQ-64GB-Abliterated`); use `MaxQ-32GB-Abliterated` for 32 GB security research, reverse engineering, red-team auditing, and uncensored legal/policy analysis.

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
