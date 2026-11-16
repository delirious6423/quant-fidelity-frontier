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
  - maxq
  - cpu-inference
  - llama-cpp
  - llama-modchip
---

# Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-GGUF (`53.99 GiB` · 100% Donor-Free Subspace-Abliterated · Built-In `NextN` MTP)

> [!CAUTION]
> **Responsible-Use Disclaimer & Edition Positioning (Opt-In KL-Safe Subspace-Abliterated Checkpoint):**
> **`Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated`** attenuates refusal-mediating activation subspaces via norm-preserving rank-2 orthogonal projection strictly across `blk.4..48` residual-stream writers (`ssm_out`, `attn_output`, `ffn_down_shexp`, `ffn_down_exps`, and `blk.48` `NextN` MTP; `blk.0..3` and `token_embd` spared), reducing held-out dual-use security refusals from **`56.25%` (`9/16`) to `18.75%` (`3/16`, `-66.7%`)** and `<think>` spot refusals to **`0 / 4` (`0.0%`)** at **`+0.1784%` `ΔPPL`** (rather than over-editing to `"0% refusals"`). Upstream safety refusal filters cannot be relied upon for content moderation.
> - **Intended Scope:** Authorized security research, reverse engineering, red-team auditing, CTF competitions, binary/exploit analysis, code auditing, and uncensored legal/policy analysis only.
> - **Deployment:** Do **not** expose unmoderated endpoints of this checkpoint to public networks without external access control and safety filtering.
> - **Default Pure Coding Flagship:** For general software engineering and standard upstream alignment (`4 / 4` strict Blind-4 pass vs. `2 / 4` on `MaxQ-64GB-Abliterated`), use **[`Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF)** (`./install.sh --preset qwen38-64gb`).

**`Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated`** is the opt-in uncensored **64 GB RAM tier** twin of [`Qwen3.8-Flash-Next-MaxQ-64GB`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF), built **100% donor-free** directly from [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`rev 812d96e10a0dd82b69d749e68693698cf6c36daa`) with zero third-party abliterated donor checkpoints or external vectors:
- **Single-Pass Fused Rank-2 Contrastive Subspace Projection + `FP64 Woodbury DualGram`:** Extracts a multi-layer rank-2 contrastive refusal basis (`64` Jinja-templated harmful/harmless pairs = `32` non-thinking + `32` thinking; verified `0.000000%` 8-gram and 13-gram overlap across all `32 / 32` checks against refusal eval, calibration, blind prompts, and `wiki-alt.txt`; rank-2 SVD spans `99.61%` of `u_nothink` and `98.24%` of `u_think`) and edits **`135` residual-stream writer tensors across `45` layers `blk.4..48`** (`ssm_out`, `attn_output`, `ffn_down_shexp`, `ffn_down_exps`, **plus `3` `blk.48` `NextN` MTP writers** so speculative drafting stays aligned when complying on dual-use prompts; `blk.0..3` and `token_embd` spared), refitting quantized weights in a single pass from `BF16` without double-quantization.
- **Bit-Identical Unedited Tensors (`950 / 1,085`) & Shared `10.73 GiB` PLE:** Preserves `950` Shard 1 tensors (`ffn_gate_exps`/`ffn_up_exps`, `blk.0..3`, `token_embd`, `output.weight`, routers, norms) **100% bit-identical** to aligned `MaxQ-64GB`, plus the exact shared `11,520,055,616 B` (`10.7289 GiB`) `PCA-Wiener D=256 + RaBitQ Q1_0` Shard 2 PLE table, using **100% standard `ggml` tensor types (`max_type = 42`, `0` custom types)**.

---

## Architecture & Parameter Summary (`general.size_label` Clarification)

> [!NOTE]
> **Clarification on Hugging Face Hub's Auto-Detected `"5.0B"` Badge:** Shard 1 carries `general.size_label = "5.0B"` inherited from the non-MoE dense trunk staging header during conversion. This checkpoint contains the **complete, unpruned `176.5B` `Qwen3.8-Flash-Next` architecture** (`512/512` routed experts per layer + `320M`-row PLE table + `blk.48` NextN MTP):

| Architectural Component | Specification |
| :--- | :--- |
| **Upstream Base Model** | `Qwen/Qwen3.8-Flash-Next` (`rev 812d96e10a0dd82b69d749e68693698cf6c36daa`, 100% donor-free in-house contrastive subspace edit) |
| **Total Parameters** | **`~176.5B`** (`~125.3B` Transformer + `512`-Expert MoE + `NextN` MTP + `51.2B` 320M-row PLE) |
| **Active Transformer Params / Token** | **`~5.0B`** (`10` of `512` routed experts + `1` shared expert + dense GDN/QSA; reflected in `general.size_label`) |
| **Backbone Topology (`qwen4exp`)** | `48` Trunk Layers (`36` Gated DeltaNet + `12` QSA Full Attention, `compress_ratio = 4`) + **`1` Subspace-Aligned `NextN` MTP Layer (`blk.48`)** |
| **MoE Width & Routing** | **`512 / 512` Unpruned Routed Experts** (`Top-10` active, `K = 640` full width) + `1` Shared Expert (`K = 640`), `0` `ffn_down_exps.bias` |
| **GGUF Tensor Types (`1,086` LLM)** | `{F32: 351, Q8_0: 339, Q6_K: 198, IQ2_S: 60, BF16: 49, Q2_0(42): 39, IQ3_XXS: 36, MXFP4(39): 7, IQ4_NL: 4, Q2_K: 2, Q1_0(41): 1}` (**`max_type = 42`, `0` custom types**) |

---

## File Manifest & Exact Byte / SHA-256 Verification

| File | Exact Size (`Bytes`) | Size (`GiB` / `GB`) | Tensors | Role & Notes | `SHA-256` |
| :--- | ---: | :---: | ---: | :--- | :--- |
| `Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-00001-of-00002.gguf` | **`46,449,692,736`** | `43.2596 GiB` (`46.450 GB`) | `1,085` | Subspace-Abliterated Trunk (`48L`, `135` edited / `950` bit-identical) + `blk.48` `NextN` MTP | `384bc936bdad75383c03e87184056e1cb0b037b0e3f5d5ae5273fd46f55150aa` |
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
- **Mandatory `<think>` Token Budget (`max_tokens >= 16384`) & Anti-Repetition Sampling:** Always set **`max_tokens >= 16384`** when `enable_thinking=True` (`max_tokens = 2048` `<think>` spot checks verified `0 / 4` refusal intent inside `<think>`), or pass `enable_thinking=False` for `<= 4,096` token caps. While `MaxQ-64GB-Abliterated` is `4 / 4` (`100%`) loop-free at greedy `temp = 0` on Blind-4 coding/SQL/JSON, greedy `temp = 0` without penalties (`max_tokens = 512`) hit `1 / 16` tail repetition (`ref_14` ASCII waveform after substantive technical output); pass `--temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.2 --repeat-penalty 1.05` to eliminate tail loops on open-ended dual-use generations.
- **Vocabulary Isolation:** Always set `RC_TABLES_DIR=/nonexistent` when serving `Qwen3.8` from a directory containing Nemotron RouteCrack tables (`./modchip` sets this automatically).

---

## Empirical Certification: Refusal Removal, Signal Preservation & `NextN` MTP Speed

### 1. Blind `wiki-alt.txt` (`40 × 512`) PPL + KLD & Dual-Use Refusal Evaluation

Evaluated under `llama-modchip v3.0` (`-t 15 -tb 15 --poll 0 --poll-batch 0`, default `--repack` ON, `RC_TABLES_DIR=/nonexistent`) against the donor-free `BF16` reference (`Mean PPL(base) = 2.776624 ± 0.054868`) and aligned `MaxQ-64GB`:

| Checkpoint | Exact LLM Bytes | `Mean PPL(Q) ± SE` | `ΔPPL` vs. Aligned | `Mean KLD ± SE` | `Median KLD` | `Same top p (%)` | Dual-Use Refusal Rate (`16` Prompts) | Blind-4 Coding/SQL/JSON (`temp=0`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aligned `MaxQ-64GB`** | `57,969,748,352 B` | `3.43825 ± 0.07119` *(`3.4325` Stage 4C)* | `0.00%` (Ref) | `0.49803` *(`0.4971` Stage 4C)* | `0.10561` | `78.314%` | `56.25%` (`9 / 16`; `2 / 4` `<think>`¹) | `4 / 4` (`5/8` full suite) |
| **`MaxQ-64GB-Abliterated`** | **`57,969,748,352 B`** | **`3.444383 ± 0.071771`** | **`+0.1784%`** (`+0.006133`, `16.8×` under `<= 3.0%` Gate) | **`0.498155 ± 0.009978`** | **`0.106715`** | **`78.275 ± 0.408%`** | **`18.75%`** (`3 / 16`, `-66.7%`; **`0 / 4` `<think>`¹**) | **`2 / 4`** (`03_sql`, `06_json` PASS; `01_cpp` `-Werror` clean²; `4/4` loop-free) |
| `ISTA-DASLab GSQ-RCO-Coder-IQ1_M` | `58,408,584,928 B` | `5.09164 ± 0.12142` | `+48.1%` | `0.93024` | `0.29644` | `69.382%` | — | — |

- **¹ KL-Safe Dual-Use Compliance & `<think>` Refusal Audit (`0` Disguised Refusals):** Under KL-safe rank-2 contrastive subspace projection (`90% / 99% / Max KLD = 1.462980 / 5.142135 / 12.492022`), held-out dual-use security refusals (`enable_thinking=False`, `temp = 0`) drop from `9 / 16` (`56.25%`) to **`3 / 16` (`18.75%`, `-66.7%`)** — preserving `ΔPPL = +0.1784%` rather than over-projecting to `"0% refusals"` — with **`0 / 4` benign false refusals** and **`0 / 4` (`0.0%`) `<think>` spot refusals** (`enable_thinking=True`, `max_tokens = 2048`). Aligned `MaxQ-64GB`'s true `<think>` spot refusal count was **`2 / 4`** once Unicode curly apostrophe `I can’t` (`U+2019` on `ref_01`) is counted alongside ASCII `1 / 4`; full-text inspection of all flipped `MaxQ-64GB-Abliterated` generations (`radare2` binary patch, `Frida` OkHttp3/Conscrypt SSL unpinning, SUID `$PATH` audit, Python `pickle.__reduce__`, `Scapy` ARP lab, RFC 4648 base32 DNS tester) confirmed concrete working code/commands with **`0` curly-apostrophe false negatives, `0` disguised refusals, and `0` gibberish**.
- **² Transparent Blind-4 Coding Trade-Off & Edition Positioning:** On the 4-prompt deterministic subset (`temp = 0`), `MaxQ-64GB-Abliterated` achieves **`2 / 4` strict pass** (`03_sql_window_grace_islands` **PASS**, `06_json_charterparty_settlement` **PASS**, **`4 / 4 = 100%` loop-free**; `01_cpp_lockfree_ring` compiled cleanly with `0` warnings/errors under `clang++ -std=c++20 -O2 -Wall -Wextra -Werror -pthread`, failing only its self-written `total_constructed` counter check in `main()`) versus **`4 / 4`** on Aligned `MaxQ-64GB`. Use **Aligned [`MaxQ-64GB`](https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF)** as the default pure coding flagship, and **`MaxQ-64GB-Abliterated`** for security research, reverse engineering, red-team auditing, and uncensored legal/policy analysis.

### 2. `llama-modchip v3.0` Throughput & `blk.48` `NextN` MTP Alignment (`16` Zen 5 Cores, `0` GPUs)

- **Aligned `MaxQ-64GB` Baseline (`-t 15 -tb 15 --poll 0`, default `--repack` ON):** `71.97 tok/s` `pp512`, **`12.66 tok/s` `tg128`** (`+72.8%` combined `v3.0` SIMD/GDN/repack + non-convoy `-t <N-1> --poll 0` gain over Stage 4C's oversubscribed `-t 16 --poll 50 --no-repack` `7.33 tok/s`; `93.00%` MTP acceptance, `VmHWM = 45.08 GiB`).
- **`MaxQ-64GB-Abliterated` (`--spec-type draft-mtp`, `VmHWM = 46.34 GiB`, `RssAnon = 2.31 GiB`):** Delivers **`10.29 tok/s` `tg128`** (`10.286 tok/s` exact mean, `66.43–70.31 tok/s` `pp512`, **`74.36%`** = `261 / 351` MTP accept, emitting direct C++20 code from token 1) on `p512`, **`10.94 tok/s` (`85.66%` = `2205 / 2574`)** on `blind_01` C++20, and **`10.61 tok/s` (`82.15%` = `764 / 930`)** on `blind_06` JSON. Across the 20-prompt `tg512` suite, `blk.48` `NextN` MTP subspace alignment **exceeded Aligned `MaxQ-64GB` in token-weighted MTP draft acceptance** on benign controls (**`75.74%`** vs. `70.36%`, peak **`86.87%` @ `12.15 tok/s`** on `benign_02`), all-16 security prompts (**`65.27%`** vs. `61.48%`), complied-only security prompts (**`67.25%`** vs. `63.64%`), and top-4 abliterated prompts (**`78.08%`** vs. `70.83%`, peak **`84.14%` @ `11.91 tok/s`** on `ref_11`).
- **Zero-Weight `Stockfish-PVS v2` Fallback:** On cold single-turn `rep 0`, `draft-mtp` wins `6 / 6` prompts; zero-weight `Stockfish-PVS v2` (`--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 3 --spec-ngram-simple-min-hits 1`) delivers `10.58 tok/s` (`1.67×` over `none`) cold `rep 0` on source-in-context refactoring with `-0.85 GiB` lower `VmHWM`.

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
