---
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- qwen3.8
- qwen3.8-flash-next
- wv-v5
- moe
- quantized
license: apache-2.0
---

# Qwen3.8-Flash-Next-WV-v5-3.5bpw-GGUF

Native **`llama.cpp` (`3.52 bpw`)** 2-shard GGUF projection of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` Total Parameters / `48` Hybrid GDN + Sparse-QSA Full-Attention Layers / `512` Routed + `1` Shared Expert / `4`-Stream mHC Hyper-Connections / Layer-1 N-Gram Per-Layer Embedding), derived from the [`WV-v5` (`3.5 bpw`)](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-v5-3.5bpw) master checkpoint.

Uses **100% standard upstream `ggml` lookup-table-free formats** (`Q3_K`, `Q4_0`, `Q2_0`, `Q6_K`, `Q8_0`, `BF16`, `F32`) for direct out-of-the-box execution in mainline `llama.cpp` (`qwen4exp` architecture) with zero custom decoders or patches required.

## WV-v5 Model Family

| Repository | Format & Target Runtime | GPU VRAM | Host RAM (PLE) | Highlights |
| :--- | :--- | :---: | :---: | :--- |
| **[`Qwen3.8-Flash-Next-WV-v5-3.5bpw-GGUF`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-v5-3.5bpw-GGUF)** | 2-Shard Standard `llama.cpp` `GGUF` (`3.52 bpw`) | `55.20 GB` | `28.80 GB` (`mmap`) | 100% upstream `ggml` (`Q3_K`/`Q4_0`/`Q2_0`/`Q6_K`/`Q8_0`) drop-in for `llama.cpp` |
| **[`Qwen3.8-Flash-Next-WV-v5-3.5bpw`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-v5-3.5bpw)** | 2-Shard `WV-v5` Master Container (`3.53 bpw`) | `55.31 GB` | `28.80 GB` (`mmap`) | Master `WZV4` non-linear checkpoint w/ `Q8_0` surgical `lm_head` & RMT Hessians |
| **[`Qwen3.8-Flash-Next-WV-v5-3.1bpw-64GB-SM120-v20`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-v5-3.1bpw-64GB-SM120-v20)** | Pre-swizzled `sm_120` `safetensors` + Host PLE | `~60 GB` | `28.80 GB` (`0 GB` VRAM) | Fits 64 GB / 96 GB Blackwell GPUs; baremetal `sm_120` tensor-core engine |
| **[`Qwen3.8-Flash-Next-WV-v5-Sub2.5bpw-SM120`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-v5-Sub2.5bpw-SM120)** | Pre-swizzled `sm_120` `safetensors` + Host PLE | `~45 GB` | `28.80 GB` (`0 GB` VRAM) | Compact sub-`2.5 bpw` `sm_120` deployment |

---

## Key Upgrades Over v4 (`Qwen3.8-Flash-Next-3.5bpw-GGUF`)

- **True `Q8_0` Surgical `output.weight` (`lm_head`, `675.4 MB`):** Preserves the `WV-v5` Pass D+ gain & 9-token double Gram-Schmidt LDA output head verbatim instead of `Q6_K`.
- **Full `FP32` Lossless Unpack + 48-Layer Activation-Aware Trajectory Compensation:** Unpacks `WZV4` rank-2 SVD + non-uniform grids purely in `FP32`, tracks live 4-stream mHC trajectories across `34 x 2048` (`69,632` tokens) genuine `<think>` math/logic/coding, 12-language polyglot code, and WikiText-2 sequences, and absorbs layer-to-layer snap drift via Cholesky cross-covariance (`C_del = X_wv^T (X_wv - X_stock)`).
- **50/50 Prose + Math/Code Hessians, RMT Shrinkage & Validation-Gated CULM:** Applies Marchenko-Pastur + Donoho-Gavish flat-bulk RMT shrinkage on Top-24 routed expert Hessians and undersampled cold experts, refits discrete `Q3_K` / `Q4_0` / `Q2_0` integer codes via validation-gated coordinate descent (`CULM`), and absorbs remaining routed snap error into the `Q8_0` Shared Expert (`ffn_down_shexp.weight`).
- **Embedded Official Jinja Chat Template & Upstream Special-Token Metadata:** Embeds the full `Qwen3.8-Flash-Next` Jinja chat template (defaulting to `reasoning_effort="xhigh"` with automatic `<think>` generation prefill and tool-call formatting) plus exact upstream `llama.cpp` `CONTROL` / `USER_DEFINED` / `UNUSED` token types directly in Shard 1.

## Shards & Memory Footprint

| File | Size | Placement | Contents |
| :--- | :---: | :---: | :--- |
| `Qwen3.8-Flash-Next-WV-v5-3.5bpw-00001-of-00002.gguf` | `55.20 GB` (`51.41 GiB`) | GPU VRAM | Token embeddings, all `48` GDN / QSA / MoE layers (`1,223` tensors, `3.518 bpw` effective), & `Q8_0` output head |
| `Qwen3.8-Flash-Next-WV-v5-3.5bpw-00002-of-00002.gguf` | `28.80 GB` (`26.82 GiB`) | Host RAM (`mmap`) | Layer-1 N-Gram Per-Layer Embedding table (`per_layer_token_embd.weight`, `Q4_0`, `1` tensor) |
| `benchmarks/qfn_wv_v5_stock_35bpw_eval_summary.json` | `~36 KB` | Reference | Full 6-suite 48-layer `BF16` vs `WV-v5` vs `Stock GGUF` PPL/KLD/JSD/Top-k/CosSim logs |

## Precision Allocation

- **Output Head (`output.weight`), Shared Experts (`ffn_*_shexp.weight`), Hyper-Connection Mixers, & PLE Projections:** `Q8_0`
- **GDN (`ssm_in`, `ssm_z`, `ssm_out`) & Sparse-QSA Attention (`attn_q`, `attn_k`, `attn_v`, `attn_output`, `wq_b`, `wk`) Projections:** `Q6_K`
- **Token Embeddings & High-Sensitivity Routed `down_exps` (`20/48` layers, incl. all `12` Full-Attention anchors + top `8` GDN sensitivity layers):** `Q4_0`
- **Routed `gate_exps` & `up_exps` (`48/48` layers):** `Q3_K`
- **Remaining Routed `down_exps` (`28/48` layers):** `Q2_0`
- **RMSNorms, Router Gates (`ffn_gate_inp.weight`), Shared Gates, QSA Indexer Projections, & GDN `A` / Conv1d / Bias Tensors:** `F32` / `BF16`

## Held-Out 6-Suite 48-Layer Logit & Perplexity Evaluation (`248,320` Vocab)

| Evaluation Suite | `BF16` PPL | `WV-v5` PPL | `Stock GGUF` PPL (`dPPL` vs `BF16`) | Mean `KLD` vs `BF16` | Median `KLD` | `Top-1` vs `BF16` | `Top-5` in `BF16` | Direct `WV-v5 <-> Stock` `KLD` / `CosSim` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Genuine Math/Code <think>` (`4x2048`)** | `1.9961` | `1.9404` | **`1.9315` (`-0.0646`)** | **`0.06389`** | `0.01268` | **`91.54%`** | **`99.77%`** | **`0.00266` / `0.99715`** |
| **`Deep Olympiad & C++20 <think>` (`4x2048`)** | `2.5498` | `2.4861` | **`2.4733` (`-0.0765`)** | **`0.06804`** | `0.02012` | **`90.82%`** | **`99.82%`** | **`0.00282` / `0.99826`** |
| **`Self-Gen Math/Code` (`2048`)** | `5.5732` | `5.6441` | **`5.6707` (`+0.0976`)** | **`0.16417`** | `0.06603` | **`86.47%`** | **`98.53%`** | **`0.00845` / `0.99467`** |
| **`External Held-Out` (`2048`)** | `11.1585` | `10.2190` | **`10.1749` (`-0.9835`)** | **`0.21475`** | `0.09584` | **`81.97%`** | **`98.29%`** | **`0.01056` / `0.99538`** |
| **`WikiText-2` (`4x2048`)** | `2.3413` | `2.3521` | **`2.3756` (`+0.0343`)** | **`0.43870`** | `0.03929` | **`83.89%`** | **`96.80%`** | **`0.02635` / `0.98986`** |
| **`WikiText-2` (`4x1024`)** | `2.5724` | `2.5473` | **`2.5831` (`+0.0107`)** | **`0.48949`** | `0.04982` | **`82.23%`** | **`96.38%`** | **`0.02973` / `0.98976`** |

## Usage (`llama.cpp`)

Point `llama-server` or `llama-cli` (`qwen4exp`-capable mainline `llama.cpp`) to Shard `00001-of-00002`. Shard `00002` (`per_layer_token_embd.weight`, `28.80 GB`) is discovered automatically in the same directory and memory-mapped into host RAM via `TENSOR_READ_LAZY`:

```bash
# Optional on network/cloud disks: pre-warm Shard 2 into host RAM pagecache so
# PLE n-gram gathers hit DDR5 RAM instead of 4 KB random disk page faults:
dd if=Qwen3.8-Flash-Next-WV-v5-3.5bpw-00002-of-00002.gguf of=/dev/null bs=16M

./llama-server \
  -m Qwen3.8-Flash-Next-WV-v5-3.5bpw-00001-of-00002.gguf \
  -ngl 999 \
  -ot "per_layer_token_embd\.weight=CPU" \
  -c 65536 -fa on -b 2048 -ub 2048 \
  -ctk bf16 -ctv bf16 \
  --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```
