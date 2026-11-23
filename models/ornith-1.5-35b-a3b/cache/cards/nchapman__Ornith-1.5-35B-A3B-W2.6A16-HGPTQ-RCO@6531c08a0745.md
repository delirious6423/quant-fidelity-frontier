---
base_model: ornith-ai/Ornith-1.5-35B-A3B
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w2a16
- w4a16
- mixed-precision
- moe
- rco
- compressed-tensors
- vllm
---

# Ornith-1.5-35B-A3B-W2.6A16-HGPTQ-RCO

Mixed 2-bit / 4-bit Hadamard + GPTQ (`W2.6A16-HGPTQ-RCO` V2: `w2_g128_sym` with symmetric half-integer `code_offset=0.5`, `w4_g128`, `H_512` Walsh–Hadamard rotation) quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (`10fbf86fed7ecee4a061f8b499a618f46001cac1`) built with Riemannian Constrained Optimization (`HGPTQ-RCO`). It compresses the 67.0 GiB BF16 256-expert MoE model down to **13.01 GiB** (`5.15×` smaller than BF16, saving `5.77 GiB` vs. pure `W4A16`; **`10.94 GiB`** pure text backbone) so the entire 35B-A3B MoE model, Vision tower, and Native MTP drafter fit on a single **16 GB GPU**.

- **VRAM & GPU fit:** `13.01 GiB` (`13,972,519,592 B`) total across all 40 MoE layers, Vision (`0.22 GiB`), and Native MTP (`0.41 GiB`), leaving ~`3.0 GiB` on a 16 GB card for KV cache and speculative decoding (for 24 GB GPUs, see [`Ornith-1.5-35B-A3B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Ornith-1.5-35B-A3B-W4A16-HGPTQ) at `18.78 GiB`).
- **V2 quantization recipe:** Moving scales from `g64` to `g128` + `H_512` rotation frees scale overhead to upgrade **12 / 40 routed-expert layers** (`3,072` experts, layers `28..39`, up from `10/40` in V1) to `W4_g128` while shrinking total size by `149 MiB`, and upgrades the remaining **28 / 40 routed-expert layers** (`7,168` experts, layers `0..27`) to symmetric half-integer 2-bit (`w2_g128_sym`, `{-1.5, -0.5, +0.5, +1.5}`, `code_offset=0.5`), keeping 100% of Attention/GDN (`130` linears), Shared Experts (`120` linears), Native MTP (`775` linears), and Vision (`110` linears) in `W4` (`lm_head` and `mtp.fc` in `W8_g128`), polished with end-to-end FP32-master Norm/GDN KL QAT.
- **Fidelity vs. BF16:** `0.2283` weighted KL (`0.3851` mean KL, `−5.65%` PPL; `-14.9%` lower KL than V1) on `probe-v4`, `0.3934` mean KL (`0.5034` weighted KL, `−11.63%` PPL; `-17.1%` lower held-out KL than V1) on `probe-guard`, `91.24%` / `66.02%` isolated/E2E MTP top-1 agreement, and `96.69%` / `94.37%` isolated/E2E vision top-1 agreement (`98.52%` / `89.63%` on `vision-guard-v2`).

## Quality vs. BF16 Base Model (`ornith-ai/Ornith-1.5-35B-A3B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model and the `18.78 GiB` pure 4-bit [`Ornith-1.5-35B-A3B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Ornith-1.5-35B-A3B-W4A16-HGPTQ) checkpoint:

| Evaluation | BF16 Base (`ornith-ai/Ornith-1.5-35B-A3B`) | `W4A16-HGPTQ` (`4.00 bpw`) | **`W2.6A16-HGPTQ-RCO` V2 (`12/40` W4)** | Notes |
|---|---:|---:|---:|---|
| **Safetensors Size** | `66.97 GiB` | `18.78 GiB` | **`13.01 GiB` (`13,972,519,592 B`)** | **`5.15×` compression** (`10.94 GiB` pure text backbone; `−5.77 GiB` vs. W4A16) |
| **Text Perplexity** (378 prompts, 842k tokens) ↓ | `10.8033` | `10.2627` (`−5.00%`) | **`10.1931` (`−5.65%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | `0.1139` / `0.2590` | **`0.2283` / `0.3851`** | `-14.9%` weighted KL vs V1 (`0.2682` / `0.4338`) |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `249.8651` | `257.9844` (`+3.25%`) | **`220.8027` (`−11.63%`)** | Token mean KL `0.3934` (weighted KL `0.5034`, `-17.1%` vs V1 `0.4744`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 4.4308`) | `91.44%` Top-1 (`KL 0.0215`) | **`91.24%` Top-1 (`NLL 4.4567`)** | Isolated MTP KL `0.0242` (end-to-end Top-1 `66.02%`, KL `0.4919`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | `96.36%` Top-1 | **`96.69%` Top-1** | Isolated ViT KL `0.0376` (end-to-end Top-1 `94.37%`, KL `0.1874`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | `97.04%` Top-1 | **`98.52%` Top-1** | Isolated ViT KL `0.0150` (end-to-end Top-1 `89.63%`, KL `0.2554`) |

## What Is Quantized (`HGPTQ-RCO` V2 16 GB Allocation)

- **Strictly Standard Bit-Widths (`{2, 4, 8}` bits, `group_size=128`, `H_512`)**:
  - **Always-Active Backbone (`100%` in `W4_g128`, `250` linears)**: All `130` hybrid GatedDeltaNet + full-attention projections (`self_attn`, `linear_attn`) and all `120` shared expert projections (`mlp.shared_expert.{gate_proj,up_proj,down_proj}`) across all 40 layers stay in `W4_g128` (`head_dim=512`, MSE scale search + 2-pass H-CD).
  - **Routed MoE Experts (`2.60 bpw` nominal across `30,720` linears)**:
    - **`W4_g128` (`12` layers = `3,072` experts = `9,216` linears)**: Layers `28..39` (`model.language_model.layers.28..39.mlp.experts.0..255.{gate_proj,up_proj,down_proj}`).
    - **`W2_g128_sym` (`28` layers = `7,168` experts = `21,504` linears)**: Layers `0..27` (`model.language_model.layers.0..27.mlp.experts.0..255.{gate_proj,up_proj,down_proj}`) with symmetric half-integer levels (`{-1.5, -0.5, +0.5, +1.5}`, `observer_kwargs.code_offset = 0.5`).
  - **Untied Output & Drafter Fusion Heads (`W8_g128`, `2` linears)**: `lm_head` (`248320 × 2048`) and `mtp.fc` (`2048 × 4096`) quantized with unrotated 8-bit Cholesky GPTQ + MSE scale search (`group_size=128`).
  - **Native MTP (`775` linears in `W4_g128`)**: `4` `self_attn` projections + `3` `mlp.shared_expert` projections + `768` routed MoE expert projections (`head_dim=512`, `group_size=128`).
  - **Vision (`110` linears in `W4`)**: All 27 ViT blocks + 2 spatial merger projections (`visual.blocks.*.mlp.linear_fc2` uses per-channel GPTQ + `head_dim=16` Hadamard rotation `v_vision_fc2`; all other Vision linears use `group_size=128` + `head_dim=128` `v_vision`).
  - **End-to-End FP32-Master Norm & GDN-Gate KL QAT**: Post-allocation KL-divergence QAT over all layer `RMSNorm`s, `q_norm`/`k_norm`, GatedDeltaNet scalar projections (`in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`), and `mlp.shared_expert_gate` against exact BF16 teacher final-norm states.
- **Kept in BF16**:
  - `embed_tokens` (`0.947 GiB`), `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `mlp.gate`, `mlp.shared_expert_gate`, `visual.patch_embed.proj`, norms, and biases.
- **Static Hot-Expert Cache Rankings**:
  - `static_hot_cache_rankings.json` (and `runtime/static_hot_cache_rankings.json` + `runtime/expert_routing_profile.json`) provides pre-profiled expert rankings (`0..255` ordered hottest-first per layer `"0".."39"`, plus `"mtp.0"` in `expert_routing_profile.json`) so inference engines can pin hot experts in VRAM from the first token.
- **Auto-loading `vLLM` plugin (`--trust-remote-code`)**:
  - Ships `vllm_warp_quant.py` wired via `auto_map` so `vllm serve nchapman/Ornith-1.5-35B-A3B-W2.6A16-HGPTQ-RCO --trust-remote-code` handles `w2_g128_sym` (`code_offset=0.5`) natively (`VLLM_WARP_QUANT_W2_MARLIN=1` enables full CUDA-graph Marlin/Humming kernels; verified on `vLLM 0.34.0`).

## Reproduction

```bash
# 1. Streaming W4A16-HGPTQ + per-layer W2 symmetric half-integer routed-expert bank
warp-quant quantize --model 35b-ornith \
  --group-size 128 --hadamard-size 512 --w2-code-offset 0.5 \
  --output-dir ./out/ornith-35b-w4 \
  --save-rco-bank-dir ./out/ornith-35b-w2-bank

# 2. Gumbel-Softmax STE RCO search (30% = 12/40 W4 routed budget) + E2E Norm/GDN KL QAT + assembly
python -c '
from pathlib import Path
from warp_quant.cachepath import resolve_snapshot
from warp_quant.models import resolve
from warp_quant.streaming_moe import run_moe_rco_search_and_qat
hub_id, info = resolve("35b-ornith")
run_moe_rco_search_and_qat(
    model_key="35b-ornith",
    base_dir=Path(resolve_snapshot(hub_id, revision=info.get("hub_revision"))),
    w4_ckpt_dir=Path("./out/ornith-35b-w4"),
    rco_w2_bank_dir=Path("./out/ornith-35b-w2-bank"),
    out_root_dir=Path("./out/rco-35b-ornith"),
    budgets=[0.30],
    assemble_names=["30pct_rco"],
)'

# 3. Teacher-forced fidelity probe
warp-quant probe --model 35b-ornith \
  --quant ./out/rco-35b-ornith/checkpoints/30pct_rco \
  --eval-mtp \
  --eval-vision data/vision-probe-v2 \
  --eval-vision-guard data/vision-guard-v2
```

Receipts: [`probe_results.json`](probe_results.json) · [`v2_receipts/probe_35b_ornith_w26_v2.json`](v2_receipts/probe_35b_ornith_w26_v2.json) · [`rco_manifest.json`](rco_manifest.json) · [`rot_receipt.json`](rot_receipt.json) · [`pareto_summary.json`](pareto_summary.json)
