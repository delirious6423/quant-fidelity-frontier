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

Mixed 2-bit / 4-bit Hadamard + GPTQ (`W2.6A16`, group size 64) quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (`10fbf86fed7ecee4a061f8b499a618f46001cac1`) built with Riemannian Constrained Optimization (`HGPTQ-RCO`). It compresses the 67.0 GiB BF16 256-expert MoE model down to **13.15 GiB** (5.09× smaller than BF16, saving 5.63 GiB vs. pure `W4A16`) so the entire 35B-A3B MoE model, Vision tower, and Native MTP drafter fit on a single **16 GB GPU**.

- **VRAM & GPU fit:** 13.15 GiB total across all 40 MoE layers, Vision, and Native MTP, leaving ~2.85 GiB on a 16 GB card for KV cache and speculative decoding (for 24 GB GPUs, see [`Ornith-1.5-35B-A3B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Ornith-1.5-35B-A3B-W4A16-HGPTQ) at 18.78 GiB).
- **Quantization recipe:** Keeps 100% of Attention/GDN (130 linears), Shared Experts (120 linears), Native MTP (775 linears), and Vision (110 linears) in `W4_g64` (`lm_head` and `mtp.fc` in `W8_g64`), while allocating the 10 most sensitive routed-expert layers (2,560 experts) to `W4_g64` and the remaining 30 routed-expert layers (7,680 experts) to `W2_g64`, polished with end-to-end FP32-master Norm/GDN KL QAT.
- **Fidelity vs. BF16:** 0.2682 weighted KL (0.4338 mean KL, −4.12% PPL) on `probe-v4`, 0.4744 mean KL (0.5918 weighted KL, −4.24% PPL) on the `probe-guard` holdout, 91.44% / 63.88% isolated/E2E MTP top-1 agreement, and 95.36% / 92.38% isolated/E2E vision top-1 agreement (97.78% / 86.67% on `vision-guard-v2`).

## Quality vs. BF16 Base Model (`ornith-ai/Ornith-1.5-35B-A3B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model and the `18.78 GiB` pure 4-bit [`Ornith-1.5-35B-A3B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Ornith-1.5-35B-A3B-W4A16-HGPTQ) checkpoint:

| Evaluation | BF16 Base (`ornith-ai/Ornith-1.5-35B-A3B`) | `W4A16-HGPTQ` (`4.00 bpw`) | **`W2.6A16-HGPTQ-RCO` (`2.62 bpw`)** | Notes |
|---|---:|---:|---:|---|
| **Safetensors Size** | `66.97 GiB` | `18.78 GiB` | **`13.15 GiB` (`14,121,080,536 B`)** | **`5.09×` compression** (`−53.82 GiB` vs. BF16, `−5.63 GiB` vs. W4A16) |
| **Text Perplexity** (378 prompts, 842k tokens) ↓ | `10.8033` | `10.2627` (`−5.00%`) | **`10.3577` (`−4.12%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | `0.1139` / `0.2590` | **`0.2682` / `0.4338`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `249.8651` | `257.9844` (`+3.25%`) | **`239.2615` (`−4.24%`)** | Token mean KL `0.4744` (weighted KL `0.5918`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 4.4308`) | `91.44%` Top-1 (`KL 0.0215`) | **`91.44%` Top-1 (`NLL 4.4458`)** | Isolated MTP KL `0.0215` (end-to-end Top-1 `63.88%`, KL `0.5776`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | `96.36%` Top-1 | **`95.36%` Top-1** | Isolated ViT KL `0.0483` (end-to-end Top-1 `92.38%`, KL `0.1917`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | `97.04%` Top-1 | **`97.78%` Top-1** | Isolated ViT KL `0.0231` (end-to-end Top-1 `86.67%`, KL `0.2902`) |

## What Is Quantized (`HGPTQ-RCO` 16 GB Allocation)

- **Strictly Standard Bit-Widths (`{2, 4, 8}` bits, `group_size=64`)**:
  - **Always-Active Backbone (`100%` in `W4_g64`, `250` linears)**: All `130` hybrid GatedDeltaNet + full-attention projections (`self_attn`, `linear_attn`) and all `120` shared expert projections (`mlp.shared_expert.{gate_proj,up_proj,down_proj}`) across all 40 layers stay in `W4_g64` (`head_dim=64`, MSE scale search + 2-pass H-CD).
  - **Routed MoE Experts (`2.50 bpw` average across `30,720` linears)**:
    - **`W4_g64` (`10` layers = `2,560` experts = `7,680` linears)**: Layers `30..39` (`model.language_model.layers.30..39.mlp.experts.0..255.{gate_proj,up_proj,down_proj}`).
    - **`W2_g64` (`30` layers = `7,680` experts = `23,040` linears)**: Layers `0..29` (`model.language_model.layers.0..29.mlp.experts.0..255.{gate_proj,up_proj,down_proj}`).
  - **Untied Output & Drafter Fusion Heads (`W8_g64`, `2` linears)**: `lm_head` (`248320 × 2048`) and `mtp.fc` (`2048 × 4096`) quantized with unrotated 8-bit Cholesky GPTQ + MSE scale search (`group_size=64`).
  - **Native MTP (`775` linears in `W4_g64`)**: `4` `self_attn` projections + `3` `mlp.shared_expert` projections + `768` routed MoE expert projections (`head_dim=64`, `group_size=64`).
  - **Vision (`110` linears in `W4_g64`)**: All 27 ViT blocks + 2 spatial merger projections (`visual.blocks.*.mlp.linear_fc2` uses per-channel GPTQ + `head_dim=16` Hadamard rotation `v_vision_fc2`; all other Vision linears use `group_size=64` + `head_dim=64`).
  - **End-to-End FP32-Master Norm & GDN-Gate KL QAT**: Post-allocation KL-divergence QAT over all layer `RMSNorm`s, `q_norm`/`k_norm`, GatedDeltaNet scalar projections (`in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`), and `mlp.shared_expert_gate` against exact BF16 teacher final-norm states.
- **Kept in BF16**:
  - `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `mlp.gate`, `mlp.shared_expert_gate`, `visual.patch_embed.proj`, norms, and biases.
- **Static Hot-Expert Cache Rankings**:
  - `static_hot_cache_rankings.json` (and `runtime/static_hot_cache_rankings.json` + `runtime/expert_routing_profile.json`) provides pre-profiled expert rankings (`0..255` ordered hottest-first per layer `"0".."39"`, plus `"mtp.0"` in `expert_routing_profile.json`) so inference engines can pin hot experts in VRAM from the first token.

## Reproduction

```bash
# 1. Streaming W4A16-HGPTQ + per-layer W2 routed-expert bank
warp-quant quantize --model 35b-ornith \
  --output-dir ./out/ornith-35b-w4 \
  --save-rco-bank-dir ./out/ornith-35b-w2-bank

# 2. Gumbel-Softmax STE RCO search (25% W4 routed budget) + E2E Norm/GDN KL QAT + assembly
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
    budgets=[0.25],
    assemble_names=["25pct_rco"],
)'

# 3. Teacher-forced fidelity probe
warp-quant probe --model 35b-ornith \
  --quant ./out/rco-35b-ornith/checkpoints/25pct_rco \
  --eval-mtp \
  --eval-vision data/vision-probe-v2 \
  --eval-vision-guard data/vision-guard-v2
```

Receipts: [`probe_results.json`](probe_results.json) · [`rco_manifest.json`](rco_manifest.json) · [`rot_receipt.json`](rot_receipt.json) · [`pareto_summary.json`](pareto_summary.json)
