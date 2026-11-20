---
base_model: Qwen/Qwen3.8-Flash-Next
library_name: transformers
pipeline_tag: text-generation
tags: [autoround, int3, mixed-precision, moe, vllm, fp8]
---

# Qwen3.8-Flash-Next-3.55BPW-MTP

DONT USE THIS MODEL, IT WAS TRAINED ON A TRAINING CORPUS TOKENIZED TO LING, BEHAVIOR IS UNPREDICTABLE


Mixed-precision **AutoRound** quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
for vLLM. Text-only (vision and MTP removed). Non-PLE weights: **51.99 GiB**, **3.55 bits per
weight** on average (packed weights, scales, zero points and BF16 parts). The 51.2 B-parameter per-layer embedding (PLE)
is Qwen's own FP8 table with its global scale (~47.7 GiB, offloaded to host memory by vLLM).

## Precision layout

| Component | Format |
|---|---|
| Routed experts, all 46 layers | W3A16, group 128, symmetric |
| GDN linear attention, QSA self-attention projections | W8A16, group 128, symmetric |
| Indexer, decay/gate controls, hyper-connections, shared experts, routers, embeddings, LM head, norms | BF16 |
| Per-layer embedding (PLE) | FP8 E4M3 + global scale (official Qwen tensors) |

AutoRound 0.16.1 (SignRound), 400 iterations/block, 256 x 8192-token
calibration samples (chat/code/reasoning/long-context/wikitext prose, production chat template; best-MSE=True). Format `auto_round:auto_gptq`.

## Fidelity vs. BF16

Teacher-forced **full-vocabulary** KL(P_bf16 ‖ Q_quant) on a held-out set (24 × 1,024 tokens; sources and splits disjoint
from calibration). Both models run **without the PLE injection** (the BF16 PLE table does not fit the quantization host),
so this measures the quantization of the transformer blocks only:

| Domain | Positions | Mean KL | Median KL | P99 KL | Top-1 agreement | PPL bf16 → quant |
|---|---:|---:|---:|---:|---:|---:|
| **all** | 24552 | **0.15897** | 0.01133 | 2.4166 | 90.26% | 3.524 → 3.537 |
| chat | 10230 | 0.17180 | 0.03464 | 2.0036 | 86.70% | 7.864 → 7.702 |
| code | 5115 | 0.13346 | 0.00040 | 2.7081 | 94.68% | 1.806 → 1.836 |
| math | 5115 | 0.10857 | 0.00353 | 2.1411 | 93.59% | 2.302 → 2.307 |
| prose | 4092 | 0.22178 | 0.00655 | 2.9360 | 89.49% | 1.861 → 1.957 |

P99.9 KL 5.9165, max 17.358.

### Progression Across `3.48BPW` Quantizations

| Version | Run | Config | Mean KL | P99 KL | P99.9 KL | Top-1 | Chat Top-1 | Code Top-1 | Math Top-1 | Prose Top-1 | PPL bf16 → quant |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 (Baseline) | `20261114-213016` | 104 samples, 250 iters, final-iter | 0.11572 | 1.5207 | 4.5617 | 88.78% | 85.61% | 94.43% | 92.98% | 84.38% | 5.431 → 5.677 |
| v2 | `run-20261116-132458` | 236 samples, 400 iters, best-MSE=True | 0.10026 | 1.4335 | 4.2587 | 90.47% | **89.32%** | 94.41% | 93.76% | 84.31% | 5.481 → 5.608 |
| **v3 (Final)** | `run-20261118-162556` | 256 samples, 400 iters, best-MSE=True | **0.15897** | **2.4166** | **5.9165** | **90.26%** | 86.70% | **94.68%** | **93.59%** | **89.49%** | **3.524 → 3.537** |
| **Δ (v1 → v3)** | — | — | **-17.2%** | **-11.7%** | **-7.6%** | **+1.75 pp** | **+3.15 pp** | **+0.55 pp** | **+1.08 pp** | **+0.64 pp** | — |

## Serving (vLLM)

The checkpoint ships the official FP8 per-layer-embedding (PLE) shards with their global scale
(`ple-*.safetensors`) and `ple_embedding_dtype: float8_e4m3fn` in `config.json`.

```bash
python vllm-patch.py              # run with the python of your vLLM environment (see below)
vllm serve Beling86/Qwen3.8-Flash-Next-3.55BPW-MTP --trust-remote-code --dtype bfloat16 --engram-config '{"cpu_offload": true, "use_thp": true}'
```

`vllm-patch.py` (in this repo; idempotent, keeps `.orig` backups, `--check` / `--revert`) fixes a vLLM bug we hit
while serving a sibling checkpoint with per-expert mixed bit-widths: vLLM's INC parser reads the per-layer bits of a
fused MoE layer only if the class name contains `fusedmoe`, but the class is now `RoutedExperts`, so every MoE layer
falls back to the global `bits: 4` and the 2/3-bit experts are allocated with 4-bit containers (CUDA OOM while the model
is constructed, before any weight is read). This checkpoint has the same layout (W2/W3 experts under a global
`bits: 4`), so expect the same failure on stock vLLM without the patch. The Ling-specific KDA patch in that script
does not apply to this model and reports "not applicable".

2/3-bit experts are meant to run on the Humming kernels, 8-bit projections on Marlin. If you serve this model, please
report what happens.

## Validation gates passed

Coverage, packed layout vs. config, zero points vs. `sym`, fused-group consistency, size ≤ 52.0 GiB
(non-PLE weights), plan conformance, strict reload (no missing keys) for the KL evaluation, PLE shards and global scale.
