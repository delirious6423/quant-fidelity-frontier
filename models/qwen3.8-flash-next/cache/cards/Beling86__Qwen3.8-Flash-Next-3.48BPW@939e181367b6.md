---
base_model: Qwen/Qwen3.8-Flash-Next
library_name: transformers
pipeline_tag: text-generation
tags: [autoround, int3, mixed-precision, moe, vllm, fp8]
---

# Qwen3.8-Flash-Next-3.48BPW

Mixed-precision **AutoRound** quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
for vLLM. Text-only (vision and MTP removed). Non-PLE weights: **51.07 GiB**, **3.48 bits per
weight** on average (packed weights, scales, zero points and BF16 parts). The 51.2 B-parameter per-layer embedding (PLE)
is Qwen's own FP8 table with its global scale (~47.7 GiB, offloaded to host memory by vLLM).

## Precision layout

| Component | Format |
|---|---|
| Routed experts, all 48 layers | W3A16, group 128, symmetric |
| GDN linear attention, QSA self-attention projections | W8A16, group 128, symmetric |
| Indexer, decay/gate controls, hyper-connections, shared experts, routers, embeddings, LM head, norms | BF16 |
| Per-layer embedding (PLE) | FP8 E4M3 + global scale (official Qwen tensors) |

AutoRound 0.16.1 (SignRound), 250 iterations/block, 104 x 2048-token
calibration samples (chat/code/reasoning/long-context, production chat template). Format `auto_round:auto_gptq`.

## Fidelity vs. BF16

Teacher-forced **full-vocabulary** KL(P_bf16 ‖ Q_quant) on a held-out set (24 × 1,024 tokens; sources and splits disjoint
from calibration). Both models run **without the PLE injection** (the BF16 PLE table does not fit the quantization host),
so this measures the quantization of the transformer blocks only:

| Domain | Positions | Mean KL | Median KL | P99 KL | Top-1 agreement | PPL bf16 → quant |
|---|---:|---:|---:|---:|---:|---:|
| **all** | 24552 | **0.11572** | 0.01821 | 1.5207 | 88.78% | 5.431 → 5.677 |
| chat | 10230 | 0.13859 | 0.03363 | 1.6373 | 85.61% | 10.235 → 11.257 |
| code | 5115 | 0.08184 | 0.00106 | 1.3862 | 94.43% | 2.773 → 2.773 |
| math | 5115 | 0.07272 | 0.00463 | 1.2121 | 92.98% | 2.675 → 2.612 |
| prose | 4092 | 0.15467 | 0.05358 | 1.5946 | 84.38% | 6.255 → 6.622 |

P99.9 KL 4.5617, max 12.482. vLLM load/generation was **not** validated for this release (smoke test skipped).

## Serving (vLLM)

**Status: not validated on vLLM for this release** (the quantization host has a 64 GB GPU, too small for the
~50 GiB of weights plus vLLM's startup; only the KL evaluation and the structural gates below were run).
The checkpoint ships the official FP8 per-layer-embedding (PLE) shards with their global scale
(`ple-*.safetensors`) and `ple_embedding_dtype: float8_e4m3fn` in `config.json`, so vLLM's FP8 PLE loader should be selected (from reading vLLM's source; not tested).

```bash
python vllm-patch.py              # run with the python of your vLLM environment (see below)
vllm serve Beling86/Qwen3.8-Flash-Next-3.48BPW --trust-remote-code --dtype bfloat16 --engram-config '{"cpu_offload": true, "use_thp": true}'
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
