---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - qwen3.8
  - image-text-to-text
  - fp8
  - compressed-tensors
  - llm-compressor
  - gdn
  - quantized
---
# Qwen3.8-27B FP8-BLOCK

Data-free (RTN) compressed-tensors FP8_BLOCK quantization of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) @
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, with **all Gated DeltaNet
linear-attention projections kept in BF16**. Built 2026-08-17 with
llmcompressor `model_free_ptq`, scheme `FP8_BLOCK` (static 128x128 E4M3
weight blocks, dynamic group-128 FP8 input activations).

The original Qwen/Qwen3.8-27B model card is preserved unchanged in
[`README-upstream.md`](README-upstream.md) and is the authority on the base
model's capabilities, usage, and license.

## Recipe

| role | representation |
| --- | --- |
| `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (48 GDN layers) + state path (`in_proj_a/b/ba`, `A_log`, `conv1d`, `dt_bias`, `norm`) | **BF16** (all GDN linear-attention modules unquantized) |
| `self_attn.{q,k,v,o}_proj` (16 full-attn layers + MTP layer) | FP8 E4M3 128x128 blocks |
| `mlp.{gate,up,down}_proj` (64 layers + MTP layer) | FP8 E4M3 128x128 blocks |
| `lm_head`, `embed_tokens`, vision tower, `mlp.gate`/`shared_expert_gate`, `mtp.fc`, all norms | BF16 |

Same policy as the official `Qwen/Qwen3.8-27B-FP8` except the official export
quantizes the GDN `in_proj_qkv/z/out_proj` projections to FP8; this checkpoint
does not. Weights 33.9 GiB (official FP8: 28.75 GiB). Build ~1.6 min on one
RTX PRO 6000 Blackwell. The per-tensor composition is in `quant-manifest.json`.

## Measured quality — distribution fidelity (v5 shard-0 KLD replay)

KL(BF16 base ‖ candidate) over 512 contexts / 1,048,064 positions, full
248,320 vocab, **one shared BF16 LM head**. Every checkpoint below was
captured and replayed on the same hardware and protocol (RTX PRO 6000
Blackwell, vllm 0.27.1, bf16 KV, eager, `max_num_seqs=1`, one 2,048-token
prefill chunk per context). Absolute values are suite-specific; orderings and
differences between rows are the reliable signal (resolvable to ~1e-3).

| checkpoint (all captured/replayed locally) | weights | mean KLD | median KLD | p99 KLD | top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.8-27B` (BF16 base @ `1d4bf0f2`) | 51.75 GiB | 0 (ref) | 0 (ref) | 0 (ref) | 100% |
| **`Qwen3.8-27B-FP8-BLOCK` (this model)** | **33.90 GiB** | **0.004110** | **0.001737** | **0.041333** | **97.18%** |
| `Qwen/Qwen3.8-27B-FP8` (official @ `017b9c7a`) | 28.75 GiB | 0.005189 | 0.002024 | 0.052687 | 96.93% |
| `Qwen3.8-27B-FP8-BLOCK-GDN8` (official-policy clone, ours) | 28.75 GiB | 0.005253 | 0.002033 | 0.053828 | 96.86% |
| `unsloth/Qwen3.8-27B-NVFP4` (@ `7d6f8d4d`) | 21.81 GiB | 0.030155 | 0.009493 | 0.335824 | 93.14% |
| `RadixArk/Qwen3.8-27B-NVFP4` (@ `52d1adc5`) | 20.42 GiB | 0.041985 | 0.015005 | 0.465240 | 91.55% |

**Determinism:** the harness is deterministic. Two independent captures of
RadixArk agreed to Δmean-KLD = 3.3e-5, Δmedian = 1.5e-5, Δtop-1 = 3e-5 (all
well inside the 512-context bootstrap CI, ±2e-3) — single-run values are
reportable as-is. p99 is shown rather than max because max is one degenerate
token-position (RadixArk max 10.34 ≈ 22x its p99) and not reproducible;
p99 is the robust worst-case tail.

**Reading the table:** this checkpoint is the closest locally-measured
candidate to the BF16 base, ~0.001 lower mean KLD than the official FP8
export (the entire gain is the GDN-BF16 projection policy), and ~10x tighter
than the NVFP4 rows we measured. Both NVFP4 rows are measurably worse
(aggregate MLP NVFP4 error dominates on the 64-layer MoE); among the two,
`unsloth/Qwen3.8-27B-NVFP4` is notably better than `RadixArk` (mean 0.030 vs
0.042). Toolchain validated by reproducing the published official-FP8 row
(0.005189 vs published 0.005197) and the published unsloth row (0.030155 vs
0.030115) to <5e-5 in the same session.

## Serving performance — SGLang MTP (2026-08-17, RTX PRO 6000)

Matched methodology across all rows: 128 ShareGPT prompts, seed 42,
32768-token context, FP8 KV cache, flashinfer, MTP3 (SGLang EAGLE 3-step /
topk-1 / 4 draft tokens), language-only, thinking off. All 128 requests
completed in every row with 0 failures.

| metric | SGLang MTP | vLLM MTP |
| --- | ---: | ---: |
| c=1 prefill tok/s | **84.3** | 79.0 |
| c=1 decode tok/s | **91.4** | 85.7 |
| c=32 decode tok/s | **1207.4** | 1017.5 |
| c=1 TTFT p50 ms | 66.6 | 70.6 |
| c=1 TPOT p50 ms | 10.9 | 11.6 |

**As of 2026-08-17 this model with MTP runs measurably faster under SGLang
than vLLM** (+7% c=1 decode, +19% c=32 decode). vLLM MTP acceptance
59.8–61.3%; SGLang MTP acceptance isn't reported through the vLLM bench
client (same reporting gap as vLLM historical rows) — the decode speedup
above is the direct measured benefit.

**Memory / draft-copy status (checked 2026-08-17):** SGLang 0.5.17 EAGLE with
a same-checkpoint draft (`--speculative-draft-model-path <this model>`) still
loads a **full separate draft weight copy** (`is_draft_worker=True`, draft
weights counted by the KV configurator) — **not patched in this build**.
Target + draft ≈ 67.8 GB, leaving ~17 GB KV at mem-fraction 0.95 (≈ one
full-length 262,144 request). The DFlash/DSPark small-draft workers exist in
0.5.17 (the path that would fix this) but a matching small draft for Qwen3.8
is not yet staged here; multi-stream agentic stays on NVFP4 until then. vLLM
MTP shares target weights (no second copy), so it fits but is slower.

## Replicating the quantization

The checkpoint was built with data-free (RTN) model-free PTQ, no calibration
examples or architecture-specific Transformers code. The reproducible core
call with `llmcompressor` is:

```python
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="snapshot of Qwen/Qwen3.8-27B pinned to 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    save_directory="./Qwen3.8-27B-FP8-BLOCK",
    scheme="FP8_BLOCK",
    ignore=[
        # official export's GDN state path + norms/head/tower etc.
        "re:.*lm_head$",
        "re:.*embed_tokens$",
        "re:.*linear_attn.*",      # keep all GDN linear-attn (incl. proj) BF16
        "re:.*mlp\.gate$",
        "re:.*mlp\.shared_expert_gate$",
        "re:.*mtp\.fc$",
        "re:.*norm$",
        "re:.*visual.*",
        # plus the GDN state path: A_log, conv1d, dt_bias, in_proj_a/b/ba
    ],
    max_workers=1,
    device="cuda:0",
)
```

The `ignore` policy is derived from the official `Qwen/Qwen3.8-27B-FP8`
882-entry ignore list; the only structural difference from the official
export is that the GDN `in_proj_qkv/z/out_proj` projections are excluded
(kept BF16) rather than quantized. Saved scheme is compressed-tensors
`FP8_BLOCK`; per-tensor composition is in `quant-manifest.json`.

## Serving with SGLang MTP

On NVIDIA Blackwell / SM120 GPUs both engines need the hybrid-architecture
recipe (`--language-model-only`, `--mamba-ssm-dtype bfloat16` for the GDN
SSM state, `--kv-cache-dtype fp8_e4m3`, and the FlashInfer version-check
disclaimer; SGLang also `--max-mamba-cache-size` = max-running-requests ×
(5+4) for MTP so the GDN state pool does not clamp concurrency).

```bash
# SGLang MTP (fastest measured lane, EAGLE 3-step / topk-1 / 4 draft tokens)
python -m sglang.launch_server \
  --model-path ./Qwen3.8-27B-FP8-BLOCK \
  --trust-remote-code --tp-size 1 \
  --context-length 32768 --mem-fraction-static 0.90 \
  --max-running-requests 32 \
  --attention-backend flashinfer \
  --mamba-ssm-dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 \
  --language-only \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path ./Qwen3.8-27B-FP8-BLOCK \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

vLLM MTP uses the same target + `--speculative-config
'{"method":"mtp","num_speculative_tokens":3}'`.

## Runtime notes

- Loads as compressed-tensors in vLLM 0.27.1 and SGLang 0.5.17.
- SGLang EAGLE same-checkpoint draft loads a full draft copy (see above);
  vLLM MTP shares target weights.
- Not measured: multimodal generation quality, long-context quality, safety,
  task-level benchmark deltas vs BF16 (distribution fidelity only).
