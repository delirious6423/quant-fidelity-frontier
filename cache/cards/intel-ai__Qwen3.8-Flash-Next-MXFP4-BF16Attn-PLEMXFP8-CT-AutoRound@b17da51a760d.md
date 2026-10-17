---
base_model:
- Qwen/Qwen3.8-Flash-Next
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: auto-round
tags:
- auto-round
- compressed-tensors
- mxfp4
- mxfp8
- mixed-precision
- moe
- qwen4_exp
- ple
---

# Qwen3.8-Flash-Next — MXFP4 Experts + MXFP8 PLE Table, Everything Else BF16 (AutoRound)

Quantization of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(bf16, 180 B params total / ~6.7 B activated). This is the closest analog of NVIDIA's NVFP4
checkpoint in our MXFP4-Mixed family: **routed experts → MXFP4, PLE n-gram table → MXFP8,
everything else (including all attention projections) bf16.**

**Artifact size: 133.0 GB** (bf16 source: 360 GB; NVIDIA's NVFP4 checkpoint: 132.7 GB).
Produced with **Intel AutoRound 0.16.0** (`--model_free`) for the weights, plus an offline
converter (`convert_ple_to_mxfp8.py`) for the PLE table (quantization fidelity: **SNR 53.6 dB**).

> **READ BEFORE USE — runtime patch required (PLE table only).** Current vLLM (≤ 0.29.1rc1.dev528,
> and `main` as of 2026-10-04) loads compressed-tensors checkpoints with the PLE table **always as
> unquantized**. Without the patch in `patches/`, this checkpoint's FP8 table bytes are cast to bf16
> **without their scales** and the model degenerates *silently*. See §3.

## 1. Quantization scheme

| Module | Format | Notes |
|---|---|---|
| Routed experts `mlp.experts` (48 layers, 120.8 B params) | **MXFP4** W4A4, group 32, UE8M0 scales, dynamic activations | `config_groups.group_0` (`mxfp4-pack-quantized`, catch-all `Linear` target) |
| **PLE n-gram embedding table** (51.2 B params, 128 shards) | **MXFP8 weights**: E4M3 + UE8M0 group-32 scales (`shard_N.weight` F8_E4M3 + `shard_N.weight_scale` U8) | `config_groups.group_1` (`mxfp8-quantized`, explicit module target); removed from `ignore` |
| Full-attention `self_attn.{q,k,v,o}_proj` (12 layers) | **BF16** | same choice as NVIDIA's NVFP4 checkpoint |
| GDN `linear_attn.*` (36 layers) | **BF16** | |
| `self_attn.indexer.index_qk_proj` / `mlp.shared_expert.*` | **BF16** | |
| `mlp.gate` (router), `mlp.shared_expert_gate` | BF16 | |
| `linear_attn.in_proj_a/b`, `hyper_connection.*` | BF16 | |
| `mtp.*` (incl. its 512 experts) | BF16 | |
| `embed_tokens` / `lm_head` / `visual.*` | BF16 | |

KV cache: **not quantized**. Attention activations: **not quantized**.

Sanity fingerprints: `ignore` = 2,479 entries; top-level `format: "mixed-precision"`; the only 8-bit
weights in the checkpoint are the 128 PLE shards (`float8_e4m3fn [2500012,160]` + `uint8 [2500012,5]`);
150,680 tensors.

## 2. Reproduce the quantization

```bash
# Step 1 — weights (AutoRound 0.16.0, model-free)
auto_round \
  --model_name Qwen/Qwen3.8-Flash-Next \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers "visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,ple,mtp,hyper_connection,indexer,shared_expert,self_attn,linear_attn" \
  --layer_config "{mlp.experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor \
  --output_dir ./Qwen3.8-Flash-Next-MXFP4-Mixed-BF16Attn

# Step 2 — PLE table to MXFP8 (offline converter; also upgrades the single-group
# config to two groups: mxfp4 experts + mxfp8 PLE, top-level mixed-precision)
python3 convert_ple_to_mxfp8.py \
  ./Qwen3.8-Flash-Next-MXFP4-Mixed-BF16Attn \
  ./Qwen3.8-Flash-Next-MXFP4-Mixed-BF16Attn-PLEMXFP8
```

## 3. Inference — required vLLM patch

**Base:** vLLM `0.29.1rc1.dev528` (any `qwen4_exp`-capable build ≥ 0.29.0 should patch cleanly, but only
dev528 was validated). The patch adds an `MXFP8 PLE` embedding method, a fused-dequantize lookup kernel,
compressed-tensors detection of the n-gram target, and fused loading of `shard_<i>.weight_scale`.

```bash
./patches/apply_qwen38fn_ple_mxfp8.sh          # apply | check | revert
```

Verified load line: `Qwen4Exp PLE embedding: using MXFP8 n-gram table from compressed-tensors checkpoint`.

- **TP=1 recommended** (upstream W4A4 CUTLASS MoE scale-padding bug at TP>1; not a checkpoint defect).
- With `EngramConfig(cpu_offload=True)` the MXFP8 table is pinned on host (~48 GiB instead of ~96 GiB);
  a 133 GB checkpoint fits on a single 275 GB B300.
- KV cache `auto` (bf16) or `fp8` both work; the QSA indexer's `indexer_kv_dtype="fp8"` runtime switch
  is independent of this checkpoint.

## 4. Evaluation

Harness: `lm-eval 0.4.13` + vLLM `0.29.1rc1.dev528` **with the §3 patch applied**, TP=1, bf16 KV cache,
`seed=42`, `max_model_len=8192`, `max_num_seqs=64`, `gpu_memory_utilization=0.85`,
`language_model_only=true`, `reasoning_parser=qwen3`, `enable_thinking=false`.

| Model | GSM8K (strict / flexible) | MMLU | PIQA (acc / acc_norm) | HellaSwag (acc / acc_norm) |
|---|---|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` (bf16 baseline) | 0.9674 / — | 0.8652 | 0.8194 / — | 0.6927 / — |
| Sibling `-BF16Attn` (PLE table bf16) | 0.9644 / 0.9636 | 0.8641 | 0.8248 / 0.8308 | 0.6856 / 0.8726 |
| **This checkpoint (PLE table MXFP8)** | **0.9636 / 0.9636** | **0.8650** | **0.8150 / 0.8319** | **0.6834 / 0.8726** |

All deltas vs the bf16-table sibling are within ~1 σ and mixed in sign — **the MXFP8 PLE table costs no
measurable accuracy on these benchmarks** while saving 49.6 GB.

### Reproduce

```bash
lm_eval --model vllm \
  --model_args '{"pretrained": "<this-checkpoint>", "tensor_parallel_size": 1, "max_model_len": 8192, "max_num_seqs": 64, "max_num_batched_tokens": 16384, "gpu_memory_utilization": 0.85, "dtype": "bfloat16", "trust_remote_code": true, "add_bos_token": true, "enable_prefix_caching": false, "max_gen_toks": 2048, "language_model_only": true, "reasoning_parser": "qwen3", "enable_thinking": false, "safetensors_load_strategy": "prefetch"}' \
  --tasks gsm8k --batch_size 32 --seed 42 --apply_chat_template --fewshot_as_multiturn

lm_eval --model vllm --model_args '<same JSON as above>' \
  --tasks piqa,mmlu,hellaswag --batch_size 32 --seed 42
```

## 5. Comparison with NVIDIA's NVFP4 checkpoint

Same coverage except two items: MTP experts (bf16 here, FP8 in NVIDIA's) and the experts' 4-bit format
(MXFP4 group-32 UE8M0 dynamic-activation here, NVFP4 group-16 with FP8 micro-scales + FP32 global
scales + statically calibrated activations in NVIDIA's). The PLE table is FP8-quantized by both —
NVIDIA uses plain FP8 (single per-tensor FP32 scale); this checkpoint uses MXFP8 (per-32 UE8M0 group
scales, finer granularity). Size: **133.0 GB here vs 132.7 GB**.

## 6. Notes and caveats

- **The §3 patch is mandatory.** Unpatched vLLM does not error; it silently misreads the table.
- Validated platform: B300 (sm103), TP=1, bf16 KV, context ≤ 8k. Long-context (RULER-style 128k)
  behaviour of the MXFP8 PLE table has **not** been separately validated — it is the model's
  long-range n-gram memory, so validate there before production long-context use.
- MTP is entirely bf16: enable speculative decoding only after separate validation.

## 7. Citation

Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) —
Qwen Community License 1.0. Quantization toolkit: [Intel AutoRound](https://github.com/intel/auto-round).
