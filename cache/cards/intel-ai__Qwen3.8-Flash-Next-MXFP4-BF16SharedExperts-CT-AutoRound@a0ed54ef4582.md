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
---

# Qwen3.8-Flash-Next — MXFP4/MXFP8 Mixed Checkpoint (AutoRound, model-free)

Mixed-precision quantization of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(bf16, 180 B params total / ~6.7 B activated) to the **compressed-tensors `mixed-precision`** format,
produced with **Intel AutoRound 0.16.0** in `--model_free` mode (RTN, no calibration data, no model
instantiation). Weight values are re-encoded only where stated below; every module not listed as
quantized is kept bit-identical to the official checkpoint.

> Naming note: the local build directory carried an `-FP8Attn` suffix from an earlier experiment —
> **this checkpoint contains no FP8 attention quantization** (no `q_scale`, no `kv_cache_scheme`).
> FP8 here refers only to the MXFP8 weight format of the non-expert Linear layers.

## 1. Quantization scheme

| Module | Format | Notes |
|---|---|---|
| Routed experts `mlp.experts` (48 layers, 120.8 B params) | **MXFP4** W4A4, group 32, UE8M0 scales, dynamic activations | 73,728 per-expert Linear targets; stored as `weight_packed` + `weight_scale` |
| Full-attention `self_attn.{q,k,v,o}_proj` (12 layers) | **MXFP8** W8A8, group 32, UE8M0 scales, dynamic activations | |
| GDN `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (36 layers) | **MXFP8** W8A8, group 32 | |
| `mlp.shared_expert.{gate,up,down}_proj` (48×3) | **BF16** | kept in full precision |
| `self_attn.indexer.index_qk_proj` (12 layers) | **BF16** | top-k selector of the sparse attention |
| PLE n-gram embedding table (51.2 B params) | **BF16** | lookup table, not a Linear; 102 GB of the artifact |
| `mlp.gate` (router), `mlp.shared_expert_gate` | BF16 | routing/gating |
| `linear_attn.in_proj_a/b`, `hyper_connection.*` (incl. `block_inject_weight`) | BF16 | N=48/4 blocks, fused by the engine; cannot take MX kernels |
| `mtp.*` (incl. its 512 experts) | BF16 | speculative-decoding module |
| `embed_tokens` / `lm_head` / `visual.*` | BF16 | |

KV cache: **not quantized** (no `kv_cache_scheme`). Attention activations: **not quantized**.

**Artifact size: 180.0 GB** (bf16 source: 360 GB). Logical parameter count is unchanged (180 B);
compression is capped at ~2× by the 102 GB PLE n-gram table, which has no MX-format representation.

## 2. Reproduce the quantization

```bash
# AutoRound 0.16.0 (e.g. conda env with `pip install auto-round==0.16.0`), CPU-only is fine
auto_round \
  --model_name Qwen/Qwen3.8-Flash-Next \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers "visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,ple,mtp,hyper_connection,indexer,shared_expert" \
  --layer_config "{mlp.experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor \
  --output_dir ./Qwen3.8-Flash-Next-MXFP4-Mixed
```

Sanity fingerprints of this artifact: `ignore` = 2,451 entries; `config_groups.group_0` =
`mxfp4-pack-quantized` with 73,729 targets; `group_1` = `mxfp8-quantized` with `targets=["Linear"]`;
150,708 tensors total (156 `float8_e4m3fn` weights, 147,612 `uint8` packed/scale tensors).

## 3. Inference

- **vLLM ≥ 0.29.0** (first release with `qwen4_exp` support). Evaluated on
  `0.29.1rc1.dev528` (nightly); that build's `EngramConfig(cpu_offload=True)` is what makes TP=1 fit.
- **TP=1 recommended.** On current vLLM the MXFP4 W4A4 MoE path (`CutlassExpertsMxfp4`) has an
  unresolved scale-padding misalignment at TP>1 for this model's expert shapes — outputs degenerate.
  This is an upstream limitation, not a checkpoint defect (same failure occurs with the INC reference
  checkpoints). A 180 GB checkpoint with the PLE table offloaded fits on a single 275 GB B300.
- The 102 GB PLE n-gram table is BF16 by design and dominates memory; KV cache can be `auto`
  (bf16) or `fp8` on QSA builds that support it — both work with this artifact.
- The QSA indexer (top-k selector) supports `indexer_kv_dtype="fp8"` as a pure runtime switch
  (fp8×fp8 scoring); independent of this checkpoint and of KV-cache dtype.
- No chat template changes needed: ship the stock `chat_template.jinja`; for plain-text harness runs
  do not pass `--apply_chat_template`.

## 4. Evaluation

Harness: `lm-eval 0.4.13` + vLLM `0.29.1rc1.dev528`, TP=1, bf16 KV cache, `seed=42`,
`max_model_len=8192`, `max_num_seqs=64`, `gpu_memory_utilization=0.85`, `language_model_only=true`,
`reasoning_parser=qwen3`, `enable_thinking=false`. GSM8K ran with chat template + few-shot multiturn.

| Model | GSM8K (strict / flexible) | MMLU | PIQA (acc / acc_norm) | HellaSwag (acc / acc_norm) |
|---|---|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` (bf16 baseline) | 0.9674 / — | 0.8652 | 0.8194 / — | 0.6927 / — |
| **This checkpoint** | **0.9666 / 0.9659** | **0.8643** | **0.8145 / 0.8275** | **0.6832 / 0.8694** |
| Sibling variant (shared_expert = MXFP8) | 0.9621 / 0.9621 | 0.8626 | 0.8226 / 0.8270 | 0.6832 / 0.8707 |

All deltas vs the sibling variant are ≤ 0.6 σ (independent-run standard errors), i.e. keeping
`shared_expert` + `indexer` in BF16 costs no measurable accuracy and recovers most of the GSM8K gap
to the bf16 baseline.

### Reproduce

```bash
lm_eval --model vllm \
  --model_args '{"pretrained": "<this-checkpoint>", "tensor_parallel_size": 1, "max_model_len": 8192, "max_num_seqs": 64, "max_num_batched_tokens": 16384, "gpu_memory_utilization": 0.85, "dtype": "bfloat16", "trust_remote_code": true, "add_bos_token": true, "enable_prefix_caching": false, "max_gen_toks": 2048, "language_model_only": true, "reasoning_parser": "qwen3", "enable_thinking": false, "safetensors_load_strategy": "prefetch"}' \
  --tasks gsm8k --batch_size 32 --seed 42 --apply_chat_template --fewshot_as_multiturn

lm_eval --model vllm --model_args '<same JSON as above>' \
  --tasks piqa,mmlu,hellaswag --batch_size 32 --seed 42
```

## 5. Comparison with NVIDIA's NVFP4 checkpoint

[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (ModelOpt,
132.7 GB) makes the same core choice (experts → 4-bit, `shared_expert`/`indexer`/router → BF16) but
differs on three axes: it keeps **all attention projections in BF16** (this checkpoint uses MXFP8),
it FP8-quantizes the **PLE n-gram table** (this checkpoint keeps it BF16 — the main size difference,
132.7 vs 180.0 GB), and its experts use **NVFP4** (group 16, FP8 micro-scales, statically calibrated
activations) vs this checkpoint's **MXFP4** (group 32, UE8M0 scales, dynamic activations). Its MTP
experts are FP8; this checkpoint keeps MTP fully BF16.

## 6. Notes and caveats

- The quantization-side tool (AutoRound) and runtime kernel (vLLM `CutlassExpertsMxfp4`) use the
  OCP MXFP4/MXFP8 formats with UE8M0 scales; SM100+ (Blackwell) is the validated platform.
- MTP is entirely BF16: enable speculative decoding only after validating it separately.
- Vision tower is included (BF16) but was not exercised in the text-only evaluation above.
- Long-context behaviour (RULER-style, 128k) follows the bf16 baseline's requirements: serve with
  chat template applied and `enable_thinking=false` budgeted into generation length.

## 7. Citation

Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) —
Qwen Community License 1.0. Quantization toolkit: [Intel AutoRound](https://github.com/intel/auto-round).
