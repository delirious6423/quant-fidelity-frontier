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
- mixed-precision
- moe
- qwen4_exp
---

# Qwen3.8-Flash-Next — MXFP4 Experts, Everything Else BF16 (AutoRound, model-free)

Quantization of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(bf16, 180 B params total / ~6.7 B activated) with **Intel AutoRound 0.16.0** in `--model_free` mode
(RTN, no calibration data, no model instantiation). This is the most conservative variant in our
MXFP4-Mixed family: **only the routed experts are quantized (MXFP4 W4A4); every other module —
including all attention projections — stays bit-identical to the official checkpoint.**

**Artifact size: 182.6 GB** (bf16 source: 360 GB). The compression is capped at ~2× by the 102 GB
PLE n-gram embedding table, which has no MX-format representation and is kept bf16 by design.

## 1. Quantization scheme

| Module | Format | Notes |
|---|---|---|
| Routed experts `mlp.experts` (48 layers, 120.8 B params) | **MXFP4** W4A4, group 32, UE8M0 scales, dynamic activations | 73,728 per-expert Linear targets; `weight_packed` + `weight_scale` |
| Full-attention `self_attn.{q,k,v,o}_proj` (12 layers) | **BF16** | same choice as NVIDIA's NVFP4 checkpoint |
| GDN `linear_attn.*` (36 layers, incl. `in_proj_qkv/z`, `out_proj`) | **BF16** | |
| `self_attn.indexer.index_qk_proj` (12 layers) | **BF16** | |
| `mlp.shared_expert.{gate,up,down}_proj` (48×3) | **BF16** | |
| PLE n-gram embedding table (51.2 B params) | **BF16** | lookup table, not a Linear; 102 GB of the artifact |
| `mlp.gate` (router), `mlp.shared_expert_gate` | BF16 | |
| `linear_attn.in_proj_a/b`, `hyper_connection.*` | BF16 | N=48/4 blocks, fused by the engine |
| `mtp.*` (incl. its 512 experts) | BF16 | |
| `embed_tokens` / `lm_head` / `visual.*` | BF16 | |

KV cache: **not quantized**. Attention activations: **not quantized**.

Config structure: **single `config_group`** (`targets=["Linear"]`, W4A4 g32, dynamic activations) with
top-level `format: "mxfp4-pack-quantized"` — since every non-expert Linear is in `ignore`, the catch-all
`Linear` target resolves to exactly the experts. `ignore` = 2,607 entries. The checkpoint contains
**zero 8-bit weights** (150,552 tensors: 2,937 bf16-class + 147,456 uint8 packed/scale + 3 int64).

## 2. Reproduce the quantization

```bash
# AutoRound 0.16.0 (model-free; CPU-only is fine)
auto_round \
  --model_name Qwen/Qwen3.8-Flash-Next \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers "visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,ple,mtp,hyper_connection,indexer,shared_expert,self_attn,linear_attn" \
  --layer_config "{mlp.experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor \
  --output_dir ./Qwen3.8-Flash-Next-MXFP4-Mixed-BF16Attn
```

(The `MXFP8` scheme name only seeds the default group; with everything except experts ignored,
the effective output is the single MXFP4 group above.)

## 3. Inference

- **vLLM ≥ 0.29.0** (first release with `qwen4_exp` support). Evaluated on `0.29.1rc1.dev528` nightly;
  that build's `EngramConfig(cpu_offload=True)` is what makes TP=1 fit.
  **No patches required** — this is a vanilla compressed-tensors artifact.
- **TP=1 recommended.** On current vLLM the MXFP4 W4A4 MoE path (`CutlassExpertsMxfp4`) has an
  unresolved scale-padding misalignment at TP>1 for this model's expert shapes — an upstream
  limitation, not a checkpoint defect (the INC reference checkpoints fail the same way).
- The 102 GB PLE n-gram table dominates memory; KV cache `auto` (bf16) or `fp8` (on QSA builds that
  support it) both work. The QSA indexer's `indexer_kv_dtype="fp8"` runtime switch is independent
  of this checkpoint.
- No chat-template changes needed: ship the stock `chat_template.jinja`.

## 4. Evaluation

Harness: `lm-eval 0.4.13` + vLLM `0.29.1rc1.dev528`, TP=1, bf16 KV cache, `seed=42`,
`max_model_len=8192`, `max_num_seqs=64`, `gpu_memory_utilization=0.85`, `language_model_only=true`,
`reasoning_parser=qwen3`, `enable_thinking=false`. GSM8K ran with chat template + few-shot multiturn.

| Model | GSM8K (strict / flexible) | MMLU | PIQA (acc / acc_norm) | HellaSwag (acc / acc_norm) |
|---|---|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` (bf16 baseline) | 0.9674 / — | 0.8652 | 0.8194 / — | 0.6927 / — |
| **This checkpoint (attention = BF16)** | **0.9644 / 0.9636** | **0.8641** | **0.8248 / 0.8308** | **0.6856 / 0.8726** |


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
132.7 GB) makes the same attention choice (**all attention projections bf16**) and also keeps
`shared_expert`/`indexer`/router/hyper-connection/embeddings bf16. The remaining differences:

| | This checkpoint | NVIDIA NVFP4 |
|---|---|---|
| Routed experts | **MXFP4** W4A4, group 32, UE8M0 scales, **dynamic** activations, no calibration | **NVFP4** W4A4, group 16, FP8 micro-scales + FP32 global scales, **statically calibrated** activations |
| PLE n-gram table | bf16 (102 GB) | **FP8** (~51 GB) |
| MTP experts | bf16 | **FP8** (group 128, weight-only) |
| Size | **182.6 GB** | **132.7 GB** |

The size gap is dominated by the PLE table; a follow-up variant of this checkpoint with the table
converted to MXFP8 lands at 133.0 GB (essentially NVIDIA's size) — see the sibling
`-PLEMXFP8` artifact.

## 6. Notes and caveats

- Validated platform: B300 (sm103), TP=1, bf16 KV, context ≤ 8k. For long-context serving use the
  stock chat template and budget thinking tokens into the generation length.
- MTP is entirely bf16: enable speculative decoding only after separate validation.
- Vision tower is included (bf16) but was not exercised in the text-only evaluation above.

## 7. Citation

Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) —
Qwen Community License 1.0. Quantization toolkit: [Intel AutoRound](https://github.com/intel/auto-round).
