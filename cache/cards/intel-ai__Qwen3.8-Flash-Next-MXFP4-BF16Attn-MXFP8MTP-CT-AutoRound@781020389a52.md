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

# Qwen3.8-Flash-Next — MXFP4 Experts + MXFP8 MTP Experts, Everything Else BF16 (AutoRound)

Quantization of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(bf16, 180 B params total / ~6.7 B activated) with **Intel AutoRound 0.16.0** in `--model_free` mode
(RTN, no calibration). This is the "NVIDIA-coverage-aligned" variant of our MXFP4-Mixed family:
**main-tower routed experts → MXFP4, MTP routed experts → MXFP8, everything else — including all
attention projections and the PLE n-gram table — bf16.**

**Artifact size: 180.1 GB** (bf16 source: 360 GB).

## 1. Quantization scheme

| Module | Format | Notes |
|---|---|---|
| Routed experts `mlp.experts` (48 layers, 120.8 B params) | **MXFP4** W4A4, group 32, UE8M0 scales, dynamic activations | `config_groups.group_0` (`mxfp4-pack-quantized`, 73,729 targets) |
| **MTP routed experts** `mtp.layers.0.mlp.experts` (512, 2.5 B params) | **MXFP8** W8A8, group 32, UE8M0 scales, dynamic activations | `config_groups.group_1` (`mxfp8-quantized`); 1,536 per-expert Linears |
| Full-attention `self_attn.{q,k,v,o}_proj` (12 layers) | **BF16** | same choice as NVIDIA's NVFP4 checkpoint |
| GDN `linear_attn.*` (36 layers) | **BF16** | |
| `self_attn.indexer.index_qk_proj` / `mlp.shared_expert.*` / MTP 内同名件 | **BF16** | |
| PLE n-gram embedding table (51.2 B params) | **BF16** | 102 GB of the artifact |
| `mlp.gate` / `shared_expert_gate` / `mtp.fc_*` / `hyper_connection.*` / `in_proj_a/b` | BF16 | |
| `embed_tokens` / `lm_head` / `visual.*` / MTP 其余（norms/fc 等） | BF16 | |

KV cache: **not quantized**. Attention activations: **not quantized**.

Sanity fingerprints: `ignore` = 1,071 entries; top-level `format: "mixed-precision"`; 152,088 tensors;
the only 8-bit weights are the 1,536 MTP-expert tensors (`float8_e4m3fn` + `uint8` group-32 scales).

## 2. Reproduce the quantization

```bash
# AutoRound 0.16.0 (model-free). Note the layer_config ORDER matters: the MTP entry must come
# first — the matcher returns on first hit, and "mlp.experts" would otherwise also match the
# MTP experts (they would land in MXFP4 instead of MXFP8).
auto_round \
  --model_name Qwen/Qwen3.8-Flash-Next \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers "visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,ple,hyper_connection,indexer,shared_expert,self_attn,linear_attn,mtp.fc" \
  --layer_config '{"mtp.layers.0.mlp.experts":{"bits":8,"data_type":"mx_fp"},"mlp.experts":{"bits":4,"data_type":"mx_fp"}}' \
  --format llm_compressor \
  --output_dir ./Qwen3.8-Flash-Next-MXFP4-Mixed-BF16Attn-MTP8
```

## 3. Inference

- **vLLM ≥ 0.29.0** (first release with `qwen4_exp`). Evaluated on `0.29.1rc1.dev528` nightly.
  **No patches required** — vanilla compressed-tensors artifact.
- **TP=1 recommended** (upstream W4A4 CUTLASS MoE scale-padding bug at TP>1; also the MXFP8 MoE
  scale-swizzle padding issue at TP>1 — both are upstream limitations, not checkpoint defects).
- **MTP / speculative decoding**: off by default (the `mtp.*` weights are filtered out of the main
  model load entirely). Verified working when enabled with `speculative_config={"method":"mtp",
  "num_speculative_tokens":1}` on vLLM dev528: the `Qwen4ExpMTP` draft loads these MXFP8 experts,
  the speculator captures CUDA graphs, and generation stays sane (GSM8K limit-100: 0.98).
  Full-scale acceptance-rate/throughput validation was **not** done — treat MTP speedups as
  unverified until measured.
- KV cache `auto` (bf16) or `fp8` both work on QSA builds that support fp8.

## 4. Evaluation

Harness: `lm-eval 0.4.13` + vLLM `0.29.1rc1.dev528`, TP=1, bf16 KV cache, `seed=42`,
`max_model_len=8192`, `max_num_seqs=64`, `gpu_memory_utilization=0.85`, `language_model_only=true`,
`reasoning_parser=qwen3`, `enable_thinking=false`. GSM8K ran with chat template + few-shot multiturn.

| Model | GSM8K (strict / flexible) | MMLU | PIQA (acc / acc_norm) | HellaSwag (acc / acc_norm) |
|---|---|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` (bf16 baseline) | 0.9674 / — | 0.8652 | 0.8194 / — | 0.6927 / — |
| Sibling `-BF16Attn-1007` (MTP fully bf16) | 0.9644 / 0.9636 | 0.8641 | 0.8248 / 0.8308 | 0.6856 / 0.8726 |
| **This checkpoint** | **0.9651 / 0.9644** | **0.8655** | **0.8243 / 0.8330** | **0.6845 / 0.8708** |

All deltas vs the bf16-MTP sibling are ≤ ~0.3 σ — **quantizing the MTP experts to MXFP8 has no
measurable effect on the main path** (unsurprising: without speculative decoding those weights are
not even loaded).

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
132.7 GB). Coverage is now identical except the PLE table (bf16 here, FP8 in NVIDIA's) and the 4-bit
expert format (MXFP4 g32/UE8M0/dynamic-activation, no calibration here; NVFP4 g16 with FP8
micro-scales + FP32 global scales + statically calibrated activations in NVIDIA's). MTP experts are
FP8 in both — NVIDIA uses block-wise FP8 (group 128, weight-only, `weight_scale_inv`); this checkpoint
uses MXFP8 (group 32 UE8M0, W8A8 dynamic). See the `-PLEMXFP8` sibling for the PLE-table-MXFP8 version
(133.0→131.5 GB).

## 6. Notes and caveats

- Validated platform: B300 (sm103), TP=1, bf16 KV, context ≤ 8k. Long-context (128k) behaviour was
  not separately validated here.
- The PLE table being bf16 keeps this artifact patch-free on stock vLLM.

## 7. Citation

Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) —
Qwen Community License 1.0. Quantization toolkit: [Intel AutoRound](https://github.com/intel/auto-round).
