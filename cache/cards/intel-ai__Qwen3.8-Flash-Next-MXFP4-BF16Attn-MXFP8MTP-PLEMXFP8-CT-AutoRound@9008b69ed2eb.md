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

# Qwen3.8-Flash-Next — MXFP4 Experts + MXFP8 MTP Experts + MXFP8 PLE Table (AutoRound)

The smallest member of our MXFP4-Mixed family for
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (bf16, 180 B params):
**main-tower routed experts → MXFP4, MTP routed experts → MXFP8, PLE n-gram embedding table → MXFP8,
everything else (all attention projections included) bf16.** It matches NVIDIA's NVFP4 checkpoint's
module coverage almost exactly, on the compressed-tensors path.

**Artifact size: 131.5 GB** (bf16 source: 360 GB; NVIDIA's NVFP4 checkpoint: 132.7 GB).

Produced with **Intel AutoRound 0.16.0** (`--model_free`) for the weights, plus an offline converter
(`convert_ple_to_mxfp8.py`) for the PLE table (quantization fidelity: **SNR 53.6 dB**, max abs err 1e-4).

> **READ BEFORE USE — runtime patch required (PLE table only).** Current vLLM (≤ 0.29.1rc1.dev528, and
> `main` as of 2026-10-04) loads compressed-tensors checkpoints with the PLE table **always as
> unquantized**. Without the patch in `patches/`, this checkpoint's FP8 table bytes are cast to bf16
> **without their scales** and the model degenerates *silently*. See §3.

## 1. Quantization scheme

| Module | Format | Notes |
|---|---|---|
| Routed experts `mlp.experts` (48 layers, 120.8 B params) | **MXFP4** W4A4, group 32, UE8M0 scales, dynamic activations | `config_groups.group_0` (`mxfp4-pack-quantized`, 73,729 targets) |
| MTP routed experts `mtp.layers.0.mlp.experts` (512) | **MXFP8** W8A8, group 32, UE8M0 scales | 1,536 per-expert Linears |
| **PLE n-gram embedding table** (51.2 B params, 128 shards) | **MXFP8 weights**: E4M3 + UE8M0 group-32 scales | `config_groups.group_1` (`mxfp8-quantized`); removed from `ignore` |
| Full-attention `self_attn.{q,k,v,o}_proj` / GDN `linear_attn.*` | **BF16** | |
| `indexer` / `shared_expert` / router / `mtp.fc_*` / `hyper_connection.*` / `in_proj_a/b` | BF16 | |
| `embed_tokens` / `lm_head` / `visual.*` / MTP 其余 | BF16 | |

KV cache: **not quantized**. Attention activations: **not quantized**.

Sanity fingerprints: `ignore` = 943 entries; 8-bit weights are exactly the 1,536 MTP-expert tensors plus
the 128 PLE shards (`float8_e4m3fn` + `uint8` group-32 scales); **152,216** tensors total.

## 2. Reproduce the quantization

```bash
# Step 1 — weights (AutoRound 0.16.0, model-free; layer_config ORDER matters: MTP entry first)
auto_round \
  --model_name Qwen/Qwen3.8-Flash-Next \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers "visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,ple,hyper_connection,indexer,shared_expert,self_attn,linear_attn,mtp.fc" \
  --layer_config '{"mtp.layers.0.mlp.experts":{"bits":8,"data_type":"mx_fp"},"mlp.experts":{"bits":4,"data_type":"mx_fp"}}' \
  --format llm_compressor \
  --output_dir ./Qwen3.8-Flash-Next-MXFP4-Mixed-BF16Attn-MTP8

# Step 2 — PLE table to MXFP8 (offline converter; upgrades config to two-group mixed-precision)
python3 convert_ple_to_mxfp8.py \
  ./Qwen3.8-Flash-Next-MXFP4-Mixed-BF16Attn-MTP8 \
  ./Qwen3.8-Flash-Next-MXFP4-Mixed-BF16Attn-MTP8-PLEMXFP8
```

## 3. Inference — required vLLM patch

**Base:** vLLM `0.29.1rc1.dev528` (validated; any `qwen4_exp`-capable build ≥ 0.29.0 should patch
cleanly). The patch adds the MXFP8 PLE embedding method, a fused-dequantize lookup kernel,
compressed-tensors detection of the n-gram target, and fused `shard_<i>.weight_scale` loading.

```bash
./patches/apply_qwen38fn_ple_mxfp8.sh          # apply | check | revert
```

Verified load line: `Qwen4Exp PLE embedding: using MXFP8 n-gram table from compressed-tensors checkpoint`.

- **TP=1 recommended** (upstream MoE scale issues at TP>1; not a checkpoint defect).
- With `EngramConfig(cpu_offload=True)` the MXFP8 PLE table is pinned on host (~48 GiB); the checkpoint
  fits TP=1 on a single 275 GB B300.
- MTP / speculative decoding is off by default; verified loading with `speculative_config=
  {"method":"mtp","num_speculative_tokens":1}` (Qwen4ExpMTP draft builds and captures cleanly).
- KV cache `auto` (bf16) or `fp8` both work.

## 4. Evaluation

Harness: `lm-eval 0.4.13` + vLLM `0.29.1rc1.dev528` **with the §3 patch applied**, TP=1, bf16 KV,
`seed=42`, `max_model_len=8192`, `max_num_seqs=64`, `gpu_memory_utilization=0.85`,
`language_model_only=true`, `reasoning_parser=qwen3`, `enable_thinking=false`.

| Model | GSM8K (strict / flexible) | MMLU | PIQA (acc / acc_norm) | HellaSwag (acc / acc_norm) |
|---|---|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` (bf16 baseline) | 0.9674 / — | 0.8652 | 0.8194 / — | 0.6927 / — |
| Sibling `-MTP8-1008` (PLE table bf16) | 0.9651 / 0.9644 | 0.8655 | 0.8243 / 0.8330 | 0.6845 / 0.8708 |
| **This checkpoint** | **0.9636 / 0.9636** | **0.8656** | **0.8183 / 0.8341** | **0.6832 / 0.8728** |

All deltas vs the sibling are within ~1 σ and mixed in sign — **the MXFP8 PLE table costs no measurable
accuracy** while saving 48.7 GB (180.1 → 131.5 GB).

### Reproduce

```bash
lm_eval --model vllm \
  --model_args '{"pretrained": "<this-checkpoint>", "tensor_parallel_size": 1, "max_model_len": 8192, "max_num_seqs": 64, "max_num_batched_tokens": 16384, "gpu_memory_utilization": 0.85, "dtype": "bfloat16", "trust_remote_code": true, "add_bos_token": true, "enable_prefix_caching": false, "max_gen_toks": 2048, "language_model_only": true, "reasoning_parser": "qwen3", "enable_thinking": false, "safetensors_load_strategy": "prefetch"}' \
  --tasks gsm8k --batch_size 32 --seed 42 --apply_chat_template --fewshot_as_multiturn

lm_eval --model vllm --model_args '<same JSON as above>' \
  --tasks piqa,mmlu,hellaswag --batch_size 32 --seed 42
```

## 5. Comparison with NVIDIA's NVFP4 checkpoint

Nearly identical coverage: the only differences are the 4-bit expert format (MXFP4 g32 UE8M0
dynamic-activation here; NVFP4 g16 dual-scale statically-calibrated in NVIDIA's) and the MTP-expert FP8
flavor (MXFP8 W8A8 g32 here; blockwise FP8 WO g128 in NVIDIA's). The PLE table is FP8-quantized by both
— NVIDIA uses plain FP8 with a single per-tensor FP32 scale; this checkpoint uses MXFP8 with per-32
UE8M0 group scales. Size: **131.5 GB here vs 132.7 GB**.

## 6. Notes and caveats

- **The §3 patch is mandatory.** Unpatched vLLM does not error; it silently misreads the PLE table.
- Validated platform: B300 (sm103), TP=1, bf16 KV, context ≤ 8k. The PLE table is the model's long-range
  n-gram memory — validate on long-context tasks (RULER-style) before production long-context use.
- MTP speedups: loading verified, acceptance-rate/throughput benefit not yet measured.

## 7. Citation

Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) —
Qwen Community License 1.0. Quantization toolkit: [Intel AutoRound](https://github.com/intel/auto-round).
