---
license: mit
license_name: mit
license_link: LICENSE
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- quantized
- mxfp4
- mxfp8
- compressed-tensors
- auto-round
- vllm
- qwen4_exp
- moe
- multimodal
---

# Qwen3.8-Flash-Next-MXFP4-Mixed-CT-AutoRound

Mixed-precision quantized checkpoint of [**Qwen3.8-Flash-Next**](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
— a 125 B-parameter (6 B active) natively multimodal hybrid MoE (36× Gated DeltaNet + 12× Qwen Sparse
Attention layers, 512-expert MoE, plus a 51 B-parameter n-gram embedding table and a 1-layer MTP) —
produced with [Intel AutoRound](https://github.com/intel/auto-round) in **model-free RTN** mode (no
calibration dataset, no model load), exported as **compressed-tensors** `mixed-precision`:

- **Routed experts → MXFP4 W4A4** (E2M1, group_size 32, E8M0 scales; activations MXFP4 dynamic)
- **All other Linear layers → MXFP8 W8A8** (E4M3, group_size 32, E8M0 scales; activations MXFP8 dynamic)


**Headline accuracy vs the BF16 baseline on the same four-task protocol: AVG 0.8326 vs 0.8362 —
−0.35 pp (99.6 % of baseline).**

## 1. Model summary

| | Base model | This checkpoint |
|---|---|---|
| Base | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | this repo |
| Architecture | `Qwen4ExpForConditionalGeneration` (`qwen4_exp`): 48 layers (36× Gated DeltaNet + 12× QSA), hidden 2560, 512 routed experts (top-10) + 1 shared, PLE n-gram embedding (51 B), 1 MTP layer, 262 K native context | identical |
| Quantization | — | MXFP4 W4A4 (routed experts) + MXFP8 W8A8 (other Linear), `group_size=32`, E8M0 scales |
| Format | safetensors | compressed-tensors `mixed-precision` (`mxfp4-pack-quantized` + `mxfp8-quantized`) |
| Size on disk | ~360 GB (BF16 source) | **168 GiB**, 131 shards |
| License | Apache-2.0 / see base | same (see License) |

## 2. Quantization scope

**Quantized:**

| Modules | Format | Count |
|---|---|---:|
| `mlp.experts.{gate,up,down}_proj` (routed experts, 48 layers × 512) | **MXFP4 W4A4** | 73 728 |
| `self_attn.{q,k,v,o}_proj` (12 QSA layers) · `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (36 GDN layers) · `self_attn.indexer.index_qk_proj` · `mlp.shared_expert.{gate,up,down}_proj` | **MXFP8 W8A8** | 312 |

**Not quantized (BF16/FP32):** `lm_head`, `embed_tokens`, the **PLE n-gram embedding table** (102 GB,
the single largest BF16 component), the whole **MTP** module, MoE routers (`mlp.gate`,
`mlp.shared_expert_gate`), `linear_attn.{conv1d,in_proj_a,in_proj_b}` (in_proj_a/b fuse to N=96, below
the MX `N ≥ 128` kernel floor), hyper-connection weights, `visual.*` tower, and all norms.

Note: the PLE table is the compression ceiling of this model — it alone caps the ratio at ≈2×
regardless of weight precision.

## 3. Evaluation results

Harness `lm-eval` 0.4.13 + vLLM (`0.29.1rc1.dev528`), TP=1 on a single B300, KV cache BF16,
`enable_thinking=false`, gsm8k 5-shot with chat template (greedy), piqa/mmlu/hellaswag 0-shot,
seed 42, full sample counts. BF16 row: reference run of the unquantized base model on the same
protocol.

| | GSM8K (strict) | MMLU | PIQA | HellaSwag | **AVG** |
|---|---|---|---|---|---|
| BF16 baseline | 0.9674 | 0.8652 | 0.8194 | 0.6927 | **0.8362** |
| **MXFP4-Mixed (this checkpoint)** | 0.9621 | 0.8626 | 0.8226 | 0.6832 | **0.8326** |
| Δ | −0.53 pp | −0.26 pp | +0.33 pp | −0.95 pp | **−0.35 pp** |

All deltas are within or near one standard error of the quantization run (gsm8k ±0.53 pp). The
largest drop (HellaSwag −0.95 pp) is still inside typical W4A4-MoE noise for this protocol.

## 4. Usage (vLLM)

This checkpoint has **hard serving constraints** — read before launching:

```bash
vllm serve Qwen3.8-Flash-Next-MXFP4-Mixed-0928 \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 8192 \
  --max-num-seqs 64 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 --dtype bfloat16 \
  --language-model-only --reasoning-parser qwen3 \
  --enable-thinking false   # or chat_template_kwargs
```

- **TP=1 only.** The CUTLASS W4A4 MXFP4 MoE path misreads expert scales at TP>1 whenever
  `moe_intermediate_size / TP / 32` is not a multiple of 4 (here 640/TP/32 → TP=1: 20 ✓, TP=2: 10 ✗,
  TP=4: 5 ✗). Symptom at TP>1: garbage output / `shape '[2560, 10]' is invalid` at load.
  (Upstream vLLM main is unfixed as of 2026-09; a Marlin-based TP=2 fallback exists but is a
  different numeric path — do not mix scores across the two.)
- **TP=1 fits only because of the Engram (PLE) CPU offload** present in recent nightly vLLM builds
  (`EngramConfig(cpu_offload=True)`). Builds without it OOM at the first forward (~88 GiB extra).
  Verified build: `vllm 0.29.1rc1.dev528` (nightly cu130). PyPI releases may not work.
- `kv_cache_dtype=fp8` is supported by this build's QSA path and saves memory; omit it for BF16 KV
  (what the §3 scores used).
- Everything ran text-only (`language_model_only`); the vision tower is untouched BF16 but
  unvalidated.

## 5. Reproduce

```bash
auto-round \
  --model_name Qwen/Qwen3.8-Flash-Next \
  --model_free \
  --scheme MXFP8 \
  --layer_config '{"mlp.experts":{"bits":4,"data_type":"mx_fp"}}' \
  --ignore_layers visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,ple,mtp,hyper_connection \
  --format llm_compressor \
  --output_dir ./Qwen3.8-Flash-Next-MXFP4-Mixed
```

**Scope caveat (measured, 2026-09):** AutoRound ≥ PR
[#2327](https://github.com/intel/auto-round/pull/2327) unions a predefined `Qwen4*` ignore list into
`--ignore_layers` with no opt-out; its bare `shared_expert` substring silently keeps
`mlp.shared_expert.{gate,up,down}_proj` in BF16 (144 layers). The reference INC checkpoint predates
that PR and quantizes them. To match the reference scope byte-for-byte, remove `"shared_expert"`
from the Qwen4 entry in `auto_round/special_model_handler.py` before running. Check the run log for
`Using predefined ignore_layers from config:` to see what was actually applied.

### 5.1 Reproduce the evaluation

The exact commands behind §3 (requires the serving constraints from §4 — TP=1 and a recent nightly
vLLM; results were produced with `vllm 0.29.1rc1.dev528`):

```bash
MODEL_ARGS='{"pretrained": "Qwen3.8-Flash-Next-MXFP4-Mixed-0928",
  "tensor_parallel_size": 1, "max_model_len": 8192, "max_num_seqs": 64,
  "max_num_batched_tokens": 16384, "gpu_memory_utilization": 0.85,
  "dtype": "bfloat16", "trust_remote_code": true, "add_bos_token": true,
  "enable_prefix_caching": false, "max_gen_toks": 2048, "kv_cache_dtype": "bfloat16",
  "language_model_only": true, "reasoning_parser": "qwen3",
  "enable_thinking": false, "safetensors_load_strategy": "prefetch"}'

# gsm8k: 5-shot generative, chat template + few-shot as multi-turn
lm_eval --model vllm --model_args "$MODEL_ARGS" --tasks gsm8k \
  --batch_size 32 --seed 42 --apply_chat_template --fewshot_as_multiturn \
  --output_path ./results

# piqa / mmlu / hellaswag: 0-shot log-likelihood, no chat template
lm_eval --model vllm --model_args "$MODEL_ARGS" --tasks piqa,mmlu,hellaswag \
  --batch_size 32 --seed 42 --output_path ./results
```

`--model_args` must be a single JSON object (this lm_eval build `json.loads` the first token and
rejects comma-separated `k=v` strings containing nested dicts).

## 6. Known limitations

- Four academic benchmarks only; no long-context, agentic, code, or thinking-on evaluation; no
  throughput numbers.
- TP=1-only serving (see §4); no multi-GPU throughput data.
- Multimodal path unvalidated (`language_model_only=true` throughout).
- The 51 B n-gram table stays BF16 and dominates the on-disk size; this checkpoint's value is the
  expert compression, not a smaller footprint than what the table allows.

## 7. License and attribution

Base model [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next); see the
`LICENSE` file in this repository and the base model's terms.

Quantization with [Intel AutoRound](https://github.com/intel/auto-round) (Apache-2.0), model-free RTN
mode — no calibration corpus was used. MXFP4/MXFP8 follow the
[OCP Microscaling Formats (MX) specification](https://opencompute.org): one E8M0 shared scale per
32-element block. Serving via [vLLM](https://github.com/vllm-project/vllm); evaluation via
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
