---
license: mit
license_name: mit
license_link: LICENSE
library_name: transformers
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
- quantized
- mxfp4
- mxfp8
- compressed-tensors
- auto-round
- vllm
- deepseek_v4
- moe
---

# DeepSeek-V4-Flash-0731-MXFP4-Mixed-CT-AutoRound

MXFP4 mixed-precision re-quantization of
[**DeepSeek-V4-Flash-0731**](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) — a 304 B
total / ~33 B-active sparse MoE (43 layers, 256 routed + 1 shared experts/layer, hybrid compressed
attention + DSA indexer, 1M context, attached DSpark/MTP draft module) — produced with
[Intel AutoRound](https://github.com/intel/auto-round) in model-free RTN mode (no calibration
dataset), exported as **compressed-tensors** `mixed-precision`:

- **Routed experts (main model + MTP) → MXFP4 W4A4** (E2M1, group_size 32, E8M0 scales; activations MXFP4 dynamic)
- **All other quantized Linear layers → MXFP8 W8A8** (E4M3, group_size 32, E8M0 scales)

**Accuracy vs the official checkpoint (same protocol): AVG 0.8305 vs 0.8348 — −0.43 pp.**

> **Completeness note.** This checkpoint is a from-scratch re-export fixing a defect in the
> published community re-quant
> ([INCModel2/DeepSeek-V4-Flash-0731-MXFP4-Mixed-CT-AutoRound](https://huggingface.co/INCModel2/DeepSeek-V4-Flash-0731-MXFP4-Mixed-CT-AutoRound)),
> which **silently omits the entire `layers.29.*`** (3.57 GB) while its `config.json` still declares
> 43 layers — models served from it effectively run 42 layers deep and lose ~4.5 pp on gsm8k.
> This checkpoint carries all 72 317 tensors of the official layout (43/43 layers + full MTP/DSpark).

> **About a "BF16 baseline".** DeepSeek has never released a BF16 version of V4-Flash — the official
> checkpoint *is* natively FP4+FP8 quantized. The strongest available reference is therefore the
> official checkpoint itself, used as the baseline below.

## 1. Model summary

| | Official | This checkpoint |
|---|---|---|
| Architecture | `DeepseekV4ForCausalLM` (`deepseek_v4`): 43 MoE layers, hidden 4096, 256 routed experts (top-6) + 1 shared, head_dim 512 with per-layer KV compression (ratio 4/128 alternating) + DSA indexer on 21 layers, mHC hyper-connections, 3-layer DSpark/MTP draft module, 1M context | identical |
| Parameters | ~304.2 B total (277.0 B main routed experts + 19.8 B MTP, 91 % in experts) | identical (weights re-encoded) |
| Quantization | FP8 E4M3 128×128 block (linear) + MXFP4 experts, `expert_dtype=fp4` | **MXFP4 W4A4** (experts, incl. MTP) + **MXFP8 W8A8** (other Linear), group_size 32, E8M0 scales |
| Format | native (`quant_method: fp8` + `expert_dtype`) | compressed-tensors `mixed-precision` |
| Size | 166.89 GB | **167.08 GB / 155.61 GiB**, 48 shards |
| Tensors | 72 317 | **72 317 (complete)** |
| License | MIT | same |

## 2. Quantization scope

**Quantized:**

| Modules | Format | Count |
|---|---|---:|
| `layers.0-42.ffn.experts.*.w{1,2,3}` (routed experts, 43 × 256 × 3) | **MXFP4 W4A4** | 33 024 |
| `mtp.0-2.ffn.experts.*.w{1,2,3}` (MTP/DSpark draft experts, 3 × 256 × 3) | **MXFP4 W4A4** | 2 304 |
| `attn.{wq_a,wq_b,wkv,wo_a,wo_b}` · `attn.indexer.wq_b` · `ffn.shared_experts.w{1,2,3}` · `mtp.*` attention/main_proj | **MXFP8 W8A8** | 390 |

**Not quantized (BF16/F32; `ignore`, 196 entries):** `embed` / `head` (0.53 B each), per-layer KV
`compressor.{wgate,wkv}` (41×2) and `indexer.compressor.*` / `indexer.weights_proj` (21×3 — the
sparse-attention routing path), MoE routers `ffn.gate` (43+3), DSpark `markov_head` /
`confidence_head`, all hyper-connection (`hc_*`) and norm parameters.

## 3. Evaluation results

Same protocol for both columns: `lm-eval` 0.4.13 + vLLM, TP=2 on 2× B300, `kv_cache_dtype=fp8`,
`max_model_len=8192`, `seed=42`, gsm8k 5-shot, piqa/mmlu/hellaswag 0-shot, full sample counts
(1 319 / 1 838 / 14 042 / 10 042).

| | GSM8K (strict) | MMLU | PIQA | HellaSwag | **AVG** |
|---|---|---|---|---|---|
| Official DeepSeek-V4-Flash-0731 (native FP4+FP8) | 0.9583 | 0.8649 | 0.8341 | 0.6818 | **0.8348** |
| **MXFP4-Mixed-CT (this checkpoint)** | 0.9568 | 0.8631 | 0.8232 | 0.6788 | **0.8305** |
| Δ | −0.15 pp | −0.18 pp | −1.09 pp | −0.30 pp | **−0.43 pp** |


Reading:

- **Near-parity with the official checkpoint (99.5 % of AVG)** — the ~4 pp gap previously reported
  for CT re-quants of this model is *not* a quantization effect; it traces to the missing
  `layers.29.*` in that release (see the completeness note above). Re-running the community-format
  checkpoint through the Marlin W4A16 MoE path recovered ~1.3 pp of it (0.9257), consistent with the
  W4A4 activation cost; the remaining gap was the missing layer.
- The largest single-task delta here is PIQA (−1.09 pp, ≈1.2σ of the ±0.89 pp standard error).
  GSM8K (−0.15 pp, ±0.53 pp) is inside noise.

## 4. Usage (vLLM)

```bash
vllm serve DeepSeek-V4-Flash-0731-MXFP4-Mixed-0929 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --block-size 256 \
  --kv-cache-dtype fp8 \
  --max-model-len 8192 --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 --dtype bfloat16 \
  --enforce-eager \
  --no-enable-flashinfer-autotune
```

Serving constraints (each verified by measurement):

- **KV cache dtype is backend-dependent**: the default selected backends (FlashMLA / fp8_ds_mla
  layout, `deepseek_v4/attention.py:103-106`) accept **fp8 only** — `--kv-cache-dtype auto` fails at
  start-up with `DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache` and the requested `fp8` is
  rewritten to the packed `fp8_ds_mla` format (576 B/token slot). bf16/auto KV is legal only on the
  FlashInfer sparse backend with `use_fp4_indexer_cache` (`attention.use_fp8_ds_mla_layout=False`).
  All numbers in §3 use `fp8` — identical for both compared columns, so the comparison is apples to apples.
- **Requires the DSv4 compressed-tensors o_proj path** — the stock fused-attention output projection
  (`deep_gemm fp8_einsum`) only accepts the native 128×128 block-FP8 weight scales and dies on this
  checkpoint's per-32 MXFP8 scales (`t.dim() == N`). Use a vLLM build whose DSv4 `o_proj` dispatches
  compressed-tensors MXFP8 `wo_a` to an MXFP8 GEMM (this checkpoint was validated on such a build).
- **MoE**: the CUTLASS MXFP4 W4A4 path is correct at TP=1/2/4 here (per-rank expert scale columns
  `2048/TP/32` are always multiples of 4). Do **not** use the flashinfer `trtllm_mxfp4` auto-pick on
  builds that JIT-compile `trtllm_moe_sm100` at first use (long stalls); `--no-enable-flashinfer-autotune`
  and disabling JIT/cuteDSL warmup are recommended.
- TP must divide 64 (attention and indexer heads). DSpark speculative decoding is available with
  `--speculative-config '{"method":"dspark",...}'` (draft weights ship in this checkpoint).

## 5. Reproduce

### 5.1 Quantization

```bash
auto-round \
  --model_name deepseek-ai/DeepSeek-V4-Flash-0731 \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers compressor,indexer.weights_proj \
  --layer_config '{ffn.experts:{bits:4,data_type:mx_fp}}' \
  --format llm_compressor \
  --output_dir ./DeepSeek-V4-Flash-0731-MXFP4-Mixed
```

**Always verify layer completeness after a model-free run** — the community re-quant shipped without
`layers.29.*` and no tool warned:

```bash
python3 -c "
import json, re
wm = json.load(open('<output>/model.safetensors.index.json'))['weight_map']
layers = sorted({int(m.group(1)) for k in wm if (m := re.match(r'layers\.(\d+)\.', k))})
print('layers:', len(layers), 'missing:', [i for i in range(43) if i not in layers])"
```

### 5.2 Evaluation

```bash
MODEL_ARGS='{"pretrained": "DeepSeek-V4-Flash-0731-MXFP4-Mixed-0929",
  "tensor_parallel_size": 2, "max_model_len": 8192, "max_num_seqs": 256, "block_size": 256,
  "gpu_memory_utilization": 0.9, "dtype": "bfloat16", "trust_remote_code": true,
  "kv_cache_dtype": "fp8", "enable_prefix_caching": false, "max_gen_toks": 2048,
  "add_bos_token": true, "enforce_eager": true,
  "kernel_config": {"enable_flashinfer_autotune": false, "enable_cutedsl_warmup": false, "enable_jit_warmup": false}}'

# gsm8k: 5-shot generative, chat template + multi-turn few-shot
lm_eval --model vllm --model_args "$MODEL_ARGS" --tasks gsm8k \
  --batch_size 32 --seed 42 --apply_chat_template --fewshot_as_multiturn --output_path ./results

# piqa / mmlu / hellaswag: 0-shot log-likelihood, no chat template
lm_eval --model vllm --model_args "$MODEL_ARGS" --tasks piqa,mmlu,hellaswag \
  --batch_size 32 --seed 42 --output_path ./results
```

## 6. Known limitations

- Four academic benchmarks only; no long-context / agentic / code / thinking-on evaluation; no throughput numbers.
- `kv_cache_dtype=fp8` is forced by the architecture, so all published numbers carry fp8-KV effects (same for the official baseline — apples to apples).
- Serving requires a vLLM build with the compressed-tensors DSv4 o_proj path (§4).
- MXFP4 experts here are W4A4 (4-bit activations); the official native path runs FP4 weights with FP8 activations. PIQA (−1.09 pp) is the visible cost of that difference.

## 7. License and attribution

Base model [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) is
MIT (DeepSeek-AI); the `LICENSE` in this repository applies to this derivative unchanged.
Quantization with [Intel AutoRound](https://github.com/intel/auto-round) (Apache-2.0), model-free
RTN — no calibration corpus used. MXFP4/MXFP8 follow the
[OCP Microscaling Formats (MX) specification](https://opencompute.org). Serving via
[vLLM](https://github.com/vllm-project/vllm); evaluation via
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
