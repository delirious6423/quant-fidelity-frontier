---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model:
  - incoai/Qwen3.8-27B-DFlash2
  - Qwen/Qwen3.8-27B
inference: false
tags:
  - dflash2
  - speculative-decoding
  - draft-model
  - fp8
  - compressed-tensors
  - sglang
---

# Qwen3.8-27B-DFlash2-FP8

**+4 to +6% decode throughput on an RTX 5090** over the bf16 draft, on named datasets, at the same acceptance length. That is this draft versus `incoai/Qwen3.8-27B-DFlash2`, both against the RadixArk NVFP4 target. On the all-NVFP4 target the increment is prompt-dependent (see Throughput).

[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) with FP8 MLP and `o_proj`. Draft for Qwen3.8-27B. Runs in SGLang.

## Checkpoint

- **Base:** incoai/Qwen3.8-27B-DFlash2, 5 layers, block size 8, bf16, 3.85 GB
- **This checkpoint:** 2.41 GB. 20 tensors in `float8_e4m3fn`: `layers.*.mlp.gate_proj`, `up_proj`, `down_proj` (15) and `layers.*.self_attn.o_proj` (5). Weights use a symmetric static per-tensor scale; activations a symmetric dynamic per-tensor scale. Format: compressed-tensors `float-quantized`. No calibration data.
- **Unchanged (bf16):** `q_proj`, `k_proj`, `v_proj`, `fc`, `candidate_selector.*`, convolutions, norms. The three attention input projections stay bf16 so SGLang's fused DFlash KV materialization remains enabled.
- **Target:** Qwen3.8-27B in any precision. Measured against [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4).

## Acceptance length

Greedy, thinking enabled, `max_tokens` 1024, one request at a time. Acceptance is the token-weighted mean of SGLang's per-batch `accept len`; prompts under about 130 generated tokens produce no sample and are excluded.

| Workload | Prompts | bf16 draft | FP8 draft | Change |
| --- | ---: | ---: | ---: | ---: |
| MT-Bench, first turn | 80 | 3.80 | 3.83 | +0.8% |
| HumanEval | 164 | 4.64 | 4.58 | −1.3% |
| GSM8K, test | 100 | 5.07 | 5.02 | −1.0% |
| MATH-500 | 100 | 5.14 | 5.08 | −1.2% |

Greedy: a draft token is kept only on a target argmax match. Target tokens are unchanged.

## Throughput

One RTX 5090, SGLang `lmsysorg/sglang@sha256:616a3e97…` with the [b12x NVFP4 backend patch](https://github.com/thomgardiner/sglang-sm120-nvfp4), target RadixArk/Qwen3.8-27B-NVFP4, 8 draft tokens.

### Concurrency 1

| Workload | bf16 draft | FP8 draft | Change |
| --- | ---: | ---: | ---: |
| MT-Bench, first turn | 190.5 tok/s | 201.5 tok/s | +5.8% |
| HumanEval | 239.2 tok/s | 248.8 tok/s | +4.0% |
| GSM8K | 264.0 tok/s | 277.0 tok/s | +4.9% |
| MATH-500 | 261.6 tok/s | 272.4 tok/s | +4.1% |

Decode step measured directly (streamed chunks as verify steps, greedy, thinking off): 18.8 → 18.0 ms on the RadixArk target, 16.4 → 15.6 ms on the all-NVFP4 target. Tokens per step moves with the draft's numerics. On three fixed prompts against RadixArk the tok/s change was +0.4% (prose), +12% (code), −1.5% (math). Against the all-NVFP4 target, interleaved n=6 at 8 draft tokens: 156.6 → 168.1 prose (+7.4%), 346.4 → 356.4 code (+2.9%), 353.2 → 396.6 math (+12.3%).

On the all-NVFP4 target, raising `--speculative-num-draft-tokens` to 12 with this FP8 draft (n=2): prose 157 tok/s, code 396 tok/s, math 439 tok/s. Prose does not gain. 16 tokens is not better on code. DSpark on the same target is slower. The lead +4 to +6% is the named-dataset range on RadixArk, not a bound on every prompt.

Not measured: concurrency above 1, engines other than SGLang.

## Serving with SGLang

```bash
python3 -m sglang.launch_server \
  --model-path RadixArk/Qwen3.8-27B-NVFP4 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path thomasgardiner/Qwen3.8-27B-DFlash2-FP8 \
  --speculative-num-draft-tokens 8 \
  --attention-backend flashinfer \
  --trust-remote-code \
  --max-running-requests 1 \
  --cuda-graph-max-bs-decode 1 \
  --mem-fraction-static 0.91
```

SGLang reads the quantization config from the checkpoint. Do not pass `--speculative-draft-model-quantization`. The server log reports `DFLASH fused KV materialization enabled` when the fused path is active.

## Creation

```bash
python3 quantize.py <incoai/Qwen3.8-27B-DFlash2 dir> <output dir> tensor
```

`quantize.py` in this repository.

## Artifact identity

| File | Size | sha256 (first 16) |
| --- | ---: | --- |
| model.safetensors | 2.41 GB | 1f3636a32d866f8e |

Apache-2.0. DFlash 2 and the base draft are by [inco.ai](https://inco.ai/blog/dflash2/) and [z-lab](https://github.com/z-lab/dflash).
