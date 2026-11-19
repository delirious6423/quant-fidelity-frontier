---
base_model:
  - Qwen/Qwen3.8-27B
  - harness-hub/Qwen3.8-27B-hermes-adapted
base_model_relation: quantized
license: apache-2.0
library_name: mlx-lm
pipeline_tag: image-text-to-text
quantization_method: rco
quantization_bits: 4
tags:
  - mlx
  - rco
  - mixed-precision
  - hermes
  - agent
  - apple-silicon
---

# Qwen3.8-27B-hermes-MLX-RCO-4.5bpw

RCO mixed-precision quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with the
[harness-hub/Qwen3.8-27B-hermes-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-hermes-adapted)
RL LoRA **merged into the base weights** before quantization. Built for Apple Silicon (MLX / oMLX).

The adapter is a GRPO-trained LoRA (r=32, α=64) that makes Qwen3.8-27B a stronger agent behind
[Hermes Agent](https://github.com/NousResearch/hermes-agent) on software-engineering and terminal tasks.
Its authors report thinking-off parity with the base model's highest thinking effort at ~4× fewer tokens
(43.1% vs 45.1% SWE-bench Pro hard-51, 58.8% vs 58.8% Terminal-Bench 2.1) — those numbers are from the
[adapter card](https://huggingface.co/harness-hub/Qwen3.8-27B-hermes-adapted), measured in BF16 under the
Hermes harness, **not** re-measured here. **Keep thinking off** when serving this model, as the adapter authors recommend.

The vision tower (333 tensors) and MTP head (15 tensors) are preserved in BF16 alongside the quantized trunk.

## Quick start

```python
from mlx_lm import load, generate

model, tokenizer = load("hermitdave/Qwen3.8-27B-hermes-MLX-RCO-4.5bpw")
messages = [{"role": "user", "content": "Fix the failing test in my repo."}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=512))
```

Also serves through [oMLX](https://github.com/raullenchai/oMLX) (VLM path for image/video input, MTP
speculative decoding). No custom model code is required — standard `qwen3_5` architecture with an MLX
affine quantization map in `config.json`.

## Model details

| Field | Value |
| :--- | :--- |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (BF16) |
| Merged adapter | [harness-hub/Qwen3.8-27B-hermes-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-hermes-adapted) (LoRA r=32, α=64, 256 matrices) |
| Parameters | 27B total; 24.35B quantized |
| Layers | 64 — 16 × (3 × GatedDeltaNet → FFN + 1 × gated attention → FFN) |
| Hidden size | 5120 |
| Architecture | `qwen3_5` hybrid linear-attention VLM with MTP |
| Quantization | RCO (rate-constrained allocation), affine, group_size=64, bfloat16 scales |
| Trunk bitwidth | **4.488 bpw measured** (weights + scales/biases, quantized tensors only) |
| Refinement | imatrix-weighted LSQ scale/bias refit on all 496 quantized tensors |
| Importance matrix | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `imatrix_unsloth.gguf` (captured on the base model, 100% trunk-projection coverage) |
| Preserved BF16 | embed_tokens, lm_head, norms, conv1d, A_log/dt_bias, vision tower, MTP head |

### Bitwidth distribution (on disk)

| Bitwidth | Tensors |
| :--- | ---: |
| 8-bit | 96 |
| 6-bit | 33 |
| 4-bit | 357 |
| 3-bit | 10 |
| 2-bit | 0 (available to the allocator; not selected at this budget) |

Per-tensor allocation is in `config.json` → `quantization`. The error DB was built with candidate widths
{2,3,4,6,8}; the allocator chose the mix above by measured, importance-weighted reconstruction error.

## Fidelity (measured)

Protocol: WikiText-2 test rows (8 × 510 positions), forward KL and top-1 agreement against the merged
BF16 teacher via a streaming layer-at-a-time forward (the 54 GB teacher cannot be resident on a 64 GB
machine). Teacher PPL on the same rows: **8.650**.

| Metric | This quant | BF16 teacher |
| :--- | :--- | :--- |
| PPL (wikitext) | 8.816 | 8.650 |
| Forward KL vs BF16 | 0.0289 | — |
| Top-1 agreement vs BF16 | 92.3% | — |

### Zero-shot accuracy (5 tasks × 400 examples, raw loglikelihood)

Scored with the standard lm-eval-style raw-completion protocol (no chat template), identical examples
for both builds. The BF16 reference cannot be resident on this 64 GB machine, so these are **raw
accuracies, not BF16 recovery**; absolute values are format-sensitive for a post-trained RL model.

> A companion build restricted to two bitwidths **{4,6}** was also produced and benchmarked head-to-head.
> It showed **no noticeable difference** — fidelity (KL/top-1/PPL), MTP draft acceptance and zero-shot
> accuracy were all statistically tied with this wider mix — so it has been **withdrawn**. Its numbers
> are retained below for the record.

| Task | This quant | Narrow {4,6} (withdrawn) | Paired diff (95% CI) |
| :--- | :--- | :--- | :--- |
| arc_challenge | 0.5450 | 0.5475 | +0.0025 ± 0.0090 |
| arc_easy | 0.8100 | 0.8250 | +0.0150 ± 0.0061 |
| hellaswag | 0.5575 | 0.5525 | −0.0050 ± 0.0061 |
| winogrande | 0.5750 | 0.5625 | −0.0125 ± 0.0090 |
| mmlu | 0.5500 | 0.5425 | −0.0075 ± 0.0090 |

Mean raw accuracy: **0.6075**. All paired differences versus the withdrawn narrow build are within
noise except arc_easy (+1.5 points narrow). MTP 1-step greedy draft acceptance (grafted BF16 head,
8 × 126 WikiText positions): **63.9%** — healthy band (60–70%), statistically tied with the narrow
build's 63.7%.

Vision preservation verified through oMLX's VLM path: solid-colour PNG probes answered correctly
(this build: red).

## Performance (measured)

Hardware: Apple M3 Max, 64 GB unified memory. Runtime: mlx-lm (`mlx_lm.generate`), natural WikiText-2
prose prompts, 256 decode tokens, warmup + 3 timed runs per cell (mean, range in parentheses).
The BF16 baseline is not resident-loadable on this machine (54 GB), so no same-session BF16 TPS row
is quoted; never compare these numbers against a different runtime.

| Context | Prompt TPS | Decode TPS | Peak memory |
| :--- | ---: | ---: | ---: |
| ~1K | 209.8 (209.4–210.2) | 19.7 (19.7–19.7) | 20.66 GB |
| ~4K | 207.6 (203.6–210.9) | 19.0 (18.8–19.1) | 21.77 GB |

Prefill is bandwidth-saturated; decode reflects the dequant kernel path. bfloat16 scales are consumed
natively by MLX's dequant kernel.

## Size

| Artifact | Size |
| :--- | :--- |
| This checkpoint (incl. BF16 vision + MTP) | **20.52 GB** |
| Merged BF16 source | 55.6 GB |
| Compression | 2.71× |

## Method

1. **Merge** — the LoRA was merged into the BF16 base with exact tensor arithmetic
   (`W' = bf16(W + (α/r)·B@A)` in fp32), verified bit-exact against independent recomputation;
   untouched tensors (vision, MTP, embeddings, linear-attention projections) are byte-identical to the base.
   The adapter repo's tokenizer/chat template are adopted per its merge recipe.
2. **Error DB** — per-tensor, per-group RTN reconstruction error at {2,3,4,6,8} bits in float32,
   weighted by the published unsloth importance matrix (streaming, no model resident in memory).
3. **RCO allocation** — dynamic programming over the measured error table under a 4.5 bpw bit budget
   (parameter-weighted).
4. **Quantize + refine** — shard-by-shard MLX affine quantization with imatrix-weighted LSQ scale/bias
   refit (oQe-style) on every quantized tensor; bfloat16 scales for native dequant-kernel speed.
5. **Graft** — vision tower and MTP head restored in BF16 from the merged source; index and config audited
   (496 declared quantization entries = 496 on-disk scale modules).

Pipeline: [hermitdave/rco-mlx](https://github.com/hermitdave/rco-mlx) (`qwen3_5` dense-hybrid adapter).

## Limitations

- Lossy compression of a merged RL adapter — not a further fine-tune. Agent-task performance of this
  quantized artifact has not been re-benchmarked; the adapter card's numbers apply to BF16.
- 10 tensors sit at 3-bit (early-layer MLP projections); measured KL/top-1/PPL show no detectable damage
  on the wikitext protocol, but agentic-task fidelity of the quantized artifact is not re-benchmarked.
- The BF16 MTP head is grafted, not re-trained against the quantized trunk; draft acceptance may differ
  from the BF16 model.
- The importance matrix was captured on the base model, not the merged model — a standard approximation,
  disclosed for reproducibility.
- Thinking mode: the adapter was trained with thinking **off**; enable it only deliberately.

## Attribution

- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — Qwen team, Apache-2.0.
- **RL adapter:** [harness-hub/Qwen3.8-27B-hermes-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-hermes-adapted) — harness-hub, trained with TRL AsyncGRPO under [Hermes Agent](https://github.com/NousResearch/hermes-agent).
- **Quantization:** [rco-mlx](https://github.com/hermitdave/rco-mlx), run by [Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous Research).
- **Importance matrix:** [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).
- **Merge + validation tooling:** Hermes Agent workflows, [hermitdave/rco-mlx](https://github.com/hermitdave/rco-mlx).