---
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- awq
- int4
- w4a16
- gptq
- sglang
- vision-language
- video-understanding
- gated-deltanet
---

# Qwen3.8-27B-AWQ (INT4 W4A16)

4-bit AWQ quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
calibrated in-house on 2×RTX 3090 (Ampere, sm_86) for **single-user 256K-context
serving under SGLang**. Thinking, image, and **video** are all preserved and probe-verified.

Weights are ~18.7 GB, so the model fits two 24 GB cards at TP=2 with room for a
deep KV pool (697K tokens measured at `--max-running 1`).

## Validation

Measured on the shipped checkpoint (SGLang v0.5.17 + local patches, TP=2, RTX 3090 ×2):

| Probe | Result |
|---|---|
| MMLU (30 samples) | **0.93** |
| HumanEval (25 samples) | **0.96** |
| Needle-in-haystack | **1.0** — 6/6 found at server-verified **131,150** and **250,077** actual prompt tokens |
| Capabilities | **5/5** — basic, tool-call, thinking, **vision**, **video** |
| AWQ scale integrity | 0 of 800 `scales`/`qweight` tensors flagged (no zero/NaN/Inf scales) |
| LAB-Bench (56 questions) | 0.16 — low-ish; see caveat below |

Single-user decode throughput (M=1, 100 output tokens, depth server-verified via
`actual_input_tokens`):

| Context | tok/s | TPOT |
|---|---|---|
| 1,024 | 70.6 | 14.2 ms |
| 32,768 | 67.2 | 14.9 ms |
| **261,916** | **48.4** | 20.7 ms |

Decode holds 69% of its short-context rate at full depth. (We do not quote a
cold-prefill TTFT at 262K: our measured deep request hit the prefix cache for
261,888 of 261,916 tokens, so that timing is not a prefill number. Decode
throughput is unaffected — it attends over the whole KV either way.)

## Quantization recipe

- **Method:** GPTQ, W4A16, group size 128, symmetric int4 (`llmcompressor`), then
  converted compressed-tensors → native AWQ (`gemm`) for AWQ_Marlin kernels.
- **Calibration data (256 samples × 1024 tokens):** 30% thinking traces
  (`glaiveai/reasoning-v1-20m`), 25% image instruction (`LLaVA-Instruct-150K`),
  **20% video instruction (`lmms-lab/LLaVA-Video-178K`)**, 15% math
  (`NuminaMath-CoT`), 10% chat (`ultrachat_200k`). Video samples are deliberate:
  calibrating a video-capable model without them drifts the temporal-attention
  weights.
- **Kept in BF16 (not quantized):** `lm_head`; the whole vision tower
  (`model.visual.*`, 167 modules); and the Gated DeltaNet gating scalars
  `in_proj_a` / `in_proj_b` (48 each). The rest of `linear_attn`
  (`in_proj_qkv`, `in_proj_z`, `out_proj`) **is** INT4 — this split is what
  SGLang's `Qwen3_5GatedDeltaNet` loader expects, and getting it wrong yields a
  model that emits `!!!!!`.

## Serving (SGLang)

```bash
python -m sglang.launch_server \
  --model-path mattbucci/Qwen3.8-27B-AWQ \
  --tensor-parallel-size 2 --quantization awq_marlin \
  --context-length 262144 --max-running-requests 1 \
  --mem-fraction-static 0.85 --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder --max-mamba-cache-size 8
```

### ⚠ `--max-running-requests` caps your usable context

This model is 64 layers = **48 Gated DeltaNet + 16 full attention**, and the
DeltaNet recurrent state is replicated per concurrent slot. Combined with the
untied 248,320-token vocab (embeddings + `lm_head` stay BF16), concurrency eats
the KV pool fast. Measured on 2×24 GB:

| `--max-running-requests` | KV pool (`max_total_num_tokens`) |
|---|---|
| 1 | **697,368** (2.7× the 262,144 context) |
| 8 | **32,516** — the 262,144 context claim is then a lie |

If you raise concurrency, lower `--context-length` accordingly and read
`max_total_num_tokens` back from `/get_server_info` rather than trusting the
context flag.

## Caveats

- **No MTP.** The base ships Multi-Token-Prediction layers; `transformers` does
  not model them, so they are absent here. MTP-based speculative decoding is not
  available on this checkpoint (SGLang's main loader skips `mtp.*` weights
  anyway, so nothing breaks).
- **LAB-Bench 0.16** is on the low side. It sits inside the spread we observe
  across DeltaNet-family INT4 ships on the same 56-question probe (0.11–0.32),
  and every other instrument is top-tier, so we log it as a watch item rather
  than a defect — but if your workload is domain-heavy scientific protocol QA,
  benchmark before relying on it.
- Calibrated and validated on **Ampere (sm_86) with AWQ_Marlin**. Other
  architectures should work but are untested by us.

## License

Apache-2.0, inherited from the base model. See `LICENSE`.
