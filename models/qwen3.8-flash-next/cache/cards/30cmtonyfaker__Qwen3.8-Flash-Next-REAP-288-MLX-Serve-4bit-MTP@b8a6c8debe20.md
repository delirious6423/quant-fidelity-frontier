---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
library_name: mlx
language:
- en
- zh
tags:
- mlx
- mlx-serve
- apple-silicon
- qwen
- qwen4_exp
- moe
- 4-bit
- mtp
- reap
pipeline_tag: image-text-to-text
---

# Qwen3.8-Flash-Next REAP-288 — MLX Serve 4-bit + Native MTP

## Update — 2026-09-05

### Official MLX Core 26.9.1 desktop GUI: basic deployment verified

The corrected pack was selected and started through the **unmodified official
MLX Core 26.9.1 desktop GUI** on M4 Max, 128 GB unified memory, macOS 27.0
(`26A5425a`). It loaded successfully, reported native MTP loaded, and produced
the correct arithmetic answer in the chat window. On that GUI-launched server,
two short API cases (arithmetic and a structured tool call), each with MTP off
and on, passed **4/4**; normalized outputs matched between MTP modes.

For this tested basic deployment path, **no custom App rebuild or loader patch
is required**. This does not establish every desktop feature, vision, long-context
quality, a clean-machine installation, or new decode benchmarks. The performance
table below still describes the older modified runtime, not version 26.9.1.
The official 26.9.1 server executable SHA-256 is
`b22a0527fe76c7e14d9441875c71630db498c5ade5598dada76127b5606ea89c`.

### Required correction for older downloads

This update corrects the PLE convolution header in
`model-00037-of-00131.safetensors`: `[10240,1,4]` becomes `[10240,4,1]`.
The entire shard's tensor payload was verified byte-for-byte unchanged. Older
revisions lacked this correction and failed warmup in the official runtime.

### Historical M4 Max decode measurements — modified 26.8.11 runtime

The following measurements used a **locally modified runtime**
on M4 Max, 128 GB unified memory, macOS 27.0 (`26A5425a`), MLX Core 26.8.11
and MLX 0.32.2. They do not measure the clean official binary.

| Actual prompt tokens | Serial decode, median (range) | MTP Auto decode, median (range) | Median speedup |
|---:|---:|---:|---:|
| 2,048 | 30.825 (29.782–30.911) tok/s | 68.937 (52.736–69.470) tok/s | 2.236× |
| 5,120 | 22.510 (20.346–24.360) tok/s | 34.675 (31.202–48.037) tok/s | 1.540× |
| 10,240 | 26.145 (25.460–26.439) tok/s | 52.007 (48.156–53.793) tok/s | 1.989× |

Each cell has one warmup and three measured requests, with a 512-token output
budget, seed 42, temperature 0, top-p 0.95, top-k 20, thinking disabled,
PLD disabled, and KV cache at 8-bit. The GUI uses a 65,536-token context,
6 GB RAM prefix-cache budget and 32 GB SSD prefix-cache budget. Requests within
each pair differ only in `enable_mtp`. Prompt sizes include the chat template
and were checked with the runtime tokenizer. The prompt is synthetic padding
followed by a cache-library coding task; it does not represent every workload.

Measured requests reused hot prefixes (2,017 / 5,119 / 10,209 cached tokens
respectively). The table reports **decode**, not cold-prefill speed. Median
time to first token was 237/296 ms (serial/MTP) at 2K, 209/145 ms at 5K,
and 406/367 ms at 10K; these are hot-prefix latencies.

All measured responses reached the 512-token limit and none triggered the
server's repetition-loop stop. Three repeats within each cell produced the
same output hash, but **serial and MTP outputs differed in all nine pairs**.
This demonstrates a throughput benefit on this fixture, not exact serial/MTP
equivalence or correctness of the generated, truncated code.

Raw measurements and the aggregate report are in
[`benchmarks/20260905-long-context/`](./benchmarks/20260905-long-context/).
The reproducible harness and configuration are included there. The dynamic
policy described below remains guidance, not an installed feature of these weights.

### Dynamic sampling guidance for constrained tasks

The current day-to-day MLX Core GUI baseline is `temperature 1.00`,
`top_p 0.95`, `top_k 20`, and `repeat_penalty 1.08`. Do not permanently
lower the server-wide temperature just because one task needs stricter output.

Use a per-request `temperature: 0.2` when the task has a narrow valid answer
space and the model must follow an exact operational contract, especially:

- memory ingestion or memory updates that must call a mutation tool instead of
  replying conversationally;
- strict JSON/schema output, exact tool arguments, state-machine transitions,
  deterministic extraction, or reproducible agent/evaluation steps;
- a bounded retry after the model has shown sampling-related drift, repetition,
  unnecessary prose, or avoidable deviation from a valid schema.

Keep the normal GUI baseline for open-ended chat, creative writing,
brainstorming, and exploratory planning. Use `temperature: 0` only for a true
greedy/reproducibility control. If `0.2` still produces invalid tool names,
broken schemas, missing context, or impossible arguments, fix the tool/schema/
prompt path rather than repeatedly lowering temperature.

A practical dynamic policy is:

1. start from the GUI-managed baseline;
2. let a shared request-policy/router detect a strict-output phase from explicit
   metadata such as `response_format`, required tool choice, state mutation, or
   an observed drift/repetition failure;
3. override only that request or bounded retry with `temperature: 0.2`;
4. restore the GUI baseline immediately after the constrained phase succeeds
   or terminates.

Implement this policy in the request routing layer, not in model weights or
`config.json`. Prefer explicit task metadata over guessing from prompt text,
allow the caller to override the decision, log the selected profile and reason,
and always clear a temporary override when the request finishes. This makes the
same policy reusable across chat applications, tool agents, structured-data
workflows, and evaluation harnesses.

## About this model

This is an unofficial, experimental Apple-Silicon inference pack derived from
Qwen3.8-Flash-Next. It combines the community REAP-288 pruning work with an
MLX Serve-compatible external N-gram table and an optional converted native
MTP head.

This repository is intended for local research and agent/tool-calling tests.
It is not an official Qwen release and is not a claim of safety, correctness,
or leaderboard performance.

## What is included

- 48-layer `qwen4_exp` trunk with 288 routed experts per layer and top-10 routing
- MLX affine 4-bit weights, group size 64 for the main checkpoint
- 4-bit/group-32 learned N-gram data in `ngram_table.bin`, memory-mapped by the tested runtime
- converted one-layer Qwen3.8-Flash-Next MTP companion in `model-mtp.safetensors`
- tokenizer, chat template, vision metadata/weights, and REAP expert map
- a historical, architecture-gated conversion-tool snapshot under `tools/`;
  it is not required to load the corrected released weights

The approximately 30 GB `ngram_table.bin` contains learned model weights. It
does not contain prompts, chat history, logs, or user data.

## Provenance

- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — official base architecture and weights
- [Sawfwair/Qwen3.8-Flash-Next-MLX-4bit](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-4bit) — upstream MLX 4-bit conversion work
- [sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit) — REAP-288 selection/pruning and Qwen4-exp compatibility fixes
- [sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16) — source MTP companion
- [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) — tested inference runtime

Please retain upstream attribution and review the Qwen Community License 1.0
before redistribution or commercial use.

## MLX Serve compatibility transformation

The conversion is architecture-specific and refuses unknown layouts. It:

1. losslessly repacks 384 PLE tensors into a contiguous N-gram mmap;
2. folds 148 Qwen4-exp delta RMSNorm weights into direct gamma (`1 + w`);
3. transposes 36 depthwise linear-attention convolution layouts and one PLE
   convolution layout from
   `[C, 1, K]` to `[C, K, 1]` without changing payload values;
4. deliberately leaves the 36 gated `linear_attn.norm` tensors unchanged;
5. converts the MTP companion to the `language_model.mtp.*` indexed layout,
   folds its nine delta RMSNorm tensors, and quantizes 20 eligible matrices to
   affine 4-bit/group-64.

The trunk transform leaves all non-target tensor payloads byte-identical. See
`mlx-serve-compat-pack.json` for immutable fingerprints and counts.

## Runtime requirements

The desktop GUI deployment tested above uses official
[MLX Core v26.9.1](https://github.com/ddalcu/mlx-serve/releases/tag/v26.9.1).
It supports this corrected pack's `qwen4_exp` architecture, external N-gram
table, and indexed Qwen4 MTP head. Later releases need their own validation.

To install:

1. Install the official MLX Core app from the release linked above.
2. Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli),
   then download this complete repository into a new model folder:

   ```bash
   hf download 30cmtonyfaker/Qwen3.8-Flash-Next-REAP-288-MLX-Serve-4bit-MTP --local-dir ./models/Qwen3.8-Flash-Next-REAP-288-MLX-Serve-4bit-MTP
   ```

3. In MLX Core's model-folder settings, select the parent `models` folder,
   refresh the model list and select the downloaded model. Keep
   `ngram_table.bin`, the index, tokenizer files and MTP shard together.
4. Use the settings below; confirm the actual discovered ID using
   `/v1/models` before making API requests. Model IDs depend on the selected
   folder layout and need not equal the Hugging Face repository ID.

Existing downloads must refresh the corrected shard. Its new SHA-256 is
`f6510b472857c94b3aadb3576e299d4775f2b6ca4de294f898e4dd9cd7285484`.
No additional RMSNorm folding or quantization should be applied to this
already converted pack. The older `tools/` converter is for its declared
upstream source format; it is not a command to rerun on these released weights.

Tested GUI-managed settings:

| Setting | Value |
|---|---:|
| Context window | 65,536 |
| Default maximum output | 8,192 |
| Temperature | 1.0 |
| Top-p | 0.95 |
| Top-k | 20 |
| Repeat penalty | 1.08 |
| KV cache quantization | 8-bit |
| Maximum concurrency | 1 |
| MTP enabled | yes |
| MTP depth | Automatic |
| MTP on mixture-of-experts models | yes |
| PLD enabled | no |
| RAM prefix cache budget | 6 GB |
| SSD prefix cache budget | 32 GB |

Although the GUI can retain ANE prefill as enabled, the tested official runtime
explicitly disables ANE prefill for `qwen4_exp`; it is outside that option's
supported architecture family.

For a greedy reproducibility control, send `temperature: 0` per request. For
strict operational tasks that still benefit from a small amount of sampling,
use the phase-scoped `temperature: 0.2` policy above. Keep runtime settings in
the GUI or request payload rather than editing the model config.

## Example request

```bash
curl http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-Next-REAP-288-MLX-Serve-4bit-MTP",
    "messages": [{"role": "user", "content": "Write a Python LRU cache with tests."}],
    "temperature": 0,
    "max_tokens": 512,
    "enable_thinking": false,
    "enable_mtp": true
  }'
```

Use `"enable_mtp": false` for a serial baseline. PLD is intentionally disabled
when the native MTP path is selected; they are alternative speculative modes.

## MTP notes

The MTP head must match the Qwen3.8-Flash-Next architecture. It is not a generic
adapter that can be borrowed from an unrelated model. The companion source used
here was architecture-compatible, but its standalone keys, delta-RMSNorm
representation, and BF16 tensor layout did not match the MLX Serve checkpoint
loader. The included conversion tool performs those storage/layout changes.

MTP is most useful when draft acceptance outweighs verification cost. Long
contexts or low-acceptance prompts can be slower; use a paired warm benchmark
and a runtime fallback policy instead of forcing MTP globally.

## Limitations

- REAP pruning and 4-bit quantization can change quality, especially on rare knowledge.
- Vision weights are present but this exact pack has only been accepted on text/tool-call tests.
- Long-context behavior up to the configured model maximum has not been fully qualified.
- Reduced-refusal or “uncensored” source labeling does not make output safe or accurate.
- Do not expose the local server publicly without authentication and rate limiting.

## Privacy

The release contains model artifacts, fingerprints, documentation, and the
conversion tool only. It excludes local paths, inference logs, BFCL data,
credentials, caches, and user conversations.
