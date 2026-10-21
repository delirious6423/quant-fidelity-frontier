---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- trohrbaugh/Qwen3.8-27B-heretic-ara
pipeline_tag: image-text-to-text
tags:
- nvfp4
- vllm
- mtp
- speculative-decoding
- abliterated
- uncensored
- gb10
- dgx-spark
- qwen3
---

# Qwen3.8-27B-heretic-ara-NVFP4

NVFP4 (W4A16) quantization of
[trohrbaugh/Qwen3.8-27B-heretic-ara](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara)
— the ARA-abliterated [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) —
with the model's **MTP speculative-decoding head preserved and wired for vLLM**.
20 GB on disk versus 52 GB for the BF16 source.

Built for and benchmarked on **NVIDIA GB10 / DGX Spark** (128 GB unified
memory, aarch64), where NVFP4 is the sweet spot: the platform is
bandwidth-limited (~273 GB/s), so weight size dominates decode speed, and NVFP4
gets a native CUTLASS kernel on `sm_121`. In our testing this quant beat an FP8
build of the same model by ~29% — the opposite of results published for
datacenter GPUs, and the reason this repo exists.

## What was done

- Quantized with `nvidia-modelopt` 0.45.0 (`W4A16_NVFP4_CFG`).
- **Vision tower left unquantized** (`model.visual*` excluded) — image and
  video input verified working after conversion.
- **MTP head left unquantized and wired in.** The abliterated release ships the
  same MTP tensors as the stock model; this repo adds the `"mtp*"` entries to
  `exclude_modules` in `config.json` / `hf_quant_config.json` and maps the 15
  `mtp.*` tensors (in `model_mtp.safetensors`) into the safetensors index, so
  vLLM can use `--speculative-config` with it.
- Tokenizer, chat template, and processor configs are unchanged from the
  upstream abliterated release.

## Serving with vLLM

The known-good command (this is what produced the numbers below):

```bash
vllm serve <model-path> \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.40 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --reasoning-parser qwen3
```

To enable MTP speculative decoding (read the stability section first):

```bash
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Notes:

- **Tool-call dialect is `qwen3_coder`, not `qwen3_xml`** — the chat template
  emits `<tool_call><function=NAME><parameter=KEY>…`. The template also uses
  `<think>…</think>`, hence the `qwen3` reasoning parser. vLLM validates both
  names before loading weights, so a mismatch fails fast.
- Native context is 262,144; we serve 131,072 because KV cache is the dominant
  memory cost. Do not go below ~131k if you front an agentic coding client —
  large system prompts plus a 32k output reservation overflow a 65,536 window.
- On unified-memory platforms, vLLM sizes its pool from memory **free at
  launch**, not total — anything already resident silently shrinks your KV
  cache. Measured footprints on GB10: ~27 GiB at `--gpu-memory-utilization
  0.25`, ~43 GiB at `0.40` (both at 131k context).

## Performance (GB10, vLLM)

| | this quant | stock Qwen3.8-27B NVFP4 |
|---|---|---|
| Single-stream decode, MTP off | 12.6 tok/s | — |
| Single-stream decode, MTP on (k=3) | 19.1 tok/s | 23.6 tok/s |
| Aggregate, 4 concurrent, MTP off | **33.7 tok/s** | 30.8 tok/s |
| Aggregate, 4 concurrent, MTP on | **52.3 tok/s** | — |

The stock model is faster single-stream with MTP because its draft head was
trained on the weights it ships with; here the head drafts for abliterated
weights it never saw, which costs acceptance rate. In exchange: no refusals,
and better concurrency.

## MTP stability — read before enabling

Full transparency on the one incident: on first real-world deployment, MTP
crashed the engine once — `CUBLAS_STATUS_INTERNAL_ERROR` in the MTP head's BF16
GEMM, triggered when a new request prefilled while other requests were decoding
with draft tokens in flight. Multi-client front-ends (e.g. open-webui's
background title/tag requests racing a chat) produce exactly that pattern.

Since then it has **not reproduced**: three deliberate reproduction attempts
(synthetic mixed prefill+decode load, staggered-arrival load, and adversarial
interactive use) plus a 4-hour monitored soak under real multi-client traffic —
zero crashes, zero container restarts.

Recommendation: enable MTP together with a container restart policy
(`--restart unless-stopped` or equivalent) so a recurrence costs a reload, not
an outage. For strictly single-client use it has been trouble-free. If you need
hard guarantees, serve without `--speculative-config` — you keep the
concurrency advantage (33.7 tok/s at c=4) either way. Untried mitigations if
you do hit it: `num_speculative_tokens: 1`, `--enforce-eager`.

## Intended use and limitations

This is an **abliterated ("uncensored") derivative**: refusal behavior has been
removed by the upstream ARA process. It will comply with requests the base
model would decline. You are responsible for the legality and consequences of
what you generate with it, and for adding whatever guardrails your deployment
requires. Do not expose it unauthenticated to the public internet.

Quantization was calibrated for general use; task-critical accuracy should be
validated against the BF16 upstream for your workload. The base model's
knowledge, biases, and failure modes carry over.

## Lineage and licensing

`Qwen/Qwen3.8-27B` (Apache-2.0) → ARA abliteration by
[trohrbaugh](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara) →
NVFP4 conversion + MTP wiring (this repo). A Q6_K GGUF of the same abliterated
weights is available at
[mradermacher/Qwen3.8-27B-heretic-ara-GGUF](https://huggingface.co/mradermacher/Qwen3.8-27B-heretic-ara-GGUF).

Licensed Apache-2.0, same as the base model (LICENSE included).
