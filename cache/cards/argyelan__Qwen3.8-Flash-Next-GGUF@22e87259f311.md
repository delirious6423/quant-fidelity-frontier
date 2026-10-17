---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- llama.cpp
- qwen4
- day-0
---

# Qwen3.8-Flash-Next — GGUF (day-0 conversion)

GGUF conversion of [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
(125B MoE + 51B n-gram embeddings, 6B active, Qwen4 architecture preview) for llama.cpp.

Converted on day 0 on a DGX Spark, before quants were available anywhere — by now
several groups ship Flash-Next GGUFs (unsloth and others), so pick what fits your box.
This one is built for a single 128 GB Spark-class machine, and the FP8 converter fixes
it needed are documented below.

## Files

| File | Size | What |
|---|---|---|
| Qwen3.8-Flash-Next-IQ4XS-NGQ5-*.gguf (3 shards) | 98 GB | IQ4_XS weights + 51B n-gram table at Q5_0 (custom streaming requant). **Fits and runs on a single DGX Spark / GX10** — 34 tok/s measured. Point llama.cpp at the first shard. |
| (Q8_0 master, 176 GB — upload follows after multi-box testing) | | |

About the n-gram table: llama-quantize cannot process the single 51B-parameter tensor
(dequantizing it needs ~204 GB RAM, and its 160-wide rows make k-quants impossible).
This build uses a custom chunked streaming requantizer to bring it to Q5_0 —
tool + full walkthrough: [marknx/flash-next-gguf-tools](https://github.com/marknx/flash-next-gguf-tools). The table is lookup-only, so it tolerates this well
(all gates below passed).

## Requirements

- llama.cpp with Qwen4-Exp support: [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) (not yet in a release)
- Full recipe, converter fixes and the streaming requantizer: [marknx/flash-next-gguf-tools](https://github.com/marknx/flash-next-gguf-tools)
- Tight on memory? The tools repo also has a **two-box recipe** (DGX Spark + RTX 5090
  over 10 GbE RPC) that frees ~25 GB on the main box — that's the setup the
  eval numbers below were measured on.

## How this was made

```
python convert_hf_to_gguf.py Qwen3.8-Flash-Next-FP8 --outtype q8_0 --fp8-as-q8
llama-quantize --allow-requantize --tensor-type "per_layer_token_embd=q8_0" q8.gguf iq4xs.gguf iq4_xs
```

## Honest caveats

- Requantized from the official FP8 checkpoint (not BF16), without imatrix. Good enough
  to run day 0; expect refined quants (imatrix, tuned mixes) from the usual sources soon.
- Gates run before upload: loads clean on a DGX Spark, coherent German/English output,
  47*83 correct, JSON-schema output valid, needle retrieval clean. ~34 tok/s single stream,
  prefill ~550 tok/s.
- Full evals on this exact quant: **HumanEval 97.0 %** (159/164) · **GSM8K 96.0 %** (192/200,
  no empty answers). Run day 0 via llama.cpp server, greedy-ish defaults.
- New architecture: expect llama.cpp PR updates; rebuild when #27742 moves.

Measured and documented at [argyelan.ai](https://argyelan.ai). Numbers, warts and all.
