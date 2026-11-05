---
library_name: gguf
license: other
license_name: qwen-community-1.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - qwen4exp
  - abliterated
  - uncensored
  - mtp
  - iq3e
  - llama.cpp
  - v100
---

# Qwen3.8-Flash-Next — IQ3E-Q8D-MTP-Abliterated

Refusal-ablated build of
[`pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF`](https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF),
using the abliterated tensors from
[`huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF).

## What changed

96 `Q8_0` residual-stream write tensors were replaced with the huihui-v2
abliterated versions:

| family | count |
|---|---:|
| `blk.*.ssm_out.weight` | 36 |
| `blk.*.attn_output.weight` | 12 |
| `blk.*.ffn_down_shexp.weight` | 48 |

Everything else is unchanged:

- the 48 `blk.*.ffn_down_exps.weight` experts stay at the original **IQ4_NL**
  quantization and are **not ablated** — huihui-v2 deliberately does not ablate
  routed experts;
- the `blk.48.*` MTP layer and every other tensor are byte-identical to the
  base IQ3E file;
- no dequantize → requantize cycle was performed. The GGUF header and every
  tensor offset are identical; only the 96 payloads were overwritten.

The MTP head is not abliterated; no abliterated MTP head has been published.

## Benchmarks

Dual Tesla V100-PCIE-32GB, 262K context, q8_0 KV, MTP n=3, tensor split,
`GGML_CUDA_P2P=1`, same launch recipe as the base model. Medians of 3 runs:

| metric | base IQ3E | this build |
|---|---:|---:|
| decode | 131.1 tok/s | 123.9 tok/s |
| prefill (2.4K tokens) | 332.9 tok/s | 339.8 tok/s |
| draft acceptance | 85.7% | 86.6% |
| VRAM | 32.2 GB/GPU | 32.2 GB/GPU |

Host RAM was 15 GiB, so the PLE table pages; absolute prefill values are
host-dependent. The abliteration costs about 5% decode on this box; prefill
and draft acceptance are unchanged.

## Behavior

- The paywall-bypass probe that the unmodified base refuses is answered in
  detail.
- Code and arithmetic probes remain coherent.

This is a spot check, not a full evaluation. Abliteration can degrade
instruction-following, tool calling, and other capabilities; test against your
own prompts before adopting it.

## Requirements

- llama.cpp **v0.5.0** plus `llama-v0.5.0-v100-v3.patch` for V100 (`sm_70`);
  see the [base model card](https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF).
- Same flags as the base IQ3E model, including `--spec-type draft-mtp`.
- For tensor split: `GGML_CUDA_P2P=1` and VT-d disabled, as described in the
  base repository's field notes.

## Files

- `Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-Abliterated.gguf`
  — 85,923,705,344 bytes
  — SHA256 `4f2cb1b043d833036bf2a71c1a500614cd0e9b4434961de497fe79a55770ea4f`
- `abliterate-manifest.json` — source shards, 96 tensor names, file offsets, and
  per-tensor BLAKE2b hashes
- `SHA256SUMS`

## Safety

This model is abliterated and has no meaningful built-in guardrails. It may
comply with harmful or unsafe requests. Deploy only where you can provide your
own moderation, access control, and legal review.

## Credits

- **Qwen** — Qwen3.8-Flash-Next base model
- **pentacoxian-dev** — IQ3E-Q8D-MTP quantization and V100 llama.cpp patches
- **huihui-ai** — v2 abliterated tensors
- **llama.cpp** — GGUF format and runtime

Unofficial community derivative; not affiliated with Qwen/Alibaba.
