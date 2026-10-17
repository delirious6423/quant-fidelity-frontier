---
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- mtp
- speculative-decoding
- qwen4exp
---

# Qwen3.8-Flash-Next MTP draft head (Q8_0)

The multi-token-prediction head from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), 3.85 GiB at Q8_0.

Qwen trains it jointly with the model, so it drafts better than a separate small model
would. Use it as the draft for any Qwen3.8-Flash-Next GGUF.

## Setup

Works with **upstream llama.cpp b11386 or newer** (Qwen4Exp MTP support landed in
[ggml-org/llama.cpp#29761](https://github.com/ggml-org/llama.cpp/pull/29761)). No fork needed.

The target must be a GGUF that upstream llama.cpp loads, for example unsloth's Q8_0 or UD quants.

> Updated 2026-10-05: the file now uses upstream's tensor names for the MTP head
> (`blk.48.nextn.hc_head_*`) and marks the MTP layer as a sparse-attention layer in the
> metadata, as upstream expects. The weights are unchanged. Older downloads fail on upstream with
> `tensor 'blk.48.nextn.hc_head_norm.weight' not found`: please download the file again.

## Run

```bash
./build/bin/llama-server \
  -m Qwen3.8-Flash-Next.gguf \
  -md Qwen3.8-Flash-Next-MTP-Q8_0.gguf \
  -ngl 99 --n-gpu-layers-draft 99 \
  --spec-type draft-mtp --spec-draft-n-max 3 -c 32768
```

With `-hf`, add `--mtp` and llama-server picks up a sibling `mtp-` file from the same repo.

Adds ~3.9 GiB on top of the target.

## Measurements

Radeon 8060S, 250 tokens at temp 0, warmed up first (measured before the rename; the weights are unchanged).

| target | no draft | with this head | acceptance |
|---|---|---|---|
| UD-IQ4_XS | 24.2 t/s | 29.3 t/s | 0.623 |
| ROCmFP4-FAST | 28.1 t/s | 30.3 t/s | 0.587 |

A [4-bit version of this head](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST-GGUF)
measured faster on both targets and is 1.6 GiB smaller.

## Credits

Qwen3.8-Flash-Next support in llama.cpp is the work of **Daniel Han**
([@danielhanchen](https://github.com/danielhanchen)), from
[ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742); the MTP graph
is from [#27739](https://github.com/ggml-org/llama.cpp/pull/27739) (JJJYmmm). Base model by
the Qwen team.

Quantized and published by [Agention](https://agention.ai).

## License

Qwen Community License 1.0, included as `LICENSE`.
