---
license: other
license_name: mit-and-apache-2.0
license_link: https://huggingface.co/Sawfwair/Ornith-1.5-35B-A3B-MLX-4bit-Vision-MTP#license-and-attribution
library_name: mlx
pipeline_tag: image-text-to-text
base_model:
  - ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit
  - ornith-ai/Ornith-1.5-35B-A3B
  - shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY
tags:
  - mlx
  - mere-run
  - ornith
  - vision
  - speculative-decoding
---

# Ornith 1.5 35B-A3B MLX 4-bit: vision and distilled MTP bundle

This bundle packages the official Ornith Q4 text target, its BF16 vision component,
and Shisa's Ornith-distilled BF16 draft head for `mere.run`. The target, tokenizer,
configuration, and vision files are unchanged from bundle revision
`2323acfa0fd0a01c452c89991558e6bbd86f0f05`.

The MTP component replaces the native final shard with a compact head. Its fused
expert tensors are split into the loader's per-expert layout. All BF16 values,
including raw-HF zero-centered norm weights, are preserved; no requantization or
normalization adjustment is applied.

## Sources

| Component | Repository | Pinned revision | Destination |
|---|---|---|---|
| Q4 target | `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | `19504d912fa8fc7622bf6b1de3db5d5d890b1f02` | `/` |
| Vision | `ornith-ai/Ornith-1.5-35B-A3B` | `10fbf86fed7ecee4a061f8b499a618f46001cac1` | `vision/` |
| MTP | `shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY` | `2b19b31bfe1659c6b0d9459ec3cbd87e34a322ef` | `mtp/` |

The replacement head was initialized from Qwen3.6 and distilled against Ornith.
See [Shisa's pinned model card](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY/blob/2b19b31bfe1659c6b0d9459ec3cbd87e34a322ef/README.md)
for training provenance and the publisher's runtime caveats.

## Layout and integrity

The root contains four Q4 target shards and tokenizer files. `vision/` contains
the original BF16 vision shard, index, configuration, and processors. `mtp/`
contains `model-mtp.safetensors`, a 785-tensor index, provenance, and attribution.

`SHA256SUMS` covers all packaged files except itself. `mtp/PROVENANCE.json`
records the pinned source artifact, converted artifact hash, conversion method,
and SHA-256 of every output tensor. The source head has 19 fused BF16 tensors;
the packaged head is 1,689,376,032 bytes. Every tensor's bytes were checked after
conversion. The target verifies proposals during speculative decoding.

## Use with mere.run

```bash
mere.run text chat \
  --model text-agent-ornith-35b-mlx-4bit \
  --model-root MODEL_ROOT \
  --prompt "Reply with a short greeting."
```

Update mere.run before using this snapshot. Replace `MODEL_ROOT` with the
absolute path to the snapshot. Text and code use
verified MTP when eligible. Image requests use target-only decoding and the
unchanged vision component.

## Qualification

The [head comparison](https://github.com/sawfwair/mere-run/blob/main/docs/benchmarks/ornith-mtp-upstream-check-2026-09-05.md)
records higher draft acceptance with unchanged greedy output in 40 measured
requests. The subsequent promotion receipt records long-context replay,
tool-call, sampling, and vision checks in mere.run. These measurements do not
establish a universal speedup or certify other serving engines.

## License and attribution

The Ornith target and vision repositories declare MIT in their pinned model-card
metadata. Those repositories do not include a separate license file; this bundle
retains their source attribution without inventing a copyright notice.

The Shisa MTP component is Apache-2.0. Its license and notice are preserved in
[`mtp/LICENSE`](mtp/LICENSE) and [`mtp/NOTICE`](mtp/NOTICE), with a notice of the
layout conversion. The component licenses apply separately. See the upstream
model cards for authorship and model limitations.
