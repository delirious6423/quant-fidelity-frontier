---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: llama.cpp
pipeline_tag: text-generation
language:
  - en
tags:
  - moe
  - mixture-of-experts
  - gguf
  - qwen
  - llama.cpp
  - quantization
---

# MoEMe-27B — exact Top-12 sparse-MoE conversion of Qwen3.8-27B

Two GGUF builds of a **mechanically converted sparse-MoE** version of
[Qwen3.8-27B](https://huggingface.co/Qwen). The dense SwiGLU feed-forward
networks were re-partitioned into a MoE layout by checkpoint surgery. These are
**exact Top-12 conversions** — all 12 routed experts are active, so the model is
functionally the source dense model, reorganized and quantized. It is **not** a
new or improved model, and it is **not** a quality-preserving sparse model.

This weights release accompanies the source release and the experiment
write-up: **the conversion works, the sparsity does not**. Read the repository
before using this for anything serious.

- Code: https://github.com/trowel344/moeme

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `moeme-27b-top12-imatrix-q5_k_m.gguf` | 18.53 GiB | **Recommended.** Passes the full 16/16 capability suite and all source-logit parity gates. ~3.2–3.3 tok/s on an RTX 4060 Laptop (8 GB). |
| `moeme-27b-top12-imatrix-q4_k_m.gguf` | 16.15 GiB | Faster (~3.8 tok/s), smaller, but misses the mean-KLD parity gate by `0.000102` (`0.020102` vs `0.02`). Use if memory-constrained. |

SHA-256:

```
9996f4b352fe2c7016ecb675d11deb4e5eef99456b510478dc01b853bddffd38  moeme-27b-top12-imatrix-q5_k_m.gguf
3797152bf530563ea7787162f2f1bfd4cf2e9cf1e780b041d21b2a0a56551f94  moeme-27b-top12-imatrix-q4_k_m.gguf
```

## What was changed

Qwen3.8-27B has `num_hidden_layers = 64` (48 linear-attention + 16 full-attention),
`hidden_size = 5120`, FFN `intermediate_size = 17408`, vocab `248320`, native
context `262144`. Each dense FFN intermediate dimension `F = 17408` was split into
a disjoint partition: `F/4 = 4352` shared channels plus `12 × F/16 = 1088` routed
channels (`4352 + 12 × 1088 = 17408`). Nothing was added; channels were
reassigned. This is the root cause of the whole experiment's failure: Top-K
routing **deletes network width** instead of selecting redundant, independently
trained expert capacity.

## Requirements — patched llama.cpp

These GGUFs need a llama.cpp build that (a) loads `qwen35moe.expert_weights_scale`
in the `Qwen35MoE` loader and (b) honours the per-layer active-K override env vars
(`MOEME_ACTIVE_K`, `MOEME_ACTIVE_K_LAYERS`). A stock build will load the file but
produce wrong logits. The required one-line loader patch is included in the
source repository under `docs/patches/`.

## Run

```bash
llama-server \
  -m moeme-27b-top12-imatrix-q5_k_m.gguf \
  -c 8192 -ngl 20 -fa \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --ctx-checkpoints 2 --fit off
```

The host used for the published benchmarks was an **RTX 4060 Laptop (8 GB VRAM),
15.7 GiB RAM, 16 cores**. The flags above were measured on that host; adjust for
your hardware. Context checkpoints reserve significant RAM, so keep
`--ctx-checkpoints` low on small machines.

## Honest limitations

- **No real sparse speedup.** Only ~`49.4%` of the artifact bytes live in routed
  experts; the other ~`50.6%` is always active. The theoretical ceiling is
  `1.96x`; the measured Top-4 speedup is `1.42x`. On a bandwidth-bound single
  device, routing reduces FLOPs, not bytes moved.
- **Sparsity destroys quality.** The calibrated quality budget is a per-layer
  relative-L2 error of ~`0.01`. The static greedy-oracle floor at Top-4 has a
  **median of `0.45`** across the 64 layers (best layer `0.16`, worst `0.53`).
  No layer clears the real budget at any `K < 12`.
- **Not a chat-tuned model.** It is the source model's weights, reorganized;
  behavior and safety characteristics are inherited from Qwen3.8-27B.

## License & credits

Apache-2.0, inherited from the base model. Built on Qwen3.8-27B (Apache-2.0) and
llama.cpp (MIT).

If you have a real compute budget and want to attempt the co-activation
clustering / upcycling / sparse-aware-training route this hardware could not
reach, please do — that is why this was released.
