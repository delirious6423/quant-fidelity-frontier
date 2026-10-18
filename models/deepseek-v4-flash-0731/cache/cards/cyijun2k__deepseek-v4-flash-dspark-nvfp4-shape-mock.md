---
license: mit
library_name: transformers
pipeline_tag: text-generation
base_model:
  - deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - deepseek-v4
  - mixture-of-experts
  - nvfp4
  - dspark
  - modelopt
  - vllm
  - mock
---

# DeepSeek-V4 Flash DSpark NVFP4 shape mock

> **Random integration fixture — not a useful language model.** The tensor
> values are deterministic random data or benign constants. Do not use this
> checkpoint for quality, safety, or performance evaluation.

This compact checkpoint reproduces the production tensor schemas needed to
exercise DeepSeek-V4 Flash integration without downloading or loading the full
model. It is intended for vLLM, FlashInfer B12X, sparse-attention, and DSpark
loader/kernel testing on Blackwell systems such as DGX Spark / GB10 (SM121).

## What is represented

- Three production-shape target transformer/MoE blocks (`layers.0` through
  `layers.2`). Three is the minimum compact layout that retains the first real
  sparse MLA compressor/indexer layer.
- All three production-shape DSpark/MTP draft stages (`mtp.0` through `mtp.2`).
- Production MoE dimensions: hidden size 4096, expert intermediate size 2048,
  256 routed experts, top-k 6.
- ModelOpt NVFP4 routed target experts with group size 16. Attention, shared
  experts, and MTP retain the source mixed-precision schemas.
- `hidden_act="silu"` and `swiglu_limit=10.0`.
- DSpark configuration: target layers `[0, 1, 2]`, Markov rank 256, block size
  5, and three draft stages.
- Sparse-attention compression ratios `[0, 0, 4, 0, 0, 0]`; layer 2 includes
  the real compressor and Lightning Indexer tensor layouts.

The checkpoint contains 14,025 tensors and 24,313,583,928 bytes of tensor data
(22.65 GiB).

## Schema validation

The generator pins
[`sakamakismile/DeepSeek-V4-Flash-0731-Abliterated-NVFP4`](https://huggingface.co/sakamakismile/DeepSeek-V4-Flash-0731-Abliterated-NVFP4)
at revision `4b15c44c31f8dacdf7b0c2fa42e20d03c5f5859b` for the DSpark-inclusive
schema. Header-only validation showed:

- all 14,025 local tensor names, shapes, and dtypes exactly match the selected
  source target and MTP tensors;
- the 9,320 non-MTP tensors also exactly match
  [`nvidia/DeepSeek-V4-Flash-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4)
  revision `e3cd60e7de98e9867116860d522499a728de1cf9` for global tensors and target
  layers 0–2;
- layer 2 contains 8 compressor tensors and 7 indexer tensors;
- MTP stage tensor counts are 1,568, 1,565, and 1,572.

`tensor_manifest.json` records every expected header, and
`generate_shape_mock.py` reproduces the fixture.

## GB10 / vLLM nightly validation

Test host: NVIDIA GB10, compute capability 12.1. The cached
`vllm/vllm-openai:nightly` image was started with:

```text
--moe-backend flashinfer_b12x
--linear-backend cutlass
--kv-cache-dtype fp8
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
```

The unmodified nightly correctly reproduced its current integration guard:

```text
Model sets swiglu_limit=10.0, but the explicitly requested
moe_backend='flashinfer_b12x' does not apply the SwiGLU clamp.
```

With a clamp-capable FlashInfer B12X checkout mounted and the stale vLLM
capability guard bypassed only inside the disposable test container, vLLM:

- selected `FLASHINFER_B12X` for the target NVFP4 experts;
- initialized the FP8 Lightning Indexer cache;
- loaded all eight checkpoint shards;
- loaded the DSpark draft model (`DSpark draft model loaded: 96 params`).

The stock nightly did not complete a forward pass on GB10 because its MHC
DeepGEMM path later failed with `Unsupported architecture`, before reaching the
MoE forward. That limitation is independent of this checkpoint's B12X clamp
schema and is recorded here to avoid overstating end-to-end validation.

## Intended use

Use this checkpoint to reproduce loader selection, tensor remapping, memory
planning, and kernel integration behavior. Since the weights are random, any
generated tokens are meaningless even when a runtime completes inference.
