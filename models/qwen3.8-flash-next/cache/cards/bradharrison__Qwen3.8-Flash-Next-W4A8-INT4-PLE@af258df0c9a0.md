---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: transformers
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- gptq
- w4a8
- quantized
- ple
- amd
---

# Qwen3.8-Flash-Next W4A8 (int4 PLE sidecar included)

GPTQ **W4A8** quantization of the **BF16 OEM release of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** (`qwen4_exp` architecture): 4-bit weights, int8 activations, symmetric, group size 128, `desc_act=false`, true-sequential, activation-group-aware. Quantized with `gptqmodel 7.6.0` on a custom calibration corpus. Tokenizer, chat template and configs are carried over unchanged from the OEM checkpoint.

## What changed vs the OEM checkpoint

- The bf16 PLE n-gram embedding table (**95.4 GiB**, upstream `model-00002-of-00020.safetensors`) is **removed** from the checkpoint. It ships instead as a group-16 **int4 PLE sidecar** in [`ple-int4/`](ple-int4/) (128 shards + META.json, ~30 GB): group-16 RTN, fp16 scales, fixed zero-point 8, low-nibble-first — byte-compatible with the PLE CPU-offload worker contract.
- The small per-layer PLE projection tensors remain in-line in the model shards (they were never part of the externalized table).
- 19 safetensors shards, ~75 GB payload.

## Scope of quantization

All 74,028 language-model linears are W4A8 (83.8% of payload bytes are packed 4-bit weights
+ 2.6% fp16 scales). Left at bf16, by design (~16% of payload):

- **MTP drafter module** (~5.2 GB) — kept full precision for draft acceptance;
- **embed_tokens + lm_head** (~2.5 GB, 248320x2560 each);
- **vision tower + merger** (~0.9 GB);
- norms/biases and the 9 small PLE projection tensors.

## Serving

Requires a vLLM build with `qwen4_exp` + PLE CPU-offload support (rdna2 PLE worker). Point the worker at the sidecar:

    VLLM_PLE_QUANT_DIR=<repo>/ple-int4

Sidecar validation, two independent ways:
1. this int4 vs the published `primitive-ai/Qwen3.8-Flash-Next-PLE-quant` (AWQ-derived): mean 0.20 / worst 1.20 steps divergence over 3072 rows × 48 shards — encoder-boundary noise, i.e. the same table;
2. the published int4 decoded against our bf16 ground truth: **48/48 shards in band**.

## License

Inherits the OEM release's license: **qwen-community-1.0** (see [LICENSE](LICENSE)).

## Credits

Quantized by Brad Harrison. PLE int4 quantizer written against the rdna2 PLE-offload worker contract.
