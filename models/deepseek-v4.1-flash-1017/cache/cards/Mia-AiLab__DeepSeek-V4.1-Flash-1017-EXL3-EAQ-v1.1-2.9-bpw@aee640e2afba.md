---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
  - exl3
  - exllamav3
  - eaq
  - quantized
  - moe
  - deepseek
  - multimodal
  - experimental
---

# DeepSeek-V4.1-Flash-1017 · EXL3 · EAQ v1.1 · 2.9 bpw · EXPERIMENTAL

EAQ v1.1 quantization of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (revision `1bae0e10`). It has the same 2.9 bpw body as EAQ v1, with one change: **the sparse-attention indexer path is stored at source precision** instead of being quantized.

> **EXPERIMENTAL.** This is an experiment, not the recommended pack. EAQ v1 stays the one to use.

## Indexer advantage

The indexer chooses which earlier tokens attention is allowed to read. v1.1 keeps that path at source precision. Given the original model's hidden state, it selects the same tokens as the original on every measured row: **403,560 of 403,560** at 16k tokens and above. v1 matches **94.95%** of that set. On the hidden state each pack produces itself, the two are about equal (recall@512 about 0.73). Positions past 32k were not compared.

This is **not** a drop-in replacement for the original FP8/FP4 checkpoint. The Engram embedding tables are **not** in this repo: you still need shards 47 and 48 from the base model (about 95 GiB each).

## What is different from v1

DeepSeek's sparse attention uses an indexer to pick which tokens each attention head reads. Errors in the indexer change *what the model looks at*, so v1.1 keeps this path at source precision on the eight indexer layers (2, 8, 14, 20, 24, 28, 32, 36) and on the compressor layers that feed it:

| Tensor | Layers | v1 | **v1.1** |
|---|---|---|---|
| `attn.wq_a` (shared q down-projection) | 2, 8, 14, 20, 24, 28, 32, 36 | 5 bit | **BF16, exact source values** |
| `attn.indexer.wq_b` | same 8 layers | 5 bit | **BF16, exact source values** |
| `attn.indexer.wk` | 2, 8, 14, 20 | 8 bit | **BF16, exact source values** |
| `attn.compressor.wkv` | 2, 8, 14, 20 | 5 bit | **BF16, exact source values** |
| `attn.compressor.wgate` | 2, 8, 14 | 5 bit | **BF16, exact source values** |
| `attn.indexer.weights_proj` | the 8 index layers | BF16 | BF16 (unchanged) |

That is 35 tensors in total, 27 of which were quantized in v1. Everything else, including the routed experts at 3 bit (layers 18-22 at 2 bit), the shared experts, the other attention weights, the routers, the head (6 bit) and the MTP / DSpark blocks (4 bit), uses the same widths as v1.

## At a glance

| | |
|---|---|
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Format | EXL3 (`trellis` / `suh` / `svh` / `mul1`), codebook **mul1**, `out_scales=always`; 27 modules are stored as plain BF16 linears inside the pack |
| Average bpw | **2.90** over the quantized body (same body widths as v1) |
| Head bits | 6 |
| MTP / DSpark bits | 4 |
| Size | 210,735,685,708 bytes of tensors in 39 shards (v1: 210,580,703,416; +154,982,292) |
| Tensors | 192,414 (47,873 trellis tensors) |
| Vision | Included (tower left native) |
| Engram tables | **Not quantized.** Use original shards **47** and **48** of the base model |

## Engine requirement

v1.1 contains BF16 linears inside an EXL3 pack (27 float linears; the 8 `weights_proj` tensors are plain BF16 tensors as in v1). An engine must be able to load float linears in an EXL3 pack. TensorFold with a mixed-pack loader prints `mixed pack: 27 float linears` for this checkpoint. Stock exllamav3 builds that expect every linear to be quantized are **NOT TESTED** with this pack.

## Checks on the finished pack (MEASURED)

All of these were run on the final files:

- **Index and shards:** 192,414 keys in the index equal 192,414 in the shard headers; 39 shards.
- **Key set vs v1:** identical to v1 apart from the 27 modules whose `trellis`/`suh`/`svh`/`mul1` parts are replaced by one `.weight` each.
- **Dtype and shape vs v1:** 0 mismatches; exactly 35 tensors changed, all BF16.
- **Widths:** 47,873 trellis tensors, all matching the intended per-tensor bit widths.
- **Exact tensors:** all 35 tensors equal the source: 19 byte-identical (BF16 sources), 16 equal to the independent FP32 dequantization of the FP8 + E8M0 sources (exact in BF16).
- **Size:** tensor bytes equal the index `total_size`, and equal the v1 total plus the expected 154,982,292 bytes.
- **Config:** `config.json` identical to v1; `quantization_config.json` differs only in the 35 expected modules.

## What was NOT tested

- Recall on the 315 frozen long-context haystacks, the fresh fragile-prompt set, and the port answer-token NLL gate.
- Selection agreement beyond 32k tokens.
- Generation quality beyond the checks above, and speed.
- Loading in any engine other than the one used for these tests.

## Files

`model-0000N-of-00039.safetensors`, `model.safetensors.index.json`, `config.json`, `quantization_config.json`, tokenizer files, `LICENSE`.

## License and credits

This is a quantized derivative of DeepSeek-V4.1-Flash-1017 and does not change its license (MIT, as listed by the base model). Quantization format and kernels are from [exllamav3](https://github.com/turboderp-org/exllamav3).
