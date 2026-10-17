---
library_name: transformers
license: other
license_name: swift-open-license-1.0
license_link: LICENSE
base_model:
  - ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound
  - RadixArk/Qwen3.8-Flash-Next-NVFP4
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - qwen3.8
  - swift
  - autoround
  - int4
  - fp8
  - vllm
  - speculative-decoding
  - 256k-context
---

# Swift 1.5 Qwen3.8-Flash-Next W4A16 + FP8 PLE + MTP3

This is UkisAI's [Swift 1.5 Qwen3.8-Flash-Next](https://huggingface.co/ukisai/Swift-Qwen3.8-Flash-Next)
AutoRound W4A16 checkpoint, repackaged to match the layout of
[`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE).
That layout is meant for the 2× RTX 3090 / 128 GB host runtime.

Swift's 99 GB BF16 PLE n-gram table is replaced with RadixArk's published FP8 table, and the bundled BF16 MTP is
replaced with a compact INT4 draft. No backbone tensor was requantized or repacked.

> **Status: experimental.** The checkpoint passes structural validation, but it has not yet been validated
> end-to-end on the target runtime. There are no speed or quality measurements for it yet. See
> [Differences from the Intel-based release](#differences-from-the-intel-based-release).

## Why the PLE table and draft can be reused

Swift's post-training changed the backbone only. Swift's BF16 PLE n-gram tensors and bundled MTP tensors
were compared against `Qwen/Qwen3.8-Flash-Next` with HTTP range reads:
- PLE: the byte ranges sampled from 15 of 137 PLE tensors all matched;
- MTP: all 31 tensors matched, either in the sampled ranges or in full.

This is sampled evidence, not a full hash of every tensor. The FP8 table (quantized by RadixArk from the base
model) and the INT4 draft (built from the base MTP tensors) therefore apply to Swift unchanged.

## Composition

| Component | Format | Source |
|---|---|---|
| Backbone: routed experts, attention and linear-attention projections | AutoRound W4A16, INT4 symmetric group-128 (`auto_round:auto_gptq`) | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound@177e8a382da63080bfd2875ab379fd8cb23e0dea` |
| Excluded layers: `lm_head`, visual, hyper-connection, PLE projections, indexer, shared experts, `in_proj_a/b`, router gate | BF16, unchanged | Swift checkpoint above |
| 51.2B-parameter n-gram/PLE table | FP8 E4M3FN plus published global scale | `RadixArk/Qwen3.8-Flash-Next-NVFP4@7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| MTP draft | Routed experts INT4 symmetric group-32; other tensors unchanged | `runtime/mtp-int4-g32`, built from the base MTP |

The target contains 223,028 indexed tensors in 60 safetensors files: 50 Swift shards and 10 FP8 PLE shards.
The tensor payload is 112.497 GiB. Swift shards `00002` and `00034` also held PLE tensors; they were rewritten
without them, and every kept tensor was checked byte for byte against the source. Shard numbering has gaps by
design. `model.safetensors.index.json` is the source of truth.

`hybrid_sources.json` records the provenance, and `NOTICE` lists every change made.

## Differences from the Intel-based release

- **More layers are INT4.** Intel's checkpoint keeps the attention (`q/k/v/o_proj`) and linear-attention
  (`in_proj_qkv`, `in_proj_z`, `out_proj`) projections in BF16. Swift quantized them.
  - These layers run on vLLM's generic GPTQ/Marlin linear path, not the BF16 path, which includes the
    runtime's Triton skinny-GEMM optimization.
  - Speed and quality numbers from the Intel-based release do not transfer.
- **Different quantization run.** Swift used auto-round 0.16.0 with 50 iterations and 128×512-token
  ultrachat calibration.
- **Draft acceptance may be lower.** The MTP draft was built from the base model's MTP. Swift's post-trained
  backbone may accept fewer draft tokens than the 86–91% measured on the base target. Output stays correct
  because the target verifies every token.

## Runtime

This is not a stock Transformers checkpoint. It needs the patched vLLM runtime and digest-pinned image
recorded in `runtime/repro.lock.json`. See `runtime/README.md` and
https://github.com/DominikBucko/qwen38-flash-next-2x3090. Point `MODEL_DIR` at this repository; the launcher
finds the draft at `runtime/mtp-int4-g32`.

Swift is tuned for shorter reasoning. For its evaluation results and recommended settings, see the
[upstream model card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound).

## License

- The Swift Contribution is licensed under the **Swift Open License v1.0** (`LICENSE`). Its commercial-use
  threshold is US$1M in gross revenue; above that, a separate Swift Enterprise License from UkisAI is required.
- The base-model portions, including the FP8 PLE table and the MTP draft, remain under the **Qwen Community
  License 1.0** (`LICENSE-QWEN`).
- See `NOTICE` for attribution and the full list of changes.

"Swift" and "UkisAI" are UkisAI's trademarks and are used here only to describe where the model comes from.
This repackaging is not affiliated with or endorsed by UkisAI or Qwen.
