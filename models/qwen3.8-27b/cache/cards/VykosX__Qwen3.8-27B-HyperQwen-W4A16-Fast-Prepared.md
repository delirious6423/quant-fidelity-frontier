---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- dbirks/Qwen3.8-27B-W4A16-AutoRound
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen3
- qwen3.8
- w4a16
- autoround
- gptq
- compressed-tensors
- vllm
- hyperqwen
- rtx-3090
---

# Qwen3.8-27B HyperQwen W4A16 Fast Prepared

This repository is the **assembled target-model artifact** produced by HyperQwen's documented fast-variant preparation flow for Qwen3.8-27B on a 24 GB RTX 3090-class GPU.

It exists so the prepared target can be downloaded as one reproducible Hugging Face snapshot instead of reconstructing it from a base quantization plus delta files on every machine.

## Provenance

- Base quantized target: `dbirks/Qwen3.8-27B-W4A16-AutoRound`
- Fast-variant delta/source: `syvai/qwen3.8-27b-3090-fast-variant`
- Preparation implementation: HyperQwen commit `3acb93f7acdcdbc3cfe67e2849e27537c0524b0a`
- Prepared with HyperQwen's `prepare/fetch_fast_variant.py` flow.

The separate speculative drafter used by the Club-3090 presets is **not bundled here**. It remains available from `syvai/Qwen3.8-27B-DFlash2-W4A16`.

## Club-3090 Server validation

This exact local artifact was validated on a single RTX 3090 through Club-3090 Server with three HyperQwen presets:

- BF16 KV fast lane: 65,536 max context
- KVarN K4/V2 huge-context lane: 245,760 max context
- KVarN K8/V4 high-fidelity lane: 163,840 max context

The model artifact is shared by all three; the runtime/KV-cache configuration determines the context lane.

## License

The source model and preparation inputs used for this assembled target are published under Apache-2.0. See the source repositories above for their original model cards and notices.
