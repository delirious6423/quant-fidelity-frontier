---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- hf2q
- q4_k_m
- phantom-kv
---

# Official Qwen3.8-27B — hf2q Q4_K_M + optional PhantomKV

Q4_K_M conversion of the official [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) checkpoint at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The model is the unmodified official checkpoint: no fine-tuning, ablation, or baked-in runtime steering. Quantization changes its numeric representation; it is not byte-identical to the upstream safetensors.

## Files and provenance

- `Qwen3.8-27B-Q4_K_M.gguf`: 16,810,714,976 bytes, SHA-256 `cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7`.
- `Qwen3.8-27B-Q4_K_M.gguf.receipt.json`: genuine adjacent conversion receipt from hf2q `0.1.26` commit `9c42dc1fc19b4dbe94c63c6c4bfef7f990c8f856`. All source, converter, and output-hash fields are preserved; only `output.path` is made repository-relative. Native receipt binding deliberately ignores that portable path, not the identity fields.
- `Qwen3.8-27B-phantom-kv.gguf`: optional, separate runtime-only trained K/V bank, 17,303,808 bytes, SHA-256 `e33df498ad4abe47b7d98bf0228dc96cd629db15ec37574c2514ef88f4656ffb`. It does not replace or modify the base model weights. It was trained from the same official checkpoint for 400 steps with seed 1337 using the [phantom-kv](https://github.com/lordx64/phantom-kv) method/toolchain. The direct-K/V F32 container holds 132 slots across 16 full-attention layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63. Unbinding the bank and starting fresh runtime state removes the graft.
- `manifest.json` records checkpoint identity, artifacts, runtime dependencies, and licensing. `SHA256SUMS` covers all release files except itself. Raw training prompts, datasets, logs, and checkpoints are not distributed.

The graft concept, container, and producer/training method are credited to **lordx64 and phantom-kv contributors (MIT)**; **hf2q** supplies the conversion and independent native reader/runtime integration. See `NOTICE` and `PHANTOM-KV-LICENSE`.

## Optional GLP dependency — not redistributed

GLP stays upstream and is not included in this repository. The external GLP for the exact same official source revision is pinned to:

[Qwen3.8-27B-abliterated-cyber-GLP-49-L10-58-a1.gguf](https://huggingface.co/msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49/resolve/5f1be31cfde653f24fe74c71047433012b8ede9e/Qwen3.8-27B-abliterated-cyber-GLP-49-L10-58-a1.gguf)

SHA-256: `93c0742d597a77f9aea67298ef0abe06675da69bf8f7fa46c79c0365728f31d2`; alpha: `1`. Its actual file/license is MIT while the upstream repository metadata says Apache-2.0; this discrepancy is noted rather than silently resolved or relicensed. Consult the upstream file and repository terms independently. GLP, GCD, and PhantomKV are runtime controls, not model-weight modifications.

## Limitations and licensing

Historical MPS reference results (49/60 to 1/60 refusals; 0/20 harmless refusals) are limited probe results on that reference path, **not a native-runtime quality qualification, coding benchmark, or general capability/safety guarantee**. Native coding checks are a separate release gate; this card makes no passing claim. Grafts and runtime steering can degrade capability and change behavior. Evaluate the exact combined runtime configuration for your workload.

The official model retains its upstream **Apache-2.0** license and Alibaba Cloud attribution, supplied unchanged as `LICENSE`. The authorized operator-owned derived bank is distributed under **Apache-2.0**; the producer's MIT license and attribution remain preserved in `PHANTOM-KV-LICENSE` and `NOTICE`. No upstream GLP is redistributed or relicensed here. This is an independent conversion/runtime-artifact release, not an endorsement by Qwen, Alibaba Cloud, or the producer authors.
