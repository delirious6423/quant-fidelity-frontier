---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - mlx
  - mlx-vlm
  - omlx
  - oq
  - qwen
  - qwen3.8
  - mixture-of-experts
  - vision-language
  - quantized
  - apple-silicon
  - 6-bit
---

<p align="center">
  <a href="https://tensorfold.dev">
    <img src="https://huggingface.co/spaces/TensorFold/README/resolve/main/tensorfold-logo.png" alt="TensorFold" width="160">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Qwen-Qwen3.8-615CED?style=for-the-badge&logo=qwen&logoColor=white" alt="Qwen Qwen3.8">
  <img src="https://img.shields.io/badge/Apple_Silicon-MLX-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Apple silicon MLX">
  <img src="https://img.shields.io/badge/TensorFold-oQ-6E56CF?style=for-the-badge&logo=huggingface&logoColor=white" alt="TensorFold oQ">
</p>


<h1 align="center">Qwen3.8 Flash Next — MLX oQ6</h1>


<p align="center">
  A mixed-precision MLX conversion of <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen/Qwen3.8-Flash-Next</a>, quantised directly from the official BF16 checkpoint.
</p>


<p align="center">
  <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Original model</a> ·
  <a href="https://qwen.ai/blog?id=qwen3.8-flash-next">Qwen overview</a> ·
  <a href="https://github.com/ml-explore/mlx-vlm">MLX-VLM</a> ·
  <a href="LICENSE">Qwen Community License 1.0</a>
</p>


## About this conversion


This repository contains an oQ6 mixed-precision MLX conversion produced directly from Qwen's BF16 weights. Layer sensitivity was measured with a validated 4-bit proxy; final tensor quantisation reread the official BF16 checkpoint. Group size 32 supports the model's 160-wide hashed n-gram embedding tables.


| Item | Value |
| --- | --- |
| Base model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Format | MLX safetensors |
| Quantisation | oQ6 mixed precision; 6-bit affine base with protected modules at 6/8-bit |
| Base group size | 32 |
| Weight shards | 31 |
| Weight size | 155.78 GB (145.08 GiB) |
| Configured context | 262,144 tokens |
| Architecture | `qwen4_exp` vision-language sparse MoE |


The upstream tokenizer, chat template, vision processor, and generation configuration are preserved. This base release does not include the optional native MTP head; an explicitly named MTP build requires matching `qwen4_exp` MTP runtime support and is being handled separately.


> [!IMPORTANT]
> Qwen3.8 Flash Next uses the new `qwen4_exp` architecture. Use an oMLX or MLX-VLM build that explicitly lists `qwen4_exp` support. Older MLX-VLM releases cannot load this checkpoint.


> [!CAUTION]
> Do not attach a Qwen3.8 27B MTP drafter to this model. The hidden sizes differ and the drafter is incompatible with Flash Next.


## Quick start


```bash
hf download TensorFold/Qwen3.8-Flash-Next-MLX-oQ6 \
  --local-dir Qwen3.8-Flash-Next-MLX-oQ6
```


With a compatible MLX-VLM runtime:


```bash
python -m mlx_vlm.generate \
  --model Qwen3.8-Flash-Next-MLX-oQ6 \
  --prompt "Explain sparse mixture-of-experts routing." \
  --max-tokens 512
```


## Measured performance


Validated on an Apple M3 Studio with deterministic text-only generation after model load:


| Test path | Result |
| --- | ---: |
| Standalone MLX exact-copy smoke test | 23.1 tokens/s |
| Standalone MLX, 142-token explanatory response | 21.1 tokens/s |
| oMLX server, warmed 512-token response | 19.5 tokens/s |


The first oMLX request reported 22.77 seconds to load the model; that one-off load time is separate from generation speed. The 512-token server run is the most representative sustained result. Results vary with prompt length, cache state, sampling settings, runtime version, and memory pressure.


## Architecture


Qwen3.8 Flash Next combines Gated DeltaNet, Qwen Sparse Attention, sparse mixture-of-experts layers, widened gated residual streams, and hashed bigram/trigram embeddings.


| Architecture detail | Upstream value |
| --- | ---: |
| Language-model parameters | 125B total / 6B active |
| N-gram embedding | 51B parameters |
| Layers | 48 |
| Routed / active experts | 512 / 10, plus 1 shared |
| Attention heads / KV heads | 24 / 2 |
| Hidden size | 2,560 |
| Native configured context | 262,144 tokens |


For upstream evaluations, intended use, limitations, safety guidance, and the complete architecture discussion, see the [original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).


## Conversion and validation


- Source: official BF16 checkpoint.
- Sensitivity-guided mixed-precision allocation: 6-bit base with 228 protected modules at 6/8-bit.
- All 3,671 converted tensors and 31 indexed shards were checked locally.
- Deterministic exact-copy, explanatory, and sustained 512-token generation tests passed on Apple silicon.
- The release payload was scanned for credentials, personal contact details, private paths, private network information, logs, caches, and private organisation data.


This is a community conversion, not an official Qwen release.


## License and attribution


The upstream model is released under the **Qwen Community License 1.0**. The required licence text is included in this repository.


Model design, training, evaluations, and upstream documentation belong to Qwen and the original contributors. The MLX conversion, Apple-silicon validation, and packaging are provided by [TensorFold](https://huggingface.co/TensorFold).



<!-- TensorFold-chooser-start -->
## Choose for your Mac

[64GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-64gb-macs-6a9fefda17932216ec9ab457) · [128GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-128gb-macs-6a9ff0abd31bc9abbe7922d7) · [256GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-256gb-macs-6a9ff0ef9fed7c5bdca15e9b)

No measured memory tier is assigned here. The collections use published M3 Studio peaks with at least 25% nominal headroom; fit on other Macs is an estimate, and full context is not guaranteed. Start with short context and one request.

### Runtime and evidence

The exact tested oMLX application version is not recorded here; a library version is not an app version. The original performance tables retain their benchmark conditions and speed figures; this documentation update adds no new test results.

### Quick start and demo prompt

```bash
hf download TensorFold/Qwen3.8-Flash-Next-MLX-oQ6 --local-dir ./models/Qwen3.8-Flash-Next-MLX-oQ6
```

Add the downloaded folder to oMLX model directories, refresh the list, and follow this card's architecture and MTP compatibility requirements before loading.

Try this in a new chat with a 128-token output limit:

```text
Explain why the sky looks blue in three short sentences.
```

This is a demo prompt to try, not a recorded successful run; a captured demonstration for this documentation update is not yet available.

[Follow TensorFold for new Apple Silicon releases and fixes.](https://huggingface.co/TensorFold)
<!-- TensorFold-chooser-end -->
