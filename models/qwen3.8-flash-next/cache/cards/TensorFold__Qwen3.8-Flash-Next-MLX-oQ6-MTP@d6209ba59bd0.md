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
  - mtp
  - speculative-decoding
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
  <a href="https://qwen.ai/"><img src="qwen-logo.png" width="96" height="95" alt="Qwen"></a>
</p>


<p align="center">
  <img src="https://img.shields.io/badge/Qwen-Qwen3.8-615CED?style=for-the-badge&logo=qwen&logoColor=white" alt="Qwen Qwen3.8">
  <img src="https://img.shields.io/badge/Apple_Silicon-MLX-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Apple silicon MLX">
  <img src="https://img.shields.io/badge/Native_MTP-Included-22C55E?style=for-the-badge" alt="Native MTP included">
  <img src="https://img.shields.io/badge/TensorFold-oQ6-6E56CF?style=for-the-badge&logo=huggingface&logoColor=white" alt="TensorFold oQ6">
</p>


<h1 align="center">Qwen3.8 Flash Next — MLX oQ6 with native MTP</h1>


<p align="center">
  A mixed-precision MLX conversion of <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen/Qwen3.8-Flash-Next</a>, quantised directly from the official BF16 checkpoint with its native MTP draft block preserved.
</p>


<p align="center">
  <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Original model</a> ·
  <a href="https://qwen.ai/blog?id=qwen3.8-flash-next">Qwen overview</a> ·
  <a href="https://github.com/ml-explore/mlx-vlm">MLX-VLM</a> ·
  <a href="LICENSE">Qwen Community License 1.0</a>
</p>


## About this conversion


This repository contains an oQ6 mixed-precision MLX conversion produced directly from Qwen's BF16 weights. Layer sensitivity was measured with a validated local 4-bit proxy. Every final tensor, including the native MTP head, was rebuilt from the official BF16 checkpoint. Group size 32 supports the model's 160-wide hashed n-gram embedding tables.


| Item | Value |
| --- | --- |
| Repository | `TensorFold/Qwen3.8-Flash-Next-MLX-oQ6-MTP` |
| Base model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Format | MLX safetensors |
| Quantisation | oQ6 mixed precision, 6-bit affine base with protected modules at 6/8-bit |
| Base group size | 32 |
| Native MTP | Included, one Qwen4Exp draft block |
| Weight tensors | 3,747 total, including 76 converted MTP tensors |
| Weight shards | 31 |
| Weight size | 158.07 GB, 147.21 GiB |
| Configured context | 262,144 tokens |
| Architecture | `qwen4_exp` vision-language sparse MoE |


The upstream tokenizer, chat template, vision processor, generation configuration, and native MTP configuration are preserved.


> [!IMPORTANT]
> This checkpoint requires an oMLX or MLX-VLM runtime with explicit `qwen4_exp` native-MTP support. Stock runtimes that do not construct the Qwen4Exp MTP module may reject the 76 MTP tensors during strict weight loading.


> [!CAUTION]
> Do not attach a Qwen3.8 27B drafter to this model. Flash Next has different hidden dimensions and ships its own matching MTP block here.


## Download and use


```bash
hf download TensorFold/Qwen3.8-Flash-Next-MLX-oQ6-MTP \
  --local-dir Qwen3.8-Flash-Next-MLX-oQ6-MTP
```


Add the downloaded directory to an MTP-capable oMLX installation, select the model, and enable native MTP in its model settings.


## Apple M3 Studio performance


Validated with native MTP enabled in oMLX on an Apple M3 Studio:


| Test | Output tokens | Speed |
| --- | ---: | ---: |
| Raw completion, warmed | 32 | 29.6 tokens/s |
| Exact-instruction chat | 27 | 28.3 tokens/s |
| Casual chat | 68 | 36.6 tokens/s |


The two deterministic raw-completion runs produced identical output. The exact-instruction test returned `hello`, and the longer chat completed coherently without repetition or cache-reconciliation errors. Native MTP remained active throughout, with 70.8–84.2% draft acceptance across these runs.


For scale only, the earlier non-MTP oQ6 backbone reached 23.1 tokens/s in a short standalone MLX test, 21.1 tokens/s over a 142-token response, and 19.5 tokens/s over a warmed 512-token oMLX response on an Apple M3 Studio. Those figures are a backbone reference, not an MTP speed claim for this repository.


Results vary with prompt length, cache state, sampling settings, runtime version, and memory pressure.


## Architecture


Qwen3.8 Flash Next combines Gated DeltaNet, Qwen Sparse Attention, sparse mixture-of-experts layers, widened gated residual streams, hashed bigram and trigram embeddings, and a native next-token prediction block for speculative decoding.


| Architecture detail | Upstream value |
| --- | ---: |
| Language-model parameters | 125B total / 6B active |
| N-gram embedding | 51B parameters |
| Layers | 48 |
| Routed / active experts | 512 / 10, plus 1 shared |
| Attention heads / KV heads | 24 / 2 |
| Hidden size | 2,560 |
| Native configured context | 262,144 tokens |
| Native MTP draft blocks | 1 |


For upstream evaluations, intended use, limitations, safety guidance, and the complete architecture discussion, see the [original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).


## Conversion and validation


- The converter read the official BF16 checkpoint directly.
- Sensitivity-guided allocation used a 6-bit base and 232 protected or overridden modules.
- Structural validation checked all 3,747 indexed tensors and all 31 shards.
- MTP validation confirmed one configured draft layer and 76 converted MTP tensor entries.
- Live generation validation covered deterministic completion, exact-instruction chat, coherent chat, MTP telemetry, and cache rollback.


This is a community conversion, not an official Qwen release.


## License and attribution


The upstream model is released under the **Qwen Community License 1.0**. The required licence text is included in this repository.


Model design, training, evaluations, and upstream documentation belong to Qwen and the original contributors. The MLX conversion and packaging are provided by [TensorFold](https://huggingface.co/TensorFold).



<!-- TensorFold-chooser-start -->
## Choose for your Mac

[64GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-64gb-macs-6a9fefda17932216ec9ab457) · [128GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-128gb-macs-6a9ff0abd31bc9abbe7922d7) · [256GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-256gb-macs-6a9ff0ef9fed7c5bdca15e9b)

No measured memory tier is assigned here. The collections use published M3 Studio peaks with at least 25% nominal headroom; fit on other Macs is an estimate, and full context is not guaranteed. Start with short context and one request.

### Runtime and evidence

The exact tested oMLX application version is not recorded here; a library version is not an app version. The original performance tables retain their benchmark conditions and speed figures; this documentation update adds no new test results.

### Quick start and demo prompt

```bash
hf download TensorFold/Qwen3.8-Flash-Next-MLX-oQ6-MTP --local-dir ./models/Qwen3.8-Flash-Next-MLX-oQ6-MTP
```

Add the downloaded folder to oMLX model directories, refresh the list, and follow this card's architecture and MTP compatibility requirements before loading.

Try this in a new chat with a 128-token output limit:

```text
Explain why the sky looks blue in three short sentences.
```

This is a demo prompt to try, not a recorded successful run; a captured demonstration for this documentation update is not yet available.

[Follow TensorFold for new Apple Silicon releases and fixes.](https://huggingface.co/TensorFold)
<!-- TensorFold-chooser-end -->
