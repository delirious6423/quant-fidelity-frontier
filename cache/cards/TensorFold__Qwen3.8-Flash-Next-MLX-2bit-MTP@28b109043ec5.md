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
  - mtp
  - speculative-decoding
  - qwen
  - qwen3.8
  - qwen4-exp
  - mixture-of-experts
  - vision-language
  - quantized
  - apple-silicon
  - 2-bit
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
  <img src="https://img.shields.io/badge/TensorFold-2_bit-6E56CF?style=for-the-badge&logo=huggingface&logoColor=white" alt="TensorFold 2-bit">
</p>


<h1 align="center">Qwen3.8 Flash Next, MLX 2-bit with native MTP</h1>


<p align="center">
  A compact MLX affine conversion of <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen/Qwen3.8-Flash-Next</a> with sensitive language paths kept in BF16 and the model's own MTP block preserved.
</p>


<p align="center">
  <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Original model</a> ·
  <a href="https://qwen.ai/blog?id=qwen3.8-flash-next">Qwen overview</a> ·
  <a href="https://github.com/ml-explore/mlx-vlm">MLX-VLM</a> ·
  <a href="LICENSE">Qwen Community License 1.0</a>
</p>


## About this conversion


This is a non-sensitivity MLX conversion. It is not uniform Q2 and it is not an oQ build. Routed and shared experts plus the predictive n-gram embedding paths use affine 2-bit weights at group size 32. Attention, Gated DeltaNet, hyperconnection, routing, vision, token embedding, output head, and native MTP weights remain in BF16 or their source-compatible dtype.


| Item | Value |
| --- | --- |
| Repository | `TensorFold/Qwen3.8-Flash-Next-MLX-2bit-MTP` |
| Base model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Source revision | `f5d08274bafd880402bd16f5e3e6c514136ec06c` |
| Source precision | Official BF16 checkpoint |
| Format | MLX safetensors |
| Quantisation | Affine 2-bit, group size 32, with explicit BF16 exclusions |
| 2-bit modules | 418 |
| Native MTP | Included, one matching Qwen4Exp draft block in BF16 |
| Indexed tensors | 2,543 total, including 32 native-MTP entries |
| Weight shards | 20 |
| Weight size | 80.070 GB / 74.571 GiB |
| Configured context | 262,144 tokens |
| Architecture | `qwen4_exp` vision-language sparse MoE |


The upstream tokenizer, current chat template, image and video processor configuration, generation configuration, licence, and native MTP configuration are included.


> [!IMPORTANT]
> Use a runtime with explicit `qwen4_exp` and native-MTP support. This release was validated with oMLX 0.6.3rc3 build 2475, MLX 0.32.0, and MLX-VLM 0.6.3.


## Download and use


```bash
python -m pip install --upgrade huggingface_hub


hf download TensorFold/Qwen3.8-Flash-Next-MLX-2bit-MTP \
  --local-dir ./Qwen3.8-Flash-Next-MLX-2bit-MTP
```


Add the downloaded directory to a compatible oMLX model directory and refresh the model registry. Native MTP is optional. Keep it disabled by default for this release because the measured native path was slightly slower than baseline.


## Apple M3 Studio performance


The validation used three measured 512-token runs per mode after warm-up. The table reports median generation throughput from the exact release checkpoint.


| Runtime mode | Runs | Output per run | Median generation speed |
| --- | ---: | ---: | ---: |
| Native MTP disabled | 3 | 512 tokens | **29.3729 tokens/s** |
| Native MTP enabled | 3 | 512 tokens | **28.6199 tokens/s** |


Native MTP changed median throughput by **-2.56%** in this test. The MTP telemetry sample accepted 9 of 17 reported draft proposals, an acceptance rate of **52.94%**. Every MTP-off and MTP-on 512-token run produced the same output hash.


Instruction following, factual recall, arithmetic, concise response, and coherent-generation gates passed. Native MTP worked correctly and preserved greedy output, but it did not improve throughput on this checkpoint and runtime. The recommended default is MTP disabled.


The benchmark covers text generation. Results vary with prompt length, context growth, cache state, runtime version, memory pressure, and thermal conditions.


## Architecture


Qwen3.8 Flash Next combines Gated DeltaNet, Qwen Sparse Attention, sparse mixture-of-experts layers, widened gated residual streams, hashed bigram and trigram embeddings, and a native next-token-prediction block for speculative decoding.


| Architecture detail | Upstream value |
| --- | ---: |
| Language-model parameters | 125B total / 6B active |
| N-gram embedding | 51B parameters, 20,000,000 entries |
| Native MTP | 4B parameters, one draft layer |
| Hidden size | 2,560 |
| Layers | 48 |
| Routed / active experts | 512 / 10, plus 1 shared expert |
| Native context | 262,144 tokens, extensible upstream to 1,000,000 |


For upstream evaluations, intended use, safety guidance, and the full architecture discussion, see the [original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).


## Conversion and validation


- Converted directly from the official BF16 checkpoint.
- Quantised 418 expert and predictive-embedding modules to affine Q2 at group size 32.
- Preserved 762 sensitive or structural matrix entries in BF16, including the complete matching native MTP block.
- Verified all 2,543 indexed tensors and all 20 shards before upload.
- Loaded the checkpoint in oMLX and passed deterministic instruction, factual, arithmetic, concise-writing, and coherent-generation tests.
- Ran three 512-token measurements in each MTP mode with exact paired output parity.


This is a community conversion, not an official Qwen release.


## Limitations


- The 2-bit expert allocation is aggressive. Evaluate accuracy and visual understanding on the intended workload before deployment.
- Native MTP was 2.56% slower in the measured test. Acceptance alone does not guarantee a speedup.
- The configured 262,144-token context does not guarantee that every Apple-silicon system can run that length within available unified memory.
- Text generation was benchmarked. The vision stack loaded successfully, but visual quality was not benchmarked for this card.
- This is an MLX checkpoint for Apple silicon. It is not a GGUF, CUDA, TensorRT-LLM, or vLLM checkpoint.
- Upstream model limitations and safety considerations still apply.


## Licence and attribution


The upstream model is released under the **Qwen Community License 1.0**. The required licence text is included in this repository and should be reviewed before use or redistribution.


Model design, training, evaluations, and upstream documentation belong to Qwen and the original contributors. The MLX conversion, native-MTP preservation, Apple-silicon validation, and packaging are provided by [TensorFold](https://huggingface.co/TensorFold).



<!-- TensorFold-chooser-start -->
## Choose for your Mac

[64GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-64gb-macs-6a9fefda17932216ec9ab457) · [128GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-128gb-macs-6a9ff0abd31bc9abbe7922d7) · [256GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-256gb-macs-6a9ff0ef9fed7c5bdca15e9b)

No measured memory tier is assigned here. The collections use published M3 Studio peaks with at least 25% nominal headroom; fit on other Macs is an estimate, and full context is not guaranteed. Start with short context and one request.

### Runtime and evidence

oMLX version mentioned in the existing card: **0.6.3rc3**; consult its compatibility notes for whether this was tested and any required integrations. The original performance tables retain their benchmark conditions and speed figures; this documentation update adds no new test results.

### Quick start and demo prompt

```bash
hf download TensorFold/Qwen3.8-Flash-Next-MLX-2bit-MTP --local-dir ./models/Qwen3.8-Flash-Next-MLX-2bit-MTP
```

Add the downloaded folder to oMLX model directories, refresh the list, and follow this card's architecture and MTP compatibility requirements before loading.

Try this in a new chat with a 128-token output limit:

```text
Explain why the sky looks blue in three short sentences.
```

This is a demo prompt to try, not a recorded successful run; a captured demonstration for this documentation update is not yet available.

[Follow TensorFold for new Apple Silicon releases and fixes.](https://huggingface.co/TensorFold)
<!-- TensorFold-chooser-end -->
