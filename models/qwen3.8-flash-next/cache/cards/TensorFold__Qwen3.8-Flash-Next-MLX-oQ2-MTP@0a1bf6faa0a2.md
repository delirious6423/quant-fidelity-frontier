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
  <img src="https://img.shields.io/badge/TensorFold-oQ2-6E56CF?style=for-the-badge&logo=huggingface&logoColor=white" alt="TensorFold oQ2">
</p>


<h1 align="center">Qwen3.8 Flash Next, MLX oQ2 with native MTP</h1>


<p align="center">
  A sensitivity-aware mixed-precision MLX conversion of <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen/Qwen3.8-Flash-Next</a>, quantised from the official BF16 checkpoint with its native MTP draft block preserved.
</p>


<p align="center">
  <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Original model</a> ·
  <a href="https://qwen.ai/blog?id=qwen3.8-flash-next">Qwen overview</a> ·
  <a href="https://github.com/ml-explore/mlx-vlm">MLX-VLM</a> ·
  <a href="LICENSE">Qwen Community License 1.0</a>
</p>


## About this conversion


This release uses a 2-bit affine base and assigns more precision to sensitive modules. The final allocation includes 3, 4, 5, 6, and 8-bit modules. The native Qwen4Exp MTP block uses the same mixed-precision approach and remains part of the checkpoint.


| Item | Value |
| --- | --- |
| Repository | `TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP` |
| Base model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Source weights | Official BF16 checkpoint |
| Format | MLX safetensors |
| Quantisation | Sensitivity-aware oQ2 mixed precision |
| Base allocation | 2-bit affine, group size 32 |
| Module overrides | 297 at 3-bit, 60 at 4-bit, 36 at 5-bit, 129 at 6-bit, 224 at 8-bit |
| Override group sizes | 32, 64, or 128 according to module |
| Native MTP | Included, one Qwen4Exp draft block |
| Indexed tensors | 3,747 total, including 76 MTP entries |
| Weight shards | 15 |
| Weight size | 70.782 GB / 65.921 GiB |
| Configured context | 262,144 tokens |
| Architecture | `qwen4_exp` vision-language sparse MoE |


The upstream tokenizer, chat template, image and video processor configuration, generation configuration, licence, and native MTP configuration are included.


> [!IMPORTANT]
> Use a runtime with explicit `qwen4_exp` and native-MTP support. A runtime that does not construct the Qwen4Exp MTP module may reject the 76 MTP entries during strict loading.


## Download and use


```bash
python -m pip install --upgrade huggingface_hub


hf download TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP \
  --local-dir ./Qwen3.8-Flash-Next-MLX-oQ2-MTP
```


Add the downloaded directory to a compatible oMLX model directory and refresh the model registry. Native MTP is optional. Keep it disabled by default for this release because the measured depth-three path was slightly slower than baseline.


## Apple M3 Studio performance


The validation used three measured 512-token runs per mode. The table reports median decode throughput.


| Runtime mode | Runs | Output per run | Median generation speed |
| --- | ---: | ---: | ---: |
| Native MTP disabled | 3 | 512 tokens | **26.1398 tokens/s** |
| Native MTP enabled, depth 3 | 3 | 512 tokens | **25.9290 tokens/s** |


Native MTP changed median throughput by **-0.81%** in this test. The telemetry sample accepted 33 of 57 draft proposals, an acceptance rate of **57.89%**. Each of the three paired MTP-off and MTP-on 512-token runs produced matching output hashes.


Instruction following, factual recall, arithmetic, and coherent long-generation gates passed in both modes. MTP worked correctly, but it did not improve throughput on this checkpoint and test. The recommended default is therefore MTP disabled.


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


- The converter read the official BF16 checkpoint directly.
- The recipe used a 2-bit affine base with 746 mixed-precision module overrides.
- Full checkpoint validation covered all 3,747 indexed tensors, all 15 shards, and all 76 native-MTP tensor entries.
- Three deterministic 512-token runs passed in each MTP mode with exact paired output parity.
- Instruction, factual, arithmetic, and coherence checks passed in both modes.
- Native-MTP telemetry captured the draft and acceptance counters.


This is a community conversion, not an official Qwen release.


## Limitations


- oQ2 is an extreme-compression format. Evaluate instruction following, reasoning, factual accuracy, and visual understanding on the intended workload before deployment.
- Native MTP was 0.81% slower in the measured test. Acceptance alone does not guarantee a throughput gain.
- The configured 262,144-token context does not guarantee that every Apple-silicon system can run that length within available unified memory.
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

The exact tested oMLX application version is not recorded here; a library version is not an app version. The original performance tables retain their benchmark conditions and speed figures; this documentation update adds no new test results.

### Quick start and demo prompt

```bash
hf download TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP --local-dir ./models/Qwen3.8-Flash-Next-MLX-oQ2-MTP
```

Add the downloaded folder to oMLX model directories, refresh the list, and follow this card's architecture and MTP compatibility requirements before loading.

Try this in a new chat with a 128-token output limit:

```text
Explain why the sky looks blue in three short sentences.
```

This is a demo prompt to try, not a recorded successful run; a captured demonstration for this documentation update is not yet available.

[Follow TensorFold for new Apple Silicon releases and fixes.](https://huggingface.co/TensorFold)
<!-- TensorFold-chooser-end -->
