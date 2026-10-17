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
  - 3-bit
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
  <img src="https://img.shields.io/badge/TensorFold-oQ3-6E56CF?style=for-the-badge&logo=huggingface&logoColor=white" alt="TensorFold oQ3">
</p>


<h1 align="center">Qwen3.8 Flash Next — MLX oQ3 with native MTP</h1>


<p align="center">
  A sensitivity-guided, mixed-precision MLX conversion of <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen/Qwen3.8-Flash-Next</a>, rebuilt directly from the official BF16 checkpoint with the model's matching native MTP block preserved.
</p>


<p align="center">
  <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Original model</a> ·
  <a href="https://qwen.ai/blog?id=qwen3.8-flash-next">Qwen overview</a> ·
  <a href="https://github.com/ml-explore/mlx-vlm">MLX-VLM</a> ·
  <a href="LICENSE">Qwen Community License 1.0</a>
</p>


## About this conversion


oQ3 uses a 3-bit affine base and spends additional precision on sensitive modules. Layer sensitivity was measured with a validated quantized calibration proxy, while every released weight was quantized from the official BF16 checkpoint. The result is a compact model with 746 higher-precision module overrides rather than a uniform 3-bit layout.


| Item | Value |
| --- | --- |
| Repository | `TensorFold/Qwen3.8-Flash-Next-MLX-oQ3-MTP` |
| Base model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Source weights | Official BF16 checkpoint |
| Format | MLX safetensors |
| Quantisation | oQ3 mixed precision, 3-bit affine base |
| Base group size | 32 |
| Protected modules | 314 at 4-bit, 82 at 5-bit, 127 at 6-bit, and 223 at 8-bit |
| Native MTP | Included, one Qwen4Exp draft block |
| Indexed tensors | 3,747 total, including 76 MTP entries |
| Weight shards | 19 |
| Weight size | 92.505 GB / 86.152 GiB |
| Configured context | 262,144 tokens |
| Architecture | `qwen4_exp` vision-language sparse MoE |


The upstream tokenizer, current chat template, vision processor, generation configuration, licence, and native MTP configuration are retained.


> [!IMPORTANT]
> Use a runtime with explicit `qwen4_exp` and native-MTP support. A runtime that does not construct the Qwen4Exp MTP module may reject the 76 MTP tensor entries during strict loading.


## Download and use


```bash
python -m pip install --upgrade huggingface_hub


hf download TensorFold/Qwen3.8-Flash-Next-MLX-oQ3-MTP \
  --local-dir ./Qwen3.8-Flash-Next-MLX-oQ3-MTP
```


In a compatible oMLX build, add the downloaded directory to the model directories, refresh the registry, select the model, and enable native MTP. MTP can be disabled for baseline comparisons and troubleshooting.


## Apple M3 Studio performance


Benchmark environment: oMLX 0.6.3rc3 (build 2475), MLX-VLM 0.6.3, and MLX 0.32.0 on an Apple M3 Studio. Measurements use greedy decoding, seed 6330, identical prompts, a separate warm-up, and three measured 512-token runs per mode.


| Runtime mode | Runs | Output per run | Median generation speed | Drafted | Accepted | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native MTP disabled | 3 | 512 tokens | **26.5352 tokens/s** | Not applicable | Not applicable | Not applicable |
| Native MTP enabled | 3 | 512 tokens | **29.0822 tokens/s** | 847 | 583 | 68.83% |


Native MTP improved median decode throughput by **9.60%** in this test. The MTP-off and MTP-on runs produced exactly matching output hashes. All six sustained runs reached 512 generated tokens; exact instruction, factual, arithmetic, and coherent long-generation gates passed in both modes.


Results vary with prompt length, context growth, cache state, runtime version, memory pressure, and thermal conditions. The first request after loading includes model and kernel warm-up and is excluded from the steady-state result.


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


For upstream evaluations, intended use, limitations, safety guidance, and the full architecture discussion, see the [original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).


## Conversion and validation


- The converter read the official BF16 checkpoint directly.
- A 3-bit affine base at group size 32 was combined with 746 sensitivity-guided 4/5/6/8-bit overrides.
- Structural validation passed for all 3,747 indexed tensors, all 19 weight shards, and all 76 native-MTP tensor entries.
- Three deterministic 512-token runs passed in each MTP mode with exact output parity.
- Native-MTP telemetry, exact-answer gates, and coherent long-generation checks passed.


This is a community conversion, not an official Qwen release.


## Limitations


- Quantisation can reduce quality relative to BF16; 3-bit models should be evaluated on the intended workload.
- The native 262,144-token context does not guarantee that every Apple-silicon system can run that length within available unified memory.
- Native MTP needs a compatible runtime and may not improve every prompt or context length.
- This is an MLX release for Apple silicon, not a GGUF, CUDA, TensorRT-LLM, or vLLM checkpoint.
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
hf download TensorFold/Qwen3.8-Flash-Next-MLX-oQ3-MTP --local-dir ./models/Qwen3.8-Flash-Next-MLX-oQ3-MTP
```

Add the downloaded folder to oMLX model directories, refresh the list, and follow this card's architecture and MTP compatibility requirements before loading.

Try this in a new chat with a 128-token output limit:

```text
Explain why the sky looks blue in three short sentences.
```

This is a demo prompt to try, not a recorded successful run; a captured demonstration for this documentation update is not yet available.

[Follow TensorFold for new Apple Silicon releases and fixes.](https://huggingface.co/TensorFold)
<!-- TensorFold-chooser-end -->
