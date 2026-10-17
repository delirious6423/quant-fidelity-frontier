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
  <img src="https://img.shields.io/badge/TensorFold-6bit-6E56CF?style=for-the-badge&logo=huggingface&logoColor=white" alt="TensorFold 6-bit">
</p>


<h1 align="center">Qwen3.8 Flash Next — MLX 6-bit with native MTP</h1>


<p align="center">
  A uniform 6-bit MLX conversion of <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen/Qwen3.8-Flash-Next</a>, rebuilt directly from the official BF16 checkpoint with the model's native MTP draft block preserved.
</p>


<p align="center">
  <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Original model</a> ·
  <a href="https://qwen.ai/blog?id=qwen3.8-flash-next">Qwen overview</a> ·
  <a href="https://github.com/ml-explore/mlx-vlm">MLX-VLM</a> ·
  <a href="LICENSE">Qwen Community License 1.0</a>
</p>


## About this conversion


This repository contains the standard uniform 6-bit release. Eligible language-model and native-MTP modules use MLX affine quantisation at group size 32. Multimodal modules and MoE router gates remain in BF16. Group size 32 also covers the model's 160-wide hashed n-gram embedding tables.


| Item | Value |
| --- | --- |
| Repository | `TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP` |
| Base model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Source weights | Official BF16 checkpoint |
| Format | MLX safetensors |
| Quantisation | Uniform 6-bit affine |
| Group size | 32 |
| Retained in BF16 | Multimodal modules and `mlp.gate` router modules |
| Native MTP | Included, one Qwen4Exp draft block |
| Indexed tensors | 3,747 total, including 76 MTP entries |
| Weight shards | 30 |
| Weight size | 158.08 GB / 147.224 GiB |
| Configured context | 262,144 tokens |
| Architecture | `qwen4_exp` vision-language sparse MoE |


The upstream tokenizer, chat template, vision processor, generation configuration, licence, and native MTP configuration are retained.


> [!IMPORTANT]
> Use a runtime with explicit `qwen4_exp` and native-MTP support. A runtime that does not construct the Qwen4Exp MTP module may reject the 76 MTP tensor entries during strict loading.


## Download and use


```bash
python -m pip install --upgrade huggingface_hub


hf download TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP \
  --local-dir ./Qwen3.8-Flash-Next-MLX-6bit-MTP
```


In a compatible oMLX build:


1. Add the downloaded directory to the oMLX model directories.
2. Refresh the model registry and select `Qwen3.8-Flash-Next-MLX-6bit-MTP`.
3. Enable native MTP and set the draft depth to three tokens.
4. Load the model and start a chat or use the OpenAI-compatible API.


The model can run with native MTP disabled for baseline or troubleshooting comparisons. MTP changes the decode path, not the model's intended greedy output.


## Apple M3 Studio performance


Benchmark environment: oMLX 0.6.3rc3 (build 2475) on an Apple M3 Studio. Measurements use greedy decoding, seed 6330, identical prompts, a separate warm-up, and 3 measured 512-token runs per mode.


| Runtime mode | Runs | Output per run | Median generation speed | Drafted | Accepted | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native MTP disabled | 3 | 512 tokens | **20.6495 tokens/s** | Not applicable | Not applicable | Not applicable |
| Native MTP enabled, three draft tokens | 3 | 512 tokens | **25.935 tokens/s** | 1055 | 776 | 73.55% |


The measured throughput delta with MTP enabled was 25.6%. Exact-output parity: **Passed**. All sustained runs reached 512 generated tokens: **Passed**. Coherence checks: **Passed**. MTP telemetry capture: **Passed**.


Draft acceptance is reported beside throughput because a high acceptance rate does not by itself prove a speed improvement. Results vary with prompt length, context growth, cache state, runtime version, memory pressure, and thermal conditions. The first request after loading includes model and kernel warm-up and is excluded from the steady-state result.


## Runtime compatibility


| Component | Release used | Validation scope | Status |
| --- | --- | --- | ---: |
| oMLX | 0.6.3rc3 (build 2475) | Model load, text generation, native MTP, parity, and sustained decode | Passed |
| MLX-VLM | 0.6.3 | Qwen4Exp construction, conversion, and strict weight loading | Passed |
| MLX | 0.32.0 | Quantised tensor backend and generation kernels | Passed |


Older builds without explicit `qwen4_exp` native-MTP support are not compatible with this checkpoint. Vision input support also depends on the selected MLX-VLM and oMLX versions; the performance figures above cover text generation only.


## Architecture


Qwen3.8 Flash Next is a causal language model with a vision encoder. Its language stack combines Gated DeltaNet, Qwen Sparse Attention, sparse mixture-of-experts layers, widened gated residual streams, hashed bigram and trigram embeddings, and a native next-token-prediction block for speculative decoding.


| Architecture detail | Upstream value |
| --- | ---: |
| Language-model parameters | 125B total / 6B active |
| N-gram embedding | 51B parameters, 20,000,000 entries |
| Native MTP | 4B parameters, one draft layer |
| Hidden size | 2,560 |
| Token embedding / LM output | 248,320 padded |
| Layers | 48 |
| Layer layout | 12 × (3 × Gated DeltaNet + 1 × Qwen Sparse Attention), each followed by MoE |
| Gated DeltaNet heads | 48 V / 16 QK, head dimension 128 |
| Sparse-attention heads | 24 Q / 2 KV, head dimension 256 |
| Sparse-attention budget | 512 blocks / 2,048 tokens |
| Routed / active experts | 512 / 10, plus 1 shared expert |
| Expert intermediate size | 640 |
| Gated-residual branches / bottleneck | 4 / 320 |
| Native context | 262,144 tokens, extensible upstream to 1,000,000 |


For upstream evaluations, intended use, limitations, safety guidance, and the full architecture discussion, see the [original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).


## Conversion and validation


- The converter read the official BF16 checkpoint directly.
- Eligible modules were quantised uniformly at 6 bits using affine quantisation and group size 32.
- Multimodal modules and MoE router gates were retained in BF16.
- Structural validation of all 3,747 indexed tensors, all 30 weight shards, and all 76 native-MTP tensor entries: **Passed**.
- Exact deterministic MTP-off/on parity: **Passed**.
- 3 sustained 512-token runs per mode: **Passed**.
- MTP drafted/accepted telemetry: **Passed**.
- Coherent long-generation checks: **Passed**.


This is a community conversion, not an official Qwen release.


## Limitations


- Quantisation can reduce quality relative to BF16. The effect is usually strongest at 2-bit precision.
- The native 262,144-token context does not guarantee that every Apple-silicon system can prefill or decode that length within available unified memory.
- Native MTP needs a compatible runtime and may not improve every prompt or context length.
- The checkpoint is an MLX release for Apple silicon. It is not a GGUF, CUDA, TensorRT-LLM, or vLLM checkpoint.
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
hf download TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP --local-dir ./models/Qwen3.8-Flash-Next-MLX-6bit-MTP
```

Add the downloaded folder to oMLX model directories, refresh the list, and follow this card's architecture and MTP compatibility requirements before loading.

Try this in a new chat with a 128-token output limit:

```text
Explain why the sky looks blue in three short sentences.
```

This is a demo prompt to try, not a recorded successful run; a captured demonstration for this documentation update is not yet available.

[Follow TensorFold for new Apple Silicon releases and fixes.](https://huggingface.co/TensorFold)
<!-- TensorFold-chooser-end -->
