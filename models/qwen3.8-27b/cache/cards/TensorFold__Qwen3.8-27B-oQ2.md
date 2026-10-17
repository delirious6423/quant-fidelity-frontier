---
library_name: mlx
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - mlx
  - mlx-vlm
  - omlx
  - qwen
  - qwen3.8
  - multimodal
  - quantized
  - apple-silicon
  - oq2
  - mixed-precision
---

<p align="center">
  <a href="https://tensorfold.dev">
    <img src="https://huggingface.co/spaces/TensorFold/README/resolve/main/tensorfold-logo.png" alt="TensorFold" width="160">
  </a>
</p>

<p align="center">
  <a href="https://qwenlm.github.io/"><img src="qwen-logo.png" width="96" height="95" alt="Qwen"></a><br>
  <img src="https://img.shields.io/badge/Apple_Silicon-MLX-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Apple silicon MLX">
  <img src="https://img.shields.io/badge/TensorFold-oMLX-6E56CF?style=for-the-badge&logo=huggingface&logoColor=white" alt="TensorFold oMLX">
</p>


<h1 align="center">Qwen3.8-27B — oQ2</h1>


<p align="center">
  An oMLX-optimised mixed-precision conversion of <a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen/Qwen3.8-27B</a>, packaged for MLX-VLM and oMLX on Apple silicon.
</p>


<p align="center">
  <a href="https://huggingface.co/Qwen/Qwen3.8-27B">Original model</a> ·
  <a href="https://qwenlm.github.io/">Qwen</a> ·
  <a href="https://github.com/Blaizzy/mlx-vlm">MLX-VLM</a> ·
  <a href="https://www.apache.org/licenses/LICENSE-2.0">Apache 2.0</a>
</p>


## About this conversion


This repository contains an oQ2 mixed-precision MLX conversion of Qwen3.8-27B. The global recipe is 2-bit affine with selected sensitive modules retained at substantially higher precision. The upstream model is a dense, native vision-language model with flexible thinking control and support for text, images, and video. Its tokenizer, processor configuration, chat template, and generation configuration are preserved.


| Item | Value |
| --- | --- |
| Base model | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Format | MLX safetensors |
| Quantization | oQ2 mixed precision, affine, group size 64 |
| Precision overrides | 136 modules at 5-bit; 1 each at 3-, 6-, and 8-bit |
| Conversion stack | `mlx-vlm 0.6.3`, `mlx-lm 0.31.3`, `mlx 0.32.0` |
| Weight shards | 3 |
| Weight size | 11.29 GB (10.52 GiB) |
| Maximum configured context | 262,144 tokens |
| Architecture | `qwen3_5` / `Qwen3_5ForConditionalGeneration` |


## Apple-silicon performance


This checkpoint was load-tested and generation-tested on the following machine:


| Hardware | Configuration |
| --- | --- |
| Host | Mac Studio |
| Chip | Apple M3 Ultra |
| CPU | 32 cores (24 performance + 8 efficiency) |
| Unified memory | 256 GB |
| Runtime | MLX-VLM 0.6.3 / MLX 0.32.0 |


| Measurement | Result |
| --- | ---: |
| Decode (median) | **52.55 tokens/s** |
| Reported peak memory | **13.44 GB** |
| Timed runs | 3 × 181 generated tokens (natural EOS) |
| Warm-up | 256 generated tokens |
| Prompt | 81 tokens after chat templating |


The decode figure is the median of three greedy runs after a 256-token Metal-kernel warm-up. Each timed run ended naturally at EOS after 181 generated tokens; the runs measured 52.53, 52.61, and 52.55 tokens/s. This is a practical local reference, not a controlled cross-platform benchmark; prompt length, context growth, sampler settings, memory pressure, thermal state, and runtime versions can materially change performance.


## Quick start with MLX-VLM


```bash
python -m pip install -U mlx-vlm huggingface_hub


python -m mlx_vlm.generate \
  --model TensorFold/Qwen3.8-27B-oQ2 \
  --prompt "Explain the difference between linear and full attention." \
  --max-tokens 512
```


Download for local use:


```bash
hf download TensorFold/Qwen3.8-27B-oQ2 \
  --local-dir ~/.omlx/models/TensorFold/Qwen3.8-27B-oQ2
```


## Using it with oMLX


1. Place the model at `~/.omlx/models/TensorFold/Qwen3.8-27B-oQ2`.
2. Refresh the oMLX model registry.
3. Load `Qwen3.8-27B-oQ2` and use the chat UI or OpenAI-compatible endpoint.


```bash
curl "$OMLX_BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OMLX_API_KEY" \
  -d '{
    "model": "Qwen3.8-27B-oQ2",
    "messages": [{"role": "user", "content": "Write a short Swift actor example."}],
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 256
  }'
```


For long prompts, begin with a conservative context limit and increase it while watching memory pressure. The configured context is a model capability, not a guarantee that every host can prefill it within available unified memory.


## Architecture


Qwen3.8-27B is a dense causal language model with a vision encoder. It uses the Qwen3.5 architectural foundation, interleaving Gated DeltaNet linear-attention blocks with periodic full-attention blocks.


| Architecture detail | Upstream value |
| --- | ---: |
| Parameters | 27B |
| Language layers | 64 |
| Hidden size | 5,120 |
| Attention heads / KV heads | 24 / 4 |
| Linear-attention V / QK heads | 48 / 16 |
| FFN intermediate size | 17,408 |
| Vocabulary / padded embeddings | 248,320 |
| Configured context | 262,144 tokens |


For upstream evaluations, usage guidance, intended use, limitations, safety information, and the full architecture discussion, see the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B).


## Conversion and validation notes


- Source weights: the official Qwen checkpoint.
- Quantization: global 2-bit affine weights with group size 64, with 136 module overrides at 5-bit and one module each at 3-, 6-, and 8-bit.
- The upstream tokenizer, processor files, chat template, and generation configuration are preserved.
- Both `quantization` and `quantization_config` preserve the full per-module oQ recipe.
- All 2,180 converted tensors and all three indexed shards were checked locally.
- This is an aggressive low-bit release. Evaluate output quality carefully and prefer oQ4, oQ6, or 8-bit when fidelity matters more than footprint.
- Quantization can reduce output quality relative to the source weights; use a higher-precision variant when quality matters more than memory use.
- The model was loaded and exercised through end-to-end generation on Apple silicon.


This is a community conversion, not an official Qwen release. Validate quality and numerical behaviour on your own representative workload before production use.


## Licence and attribution


The upstream model is released under the **Apache License 2.0**. A copy is included in this repository; review it before use or redistribution.


All model design, training, benchmark, and upstream documentation credit belongs to Qwen and the original contributors. The MLX conversion, Apple-silicon validation, compatibility work, and packaging are provided by [TensorFold](https://huggingface.co/TensorFold).



<!-- TensorFold-chooser-start -->
## Choose for your Mac

[64GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-64gb-macs-6a9fefda17932216ec9ab457) · [128GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-128gb-macs-6a9ff0abd31bc9abbe7922d7) · [256GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-256gb-macs-6a9ff0ef9fed7c5bdca15e9b)

Published peak memory: **13.44 GB**; estimated starting tier: **64GB**, leaving about **50 GB** nominal headroom. The collections use published M3 Studio peaks with at least 25% nominal headroom; fit on other Macs is an estimate, and full context is not guaranteed. Start with short context and one request.

### Runtime and evidence

The exact tested oMLX application version is not recorded here; a library version is not an app version. The original performance tables retain their benchmark conditions and speed figures; this documentation update adds no new test results.

### Quick start and demo prompt

```bash
hf download TensorFold/Qwen3.8-27B-oQ2 --local-dir ./models/Qwen3.8-27B-oQ2
```

Add the downloaded folder to oMLX model directories, refresh the list, and follow this card's architecture and MTP compatibility requirements before loading.

Try this in a new chat with a 128-token output limit:

```text
Explain why the sky looks blue in three short sentences.
```

This is a demo prompt to try, not a recorded successful run; a captured demonstration for this documentation update is not yet available.

[Follow TensorFold for new Apple Silicon releases and fixes.](https://huggingface.co/TensorFold)
<!-- TensorFold-chooser-end -->
