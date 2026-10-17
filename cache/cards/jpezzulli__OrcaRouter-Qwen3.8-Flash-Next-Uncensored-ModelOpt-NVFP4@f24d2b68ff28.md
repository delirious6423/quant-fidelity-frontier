---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
  - OrcaRouter/Qwen3.8-Flash-Next-Uncensored
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
language:
  - en
  - zh
tags:
  - qwen
  - qwen3.8
  - flash-next
  - uncensored
  - modelopt
  - nvfp4
  - vision-language
  - function-calling
  - reasoning
  - mtp
---

# OrcaRouter Qwen3.8 Flash-Next — ModelOpt NVFP4

Published by request from the Pennyroyal community. This is a ModelOpt NVFP4 conversion of OrcaRouter’s uncensored Qwen3.8 Flash-Next checkpoint—the conversion I use for local agentic work with Pennyroyal/SGLang.

I’ve also heard reports from several community members of it outperforming the stock model on agentic tasks, which added to the interest in sharing this working format.

The original model modification is OrcaRouter’s work. My contribution here is the quantization, compatible packaging and runtime testing—not additional fine-tuning or uncensoring.

## Model lineage

- Modified source: [OrcaRouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/OrcaRouter/Qwen3.8-Flash-Next-Uncensored)
- Base checkpoint: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- Runtime: [Pennyroyal/SGLang](https://github.com/jpezzulli/sglang-rtxpro6000)

## Quantization and packaging

The checkpoint uses NVIDIA ModelOpt 0.46.0 NVFP4: 4-bit floating-point weights and activations, group size 16, on selected `Linear` modules. Attention and Gated-DeltaNet paths, routers, shared experts, hyper-connections, PLE, vision, embeddings, LM head and MTP are excluded from NVFP4 according to the packaged configuration. The large PLE embedding shards are FP8 E4M3. It is stored as sharded safetensors and is approximately **135.3 GB (126 GiB)**. Pennyroyal's optional online-FP8 optimizations are runtime settings, separate from this on-disk conversion. RAM versus NVMe PLE placement is also a runtime choice; see [NVMe PLE setup](https://github.com/jpezzulli/sglang-rtxpro6000/blob/pennyroyal-main-sm120-final/NVME-PLE.md).

This is the working checkpoint used for my daily agentic workload, including thinking and tool use, and tested with Pennyroyal’s Flash-Next native-MTP/FR-Spec path on an RTX PRO 6000 Blackwell. That is practical runtime evidence, not a claim of new benchmark scores or broad hardware compatibility.

## Run with Pennyroyal

For a native installation, download the repository to a local directory and set that directory as `TARGET_MODEL`; then follow [Launch Flash-Next with FR-Spec](https://github.com/jpezzulli/sglang-rtxpro6000/blob/pennyroyal-main-sm120-final/RUN.md#launch-flash-next-with-fr-spec).

For the container, mount the parent model directory as `HOST_MODELS_ROOT` and set `TARGET_MODEL` to its corresponding path below `/models`, for example `/models/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4`. See [Choose a profile and configure `.env`](https://github.com/jpezzulli/sglang-rtxpro6000/blob/pennyroyal-main-sm120-final/docker/pennyroyal/README.md#choose-a-profile-and-configure-env).

The supplied source files have conflicting license metadata: the source README labels the model Apache-2.0, while its included `LICENSE` is **Qwen Community License 1.0**. This upload preserves that license file and declares the Hub license as `other`; review the included terms before use.
