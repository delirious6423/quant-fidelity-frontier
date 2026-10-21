---
library_name: mlx
license: mit
license_link: LICENSE
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
  - en
  - zh
tags:
  - mlx
  - mlx-vlm
  - omlx
  - oq
  - glm
  - glm5
  - glm5-next
  - native-mtp
  - speculative-decoding
  - mixed-precision
  - mixture-of-experts
  - multimodal
  - vision-language
  - quantized
  - apple-silicon
---

<p align="center">
  <a href="https://tensorfold.dev">
    <img src="https://huggingface.co/spaces/TensorFold/README/resolve/main/tensorfold-logo.png" alt="TensorFold" width="160">
  </a>
</p>

<div align="center">
  <a href="https://huggingface.co/zai-org/GLM-5.3-Flash">
    <img src="zai-logo.png" width="96" height="96" alt="Z.ai logo">
  </a>
</div>


<p align="center">
  <img src="https://img.shields.io/badge/Z.ai-GLM--5.3--Flash-111827?style=for-the-badge" alt="Z.ai GLM-5.3-Flash">
  <img src="https://img.shields.io/badge/Apple_Silicon-MLX-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Apple silicon MLX">
  <img src="https://img.shields.io/badge/Native_MTP-Included-22C55E?style=for-the-badge" alt="Native MTP included">
  <img src="https://img.shields.io/badge/TensorFold-oQ-6E56CF?style=for-the-badge&logo=huggingface&logoColor=white" alt="TensorFold oQ">
</p>


<h1 align="center">GLM-5.3-Flash, MLX oQ4 with native MTP</h1>


<p align="center">
  A sensitivity-guided mixed-precision MLX conversion of <a href="https://huggingface.co/zai-org/GLM-5.3-Flash">zai-org/GLM-5.3-Flash</a>, built from the official checkpoint with its matching next-token prediction block preserved.
</p>


<p align="center">
  <a href="https://huggingface.co/zai-org/GLM-5.3-Flash">Original model</a> ·
  <a href="https://z.ai/blog/glm-5.3-flash">Z.ai overview</a> ·
  <a href="https://arxiv.org/abs/2602.15763">Technical report</a> ·
  <a href="https://github.com/ml-explore/mlx">Apple MLX</a> ·
  <a href="LICENSE">MIT licence</a>
</p>


## At a glance


| Item | Value |
| --- | --- |
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| Repository | `TensorFold/GLM-5.3-Flash-MLX-oQ4-MTP` |
| Format | MLX safetensors |
| Quantisation | oQ4 mixed precision: 4-bit affine base with 554 sensitivity-selected 5/6/8-bit overrides |
| Base group size | 64 |
| Source revision | `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a` |
| Source weight format | FP8 E4M3 with 128x128 block scaling |
| Sensitivity method | relative activation MSE measured with a validated 4-bit proxy |
| Protected or overridden modules | 554 |
| Native MTP | Included, one matching upstream prediction layer |
| MTP weight precision | 4-bit affine base with 12 native-MTP overrides at 5/6/8-bit |
| Weight tensors | 3125 total, including 59 MTP tensors |
| Weight shards | 35 |
| Download size | 185.749 GB / 172.992 GiB |
| Configured context | 1,048,576 tokens |
| Architecture | `glm5_next`, multimodal sparse MoE |


The upstream tokenizer, chat template, multimodal processor, generation configuration, and native MTP metadata are preserved.


## oQ recipe


| Component | Treatment |
| --- | --- |
| Eligible base modules | 4-bit affine, group size 64 |
| Sensitivity-protected modules | 5-bit, 6-bit, and 8-bit affine, 554 modules |
| Routed experts | 4-bit affine base with sensitivity-selected higher-precision expert projections |
| Token embeddings and output head | source-compatible precision |
| Vision encoder and projector | source-compatible precision |
| Native MTP prediction layer | 4-bit affine base with 12 native-MTP overrides at 5/6/8-bit |
| Other non-quantisable tensors | Preserved at source-compatible precision |
| Converter | TensorFold streamed oQ converter using MLX 0.32.0 |


Sensitivity was measured with the built-in multilingual code calibration set, 128 samples at 256 tokens. The allocation rule was byte-budgeted layer-sensitivity ranking under the oQ4 target and hard cap. These details are part of the release recipe and should be used when comparing oQ variants.


This is a weight-only post-training quantisation. It does not retrain or fine-tune the upstream model.


## Runtime compatibility


GLM-5.3-Flash uses the new `glm5_next` multimodal architecture, hybrid linear and sparse attention, manifold-constrained hyper-connections, and an embedded next-token prediction layer. The oQ layout also requires a runtime that understands its per-module quantisation metadata.


| Runtime path | Status |
| --- | --- |
| oMLX `0.6.3rc3 build 2475` | Validated for oQ baseline and native-MTP generation |
| MLX `0.32.0` | Mixed-precision tensor runtime |
| MLX-VLM `0.6.3 with GLM5 Next runtime integration` | Required for `glm5_next` model and vision loading |
| Older builds without oQ or GLM5 Next MTP integration | Unsupported |


> [!IMPORTANT]
> A runtime that can load an ordinary MLX conversion is not necessarily compatible with this oQ checkpoint or its embedded MTP layer. Use a build that supports the recorded per-module recipe, GLM5 Next drafting, verification, and cache reconciliation.


## Download and use


```bash
hf download TensorFold/GLM-5.3-Flash-MLX-oQ4-MTP \
  --local-dir GLM-5.3-Flash-MLX-oQ4-MTP
```


In a compatible oMLX build:


1. Add the downloaded model directory.
2. Load it once with native MTP disabled to validate the oQ baseline path.
3. Enable **Native MTP** in the model settings and select the validated draft depth shown below.
4. Start a new conversation so the runtime rebuilds the model and caches with MTP active.


## Apple M3 Studio performance


Measured on the same Apple M3 Studio, with identical prompts, sampling, context, output length, and warmed model state:


| Mode | Output tokens | Median decode |
| --- | ---: | ---: |
| oQ baseline, MTP off | 128 | 6.2155 tokens/s |
| oQ with native MTP | 128 | 4.9969 tokens/s |


| Benchmark detail | Value |
| --- | --- |
| Chip | Apple M3 Ultra |
| Unified memory | 256 GB |
| Runtime | oMLX `0.6.3rc3 build 2475`, MLX `0.32.0` |
| Draft depth | 1 |
| Draft acceptance | 100.0% on captured one-token drafts |
| Decode ratio, MTP/baseline | 0.8039x |
| Timed runs | three per mode; median reported |
| Warm-up | one 32-token greedy request discarded before each measured mode |
| Sampling | greedy decoding with temperature 0 |


These figures are a local reference, not a cross-platform guarantee. Prompt shape, image input, context growth, draft acceptance, sampling, memory pressure, thermal state, and runtime revisions can materially change performance.


## Architecture


GLM-5.3-Flash is Z.ai's first natively multimodal GLM-5 model. It combines KDA linear-attention layers with periodic sparse-attention layers, a sparse mixture-of-experts feed-forward stack, manifold-constrained hyper-connections, and a native next-token prediction layer.


| Architecture detail | Upstream value |
| --- | ---: |
| Parameters | 320B total / 18B active |
| Language layers | 45 |
| Linear / sparse-attention layers | 34 / 11 |
| Dense / sparse-MoE feed-forward layers | 3 / 42 |
| Routed / active experts | 288 / 8, plus 1 shared expert |
| Hidden size | 4,096 |
| Attention heads | 64 |
| Vocabulary size | 154,880 |
| Manifold-constrained hyper-connection width | 4 streams |
| Native MTP layers | 1 |
| Vision encoder | 24 layers, 1,024 hidden size, 448-pixel input |
| Configured maximum context | 1,048,576 tokens |


See the [official model card](https://huggingface.co/zai-org/GLM-5.3-Flash), [Z.ai overview](https://z.ai/blog/glm-5.3-flash), and [GLM-5 technical report](https://arxiv.org/abs/2602.15763) for upstream training, evaluations, intended uses, and safety guidance.


## Validation


| Check | Result |
| --- | --- |
| Config, processor, tokenizer, and chat-template files present | Passed; current upstream tokenizer, chat template, processor, generation config, and MIT licence are present |
| Safetensors index resolves every tensor to an existing shard | Passed; all 3,125 indexed tensors resolve to existing shards |
| All shards open and tensor headers match the index | Passed; all 35 shards open with no missing, duplicate, unexpected, or zero-length tensors |
| Per-module oQ metadata matches the saved tensor layouts | Passed; the 4-bit base and all 554 per-module overrides match the saved layouts |
| Native MTP configuration and 59 MTP tensors present | Passed; one configured NextN layer and 59 native-MTP tensors are present |
| Baseline text generation | Passed; exact instruction, factual, arithmetic, and sustained streaming tests completed |
| Native-MTP text generation | Passed for the same correctness gates; sustained decode is slower than baseline on this runtime |
| Greedy baseline/MTP output equivalence | Passed on instruction, factual, and arithmetic gates; sustained hash equivalence is not claimed |
| Image-and-text smoke test | Passed structurally; image generation was not benchmarked |
| Long-context smoke test | Not claimed; upstream context metadata is preserved |


Validation used oMLX `0.6.3rc3 build 2475`, MLX `0.32.0`, and MLX-VLM `0.6.3 with GLM5 Next runtime integration`.


## Limitations


- Mixed-precision quantisation can reduce quality relative to the official checkpoint. The effect can differ by task and by oQ allocation.
- Sensitivity scores are calibration-dependent; protected modules are not a guarantee of BF16-equivalent quality.
- Native MTP improves speed only when draft acceptance offsets drafting and verification cost. It may be neutral or slower for unpredictable output.
- The configured one-million-token context does not imply that every Apple-silicon system has enough memory for a full-context request.
- Image prefill has different memory and throughput characteristics from text-only generation and is not represented by the decode benchmark above.
- Runtime support for `glm5_next`, oQ metadata, and native MTP is evolving. Use the exact validated versions listed in this card.
- Validate numerical behaviour, output quality, safety, and tool use on representative workloads before production deployment.


This is a community quantisation and is not an official Z.ai release.


## Licence and attribution


The upstream model is released under the **MIT License**. The required licence text is included in this repository.


Model design, training, upstream evaluations, and documentation belong to Z.ai and the GLM-5 contributors. The oQ conversion, Apple-silicon validation, native-MTP integration work, and packaging are provided by [TensorFold](https://huggingface.co/TensorFold).


If you use this model in research, cite the upstream report:


```bibtex
@misc{glm5team2026glm5,
  title        = {GLM-5: from Vibe Coding to Agentic Engineering},
  author       = {GLM-5-Team and others},
  year         = {2026},
  eprint       = {2602.15763},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2602.15763}
}
```



<!-- TensorFold-chooser-start -->
## Choose for your Mac

[64GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-64gb-macs-6a9fefda17932216ec9ab457) · [128GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-128gb-macs-6a9ff0abd31bc9abbe7922d7) · [256GB Macs](https://huggingface.co/collections/TensorFold/mlx-models-for-256gb-macs-6a9ff0ef9fed7c5bdca15e9b)

No measured memory tier is assigned here. The collections use published M3 Studio peaks with at least 25% nominal headroom; fit on other Macs is an estimate, and full context is not guaranteed. Start with short context and one request.

### Runtime and evidence

oMLX version mentioned in the existing card: **0.6.3rc3**; consult its compatibility notes for whether this was tested and any required integrations. The original performance tables retain their benchmark conditions and speed figures; this documentation update adds no new test results.

### Quick start and demo prompt

```bash
hf download TensorFold/GLM-5.3-Flash-MLX-oQ4-MTP --local-dir ./models/GLM-5.3-Flash-MLX-oQ4-MTP
```

Add the downloaded folder to oMLX model directories, refresh the list, and follow this card's architecture and MTP compatibility requirements before loading.

Try this in a new chat with a 128-token output limit:

```text
Explain why the sky looks blue in three short sentences.
```

This is a demo prompt to try, not a recorded successful run; a captured demonstration for this documentation update is not yet available.

[Follow TensorFold for new Apple Silicon releases and fixes.](https://huggingface.co/TensorFold)
<!-- TensorFold-chooser-end -->
