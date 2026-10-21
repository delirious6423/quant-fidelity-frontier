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
  - glm
  - glm5
  - glm5-next
  - native-mtp
  - speculative-decoding
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
  <img src="https://img.shields.io/badge/TensorFold-MLX-6E56CF?style=for-the-badge&logo=huggingface&logoColor=white" alt="TensorFold MLX">
</p>


<h1 align="center">GLM-5.3-Flash — MLX 4-bit with native MTP</h1>


<p align="center">
  A native Apple-silicon conversion of <a href="https://huggingface.co/zai-org/GLM-5.3-Flash">zai-org/GLM-5.3-Flash</a>, quantised from the official checkpoint with its matching next-token prediction block preserved.
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
| Repository | `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` |
| Format | MLX safetensors |
| Quantisation | Uniform affine 4-bit for eligible modules |
| Group size | 64 |
| Source revision | `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a` |
| Source weight format | FP8 E4M3 with 128×128 block scaling |
| Native MTP | Included — one matching upstream prediction layer |
| MTP weight precision | 4-bit affine for eligible matrices; source-compatible precision for norms and routing tensors |
| Weight tensors | 114160 total, including 2641 MTP tensors |
| Weight shards | 43 |
| Download size | 181.694 GB / 169.215 GiB |
| Configured context | 1,048,576 tokens |
| Architecture | `glm5_next`, multimodal sparse MoE |


Modules that are not eligible for MLX quantisation remain at their source-compatible precision. The checkpoint preserves the upstream tokenizer, chat template, multimodal processor, generation configuration, and native MTP metadata.


## Quantisation recipe


| Component | Treatment |
| --- | --- |
| Eligible language projections | 4-bit affine, group size 64 |
| Routed experts | 4-bit affine for eligible expert projections; routing tensors preserved at source-compatible precision |
| Token embeddings and output head | BF16 |
| Vision encoder and projector | BF16, preserved from the official source |
| Native MTP prediction layer | 4-bit affine for eligible matrices; source-compatible precision for norms and routing tensors |
| Other non-quantisable tensors | Preserved at source-compatible precision |
| Converter | TensorFold streamed MLX affine converter using MLX 0.32.0 |


This is a weight-only post-training quantisation. It does not retrain or fine-tune the upstream model.


## Runtime compatibility


GLM-5.3-Flash uses the new `glm5_next` multimodal architecture, hybrid linear and sparse attention, manifold-constrained hyper-connections, and an embedded next-token prediction layer. All four features need explicit runtime support.


| Runtime path | Status |
| --- | --- |
| oMLX `0.6.3rc3` | Validated for baseline generation; native MTP tensors recognised |
| MLX `0.32.0` | Conversion and tensor runtime |
| MLX-VLM `0.6.3 with GLM5 Next runtime integration` | Required for `glm5_next` model and vision loading |
| Older or stock builds without GLM5 Next MTP integration | Unsupported for native MTP |


> [!IMPORTANT]
> A runtime that can load `glm5_next` is not automatically able to use its embedded MTP layer. Native MTP requires the GLM5 Next-specific draft, verification, and cache-reconciliation path. If that integration is absent, keep MTP disabled or use a compatible build.


## Download and use


```bash
hf download TensorFold/GLM-5.3-Flash-MLX-4bit-MTP \
  --local-dir GLM-5.3-Flash-MLX-4bit-MTP
```


In a compatible oMLX build:


1. Add the downloaded model directory.
2. Keep **Native MTP** disabled for the currently validated serving path.
3. Start a new conversation after changing model settings so the runtime rebuilds the model and caches.


For baseline generation without native MTP, a compatible MLX-VLM installation can use:


```bash
python -m mlx_vlm.generate \
  --model TensorFold/GLM-5.3-Flash-MLX-4bit-MTP \
  --prompt "Explain how hybrid attention reduces long-context serving cost." \
  --max-tokens 512
```


## Apple M3 Studio performance


Measured on an Apple M3 Studio with a warmed model state:


| Mode | Prompt tokens | Output tokens | Decode | Model memory |
| --- | ---: | ---: | ---: | ---: |
| Baseline, MTP off | 63 | 512 | 6.2657 tokens/s | 173.39 GB |


| Benchmark detail | Value |
| --- | --- |
| Chip | Apple M3 Ultra |
| Unified memory | 256 GB |
| Runtime | oMLX `0.6.3rc3`, MLX `0.32.0` |
| Timed runs | three; median reported |
| Warm-up | one 64-token greedy request discarded before measurement |
| Sampling | greedy decoding, temperature 0, top-p 1, top-k 0, fixed seed 6330 |


These figures are a local reference, not a cross-platform guarantee. Prompt shape, image input, context growth, draft acceptance, sampling, memory pressure, thermal state, and runtime revisions can materially change performance.


### Native MTP status


The checkpoint contains the official matching NextN layer. A short deterministic
MTP smoke test produced the exact requested output and recorded an accepted
draft. Sustained testing on the current oMLX build exposed repeated speculative
state reinitialisation and lower throughput than baseline decode. Native MTP is
therefore preserved for compatible future runtimes but is **not recommended for
sustained serving on the validated build**. Keep MTP disabled unless your
runtime has corrected that scheduler behaviour.


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
| Config, processor, tokenizer, and chat-template files present | Passed — official tokenizer, updated chat template, processor, generation config, and MIT licence included |
| Safetensors index resolves every tensor to an existing shard | Passed — 114,160 indexed tensors resolve exactly |
| All shards open and tensor headers match the index | Passed — all 43 shards open; no missing, duplicate, or unexpected tensors |
| Native MTP configuration and 2641 MTP tensors present | Passed — the official layer 45 NextN block is preserved and strict MTP-enabled loading succeeds |
| Baseline text generation | Passed — three identical coherent 512-token runs and all instruction, factual, and arithmetic gates |
| Native-MTP text generation | Short deterministic smoke passed with an accepted draft; sustained MTP is slower than baseline on this runtime |
| Greedy baseline/MTP output equivalence | Not claimed; sustained MTP serving is not recommended on this runtime |
| Vision tensor structure | Passed structurally — all 347 official vision tensors are preserved; image generation was not benchmarked |
| Long-context generation | Not claimed — configured context metadata is preserved, but no long-context generation result is reported |


Validation used oMLX `0.6.3rc3`, MLX `0.32.0`, and MLX-VLM `0.6.3 with GLM5 Next runtime integration`.


## Limitations


- Quantisation can reduce quality relative to the official checkpoint, especially on difficult reasoning, tool-use, multilingual, vision, and long-context workloads.
- Native MTP improves speed only when draft acceptance offsets drafting and verification cost. It may be neutral or slower for unpredictable output.
- The configured one-million-token context does not imply that every Apple-silicon system has enough memory for a full-context request.
- Image prefill has different memory and throughput characteristics from text-only generation and is not represented by the decode benchmark above.
- Runtime support for `glm5_next` and its native MTP path is evolving. Use the exact validated versions listed in this card.
- Validate numerical behaviour, output quality, safety, and tool use on representative workloads before production deployment.


This is a community quantisation and is not an official Z.ai release.


## Licence and attribution


The upstream model is released under the **MIT License**. The required licence text is included in this repository.


Model design, training, upstream evaluations, and documentation belong to Z.ai and the GLM-5 contributors. The MLX conversion, Apple-silicon validation, native-MTP integration work, and packaging are provided by [TensorFold](https://huggingface.co/TensorFold).


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
hf download TensorFold/GLM-5.3-Flash-MLX-4bit-MTP --local-dir ./models/GLM-5.3-Flash-MLX-4bit-MTP
```

Add the downloaded folder to oMLX model directories, refresh the list, and follow this card's architecture and MTP compatibility requirements before loading.

Try this in a new chat with a 128-token output limit:

```text
Explain why the sky looks blue in three short sentences.
```

This is a demo prompt to try, not a recorded successful run; a captured demonstration for this documentation update is not yet available.

[Follow TensorFold for new Apple Silicon releases and fixes.](https://huggingface.co/TensorFold)
<!-- TensorFold-chooser-end -->
