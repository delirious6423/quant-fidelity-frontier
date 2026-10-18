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
  - glm5-next
  - native-mtp
  - speculative-decoding
  - mixed-precision
  - mixture-of-experts
  - multimodal
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

<div align="center">
  <a href="https://huggingface.co/zai-org/GLM-5.3-Flash">
    <img src="zai-logo.png" width="72" height="72" alt="Z.ai logo">
  </a>
</div>


<p align="center">
  <img src="https://img.shields.io/badge/Z.ai-GLM--5.3--Flash-111827?style=flat-square" alt="Z.ai GLM-5.3-Flash">
  <img src="https://img.shields.io/badge/Apple_Silicon-MLX-000000?style=flat-square&logo=apple&logoColor=white" alt="Apple silicon MLX">
  <img src="https://img.shields.io/badge/Native_MTP-Depth_1-2563EB?style=flat-square" alt="Native MTP depth 1">
  <img src="https://img.shields.io/badge/TensorFold-Hybrid_2--bit-6E56CF?style=flat-square&logo=huggingface&logoColor=white" alt="TensorFold hybrid 2-bit">
</p>


<h1 align="center">GLM-5.3-Flash, hybrid MLX 2-bit with native MTP</h1>


<p align="center">
  A deterministic quality-protected MLX conversion of <a href="https://huggingface.co/zai-org/GLM-5.3-Flash">zai-org/GLM-5.3-Flash</a>, retaining the model's matching native next-token prediction layer.
</p>


<p align="center">
  <a href="https://huggingface.co/zai-org/GLM-5.3-Flash">Original model</a> ·
  <a href="https://z.ai/blog/glm-5.3-flash">Z.ai overview</a> ·
  <a href="https://arxiv.org/abs/2602.15763">Technical report</a> ·
  <a href="https://github.com/ml-explore/mlx">Apple MLX</a> ·
  <a href="LICENSE">MIT licence</a>
</p>


> [!IMPORTANT]
> This is a hybrid 2-bit checkpoint, not a uniform Q2 build. Routed trunk experts use Q2, non-expert language projections use Q8, native-MTP expert projections use Q4, and sensitive components remain BF16. This allocation passed live generation where broader Q2 recipes did not.


## At a glance


| Item | Value |
| --- | --- |
| Repository | `TensorFold/GLM-5.3-Flash-MLX-2bit-MTP` |
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| Source revision | [`04c4e9e95c5da8862dced7e5056455116f83a7e0`](https://huggingface.co/zai-org/GLM-5.3-Flash/commit/04c4e9e95c5da8862dced7e5056455116f83a7e0) |
| Source weight format | FP8 E4M3 with 128x128 block scaling |
| Output format | MLX safetensors, affine weight quantisation, group size 64 |
| Quantisation | Deterministic hybrid 2-bit recipe |
| Native MTP | Preserved, one upstream prediction layer, runtime depth 1 |
| Indexed tensors | 114,154 |
| Weight shards | 26 |
| Tensor payload | 111,317,859,192 bytes, 111.318 GB / 103.673 GiB |
| Configured context | 1,048,576 tokens |
| Architecture | `glm5_next`, multimodal sparse MoE |


## Precision recipe


| Component | Treatment |
| --- | --- |
| Routed trunk experts | Q2 affine, group size 64; 36,288 source matrices stacked into 126 runtime modules |
| Non-expert language projections | Q8 affine, group size 64; 540 source matrices mapped to 552 runtime modules |
| Native-MTP routed experts | Q4 affine, group size 64; 864 source matrices stacked into 3 runtime modules |
| Sparse indexer projections | All 36 at Q8 affine, group size 64 |
| Token embedding and output head | BF16 |
| Vision encoder and projector | BF16 |
| MTP fusion projection | BF16 |
| Routers, hyper-connections, norms and other non-quantisable tensors | Source precision |


This is weight-only post-training quantisation. It does not retrain or fine-tune the upstream model.


## Runtime compatibility


GLM-5.3-Flash uses the `glm5_next` multimodal architecture and a native NextN/MTP block. Use a runtime that supports this architecture and per-module MLX quantisation metadata.


| Component | Tested version |
| --- | --- |
| oMLX | 0.6.3rc3, build 2475 |
| MLX | 0.32.0 |
| mlx-lm | 0.31.3 |
| mlx-vlm | 0.6.3 |
| Native-MTP draft depth | 1 |


The current upstream chat template defaults to maximum reasoning effort. For short, direct answers, pass `reasoning_effort: low` through the chat-template arguments.


## Download and use


```bash
hf download TensorFold/GLM-5.3-Flash-MLX-2bit-MTP \
  --local-dir ./GLM-5.3-Flash-MLX-2bit-MTP
```


Add the downloaded directory to a compatible oMLX model directory, refresh the model registry, and select the model. Native MTP is optional and uses draft depth 1.


For an OpenAI-compatible request through oMLX, include the upstream low-reasoning template option when you want a concise answer:


```json
{
  "model": "GLM-5.3-Flash-MLX-2bit-MTP",
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "chat_template_kwargs": {"reasoning_effort": "low"}
}
```


## Apple M3 Studio performance


Each result is the median of three 512-token text-generation runs after a separate warm-up. Both modes used the same checkpoint, prompt, deterministic sampling settings, and current upstream chat template.


| Runtime mode | Runs | Output per run | Median decode speed | Long-run output parity |
| --- | ---: | ---: | ---: | --- |
| Native MTP disabled | 3 | 512 tokens | **6.063 tok/s** | Reference |
| Native MTP enabled, depth 1 | 3 | 512 tokens | **6.257 tok/s** | Exact match |


Native MTP improved median decode throughput by **3.21%** in this test. All six 512-token runs produced the same output hash. MTP gains depend on the prompt and draft acceptance, so treat this as a practical reference for the tested Studio rather than a universal result.


## Architecture


GLM-5.3-Flash combines KDA linear-attention layers with periodic sparse-attention layers, a sparse mixture-of-experts feed-forward stack, manifold-constrained hyper-connections, a vision encoder, and one native next-token prediction layer.


| Architecture detail | Upstream value |
| --- | ---: |
| Parameters | 320B total / 18B active |
| Language layers | 45 |
| Routed / active experts | 288 / 8, plus 1 shared expert |
| Hidden size | 4,096 |
| Attention heads | 64 |
| Native MTP layers | 1 |
| Configured maximum context | 1,048,576 tokens |


See the [official model card](https://huggingface.co/zai-org/GLM-5.3-Flash), [Z.ai overview](https://z.ai/blog/glm-5.3-flash), and [GLM-5 technical report](https://arxiv.org/abs/2602.15763) for upstream training, evaluations, intended uses, and safety guidance.


## Validation


| Check | Result |
| --- | --- |
| Official source structure | 76,108 tensors, 62 shards and 37,338 FP8 weight-scale pairs validated before conversion |
| Safetensors index and shard resolution | 114,154 entries resolve to 26 final shards |
| Saved precision layout | 36,288 Q2, 864 Q4 and 540 Q8 source matrices; all quantised matrices have matching weight, scale and bias tensors |
| Vision payload | 347 BF16 source tensors preserved |
| Native MTP structure | Complete upstream prediction layer preserved; runtime reports native-MTP compatibility |
| MTP disabled generation | Deterministic factual, arithmetic, instruction and coherence checks passed |
| MTP enabled generation | The same checks passed with exact output parity |
| Sustained generation | Three 512-token runs per mode; 6.063 tok/s off and 6.257 tok/s on |


## Limitations


- Hybrid quantisation can reduce quality relative to the official checkpoint. The effect depends on the workload.
- The configured one-million-token context does not mean every Apple silicon system has enough memory for a full-context request.
- Image and video prefill have different memory and throughput characteristics from text-only generation. The published throughput numbers are text-only.
- Native MTP may be neutral or slower on prompts with low draft acceptance.
- Runtime support for `glm5_next`, mixed per-module quantisation, and native MTP is version-sensitive.


This is a community quantisation, not an official Z.ai release.


## Licence and attribution


The upstream model uses the **MIT License**. The official licence text is included as `LICENSE`.


Model design, training, upstream evaluations, and documentation belong to Z.ai and the GLM-5 contributors. The MLX conversion, native-MTP preservation, validation, and packaging are provided by [TensorFold](https://huggingface.co/TensorFold).


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

The exact tested oMLX application version is not recorded here; a library version is not an app version. The original performance tables retain their benchmark conditions and speed figures; this documentation update adds no new test results.

### Quick start and demo prompt

```bash
hf download TensorFold/GLM-5.3-Flash-MLX-2bit-MTP --local-dir ./models/GLM-5.3-Flash-MLX-2bit-MTP
```

Add the downloaded folder to oMLX model directories, refresh the list, and follow this card's architecture and MTP compatibility requirements before loading.

Try this in a new chat with a 128-token output limit:

```text
Explain why the sky looks blue in three short sentences.
```

This is a demo prompt to try, not a recorded successful run; a captured demonstration for this documentation update is not yet available.

[Follow TensorFold for new Apple Silicon releases and fixes.](https://huggingface.co/TensorFold)
<!-- TensorFold-chooser-end -->
