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
    <img src="zai-logo.png" width="96" height="96" alt="Z.ai logo">
  </a>
</div>


<p align="center">
  <img src="https://img.shields.io/badge/Z.ai-GLM--5.3--Flash-111827?style=flat-square" alt="Z.ai GLM-5.3-Flash">
  <img src="https://img.shields.io/badge/Apple_Silicon-MLX-000000?style=flat-square&logo=apple&logoColor=white" alt="Apple silicon MLX">
  <img src="https://img.shields.io/badge/Native_MTP-Depth_1-2563EB?style=flat-square" alt="Native MTP depth 1">
  <img src="https://img.shields.io/badge/TensorFold-oQ2-6E56CF?style=flat-square&logo=huggingface&logoColor=white" alt="TensorFold oQ2">
</p>


<h1 align="center">GLM-5.3-Flash, MLX oQ2 with native MTP</h1>


<p align="center">
  A sensitivity-guided mixed-precision MLX conversion of <a href="https://huggingface.co/zai-org/GLM-5.3-Flash">zai-org/GLM-5.3-Flash</a>. The checkpoint retains the matching native next-token prediction layer.
</p>


<p align="center">
  <a href="https://huggingface.co/zai-org/GLM-5.3-Flash">Original model</a> ·
  <a href="https://z.ai/blog/glm-5.3-flash">Z.ai overview</a> ·
  <a href="https://arxiv.org/abs/2602.15763">Technical report</a> ·
  <a href="https://github.com/ml-explore/mlx">Apple MLX</a> ·
  <a href="LICENSE">MIT licence</a>
</p>


> [!IMPORTANT]
> Baseline text and image generation pass. Native MTP at depth 1 is experimental in the tested runtime: it produced coherent output and higher throughput, but it was not token-identical to baseline across a 512-token greedy run. Keep MTP disabled when exact deterministic reproducibility matters.


## At a glance


| Item | Value |
| --- | --- |
| Repository | `TensorFold/GLM-5.3-Flash-MLX-oQ2-MTP` |
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| Source revision | [`04c4e9e95c5da8862dced7e5056455116f83a7e0`](https://huggingface.co/zai-org/GLM-5.3-Flash/commit/04c4e9e95c5da8862dced7e5056455116f83a7e0) |
| Source weight format | FP8 E4M3 with 128x128 block scaling |
| Format | MLX safetensors |
| Quantisation | Sensitivity-guided oQ2 mixed precision |
| Planned effective allocation | 2.67 bits per weight |
| Sensitivity source | Retained 4-bit calibration proxy |
| Layer sensitivity scores | 45 |
| Quantised modules | 683 total: 126 Q2, 3 Q4, and 554 Q8 |
| Mandatory sparse indexers | 36 projections at Q8 affine, group size 64 |
| Native MTP | Preserved, one upstream prediction layer, runtime depth 1 |
| Indexed tensors | 3,125 |
| Native-MTP entries | 59 |
| Vision tensors | 347, all BF16 |
| Weight shards | 22 |
| Tensor payload | 110,127,789,756 bytes, 110.128 GB / 102.564 GiB |
| Weight shard files | 110,128,198,215 bytes, 110.128 GB / 102.565 GiB |
| Configured context | 1,048,576 tokens |
| Architecture | `glm5_next`, multimodal sparse MoE |


## oQ recipe


| Component | Treatment |
| --- | --- |
| Base allocation | Q2 affine, group size 64, 126 saved modules |
| Sensitivity input | 45 layer scores measured from the retained 4-bit proxy |
| Planner Q8 boosts | 555 |
| Saved Q8 overrides | 554 modules, group size 64 |
| Saved Q4 overrides | Three native-MTP switch-MLP projections, group size 64 |
| Sparse indexer projections | All 36 at Q8 affine, group size 64 |
| Routed experts | Mixed Q2 and Q8; three native-MTP switch-MLP projections use Q4 |
| Token embedding and output head | Q2 affine, group size 64 |
| Vision encoder and projector | BF16 |
| Native MTP prediction layer | 59 entries; 12 Q8 modules, 3 Q4 modules, and no Q2 modules |
| MTP fusion projection | `language_model.mtp.0.eh_proj` kept in BF16 |
| Other non-quantisable tensors | BF16 |


The planner selected 555 Q8 boosts. The saved checkpoint has 554 Q8 overrides because `language_model.mtp.0.eh_proj` remains BF16 to protect MTP fusion. This accounts for the one-module difference.


All 36 sparse indexer projections are in the saved Q8 set, including the three native-MTP indexer projections. Every quantised module has matching weight, scale, and bias entries.


This is weight-only post-training quantisation. It does not retrain or fine-tune the upstream model.


## Runtime compatibility


GLM-5.3-Flash uses the `glm5_next` multimodal architecture, hybrid linear and sparse attention, manifold-constrained hyper-connections, and one native next-token prediction layer. The oQ layout also needs per-module quantisation metadata support.


| Component | Tested version |
| --- | --- |
| oMLX | 0.6.3rc3, build 2475 |
| MLX | 0.32.0 |
| mlx-lm | 0.31.3 |
| mlx-vlm | 0.6.3 |
| Native-MTP draft depth | 1 |


Use these versions for the reported results. Other versions may not understand this checkpoint's `glm5_next`, oQ, or native-MTP paths.


## Download and use


```bash
hf download TensorFold/GLM-5.3-Flash-MLX-oQ2-MTP \
  --local-dir ./GLM-5.3-Flash-MLX-oQ2-MTP
```


Add the downloaded directory to a compatible oMLX model directory and refresh the model registry. Start with native MTP disabled. If you choose to test MTP, use draft depth 1 and read the validation note below.


## Apple M3 Studio performance


The baseline median comes from three 512-token runs. A separate clean-load pair used the same prompt and settings once per mode to compare MTP directly.


| Mode | Runs | Output per run | Sustained throughput | Drafted | Accepted | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP disabled | 3 | 512 tokens | 6.1440 tok/s median | n/a | n/a | n/a |
| MTP disabled, matched pair | 1 | 512 tokens | 6.1287 tok/s | n/a | n/a | n/a |
| Native MTP, depth 1, matched pair | 1 | 512 tokens | 6.5629 tok/s | 323 | 187 | 57.9% |


The matched MTP run was 7.1% faster. It is one run, not a median. Both modes generated 512 coherent tokens, but their exact greedy token sequences first differed at zero-based output index 313. Treat the speed result as experimental rather than a bit-exact acceleration claim.


## Architecture


GLM-5.3-Flash combines KDA linear-attention layers with periodic sparse-attention layers, a sparse mixture-of-experts feed-forward stack, manifold-constrained hyper-connections, and one native next-token prediction layer.


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
| Safetensors index and shard resolution | 3,125 entries resolve to 22 final shards |
| Shard and tensor-header integrity | Header and index key sets match |
| Saved mixed-precision layouts | 683 complete weight, scale, and bias triplets; 126 Q2, 3 Q4, and 554 Q8 |
| Planner and saved Q8 reconciliation | 555 planned; 554 saved because `language_model.mtp.0.eh_proj` is BF16 |
| Sparse indexer invariant | Exactly 36 modules; all Q8 affine with group size 64 |
| Native MTP structure | 59 entries; 12 Q8 modules, 3 Q4 modules, and no Q2 modules |
| Baseline sustained generation | 3 x 512 tokens; 6.1440 tok/s median |
| Native-MTP sustained generation | 1 x 512 tokens at depth 1; 6.5629 tok/s; 187 of 323 drafts accepted |
| Fresh MTP-off 512-token parity | Both modes were coherent and deterministic per mode, but exact token parity failed; the first difference was at zero-based output index 313 |
| Image input | Standard one-image chat request with no manual placeholder completed successfully and identified the test image as red and blue |


## Limitations


- Mixed-precision quantisation can reduce quality relative to the official checkpoint. The effect depends on the workload and allocation.
- Proxy-derived sensitivity scores depend on the calibration protocol and do not guarantee BF16-equivalent output.
- Native MTP helps only when draft acceptance offsets drafting and verification cost. It may be neutral or slower for unpredictable output.
- The current native-MTP throughput result is one matched run. Compare more runs before drawing a firm performance conclusion.
- In the tested runtime, native MTP was not token-identical to baseline over 512 greedy output tokens. Leave it disabled for bit-exact deterministic workflows.
- The configured one-million-token context does not mean every Apple silicon system has enough memory for a full-context request.
- Image and video prefill have different memory and throughput characteristics from text-only generation.
- Runtime support for `glm5_next`, oQ metadata, and native MTP is version-sensitive.


This is a community quantisation, not an official Z.ai release.


## Licence and attribution


The upstream model uses the **MIT License**. The official licence text is included as `LICENSE`.


Model design, training, upstream evaluations, and documentation belong to Z.ai and the GLM-5 contributors. The oQ conversion, native-MTP preservation, validation, and packaging are provided by [TensorFold](https://huggingface.co/TensorFold).


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
hf download TensorFold/GLM-5.3-Flash-MLX-oQ2-MTP --local-dir ./models/GLM-5.3-Flash-MLX-oQ2-MTP
```

Add the downloaded folder to oMLX model directories, refresh the list, and follow this card's architecture and MTP compatibility requirements before loading.

Try this in a new chat with a 128-token output limit:

```text
Explain why the sky looks blue in three short sentences.
```

This is a demo prompt to try, not a recorded successful run; a captured demonstration for this documentation update is not yet available.

[Follow TensorFold for new Apple Silicon releases and fixes.](https://huggingface.co/TensorFold)
<!-- TensorFold-chooser-end -->
