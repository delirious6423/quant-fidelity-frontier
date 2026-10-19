---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- qwen4_exp
- quantized
- 4-bit
- apple-silicon
---

# Qwen3.8-Flash-Next, MLX 4-bit (mixed precision)

This is a quantized copy of **[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, made by
the **Qwen Team (Alibaba Group)**. Qwen designed and trained the model, and all credit for it goes to them. Nace AI
only quantized the published weights to run on Apple silicon Macs. We did no training, fine-tuning or merging.
This is not an official Qwen release, and Qwen has not reviewed it.

For the model itself (architecture, benchmarks, intended use, limitations), see Qwen's
[model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and the
[Qwen3.8-Flash-Next blog post](https://qwen.ai/blog?id=qwen3.8-flash-next).

* Source: `Qwen/Qwen3.8-Flash-Next` at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` (bf16)
* License: the [Qwen Community License 1.0](LICENSE), unchanged from the source. Its terms apply to this copy as they
  do to the original. That includes condition 2: a Model-as-a-Service or AI Work Assistant business needs a separate
  license from Qwen for commercial use.
* Size: 106 GB on disk. About 73 GB of it goes in GPU memory: 71.4 GB of text model plus a 1.5 GB MTP head, and
  the vision tower (0.9 GB) loads on the first image. The 32 GB n-gram embedding table (`ple.safetensors`) is read
  from disk through memory mapping and never placed on the GPU. This needs a Mac with 128 GB of memory.

## What was done

MLX affine quantization (scale and bias in bf16 per group). The precision of each tensor type was chosen by
measuring each candidate against the bf16 model:

| Tensors | Format |
| --- | --- |
| Routed experts | 4-bit, group 64 |
| Gated DeltaNet projections, shared experts, `lm_head` | 4-bit, group 64 |
| Hyper-connection mixes, attention `q/k/v/o_proj` | 6-bit, group 64 |
| Router, shared-expert gate, sparse-attention indexer, n-gram key/value projections, `embed_tokens` | 8-bit, group 64 |
| n-gram embedding table | 4-bit, group 32 |
| MTP head | experts 4-bit, router and other linears 8-bit |
| Vision tower, norms, conv1d, `A_log`, `dt_bias` | bf16, as published |

Rounding is not plain round-to-nearest. For each group the quantizer picks the best of a min-max grid, a symmetric
grid and least-squares refits, after rounding scale and bias to bf16.

The layout follows mlx-vlm's `qwen4_exp` conventions:
* Norm weights stay zero-centred, as the source stores them.
* Fused experts are split into `switch_mlp.{gate,up,down}_proj`.
* The n-gram table is in `ple.safetensors`, described by `ple_manifest.json`.
* The MTP head is in `mtp.safetensors`.

`conversion.json` lists the source revision, the recipe, and every file's size and SHA-256.

## Quality

Next-token distributions over 4 x 1,024 tokens of C++, Python, English prose and agent chat with tool calls,
against the bf16 model run in fp32. KL is exact over the full 248,320-token vocabulary.

| Version | KL vs bf16 | Top-1 agreement | Perplexity (bf16: 4.054) | Decode bytes / token |
| --- | ---: | ---: | ---: | ---: |
| bf16 weights, bf16 activations (noise floor) | 0.024 | 94.6% | 4.021 | 13.3 GB |
| **This version** | **0.129** | **86.8%** | **4.109** | **4.13 GB** |
| Uniform 4-bit, group 32, round-to-nearest | 0.232 | 83.6% | 4.335 | 4.21 GB |
| Uniform 4-bit, group 64, round-to-nearest | 0.288 | 81.6% | 4.527 | 3.79 GB |

This model is more sensitive to quantization than earlier Qwen MoE models. The 6-bit hyper-connections and
attention account for most of the difference.

## Running it

Made for [Kapitsa](https://github.com/nace-ai/kapitsa), Nace AI's inference engine for Apple silicon. Support for
this model comes in Kapitsa's next release:

```sh
kapitsa serve --model flash-next
```

The files also follow mlx-vlm's layout. We loaded them in mlx-vlm 0.7.8 to produce our reference outputs. mlx-vlm
drops the MTP head and reads the n-gram table through `ple_manifest.json`.

## Citation

Please cite Qwen's work:

```bibtex
@techreport{qwen2026design,
    title       = {On the Design of {Qwen3.8-Next} Architecture: Evaluation, Efficiency, and Training Stability},
    author      = {{Qwen Team}},
    institution = {Alibaba Group},
    month       = {August},
    year        = {2026}
}

@misc{qwen3.8flashnext,
    title  = {{Qwen3.8-Flash-Next}: A New Architecture, Towards Ultimate Cost-Efficiency},
    author = {{Qwen Team}},
    month  = {August},
    year   = {2026},
    url    = {https://qwen.ai/blog?id=qwen3.8-flash-next}
}
```
