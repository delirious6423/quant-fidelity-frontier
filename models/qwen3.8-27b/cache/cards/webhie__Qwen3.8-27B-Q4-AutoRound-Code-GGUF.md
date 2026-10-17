---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: llama.cpp
tags:
- qwen3_8
- autoround
- gguf
- q4_0
- quantization
- llama.cpp
- multimodal
- vision
- mtp
---
# Qwen3.8-27B Q4_0 AutoRound Code GGUF

A **mixed GGUF Q4_0 quantization** of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), produced with [Intel's AutoRound](https://github.com/intel/auto-round). The main linear weights use GGUF `Q4_0`; `output.weight` is retained as `Q6_K`, while normalization and selected state-space auxiliary tensors remain `F32`.

> **Key difference from a Transformers-format AutoRound checkpoint:** This release is packaged for direct use with **[llama.cpp](https://github.com/ggml-org/llama.cpp)**. It includes the language-model GGUF and the separate `mmproj-model.gguf` vision projector. The quantization used the `auto-round-best` recipe with 512 calibration samples from a code-heavy, mixed code/math/science calibration set. No benchmark results are reported for this release.

## TL;DR

- **Base**: Qwen3.8-27B (27B dense vision-language model)
- **Quant**: GGUF `Q4_0` mixed; symmetric 32-value blocks for the primary quantized tensors
- **Tool**: `auto-round-best` from Intel AutoRound 0.14.2 (1000 iterations, 512 samples)
- **Size**: 15.72 GB language-model GGUF plus 1.84 GB `mmproj` projector
- **Context**: 262,144 tokens natively
- **MTP**: One Multi-Token Prediction layer is retained in the language-model GGUF
- **Vision**: Image input is supported through the included `mmproj-model.gguf`; video input depends on the llama.cpp build's video support
- **Calibration**: 50% Open-SWE code traces, 25% OpenMathReasoning, and 25% OpenThoughts3 science examples

## Quick inference with llama.cpp

Use a recent llama.cpp build with Qwen3.8 support. The GGUF already contains the tokenizer and chat template, so no separate tokenizer files are required.

### Text and code

```bash
llama-cli \
  -m Qwen3.8-27B-Q4_0-AutoRound-Code.gguf \
  -ngl 999 \
  -c 262144 \
  -n -1 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  -cnv
```

The model's chat template enables thinking by default. Use llama.cpp's chat-template options when you need to change reasoning behavior, for example:

```bash
llama-cli \
  -m Qwen3.8-27B-Q4_0-AutoRound-Code.gguf \
  --chat-template-kwargs '{"enable_thinking":false}' \
  -p "Write a quicksort in Python."
```

### Image input

Download both GGUF files from this repository and pass the projector with `--mmproj`:

```bash
llama-cli \
  -m Qwen3.8-27B-Q4_0-AutoRound-Code.gguf \
  --mmproj mmproj-model.gguf \
  --image path/to/image.jpg \
  -p "Describe this image."
```

For video input, use `--video` instead of `--image` with a llama.cpp build compiled with video support.

## Quantization details

| Field | Value |
|---|---|
| Base | `Qwen/Qwen3.8-27B` |
| Method | AutoRound (`intel/auto-round`), **best recipe** |
| AutoRound version | 0.14.2 |
| Scheme | `GGUF:Q4_0` |
| Primary tensor type | `Q4_0` |
| Q4_0 block size | 32 values per block |
| Symmetric | yes, for Q4_0 tensors |
| Other tensor types | `output.weight`: `Q6_K`; selected tensors: `F32` |
| Language-model GGUF | 866 tensors; 15,715,853,632 bytes |
| Vision projector | 334 `F32` tensors; 1,842,940,160 bytes |
| Context length | 262,144 tokens |
| MTP | 1 prediction layer |
| Calibration samples | 512 |
| Calibration sequence length | 2,048 tokens |
| Calibration batch size | 1 |
| Gradient accumulation | 8 steps |
| Low GPU memory mode | enabled |
| torch.compile | not enabled in the quantization command |

### Tensor precision breakdown

The language-model GGUF contains:

- 505 `Q4_0` tensors: 26,046,627,840 parameters
- 1 `Q6_K` tensor (`output.weight`): 1,271,398,400 parameters
- 360 `F32` tensors: 2,671,616 parameters

The `F32` tensors include normalization weights and state-space auxiliary tensors such as `ssm_dt`, `ssm_a`, `ssm_conv1d`, and `ssm_norm`. The separate vision projector is fully stored as `F32`.

## Calibration data

The calibration file used for this quantization was `mixed-code-math-science-512.jsonl`, generated with seed 42 and pre-packed to approximately 2,304 tokens per row. AutoRound consumed it with `seqlen=2048` and `num=512`; the pre-packed data was used without concatenating rows.

| Source | Share | Rows | Details |
|---|---:|---:|---|
| [`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | 50% | 256 | `openhands` config, `qwen35_122b` split; Python, TypeScript, Rust, JavaScript, Go, Java, and PHP |
| [`nvidia/OpenMathReasoning`](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | 25% | 128 | Mathematical reasoning examples |
| [`open-thoughts/OpenThoughts3-1.2M`](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | 25% | 128 | Science-domain examples, including organic chemistry and physics |

The SHA-256 of the exact calibration JSONL used was:

```text
022cbb61e5c599a98c92f11304368f6352a2e1a575cffb47149e7f5ef15c5258
```

## Reproduction

The quantization was run from the project workspace with the following command:

```bash
auto-round-best \
  --model ./models/Qwen3.8-27B \
  --scheme "GGUF:Q4_0" \
  --format "gguf:q4_0" \
  --dataset "./calib/mixed-code-math-science-512.jsonl:num=512" \
  --output_dir ./quants/Qwen3.8-27B-AutoRound-Best-Q4_0-Mixed \
  --seqlen 2048 \
  --nsamples 512 \
  --bs 1 \
  --gradient_accumulate_steps 8 \
  --low_gpu_mem_usage
```

The source model is [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). The `mmproj-model.gguf` projector was exported from the same source model for llama.cpp multimodal inference.

## Acknowledgements

- [Alibaba / Qwen team](https://huggingface.co/Qwen) for the base [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) model
- [Intel AutoRound](https://github.com/intel/auto-round) for the quantization framework
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for GGUF inference and multimodal support
- The maintainers of [Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces), [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning), and [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) for the public calibration sources

## License

Apache 2.0 - same as the [Qwen3.8-27B base model](https://huggingface.co/Qwen/Qwen3.8-27B).

## Citation

If you use this quant, please cite the original Qwen3.8 release (see the [base model card](https://huggingface.co/Qwen/Qwen3.8-27B)) and the AutoRound paper:

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}

@article{cheng2023autoround,
  title   = {Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs},
  author  = {Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao and Liu, Yi},
  journal = {arXiv preprint arXiv:2309.05516},
  year    = {2023}
}
```
