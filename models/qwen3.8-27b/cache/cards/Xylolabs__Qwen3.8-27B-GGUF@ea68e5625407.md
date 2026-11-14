---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
language:
- en
- ko
tags:
- gguf
- llama-cpp
- quantized
- qwen38
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>Qwen3.8-27B GGUF</h1>

[![Base model](https://img.shields.io/badge/base-Qwen%2FQwen3.8--27B-blue)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![Formats](https://img.shields.io/badge/formats-IQ4__XS%20%2F%20Q5__K__M-5B2C83)](#formats)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/blob/main/LICENSE)
[![Code](https://img.shields.io/badge/code-llama.cpp-black)](https://github.com/ggml-org/llama.cpp/tree/cd959bae99ff4bed22bef2604f45241c09bbb8c8)

`gguf` · `llama-cpp` · `quantized` · `qwen38`

</div>

Three existing GGUF quantizations of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), converted and
quantized by Xylolabs with llama.cpp. Choose plain IQ4_XS, IQ4_XS with the existing importance matrix, or
plain Q5_K_M. Each retains the text decoder and MTP weights; generation checks use the main decoder.

This export needs about 19.52 GB of device memory for its largest GGUF weight choice alone, plus room for the
KV cache and buffers.

## Formats

Each file contains 866 tensors from the 64-layer text decoder and one MTP module, plus the tokenizer and the
original chat template. Vision encoder/projector weights are omitted. The files are alternative checkpoints;
use one at a time. Average bits per weight include every stored tensor, including embedding, output head and
float32 parameters.

| File | Download GB | Average bits/weight | Mean KL | KL / Q4_K_M |
|---|---:|---:|---:|---:|
| `Qwen3.8-27B-IQ4_XS.gguf` | 15.420 | 4.512 | 0.024978 | 1.198× |
| `Qwen3.8-27B-IQ4_XS-imat.gguf` | 15.309 | 4.480 | 0.019497 | 0.935× |
| `Qwen3.8-27B-Q5_K_M.gguf` | 19.536 | 5.717 | 0.006743 | 0.323× |

Actual tensor types in these mixed presets:

| Choice | Tensor types and counts |
|---|---|
| `IQ4_XS` | F32: 360, IQ4_XS: 432, Q5_K: 73, Q6_K: 1 |
| `IQ4_XS-imat` | F32: 360, IQ4_XS: 440, Q5_K: 65, Q6_K: 1 |
| `Q5_K_M` | F32: 360, Q5_K: 439, Q6_K: 67 |

The `-imat` file preserves the existing importance-matrix result. Historical relative calibration filenames in
its metadata are provenance labels; no calibration dataset is included. No new training, calibration or
quantization was performed for publication.

All stored MTP tensors count toward the sizes above. An image-input path or an MTP drafter requires its own
compatible implementation and evaluation.

## Quickstart

With a [llama.cpp build](https://huggingface.co/docs/hub/gguf-llamacpp), download one file and run:

```bash
llama-cli -m Qwen3.8-27B-Q5_K_M.gguf -c 2048 -n 32 --temp 0 --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --single-turn -p "Name the capital of France. Reply with only the city."
```

This command is checked with llama.cpp v0.9.0. Select one quantization for inference. The example explicitly
limits context to 2,048 tokens; the original context metadata remains unchanged. Additional memory is required
for KV, recurrent state and working buffers.

## Evaluation

**All three files pass the relative reference check: 33/33 answers plus 7/7 thinking-on rechecks, with zero
counted losses or broken answers.** The stored BF16 pass and current Q4_K_M reference also answer 33/33 plus
7/7.

The two matched 2,048-token sequences cover 4,094 scored positions, one English text and one Korean text. Mean
KL is measured against the stored BF16 logits. Current Q4_K_M gives 0.020845; the ratios in the table are
below the 1.5× bar. The answer gate allows at most one more counted loss than Q4_K_M. All 33 prompt token
sequences, seven reasoning rechecks, stop IDs and generation budgets match the retained reference contract.

IQ4 checks use an 8,192-token context. Q5_K_M and its fresh paired Q4_K_M check use 4,352 tokens, which covers
the largest prompt plus generation budget (4,193 tokens). Both current Q4 runs produce the same KL and answer
totals. Counted losses exclude BF16 near ties (top-two gap below 1.0 at the first divergent token) and
reasoning items recovered with thinking on; broken answers count where BF16 is right.

The exact Q5_K_M CLI example loads the released file and completes one query using the main text decoder.
These checks concern stored texts and short prompts; an integration requires its own task evaluation. Speed is
not reported here.

## License

[Apache 2.0](LICENSE), as the original Qwen3.8-27B release. Full original terms and source/conversion
attribution are retained in `LICENSE` and `NOTICE`. Provided as is, without warranty or upstream endorsement.

## Citation

Use the citation supplied in the [original Qwen3.8-27B model
card](https://huggingface.co/Qwen/Qwen3.8-27B#citation):

```bibtex
@misc{qwen38,
  title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
  url = {https://qwen.ai/blog?id=qwen3.8},
  author = {{Qwen Team}},
  month = {August},
  year = {2026}
}
```

