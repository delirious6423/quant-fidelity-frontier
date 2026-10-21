---
license: apache-2.0
library_name: mlx
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - mlx
  - mlx-vlm
  - qwen3_5
  - vision-language
  - quantized
  - 8-bit
  - safetensors
---

# Qwen3.8-27B MLX 8-bit

Unofficial community conversion of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) for
Apple Silicon using MLX and MLX-VLM.

This is a format conversion and affine 8-bit quantization of the language
model weights. The vision tower remains in BF16. No fine-tuning, merging,
abliteration, distillation, or additional training was performed.

> This repository is not an official Qwen, Alibaba Cloud, Apple, MLX,
> MLX-VLM, Hugging Face, or LM Studio release, and those parties do not
> endorse it.

## Model details

| Field | Value |
| --- | --- |
| Upstream model | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Upstream revision | [`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) |
| Conversion date | 2026-08-20 |
| Format | MLX, sharded Safetensors |
| Language-model quantization | 8-bit affine, group size 64 |
| Vision tower | BF16 |
| Native context in the upstream configuration | 262,144 tokens |
| Repository size | Approximately 27.5 GiB |
| License | Apache License 2.0 |

The upstream model is a 27B native vision-language model. It supports text,
image, and video understanding, flexible thinking control, and agentic/tool
use through compatible serving stacks. See the
[`Qwen3.8-27B` model card](https://huggingface.co/Qwen/Qwen3.8-27B) for the
architecture, upstream evaluations, recommended sampling settings, intended
uses, and model-family limitations.

## What changed

- The language-model tensors were converted to MLX and quantized with affine
  8-bit weights using a group size of 64.
- The vision encoder was retained in BF16 by the converter.
- `tokenizer_config.json` embeds the same chat template also distributed as
  `chat_template.jinja`, improving compatibility with offline and local
  serving stacks.
- The generated configuration records the quantization parameters.
- Release documentation and SHA-256 checksums were added.

The exact file-level changes and reproducible conversion command are recorded
in [`MODIFICATIONS.md`](./MODIFICATIONS.md).

## Requirements

- A Mac with Apple Silicon.
- A current version of `mlx-vlm`; this conversion was produced and validated
  with `mlx-vlm==0.6.15` and `mlx==0.32.1`.
- Sufficient unified memory. Short-context validation peaked at approximately
  35.8 GB. At least 48 GB of unified memory is recommended for practical use.
  Long contexts and concurrent requests require substantially more memory.

## Use with MLX-VLM

Install the runtime:

```bash
python -m pip install -U "mlx-vlm>=0.6.15"
```

### Text

```bash
python -m mlx_vlm.generate \
  --model z4bfg/Qwen3.8-27B-MLX-8bit \
  --prompt "Explain affine weight quantization in three concise points." \
  --max-tokens 512 \
  --temperature 0.0
```

Add `--enable-thinking` when explicit thinking mode is desired and supported
by the installed MLX-VLM version.

### Image

```bash
python -m mlx_vlm.generate \
  --model z4bfg/Qwen3.8-27B-MLX-8bit \
  --image /path/to/image.jpg \
  --prompt "Describe the visible objects, their relationships, and the scene." \
  --max-tokens 512 \
  --temperature 0.0
```

### Python

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model_id = "z4bfg/Qwen3.8-27B-MLX-8bit"
model, processor = load(model_id)

images = ["/path/to/image.jpg"]
prompt = apply_chat_template(
    processor,
    model.config,
    "Describe the visible objects and the scene.",
    num_images=len(images),
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    image=images,
    max_tokens=512,
    temperature=0.0,
)
print(result.text)
```

For server usage, multi-image input, video handling, thinking controls, and
streaming, follow the current
[`mlx-vlm` documentation](https://github.com/Blaizzy/mlx-vlm).

## Local validation

These are local smoke and regression checks, not official Qwen benchmarks and
not a substitute for a broad accuracy evaluation.

The converted checkpoint and the exact upstream BF16 revision were run with
deterministic decoding on the same Apple Silicon host and test harness.

| Check | This 8-bit build | Upstream BF16 |
| --- | ---: | ---: |
| Functional cases passed | 5 / 6 | 5 / 6 |
| Peak memory observed in the suite | 35.78 GB | 56.55 GB |
| Typical generation rate in the suite | about 20-23 tok/s | about 8-11 tok/s |

Five of six generated outputs were byte-identical between the two builds. The
remaining failed probability case produced the same incorrect answer in both
builds, so that specific failure was not introduced by quantization. This
small result does not establish general accuracy parity.

Additional deployment checks completed successfully:

- direct MLX-VLM text and single-image inference;
- strict JSON output and syntactically valid Python generation;
- Russian-language instruction following;
- LM Studio text, image, and required tool-call requests;
- repeated unload/load cycles with a stable model identifier;
- four parallel prediction slots and an eight-request queue on a 512 GB M3
  Ultra host.

Performance depends on prompt length, output length, runtime version, thermal
state, context allocation, and concurrent workloads. The figures above should
not be treated as cross-platform benchmarks.

## Limitations and responsible use

- Quantization can change logits and outputs. Validate this checkpoint on your
  own task before relying on it.
- The model can hallucinate, produce incorrect code or factual claims, reflect
  training-data biases, and generate unsafe or inappropriate content.
- Tool calls are model-generated data. Validate tool names and arguments, apply
  least-privilege permissions, and require confirmation for consequential
  actions.
- The advertised context length is an architectural capability, not a memory
  guarantee. KV-cache and media-token memory can dominate at long contexts.
- Text and image paths were validated for this release. Video artifacts are
  retained, but video inference was not independently validated as part of
  this conversion release.
- This repository does not reproduce the upstream benchmark suite. Use the
  upstream card for official benchmark results and evaluation methodology.

Do not use model output as the sole basis for medical, legal, financial,
safety-critical, or other high-impact decisions.

## Reproducibility and integrity

The conversion environment, command, source revision, and modified files are
documented in [`MODIFICATIONS.md`](./MODIFICATIONS.md). File hashes are listed
in [`SHA256SUMS`](./SHA256SUMS).

## License and attribution

The upstream checkpoint is distributed under the Apache License 2.0. This
repository retains that license in [`LICENSE`](./LICENSE), preserves upstream
attribution, identifies the modified artifacts, and uses the same
`apache-2.0` license metadata.

Copyright in the original model remains with its original rightsholders. The
maintainer of this repository claims only responsibility for the conversion,
packaging, and release documentation, not ownership of the upstream model or
its trademarks.

See [`NOTICE`](./NOTICE) and [`MODIFICATIONS.md`](./MODIFICATIONS.md) for
additional attribution and modification notices.

## Citation

Please cite the upstream Qwen work when using this conversion:

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```

## Acknowledgements

- [Qwen Team](https://huggingface.co/Qwen) and Alibaba Cloud for the original
  model.
- [Apple ML Research](https://github.com/ml-explore) for MLX.
- The [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) maintainers and
  contributors for the conversion and inference tooling.
- [LM Studio](https://lmstudio.ai/) for one of the local compatibility and
  serving environments used during validation.

