---
language:
- multilingual
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- mxfp4
- quantized
- qwen3_5
---

# Qwen3.8-27B MXFP4 MLX

An unofficial MLX MXFP4 conversion of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced for
Apple Silicon inference. This repository is a quantized conversion, not a
fine-tune, merge, or retraining of the original model. It is not affiliated
with or endorsed by Qwen.

## Quantization

| Setting | Value |
| --- | --- |
| Source | `Qwen/Qwen3.8-27B` BF16 checkpoint |
| Original architecture parameter count | 27B |
| Conversion tool | `mlx-lm` 0.31.3 |
| Weight format | MXFP4 |
| Nominal bits | 4 |
| Group size | 32 |
| Non-quantized dtype | BF16 |
| Resulting disk size | approximately 14 GB |

The conversion command was:

```bash
mlx_lm.convert \
  --hf-path Qwen/Qwen3.8-27B \
  --quantize \
  --q-bits 4 \
  --q-mode mxfp4 \
  --q-group-size 32 \
  --dtype bfloat16 \
  --mlx-path Qwen3.8-27B-mxfp4
```

## Parameter-count note

This remains the full **27B-parameter Qwen 3.8 architecture**. Quantization
changes how the values are stored; it does not remove transformer layers or
turn the model into a 5B architecture.

The Hugging Face **“Model size 5B params”** badge is generated automatically
from the shapes of the packed MLX 4-bit safetensors. A packed 4-bit tensor holds
multiple original weight values in each stored element, so that generic counter
under-reports MLX quantized models. It is a Hub display limitation and cannot
be overridden from model-card metadata. The 13.4 GB MLX file-size badge is the
relevant storage figure for this conversion; the original parameter count is
27B.

## Usage

Load this repository with a current MLX-compatible runtime such as
[MLX-LM](https://github.com/ml-explore/mlx-lm) or
[oMLX](https://github.com/jundot/omlx). Qwen 3.8 requires a recent MLX-LM
release that supports the `qwen3_5` architecture.

This target can run on its own. For the long-context DFlash2 profile described
below, it is paired with a separately published, matched DFlash2 draft model.
The draft is an optional speculative-decoding helper; it is not the chat model.

## DFlash2 companion (optional)

For oMLX speculative decoding, pair this target with the separately published
[`deresolution/Qwen3.8-27B-DFlash2-mxfp4`](https://huggingface.co/deresolution/Qwen3.8-27B-DFlash2-mxfp4)
draft sidecar. It is an MLX MXFP4 conversion of the upstream
[`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).
Configure the sidecar as this target's DFlash draft model. Do not enable
Lightning MTP and DFlash simultaneously.

## Long-context configuration notes

On a 32 GB Apple Silicon Mac, this target was tested in oMLX with a 49,152-token
server window, TurboQuant KV cache at 3.5 bits, and a pre-quantized DFlash2
helper. That is a machine-specific configuration, not a claim that the model
will fit that context on every Mac. See the full build guide for the commands,
memory tradeoffs, and staged verification procedure.

## License and attribution

This derivative conversion retains the upstream
[Apache License 2.0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE).
See the source model's model card for its intended use, limitations, and
upstream documentation.
