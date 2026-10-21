---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
license: mit
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - deepseek
  - deepseek-v4
  - moe
  - quantized
---

# DeepSeek V4 Flash 0731 - Ombro v1 GGUF

This repository contains **Ombro v1**, an experimental mixed-precision GGUF
quantization of the 304B-parameter
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
release. It is not based on the earlier DeepSeek V4 Flash preview. Ombro v1 is
the recipe name, not an official GGUF quant type, and is not affiliated with
DeepSeek or Unsloth.


*Included a new jinja I wrote, it allows Low, Medium, High and Max, based on unsloth's template*

## File

| File | Size | Shards |
| --- | ---: | ---: |
| `DeepSeek-V4-Flash-0731-Ombro-v1.gguf` | 81,372,387,840 bytes (81.37 GB / 75.79 GiB) | 1 |

## Quantization

The official weights were first converted to a mixed-precision intermediate
GGUF with `--outtype auto`. A newly generated importance matrix was then
collected from 24 calibration chunks at a context size of 512. The private
calibration text mixes conversational, agentic, and coding material and is not
distributed with this repository.

The final quantization used `llama.cpp` commit
`876a4321163249c43ca4e986818fab5ab081f282` and uses the following layout:

DeepSeek V4 Flash has 43 transformer layers. Its attention path, embeddings,
router, and shared experts form the always-active backbone. The much larger
routed-expert section contains 256 experts per layer, of which only a small
subset is selected for each token. Within each expert, `gate` and `up` expand
and transform the hidden state, while `down` projects it back to the model
width.

| Part | What it contains | Precision | Approx. size |
| --- | --- | --- | ---: |
| Always-active backbone | Embeddings and output (`Q8_0`), attention/indexer and shared experts (`Q6_K`), router inputs (source-compatible), hyper-connection function weights (`Q8_0`), norms, metadata, and tokenizer | Mixed | 6.33 GB / 5.89 GiB |
| Edge routed experts | All three expert projections in layers 0-4 and 38-42 (10 layers, 30 tensors) | `IQ3_XXS` | 24.66 GB / 22.97 GiB |
| Middle expert `down` | Down projections in layers 5-37 (33 tensors) | `IQ2_S` | 22.70 GB / 21.14 GiB |
| Middle expert `gate` + `up` | Gate and up projections in layers 5-37 (66 tensors) | `IQ1_S` | 27.68 GB / 25.78 GiB |
| **Complete GGUF** | All 1,328 tensors plus GGUF metadata and alignment | Mixed | **81.37 GB / 75.79 GiB** |

The sizes above are derived from the final tensor allocations and rounded to
two decimals. They are meant to show where the file size goes; the quant names
describe storage formats, not separate downloadable files.

One-dimensional norms, biases, and scalar tensors retain the source-compatible
types selected by `llama.cpp`. `IQ1_M` was supplied as the quantizer's fallback
type, but the completed quantization log contains no tensor converted to
`IQ1_M`: all eligible weight tensors were covered by the explicit map above.

## Running with llama.cpp

Use a recent `llama.cpp` build with DeepSeek V4 Flash 0731 support:

```bash
llama-server \
  --model DeepSeek-V4-Flash-0731-Ombro-v1.gguf \
  --jinja \
  --ctx-size 65536 \
  --flash-attn on \
  --no-repack \
  --temp 1.0 \
  --top-p 1.0
```

The model is larger than typical consumer VRAM. Select GPU offload, CPU-MoE
offload, context size, and KV-cache precision for your hardware. DeepSeek
recommends `temperature=1.0`, with `top_p=0.95` for agentic scenarios and
`top_p=1.0` otherwise.

## Validation Status

The GGUF has been verified to load and generate with `llama.cpp`. Multiple
local one-shot visual coding runs have completed successfully, including
interactive 3D output, but results remain preliminary. Ombro v1 has not yet
received broad perplexity, KL-divergence, knowledge, coding, agentic, or
source-model comparison testing. No equivalence to a higher-precision quant is
claimed.

## License and Attribution

The original model and weights are released under the MIT License. See the
[original model repository](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
for its model card, technical report, license, intended use, and limitations.
