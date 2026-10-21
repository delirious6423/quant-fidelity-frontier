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

# DeepSeek V4 Flash 0731 - Ombro v3c GGUF

Ombro v3c is an experimental mixed-precision GGUF quantization of the
304B-parameter
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
It targets consumer systems that must keep much of the routed-expert weights in
host memory while preserving the attention path and selected expert layers at
higher precision.

Ombro is a community quantization recipe, not an official GGUF quant type. This
repository is not affiliated with DeepSeek, Unsloth, or llama.cpp.

## File

| File | Size | Shards |
| --- | ---: | ---: |
| `DeepSeek-V4-Flash-0731-Ombro-v3c.gguf` | 85,801,572,832 bytes (85.80 GB / 79.91 GiB) | 1 |

## What changed from Ombro v1

Ombro v1 used very low precision for most routed experts and a higher-precision
band at the first and last five transformer layers. V3 first raised all routed
experts to `IQ2_XXS`. V3c retains that uniform baseline while restoring layers
0-4 and 38-42 to `IQ3_XXS`.

This choice is empirical rather than cosmetic. On the same held-out perplexity
set, v3 improved over v1 and restoring the edge layers improved the score again.

| Quant | File size | PPL |
| --- | ---: | ---: |
| Ombro v1 | 81.37 GB | 3.6723 +/- 0.06077 |
| Ombro v3 | 77.75 GB | 3.6002 +/- 0.05939 |
| **Ombro v3c** | **85.80 GB** | **3.5395 +/- 0.05809** |

Ombro v3 was an intermediate experiment and is not being released because its
generation results were not good enough. V3c is the release candidate derived
from that line of experiments.

The test used nine 4096-token chunks from a holdout that was excluded from the
importance-matrix corpus. These numbers establish a local relative ordering;
they are not a claim of parity with the original model or a broad public
benchmark result.

## Quantization recipe

The official checkpoint was converted to a mixed-precision GGUF intermediate.
The final quantization used a newly generated importance matrix covering 49
chunks of 4096 tokens (200,704 evaluated tokens). The calibration material mixed
English and German conversation, agentic/code material, and visual coding
prompts. The private calibration text is not distributed.

Importance-matrix coverage improved substantially over the matrix used for v1:

| Coverage measure | Ombro v1 matrix | V3/v3c matrix |
| --- | ---: | ---: |
| Routed-expert slots never activated | 160 | 0 |
| Slots with fewer than 50 activations | 2,784 | 25 |
| Median activations per expert | 142 | 3,176 |

DeepSeek V4 Flash has 43 transformer layers. Its attention path, embeddings,
router, and shared experts form the always-active backbone. The much larger
routed-expert section contains 256 experts per layer, of which only a small
subset is selected for each token. Within each expert, `gate` and `up` expand
and transform the hidden state, while `down` projects it back to the model
width.

| Part | What it contains | Precision | Approx. size |
| --- | --- | --- | ---: |
| Always-active backbone | Embeddings and output (`Q8_0`), attention/indexer and shared experts (`Q6_K`), router inputs (source-compatible), hyper-connection weights (`Q8_0`), norms, metadata, and tokenizer | Mixed | 6.33 GB / 5.89 GiB |
| Edge routed experts | All three expert projections in layers 0-4 and 38-42 (10 layers, 30 tensors) | `IQ3_XXS` | 24.66 GB / 22.97 GiB |
| Middle routed experts | All three expert projections in layers 5-37 (33 layers, 99 tensors) | `IQ2_XXS` | 54.81 GB / 51.05 GiB |
| **Complete GGUF** | All 1,328 tensors plus GGUF metadata and alignment | Mixed | **85.80 GB / 79.91 GiB** |

The sizes above are derived from the final tensor allocations and rounded to
two decimals. They are meant to show where the file size goes; the quant names
describe storage formats, not separate downloadable files.

The exact `llama-quantize` tensor matching rules are:

```text
^token_embd\.weight$=Q8_0
^output\.weight$=Q8_0
^blk\.[0-9]+\.(attn|indexer).*\.weight$=Q6_K
^blk\.[0-9]+\.ffn_(gate|up|down)_shexp\.weight$=Q6_K
^.*hc_.*\.weight$=Q8_0
^blk\.(0|1|2|3|4|38|39|40|41|42)\.ffn_(gate|up|down)_exps\.weight$=IQ3_XXS
^blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight$=IQ2_XXS
```

The quantization source, importance matrix, fallback type, and explicit tensor
map are preserved. The original one-off v3c shell invocation was not retained,
so the public recipe is a reconstruction from those artifacts rather than a
claim of byte-for-byte reproducibility across llama.cpp versions.

## MTP and speculative decoding

This GGUF contains the 43 main model layers only. The source checkpoint's MTP,
NextN, and DSpark draft tensors are not included by the conversion path used for
this file. Built-in speculative decoding is therefore unavailable; an external
compatible draft model is required if speculative decoding is desired.

## Running with llama.cpp

Use a recent CUDA-enabled llama.cpp build with DeepSeek V4 Flash 0731 support.
The following example matches a two-GPU consumer system where much of the model
must remain memory-mapped from fast local storage:

```bash
llama-server \
  --model DeepSeek-V4-Flash-0731-Ombro-v3c.gguf \
  --jinja \
  --ctx-size 131072 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k bf16 \
  --cache-type-v bf16 \
  --load-mode mmap \
  --no-repack \
  --fit on \
  --split-mode layer \
  --threads 16 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --ctx-checkpoints 0 \
  --cache-ram 0 \
  --temp 1.0 \
  --top-p 0.95
```

Adjust GPU offload, thread count, context, and batch sizes for your hardware.
Keep K and V cache types identical. **`bf16` is the recommended KV-cache type**
for Ombro v3c: it produced the most dependable local results and is the default
used for validation. `q8_0` also produced a usable manually reviewed result, but
the available Q8 testing is too limited and not prompt-matched closely enough to
claim equivalent quality. Use it when reducing KV-cache memory is more important
than staying on the best-tested path.

DeepSeek recommends `temperature=1.0`, with `top_p=0.95` for agentic scenarios
and `top_p=1.0` for other workloads. Do not add sampler values merely because a
frontend exposes them; explicit zero is not the same as an unset default.

## Validation status

- Loads and generates successfully with llama.cpp.
- Perplexity measured on a held-out local corpus as shown above.
- Local runtime tuning and generation tested at up to 128k context.
- No formal long-context accuracy, KL-divergence, knowledge, coding, agentic, or
  source-model equivalence benchmark has been completed.
- Visual one-shot benchmarks are useful smoke tests but are not treated as
  sufficient evidence of general model quality.

### Local visual smoke-test summary

Across three shared visual prompt families, the mean local Auto-Eval score was
53.3/100 for Ombro v1 and 67.3/100 for Ombro v3c. These checks primarily measure
whether generated applications load, render, move, and avoid runtime errors;
they do not directly score reasoning quality, code quality, or visual fidelity.
Run-to-run variance was substantial, so this aggregate is included only as a
smoke-test signal. The held-out perplexity result above is the cleaner controlled
comparison between the quantization recipes.

## License and attribution

The original model and weights are released under the MIT License. See the
[DeepSeek model repository](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
for the original model card, technical information, intended use, limitations,
and license terms.
