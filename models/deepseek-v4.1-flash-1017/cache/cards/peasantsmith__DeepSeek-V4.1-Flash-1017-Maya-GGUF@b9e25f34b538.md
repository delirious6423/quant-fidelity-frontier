---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
quantized_by: peasantsmith
tags:
  - gguf
  - imatrix
  - mixture-of-experts
  - deepseek
  - project-maya
---

# DeepSeek-V4.1-Flash-1017 · Maya-S

**Maya-S picks the same next token as the original model 92.6% of the time** on held-out text (KL divergence
0.120). One set of files, 339.5 GB: the model in 178.2 GB, its Engram tables in 161.3 GB.

A GGUF quantization of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(a mixture of experts: 384 routed experts per layer, 6 active per token, plus a shared expert), made for
**[Project Maya](https://github.com/mw00/project-maya)**. Project Maya keeps the most-used experts on the GPU and
reads the rest from RAM and the SSD, so the model runs well below its size. More memory is faster, and every machine
is different: try it on yours. Made from DeepSeek's own release (FP4 experts, FP8 dense weights), not from a
re-quantized file.

- **Maya-S** (339.5 GB, one GGUF split of 6 files): the routed experts - most of the model - in about 2 to 3 bits
  (IQ2_XXS gate/up, IQ3_XXS down; IQ2_S / IQ3_S in the first and last four layers), the attention, shared experts and
  dense weights in 8-bit (Q8_0), and DeepSeek's two Engram tables (its n-gram memory) in 6-bit (Q6_K). The Engram
  tables are read a few rows a token from the SSD, so they cost disk space, not memory.

> **Built partly on the IST Austria DAS Lab ([ISTA-DASLab](https://huggingface.co/ISTA-DASLab)) recipe.** The experts
> are rounded with their GPTQ-style error feedback (each expert against its own input statistics from the original
> model), and the result is measured against the full-precision model.

## Against the original

Token by token, on held-out text never used for calibration (10 sequences of 2,047 tokens), run through Project
Maya's engine, with the original model's own predictions as the reference (DeepSeek's reference implementation
running the original weights):

| | Same top token as the original | KL divergence from the original |
| --- | ---: | ---: |
| The original weights, through Project Maya's engine | 96.4% | 0.047 |
| **Maya-S** | **92.6%** | **0.120** |

The first row is the yardstick's own floor: the engine's rounding against the reference implementation, with nothing
quantized. The rest of Maya-S's distance is the quantization.

**The Engram tables in Q6_K** cost no measurable quality: against the tables as DeepSeek released them (FP8), the model's
predictions differ by a KL divergence of 0.011 with 97.5% the same top token - the measurement's own noise floor. Against
the original, Maya-S scores a KL divergence of 0.0858 with the Q6_K tables and 0.0866 with the FP8 ones (4 held-out
sequences).

## Files

| Folder | Files | Size |
| --- | --- | ---: |
| `Maya-S/` | `DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00001-of-00006.gguf` to `-00006-of-00006.gguf`: the model in files 1-4 (178.2 GB), the Engram tables in files 5 and 6 (161.3 GB) | 339.5 GB |
| `dspark/` | `dspark.bin`, `dspark.json`: the model's DSpark drafter (its multi-token draft blocks, for speculative decoding) - DeepSeek's FP4 experts as released, its dense weights in 8-bit | 8.0 GB |

The vision tower comes from DeepSeek's own checkpoint, unchanged; Project Maya's setup fetches it.

## What is in it

| Tensors | Maya-S |
| --- | --- |
| Routed experts, gate and up | IQ2_XXS, rounded with error feedback (IQ2_S in the first and last four layers) |
| Routed experts, down | IQ3_XXS, rounded with error feedback (IQ3_S in the first and last four layers) |
| Attention, shared experts, embeddings, output, the Engram value projection | Q8_0 |
| Router, norms, hyper-connection mixing, attention sinks | F32, as released |
| The indexer's small projections | BF16, as released |
| Engram tables | Q6_K (files 5 and 6), from DeepSeek's released tables |

## How it was made

1. **Calibration text**: 128 sequences of 2,048 tokens in DeepSeek's own chat template, thinking included - a mix of
   chat, multilingual chat, reasoning, code and tool calls (in the model's own format). A separate held-out set, never
   used for calibration, is kept for evaluation.
2. **Statistics from the original model itself**, layer by layer: an importance matrix for every expert separately,
   not one per layer.
3. **Error-feedback rounding** of the experts (GPTQ-style, at the quantizer's 256-weight blocks, each expert with its
   own input statistics): on held-out activations it lowers the experts' output error by 35-50% against plain
   importance-weighted rounding at the same size.
4. llama.cpp's own quantizers (ggml), so the files are ordinary GGUFs.

## Running it

**[Project Maya](https://github.com/mw00/project-maya)** sets everything up and runs it (v1.1.17 or later for this
one-set layout; an install of v1.1.16's keeps working as it is):

```sh
git clone https://github.com/mw00/project-maya.git && cd project-maya
./setup.sh --setup --model DeepSeek-V4.1-Flash-1017 --ds4-quant maya-s     # Windows: START-MAYA.bat
```

It checks the PC, downloads the files (every one verified by its sha256, resumed if a download stops), compiles the
engine for your GPU(s) and starts it: the dashboard at `http://127.0.0.1:8080`, or the OpenAI- and
Anthropic-compatible API at the same address (streaming, tool calls, pictures).

## License

The model is DeepSeek's DeepSeek-V4.1-Flash-1017 under the MIT license; these quantizations are released under the
same license.
