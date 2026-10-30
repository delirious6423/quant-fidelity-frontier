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

# DeepSeek-V4.1-Flash-1017 · Maya GGUF

GGUF quantizations of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(a mixture of experts: 384 routed experts per layer, 6 active per token, plus a shared expert), made for
**[Project Maya](https://github.com/mw00/project-maya)**. Project Maya keeps the most-used experts on the GPU and reads
the rest from RAM and the SSD, so the model runs well below its size. Both quants are made from DeepSeek's own release
(FP4 experts, FP8 dense weights), not from a re-quantized file, and keep every expert.

## Two quants

| | **Maya-S** | **Maya-XS** |
| --- | --- | --- |
| Made for | the closest answers to the original | the whole model in a 128 GB PC's memory |
| Model in memory | 178.2 GB | 115.3 GB |
| DSpark drafter | 8.0 GB (DeepSeek's FP4 experts) | 4.3 GB (its own, 2-bit experts) |
| Engram tables (read from the SSD) | 161.3 GB | 161.3 GB |
| Download, all files | 347.5 GB | 280.9 GB |
| Routed experts | IQ2_XXS gate/up, IQ3_XXS down (IQ2_S / IQ3_S in the first and last four layers) | IQ1_S |
| Attention, shared experts, embeddings, output | Q8_0 | Q8_0 |
| Same top token as the original | **92.6%** | **87.5%** |
| Next-token loss (NLL) vs the original | **+3.2%** | **+8.7%** |

Measured on held-out text never used for calibration, against the original model's own predictions (DeepSeek's
reference implementation running the original weights). As a yardstick, the original weights run through Project
Maya's engine give 96.4% - the engine's own rounding; the rest of each quant's distance is the quantization.

> **Built partly on the IST Austria DAS Lab ([ISTA-DASLab](https://huggingface.co/ISTA-DASLab)) recipe.** The experts
> are rounded with GPTQ-style error feedback (each against its own input statistics), and the result is measured
> against the full-precision model.

## How they were made

1. **Calibration text** in DeepSeek's own chat format, thinking included: code, tool calls and agent traces,
   reasoning, chat, multilingual. Maya-S: 128 sequences of 2,048 tokens; Maya-XS: three times as many, plus the
   model's own answers. A separate held-out set is kept for evaluation.
2. **Statistics from the model itself**, layer by layer, an importance matrix for every expert separately. Maya-XS is
   calibrated on the quantized model itself: each layer sees what the already-quantized layers before it produce.
3. **Error-feedback rounding** of every expert, at the quantizer's 256-weight blocks, each expert against its own
   input statistics.
4. **The Engram tables** (DeepSeek's n-gram memory) in Q6_K: against the tables as released (FP8) they cost no
   measurable quality (97.5% the same top token - the measurement's own noise floor). They are read a few rows a token
   from the SSD, so they cost disk space, not memory.
5. **llama.cpp's own quantizers (ggml)**, so the files are ordinary GGUFs.

## Files

| Folder | Files | Size |
| --- | --- | ---: |
| `Maya-S/` | `DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00001-of-00006.gguf` … `-00006-of-00006.gguf`: the model in files 1-4, the Engram tables in files 5-6 | 339.5 GB |
| `Maya-XS/` | `DeepSeek-V4.1-Flash-1017-Maya-XS-IQ1_S-00001-of-00005.gguf` … `-00005-of-00005.gguf`: the model in files 1-3, the Engram tables in files 4-5 | 276.6 GB |
| `dspark/` | `dspark.bin` / `.json`: Maya-S's drafter; `dspark-xs2.bin` / `.json`: Maya-XS's drafter | 8.0 / 4.3 GB |

Router, norms, hyper-connection mixing and attention sinks stay as released (F32 / BF16). The vision tower comes from
DeepSeek's own checkpoint, unchanged; the setup fetches it.

## Running it

**[Project Maya](https://github.com/mw00/project-maya)** sets everything up and runs it:

```sh
git clone https://github.com/mw00/project-maya.git && cd project-maya
./setup.sh --setup --model DeepSeek-V4.1-Flash-1017 --ds4-quant maya-s     # Windows: START-MAYA.bat
./setup.sh --setup --model DeepSeek-V4.1-Flash-1017 --ds4-quant maya-xs    # v1.1.22 or later
```

It checks the PC, downloads the files (every one verified by its sha256, resumed if a download stops), compiles the
engine for your GPU(s) and starts it: the dashboard at `http://127.0.0.1:8080`, or the OpenAI- and
Anthropic-compatible API at the same address (streaming, tool calls, pictures).

## License

The model is DeepSeek's DeepSeek-V4.1-Flash-1017 under the MIT license; these quantizations are released under the
same license.
