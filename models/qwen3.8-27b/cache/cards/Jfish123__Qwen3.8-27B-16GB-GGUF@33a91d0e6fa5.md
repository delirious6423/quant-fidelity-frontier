---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- ollama
- llama.cpp
- qwen3.8
- coding
- 16gb-vram
---

# Qwen3.8-27B for 16 GB GPUs: 125k context, fixed chat template

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) squeezed onto a single 16 GB card (tested on an RTX 5080) with
**125,000 tokens of context, 100% on the GPU**, plus two chat-template fixes so it works in **Claude Code** and
**OpenAI Codex** through Ollama.

Run it with Ollama:

```
ollama run awakenedsniper78/qwen3.8-coder
```

## What this is

- **Weights:** [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `UD-IQ3_XXS`, byte-for-byte
  unchanged (all 866 tensors verified). Only the embedded `tokenizer.chat_template` was edited.
- **Text only:** the vision projector is left out; its ~0.9 GB of VRAM goes to context instead.
- **Thinking on by default**, sampling as Qwen recommends for thinking mode (temperature 1.0, top_p 0.95, top_k 20).

## Measured on an RTX 5080 (16 GB)

| | |
|---|---|
| Context | 125,000 tokens |
| GPU offload | 66/66 layers (100% GPU) |
| VRAM | ~10.2 GB weights + 2.1 GB KV cache (q4_0) + 0.7 GB compute |
| Generation speed | ~52 tokens/s |
| Prompt processing | ~1,000–2,000 tokens/s |
| Load time | ~5 s from NVMe |

Qwen3.8's hybrid design makes long context cheap: only 16 of its 64 layers keep a KV cache (the rest are Gated
DeltaNet), so 125k tokens of q4_0 cache is about 2.1 GB.

## Required Ollama server settings

The 125k context only fits with a quantized KV cache. Set these environment variables for the Ollama server, then
restart Ollama:

```
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q4_0
OLLAMA_GPU_OVERHEAD=536870912
```

- `OLLAMA_KV_CACHE_TYPE=q4_0`: at the default (f16) or q8_0, 125k context doesn't fit next to the weights and layers
  spill to the CPU.
- `OLLAMA_GPU_OVERHEAD` (512 MB): with the default of 0, Ollama packs VRAM completely full, and the CUDA backend then
  crashed on load with `ptxas fatal: Memory allocation failure` / `CUDA error: a PTX JIT compilation failed`.

If your desktop already uses a lot of VRAM, lower the context: `/set parameter num_ctx 65536`.

## Chat template fixes

Unsloth's template raises errors that break popular agent tools. Two changes (full diff in
[`chat_template.fixes.diff`](chat_template.fixes.diff)):

1. **System messages after the start of the conversation.** The template raised
   `System message must be at the beginning.` Claude Code (via Ollama's Anthropic-compatible `/v1/messages`) sends
   these, so every request failed with HTTP 500. Late system messages are now rendered as normal
   `<|im_start|>system` turns.
2. **Unknown reasoning efforts.** The template only accepted `xhigh`, `high`, `medium` and `low`, and raised
   `Unexpected reasoning effort max` for anything else. Codex sends `max`. Now `max`, `ultra` and other unknown values
   map to `xhigh`, and `minimal`/`none` map to `low`.

Ollama's Modelfile `TEMPLATE` only accepts Go templates, so the Jinja template had to be rewritten inside the GGUF.
[`tools/gguf_set_template.py`](tools/gguf_set_template.py) does that with the standard library only, and checks that
every other key and all tensor data are unchanged:

```
python tools/gguf_set_template.py original.gguf fixed.gguf chat_template.jinja
```

## Run from this repo instead of the Ollama registry

```
hf download Jfish123/Qwen3.8-27B-16GB-GGUF --local-dir qwen3.8-coder
cd qwen3.8-coder
ollama create qwen3.8-coder -f Modelfile
```

llama.cpp:

```
llama-server -m Qwen3.8-27B-UD-IQ3_XXS-fixed.gguf -c 125000 -ngl 99 -fa on -ctk q4_0 -ctv q4_0
```

## Benchmark

[`benchmark/lcb_score.py`](benchmark/lcb_score.py) runs all 175 problems of
[LiveCodeBench](https://huggingface.co/datasets/livecodebench/code_generation_lite) v6 against a local Ollama
model, grades each one against its hidden tests, and prints a pass@1 score. It saves results as it goes, so a stopped
run resumes where it left off.

**Score for this quant: run in progress.**

Published references, at full precision (from the Qwen3.8-27B model card): Qwen3.8-27B **90.3**, Claude Opus 4.6
Max **88.8**. This harness is close to, but not identical to, the official one, so treat differences of a few points
as noise.

## Credits

- [Qwen team](https://huggingface.co/Qwen) for Qwen3.8-27B (Apache 2.0)
- [Unsloth](https://huggingface.co/unsloth) for the dynamic IQ3_XXS quantization
