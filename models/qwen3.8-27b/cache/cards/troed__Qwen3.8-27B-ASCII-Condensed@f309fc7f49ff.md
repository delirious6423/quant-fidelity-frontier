---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - qwen3
  - long-context
  - vocabulary-pruning
  - speculative-decoding
  - dflash2
  - llama.cpp
---

# Qwen3.8-27B-ASCII-Condensed

Qwen3.8-27B with a condensed ASCII-only vocabulary and a matching DFlash2 draft model. Meant to be used together with [troed/llama.cpp-adaptive-kv-streaming](https://github.com/troed/llama.cpp-adaptive-kv-streaming), a fork of [Raymond's KV streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming).

- vocabulary reduced from 248,320 to 129,006 rows in the embedding and output head
- the freed VRAM becomes KV cache: 160K context on a 16 GB GPU
- no retraining: surviving weights are bit-identical to the sources below

Note: In v2 of Raymond's phase arena there's support for MTP but currently not DFlash2. You thus only need the model file
if using my settings. The DFlash2 of course works fine with regular llama.cpp.

## Files

| File | Size | Role |
| --- | --- | --- |
| `Qwen3.8-27B-ASCII-Condensed-IQ4_XS-3.84bpw.gguf` | 11.4 GiB | target model |
| `Qwen3.8-27B-ASCII-Condensed-DFlash2-Q2_K_S-MIX.gguf` | 511 MiB | DFlash2 draft model (load with `md =`) |

## How the files were made

### Target

Created from [byteshape/Qwen3.8-27B-GGUF](https://huggingface.co/byteshape/Qwen3.8-27B-GGUF) (its `Qwen3.8-27B-IQ4_XS-3.84bpw.gguf`) using [bsaleh03's ASCII-Condensed-prune-tools](https://github.com/bsaleh03/ASCII-Condensed-prune-tools). The vocab rows were gathered directly in quantized space and the tokenizer was rewritten to match, so every surviving weight is bit-identical to the source: no dequantization, no requantization, no retraining.

### Draft

Created from the original [HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF](https://huggingface.co/HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF), with the same condensation applied to the draft's vocab tensors and tokenizer. No weights were changed beyond the row subset.

## Requirements

A build of [troed/llama.cpp-adaptive-kv-streaming](https://github.com/troed/llama.cpp-adaptive-kv-streaming):

```sh
cmake -B build -DGGML_NATIVE=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_QUANTS=q8_0-q4_0,q8_0-q8_0,q4_0-q4_0
cmake --build build --config Release -j
```

The files are standard GGUFs and load in any recent llama.cpp, but the `shared-device-memory-mib` settting need the fork.

## Usage

```sh
hf download troed/Qwen3.8-27B-ASCII-Condensed --local-dir models
```

Config for the fork's `llama-server` (local paths adjusted):

```ini
[Qwen3.8-27B]
spec-type = ngram-simple,draft-mtp
spec-draft-n-max = 5
spec-draft-p-min = 0.8
device-draft = CUDA0
n-gpu-layers-draft = all
# If all 16GB are available to the model, else lower this value
shared-device-memory-mib = 3904
# Any suitable template
chat-template-file = chat_template_qwen3.8.jinja
m = Qwen3.8-27B-ASCII-Condensed-IQ4_XS-3.84bpw.gguf
ctx-size = 200192
n-gpu-layers = 99
batch-size = 256
ubatch-size = 256
cache-type-k = q8_0
cache-type-v = q4_0
fit = off
parallel = 1
temp = 1.0
top-p = 0.95
top-k = 20
min-p = 0.0
presence-penalty = 0.0
repeat-penalty = 1.0
reasoning = on
reasoning-preserve = on
# Original GGUF-converted mmproj
mmproj = Qwen3.8-mmproj-BF16.gguf
load-mode = none
flash-attn = on
```

`chat-template-file` is any Qwen 3.8 jinja chat template. No mmproj is provided here; use e.g. the one from [byteshape/Qwen3.8-27B-GGUF](https://huggingface.co/byteshape/Qwen3.8-27B-GGUF) (`mmproj-bf16.gguf`).

## Performance

On an RTX 5060 Ti 16 GB + 96 GB DDR5, with the config above:

- prompt processing: ~500-1000 t/s
- token generation: ~30-50 t/s

Full setup walkthrough: [16 GB VRAM llama-server configs](https://blog.troed.se/projects/llama-server-16gb-vram-configs/).

## Language support

The vocabulary is ASCII only. The 256 byte-level fallback tokens are always kept, so non-ASCII text still decodes correctly, it just costs more tokens per character. If you need Latin-extended, Greek, currency or box-drawing characters, re-run the [prune tools](https://github.com/bsaleh03/ASCII-Condensed-prune-tools) with a wider policy.

## Credits

- [Qwen](https://huggingface.co/Qwen) for Qwen3.8-27B
- [Raymond Huang](https://github.com/RaymondHuang210129) for the KV cache streaming work
- [ByteShape](https://huggingface.co/byteshape/Qwen3.8-27B-GGUF) for the IQ4_XS-3.84bpw quantization (weights unaltered)
- [bsaleh03](https://github.com/bsaleh03/ASCII-Condensed-prune-tools) for the vocabulary pruning tools
- [HermiHg](https://huggingface.co/HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF) for the Q2_K_S-MIX DFlash2 draft (weights unaltered)

Licensed under Apache-2.0, inherited from the base model.