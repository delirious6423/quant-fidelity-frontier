---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - harness-hub/Qwen3.8-27B-super-adapter
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - quantized
  - merged-lora
  - agents
  - opencode
  - nvfp4
  - kv-cache
  - blackwell
  - qwen3.8
---

# Qwen3.8-27B-DMAIC_ASI-S16-SA (GGUF)

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with the
[harness-hub super-adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter) **merged into the weights**
before quantizing, then built to our 16 GB recipe ([S16](https://huggingface.co/DMAIC-ASI/Qwen3.8-27B-DMAIC_ASI-S16-GGUF)):
one 13.2 GB file for agentic coding on a **16 GB GPU with over 100k tokens of context**. Built by DMAIC ASI Research.

The super-adapter is a LoRA trained for agent harnesses (pi, Hermes Agent, OpenCode). Its authors report large gains
on SWE-bench Pro and Terminal-Bench 2.1 over the base model. Usually you run it as a separate `--lora` file on top of
a quantized base. This file has it built in.

## Why a merged build instead of base + `--lora`

| | Base GGUF + `--lora` adapter | This file |
|---|---|---|
| Files to manage | two (base + 956 MB F16 adapter) | one |
| Extra VRAM for the adapter | about 0.96 GB of F16 weights on the GPU | none: same 13,207,096,320 bytes as S16 |
| Where the adapter is applied | on top of weights quantized **without** it, so the quantization error was never fitted to it | merged in full BF16 precision (scale 2.0 = alpha 192 / rank 96), **then** quantized |
| Quantization tuned for the adapted model | no | yes: importance matrix and GPTQ-style rounding of all 48 slimmed FFN tensors and the output head, run on the merged weights |
| Extra compute per token | low-rank matmuls on every adapted tensor | none |
| Built-in MTP speculative decoding | depends on your llama.cpp build applying the LoRA to the draft path | works as in S16 (the adapter does not touch the MTP head) |

On a 16 GB card, the adapter's 0.96 GB costs you a large slice of context. Merging first also lets the quantizer
optimise for the weights you actually run, rather than adding an unquantized correction on top of a quantization
fitted to different weights. The adapter's own card notes that the further the base is from BF16, the further you
drift from the evaluated behaviour.

## You need two things, from two places

| What | Where |
|---|---|
| The model: `Qwen3.8-27B-DMAIC_ASI-S16-SA-Q3_K_XL.gguf` (this repo) | https://huggingface.co/DMAIC-ASI/Qwen3.8-27B-DMAIC_ASI-S16-SA-GGUF |
| The llama.cpp build with the NVFP4 KV cache | https://github.com/DMAIC-ASI-Research/llama.cpp-nvfp4-kv |

The file is a standard GGUF and loads in stock llama.cpp. The NVFP4 KV cache, which is what makes 100k+ of context fit
on 16 GB, needs the build above (RTX 50-series GPUs). Step-by-step guide:
[NVFP4-HOWTO.md](https://github.com/DMAIC-ASI-Research/llama.cpp-nvfp4-kv/blob/main/NVFP4-HOWTO.md); for an AI
assistant doing the setup: [SETUP-AGENT.md](https://github.com/DMAIC-ASI-Research/llama.cpp-nvfp4-kv/blob/main/SETUP-AGENT.md).

## Quick start (RTX 50-series, 16 GB)

```bash
git clone https://github.com/DMAIC-ASI-Research/llama.cpp-nvfp4-kv
cd llama.cpp-nvfp4-kv
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$(command -v nvcc)
cmake --build build --config Release -j 4 --target llama-server llama-cli

hf download DMAIC-ASI/Qwen3.8-27B-DMAIC_ASI-S16-SA-GGUF Qwen3.8-27B-DMAIC_ASI-S16-SA-Q3_K_XL.gguf --local-dir models

build/bin/llama-server -m models/Qwen3.8-27B-DMAIC_ASI-S16-SA-Q3_K_XL.gguf \
  -ngl 99 -c 114688 --fit off -fa on -ctk nvfp4 -ctv nvfp4 -b 2048 -ub 256 -np 1 -kvu \
  --spec-type draft-mtp,ngram-simple -ctkd nvfp4 -ctvd nvfp4 --spec-draft-n-ubatch 64 \
  --alias qwen3.8-s16-sa --host 127.0.0.1 --port 8080
```

If loading runs out of memory (a desktop on the same GPU uses some), lower `-c` in steps of 8192. On a GPU with
nothing else on it, you can try up to `-c 120064`, as with S16.

On other GPUs, or in stock llama.cpp, use an ordinary KV cache, for example `-ctk q8_0 -ctv q8_0`, with a smaller
context.

Thinking is on by default, which is the setting the adapter's authors recommend. To point a harness at the server
(OpenCode, pi, Hermes Agent), use `http://127.0.0.1:8080/v1` as an OpenAI-compatible endpoint. The
[adapter card](https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter-GGUF) has ready-made configs; use the
alias above as the model name.

## The file

| | |
|---|---|
| File | `Qwen3.8-27B-DMAIC_ASI-S16-SA-Q3_K_XL.gguf` |
| Size | 13,207,096,320 bytes (3.86 bits per weight) |
| sha256 | `32a75ae495933829dca28472cab12d5099135987c3e80dd9e7366068d98521a2` |
| Quant | the S16 per-tensor layout, unchanged: mostly IQ4_XS, IQ3_S and IQ3_XXS, with Q5_K, Q4_K and Q3_K ("Q3_K_XL" is the size class, not a single llama.cpp type) |
| Contents | text model with the super-adapter merged, plus the built-in multi-token-prediction (MTP) head; no vision projector |
| Tested on | 16 GB RTX 50-series card that also drives a desktop: served at 115,712 tokens of context with MTP |

## Quality and speed

KL divergence against a Q8_0 reference of the same merged BF16 model (`llama-perplexity --kl-divergence`, wiki text,
20 chunks of 1024 tokens; lower is better). S16 measured the same way against its own Q8_0 reference:

| Model | Mean KLD (± se) | p99 KLD | Same top token |
|---|---:|---:|---:|
| **S16-SA (this file)** | 0.0349 ± 0.0012 | 0.343 | 92.5 % |
| S16 (no adapter) | 0.0340 ± 0.0008 | — | 92.5 % |

The merged model quantizes about as well as the base did: the same layout loses about the same information.

Against S16 on the same machine and settings, measured once each: MTP draft acceptance 0.53 vs 0.56; generation speed
about 8 % lower at a 512-token context, 4 % lower at 8k, and equal at 32k.

**Not measured yet:** agentic benchmark scores for this file. The SWE-bench Pro and Terminal-Bench numbers on the
adapter's card come from the BF16 adapter served by vLLM, not from a quantized build. Expect this file to follow them
more closely than a quantized base with the adapter added on top, but we have not run those benchmarks.

## How it was made

1. The two BF16 shards of `unsloth/Qwen3.8-27B-GGUF` (pinned revision `4ca7207`) were joined into one BF16 file.
2. The super-adapter (`Qwen3.8-27B-super-adapter-lora-F16.gguf`, rank 96, alpha 192) was merged in at BF16 with
   `llama-export-lora`. It touches attention q/k/v/o in the 16 full-attention layers and the FFN in all 64 blocks.
3. **Control check.** We first re-quantized the original, unadapted BF16 with the S16 type map. 817 of S16's 866
   tensors came out byte-identical to the published S16, which proves that the pipeline reproduces S16 exactly. The 49
   that cannot be replayed are named in the run record.
4. The merged model was quantized to the identical S16 layout with Unsloth's importance matrix.
5. All 48 slimmed IQ3_XXS FFN tensors and the output head were re-rounded GPTQ-style, using activation covariances
   measured **on the merged model** across code, tool-use and general text.
6. Gates: tensor types and file size equal to S16; only the 49 GPTQ-patched tensors differ from a plain quantize;
   KLD as above.

## Credits and license

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen Team, Apache-2.0.
- Adapter: [harness-hub/Qwen3.8-27B-super-adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter),
  Apache-2.0. All credit for the agentic training goes to its authors; this repo only merges and quantizes it.
- Uses [Unsloth](https://huggingface.co/unsloth)'s published imatrix and BF16 GGUF.
- Runs on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
- This build is released under Apache-2.0.

Sampling the base model recommends: thinking mode temperature 1.0, top_p 0.95, top_k 20; non-thinking temperature
0.7, top_p 0.8, top_k 20, presence_penalty 1.5.
