---
base_model:
- Qwen/Qwen3.8-Flash-Next
- unsloth/Qwen3.8-Flash-Next-GGUF
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: gguf
pipeline_tag: text-generation
language:
- de
- en
- zh
tags:
- gguf
- qwen4exp
- qwen3.8-flash-next
- unsloth
- sakura
- experimental
---

# Sakura Qwen3.8 Flash-Next Compact UD-IQ1_S CP2

Experimental community repack of **Unsloth UD-IQ1_S**, retaining all **512 routed experts**. Shards 1 and 3 are copied unchanged. Shard 2 keeps every non-table tensor and replaces only `per_layer_token_embd.weight` with the existing Sakura CP2 table.

**This exact UD-IQ1_S + CP2 package has not been loaded for inference or tested for generation, perplexity, quality or speed.** This is an experimental structural release, not a model-quality or hardware qualification.

All three model files total **53.259 GiB**. Disk size is not running memory and does not prove that this configuration fits a 32 GB or 24 GB GPU.

## Hardware compatibility

The supplied binaries are the same Windows x64 Vulkan CP2 runtime as [Sakura K512 CP2](https://huggingface.co/webmp3/Sakura-Qwen3.8-Flash-Next-Compact-47.6GiB-K512-CP2-v1-GGUF). Vulkan support in a GPU driver alone does not establish compatibility. **This exact Unsloth + CP2 weight combination has not been loaded.** No speed, fit or quality number from K512, ISTA Compact or Gyro is a result for this package.

| Hardware / platform | Status | What to expect |
|---|---|---|
| AMD Strix Halo, Radeon 8060S, 64 GB UMA, Windows x64 | Runtime tested on the K512 CP2 package only | Same supplied runtime. This package is 53.3 GiB of model files and has not been loaded here. |
| Other Strix Halo configurations | Not tested | Similar hardware is not a separate validation. |
| Other AMD, NVIDIA or Intel GPUs | Not tested | No CUDA build is supplied. No hardware-specific profile has been qualified. |
| GPUs with 24 or 32 GB dedicated VRAM | Default profile not validated | Disk size is not running memory. Smaller cards may need different offloading; such profiles have not been tested. |
| CPU-only, Linux, macOS, Windows on ARM | No matching native binary supplied | This package contains Windows x64 binaries. |

Use the included runtime. Stock llama.cpp, LM Studio, Ollama and unrelated forks are not qualified for this custom CP2 table. Start `START.cmd`, then open **http://127.0.0.1:18121**. Alias `sakura-flash-next-ud-iq1-s-cp2`.

Backbone: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), revision `766911a6b7369840a91dbcd95f9f997acaab6cd6`. Existing CP2 table and runtime: [Sakura K512 CP2](https://huggingface.co/webmp3/Sakura-Qwen3.8-Flash-Next-Compact-47.6GiB-K512-CP2-v1-GGUF). Model weights retain **Qwen Community License 1.0**.

[简体中文](README_zh.md)
