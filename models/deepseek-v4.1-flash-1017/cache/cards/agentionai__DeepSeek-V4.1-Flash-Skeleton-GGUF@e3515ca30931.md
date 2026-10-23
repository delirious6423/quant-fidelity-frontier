---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- gguf
- deepseek
- test-fixture
- skeleton
- halloween
---

# 💀 DeepSeek-V4.1-Flash Skeleton GGUF

*All bones, no flesh.* 🎃

These are GGUF files with the exact architecture of
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): the same metadata, tokenizer,
tensor names, shapes and types. **The weights are not the real model.** They are for people building or testing
runtimes, converters and quantizers who don't want to download 510 GB first.

They will not answer your questions. They speak only in tongues.

## 🪦 What's in the crypt

| file | layers | experts | weights | size on HF |
|---|---|---|---|---|
| `DS41-small.gguf` | 6 | 128 | random (sane scales) | 20.4 GB |
| `make_skeleton.py` | 40 | 384 | all zero | builds the full 508 GB skeleton locally as a sparse file (~6 MB on disk) |

**The full skeleton** is built on your machine instead of downloaded:
- The script reads the real tensor headers from the official repo over HTTP range requests.
- It writes fake shards whose data regions are holes, then runs the converter with a writer that seeks over zero
  chunks.
- The result is a 508.4 GB file that takes about 6 MB on disk.

| type | tensors | bytes |
|---|---|---|
| MXFP4 (routed experts) | 120 | 288.8 GB |
| MXFP8 (Engram n-gram table) | 1 | 202.8 GB |
| BF16 | 351 | 16.4 GB |
| F32 | 571 | 0.5 GB |
| **total** | **1043** | **508.4 GB** |

**`DS41-small.gguf`** has the same tensor names and per-tensor shapes as the full model, scaled down:
- 6 layers, which cover every attention-layer kind (compress ratios 0/2/1, KV and index source layers).
- 128 routed experts instead of 384.
- Engram on layers 1 and 3, with the vocab cut down to about 1.2M rows per table.
- No MTP or vision tensors.
- Weights are random, with scales of about 1/sqrt(fan_in), so numerics don't degenerate.

## 🧟 Raising the dead

Both files need DeepSeek-V4.1 support in llama.cpp. At the time of writing that is the open upstream PR
[ggml-org/llama.cpp#30817](https://github.com/ggml-org/llama.cpp/pull/30817).

```bash
# small model, CPU, a few (meaningless) tokens
llama-cli -m DS41-small.gguf -ngl 0 -c 512 -n 8 -st -p "Hello" --temp 0

# full skeleton: loads and steps the whole 40-layer graph (output is noise)
python make_skeleton.py --out DS41-skeleton.gguf   # needs the PR's converter, torch, safetensors
llama-cli -m DS41-skeleton.gguf -ngl 0 -c 256 -n 2 -st -p "Hello" --temp 0
```

The converter drops the chat template on purpose, so use `-st`. The full skeleton ran 2 tokens on CPU under a 12 GB
memory cap with mmap.

## 🕸️ Good for / not good for

**Good for:**
- loader, graph and memory-planning tests;
- speed tests (zero weights cost the same to multiply);
- converter and quantizer plumbing;
- CI fixtures.

**Not good for:**
- anything about output quality, perplexity or KLD;
- the full skeleton's numerics. Every weight is zero, including norms and the router, so routing is degenerate.

## ⚰️ Notes
- Zero FP8 scales fail the converter's range check, so the skeleton writes 1.0 (0x7F) into the FP8 linear scale
  regions (7.2 MB of real bytes). Expert and Engram scales stay as holes.
- License: MIT, the same as the source model. No weights from the source model are included.



## Credits
Made by AgentionAI. Happy Halloween. 🦇

Built and tested on AMD Ryzen AI Max (Strix Halo). Hardware supported by AMD


### Support AgentionAI

These quants are released freely. If they save you VRAM or make Qwen more useful, you
can buy me a coffee or some GPU time and sponsor continued quantization and benchmarking on
[Buy Me A Coffee](https://buymeacoffee.com/agentionai). AgentionAI is a one-person team and can use your help.



