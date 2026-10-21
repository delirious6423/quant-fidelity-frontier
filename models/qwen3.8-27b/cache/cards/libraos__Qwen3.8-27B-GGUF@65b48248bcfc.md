---
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
library_name: gguf
pipeline_tag: image-text-to-text
quantized_by: unsloth
tags:
  - gguf
  - llama.cpp
  - qwen3
  - vision
---

# Qwen3.8-27B-GGUF

> ### Mirror
>
> These files are copies of [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF),
> re-hosted here because this is the exact build a Meganova node serves on the
> Swan Chain network, pinned so the running configuration stays reproducible.
> The quantization is **unsloth's** work, not ours, and the weights are
> unmodified.
>
> **If you just want the model, use
> [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)** —
> it carries the full quant ladder. This repo holds only the two files the node
> actually loads.

## Files

| file | size | what it is |
|---|---|---|
| `Qwen3.8-27B-UD-Q4_K_S.gguf` | 15.36 GB | the model, unsloth Dynamic 4-bit |
| `mmproj-BF16.gguf` | 931 MB | vision projector — required for image input |

Both are needed for the multimodal setup below. Text-only use needs just the first.

## What it actually costs, measured

Measured on the reference node, not estimated: **2x RTX 3080 (10 GB each)**.

| | |
|---|---|
| VRAM | **17.4 GiB** across two cards (9228 + 8594 MiB) |
| Context | 65536, KV cache at `q8_0` |
| Decode | **~33.5 tok/s** |
| Prefill | ~1.2–1.7k tok/s |

The vision encoder runs on **CPU** (`--no-mmproj-offload`): it does not fit on
the GPU alongside a 64k context on 10 GB cards. Expect ~10–25 s per 1280x960
image. On larger cards, drop that flag.

## The exact command that produces those numbers

```bash
docker run -d --name llama-qwen38 --restart unless-stopped \
  --gpus '"device=2,3"' -p 30001:8080 \
  -v /path/to/weights:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Qwen3.8-27B-UD-Q4_K_S.gguf \
  --mmproj /models/mmproj-BF16.gguf --no-mmproj-offload \
  --alias Qwen/Qwen3.8-27B --host 0.0.0.0 --port 8080 \
  -ngl 99 --tensor-split 11,9 -c 65536 --parallel 2 \
  -t 16 -tb 16 -fa on -ctk q8_0 -ctv q8_0 --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5
```

Notes that cost real time to find:

- **`--tensor-split 11,9`, not `1,1`.** The vision projector and the KV cache do
  not land evenly; an even split leaves card 0 short and the load fails.
- **Uniform `q8_0` KV cache.** Mixed or `q4_0` KV makes llama.cpp's CUDA
  flash-attention fall back to CPU, and prefill collapses to ~50 tok/s. `f16` or
  uniform `q8_0` are the safe choices.
- **`--alias` must be the model ID the router expects**, or requests arrive for
  a name the server rejects as unknown.

## Licence

Apache 2.0, from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
