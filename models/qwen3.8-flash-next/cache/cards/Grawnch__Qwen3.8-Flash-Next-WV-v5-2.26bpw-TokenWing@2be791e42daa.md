---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
  - gguf
  - tokenwing
  - moe
  - qwen3.8-flash-next
  - wv-v5
  - quantized
---

# Qwen3.8-Flash-Next-WV-v5-2.26bpw-TokenWing

Pre-transcoded 2-shard **TokenWing (`wv-plugin-v1.1`)** GGUF checkpoint of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` total parameters, `48` hybrid `GDN` + sparse-`QSA` layers, `512` routed experts top-`10`), with embedded `vram_experts`, `usage_seed`, and `layer_order` manifests for high-throughput NVMe/RAM MoE streaming on 12 GB–24 GB consumer GPUs.

## Shards
- `Qwen3.8-Flash-Next-WV-v5-2.26bpw.tokenwing-00001-of-00002.gguf` (`35.39 GiB` — 48-layer trunk + `2.26 bpw` MoE experts + embedded TokenWing manifests)
- `Qwen3.8-Flash-Next-WV-v5-2.26bpw.tokenwing-00002-of-00002.gguf` (`26.82 GiB` — Layer-1 PLE table)

## Runtime Requirements
Requires `runmyai/tokenwing` (`feat/quant-plugin`, PR #8) + `wv-plugin-v1.1` (`libqfn_wv_plugin.so` on Linux or `qfn_wv_plugin.dll` on Windows) from [`grawnchtoader/tokenwing-assets`](https://github.com/grawnchtoader/tokenwing-assets/releases/tag/wv-plugin-v1.1).

### 1. Whole-Layer GPU + On-Load VNNI RAM Repack (12 GB VRAM / 16 GB RAM Laptop)
```bash
llama-completion -m Qwen3.8-Flash-Next-WV-v5-2.26bpw.tokenwing-00001-of-00002.gguf \
  --quant-plugin qfn_wv_plugin.dll \
  -ngl 99 -c 4096 -t 8 \
  --moe-stream model --moe-stream-ram 6GiB \
  --moe-stream-overlap 1 --moe-stream-cpu-fuse 1 --moe-stream-cpu-dispatch 1 \
  --moe-stream-cpu-repack 1 --moe-stream-cpu-x4 2 --moe-stream-idle-load 2 \
  --moe-router-min-w 0.015 --moe-router-k-min 4 --moe-stream-timing 1
```

### 2. Hybrid `H=96` Hot-Expert VRAM Mode (Lowest NVMe Read Bandwidth)
```bash
llama-completion -m Qwen3.8-Flash-Next-WV-v5-2.26bpw.tokenwing-00001-of-00002.gguf \
  --quant-plugin qfn_wv_plugin.dll \
  -ngl 99 -c 4096 -t 8 \
  --moe-stream model --moe-stream-ram 6GiB \
  --moe-stream-gpu-layers 2 --moe-stream-vram-experts 96 \
  --moe-stream-vram-experts-file model --moe-stream-vram-parallel 1 \
  --moe-stream-overlap 1 --moe-stream-cpu-fuse 1 --moe-stream-cpu-dispatch 1 \
  --moe-stream-cpu-repack 1 --moe-stream-cpu-x4 2 --moe-stream-idle-load 2 \
  --moe-router-min-w 0.015 --moe-router-k-min 4 --moe-stream-timing 1
```
