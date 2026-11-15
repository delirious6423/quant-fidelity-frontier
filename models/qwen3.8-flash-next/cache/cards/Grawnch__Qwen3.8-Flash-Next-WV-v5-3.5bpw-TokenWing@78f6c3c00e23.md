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

# Qwen3.8-Flash-Next-WV-v5-3.5bpw-TokenWing

Pre-transcoded 2-shard **TokenWing (`wv-plugin-v1.1`)** GGUF checkpoint of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` total parameters, `48` hybrid `GDN` + sparse-`QSA` layers, `512` routed experts top-`10`) at full `WV-v5` `3.5 bpw` precision.

## Shards
- `Qwen3.8-Flash-Next-WV-v5-3.5bpw.tokenwing-00001-of-00002.gguf` (`51.41 GiB` — 48-layer trunk + `3.5 bpw` MoE experts)
- `Qwen3.8-Flash-Next-WV-v5-3.5bpw.tokenwing-00002-of-00002.gguf` (`26.82 GiB` — Layer-1 PLE table)

## Runtime Requirements
Requires `runmyai/tokenwing` (`feat/quant-plugin`, PR #8) + `wv-plugin-v1.1` (`libqfn_wv_plugin.so` on Linux or `qfn_wv_plugin.dll` on Windows) from [`grawnchtoader/tokenwing-assets`](https://github.com/grawnchtoader/tokenwing-assets/releases/tag/wv-plugin-v1.1).

```bash
llama-completion -m Qwen3.8-Flash-Next-WV-v5-3.5bpw.tokenwing-00001-of-00002.gguf \
  --quant-plugin qfn_wv_plugin.dll \
  -ngl 99 -c 4096 -t 8 \
  --moe-stream model --moe-stream-ram 6GiB \
  --moe-stream-overlap 1 --moe-stream-cpu-fuse 1 --moe-stream-cpu-dispatch 1 \
  --moe-stream-cpu-x4 2 --moe-stream-idle-load 2 --moe-stream-timing 1
```
