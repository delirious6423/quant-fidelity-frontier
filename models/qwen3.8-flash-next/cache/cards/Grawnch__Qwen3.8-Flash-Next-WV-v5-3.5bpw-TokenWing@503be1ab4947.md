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

# Qwen3.8-Flash-Next-WV-v5-3.5bpw-TokenWing (`wv-plugin-v1.3`, ABI v2)

2-shard **TokenWing (`wv-plugin-v1.3`, `GGML_QUANT_PLUGIN_ABI_VERSION = 2`)** GGUF checkpoint of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` total parameters, `48` hybrid `GDN` + sparse-`QSA` trunk layers, `512` routed experts top-`10`) at `WV-v5` `3.5 bpw` precision, built for [`runmyai/tokenwing`](https://github.com/runmyai/tokenwing) (`dev` `>= 6a3e018f`, Linux & Windows).

## What's Included
- **TokenWing ABI v2 Ready:** Includes `tokenwing.quant_plugin.types`, embedded `tokenizer.chat_template`, `tokenizer.ggml.eos_token_id = 248046` (`<|im_end|>`), and `tokenizer.ggml.add_eos_token = False`.
- **Pre-Computed Routing & Cache Metadata:** Pre-embeds expert cache ordering profiles directly in Shard 1 (`53.48 GiB` Shard 1 + `26.82 GiB` Shard 2 Layer-1 `Q4_0` PLE).
- **Verified Logit Parity & Perplexity:**
  - Full-model 48-layer CPU (`-ngl 0`) vs CUDA (`-ngl 99`) probability cosine: **`1.000000`**
  - WikiText-2 PPL (`c=512`): **`3.1299`** | Math & Systems Code PPL (`c=512`): **`2.4875`**

## Measured Throughput (`wv-plugin-v1.3` on `runmyai/tokenwing` `dev`)
- **Shard Sizes:** Shard 1 = **`53.48 GiB`**; Shard 2 = **`26.82 GiB`** (Layer-1 PLE `Q4_0` table).
- **Low-VRAM Cold Direct-I/O NVMe Streaming (`<12 GB` VRAM + `6 GiB` RAM, `-t 8`):**
  - Technical prompt: **`19.51 tok/s`** (`285.52 MiB/tok` NVMe reads, `92.31%` cache hit)
  - Systems/CUDA code prompt: **`19.42 tok/s`** (`272.02 MiB/tok` NVMe reads, `92.10%` cache hit)
- **`wv-plugin-v1.3` (`< 12 GB` VRAM + `20 GiB` RAM Pool, `-t 10`, `--spec-type ngram-mod --spec-ngram-mod-n-max 8`):** **`25.06–25.87 tok/s`** decode (`97.4%` cache hit rate).
- **Full-VRAM GPU (`-ngl 99`, `RTX PRO 6000 Blackwell`):** **`67.40 tok/s`** (`llama-completion` 256-tok decode) | **`52.82 tok/s`** (`llama-bench tg128`) | **`107.30 tok/s`** (`pp512`).
- **Optional `blk.48` MTP Draft Sidecar (`>= 24 GB` VRAM GPUs):** [`Qwen3.8-Flash-Next-WV-v5-MTP-Draft.gguf`](https://huggingface.co/Grawnch/TokenWing-Model-Assets) (`2.39 GiB` GGUF, `77–86.8%` acceptance via `-md Qwen3.8-Flash-Next-WV-v5-MTP-Draft.gguf --spec-type draft-mtp --spec-draft-n-max 1 -ngld 99`).

## Runtime Usage (`wv-plugin-v1.3`, Linux `.so` & Windows `.dll`)
Download `libqfn_wv_plugin.so` (Linux x86_64, `GLIBC <= 2.34`) or `qfn_wv_plugin.dll` (Windows x86_64) from [`grawnchtoader/tokenwing-assets` (`wv-plugin-v1.3`)](https://github.com/grawnchtoader/tokenwing-assets/releases/tag/wv-plugin-v1.3):

```bash
# Recommended 12 GB GPU + 32 GB RAM Laptop Command (~25.9 tok/s decode, 97.4% cache hit; use qfn_wv_plugin.dll on Windows):
llama-server -m Qwen3.8-Flash-Next-WV-v5-3.5bpw.tokenwing-00001-of-00002.gguf \
  --quant-plugin libqfn_wv_plugin.so \
  -ngl 99 --moe-stream model --moe-stream-ram 20GiB \
  --moe-stream-overlap 1 --moe-stream-cpu-fuse 1 --moe-stream-cpu-dispatch 1 \
  --moe-stream-cpu-x4 0 --moe-stream-idle-load 2 \
  --moe-stream-fused-route 1 --moe-stream-pred-shared 1 --moe-stream-hc-share 1 \
  --moe-stream-plan 1 --moe-stream-drop-stale 1 --moe-stream-needed-early 1 \
  --moe-stream-lookahead 2 --moe-stream-guess-combined 1 \
  --moe-stream-protect-named 1 --moe-stream-policy decay --moe-stream-promote 1 \
  --moe-stream-rs-window 1 --moe-stream-ple-late 1 --moe-stream-hc-silu 1 \
  --moe-stream-gpu-share-q8 1 --moe-stream-hc-q8 1 --moe-stream-rs-inplace 1 \
  --spec-type ngram-mod --spec-ngram-mod-n-max 8 -t 10 -c 8192

# Low-VRAM (<12 GB GPU + 6 GiB RAM, ~19.5 tok/s):
llama-cli -m Qwen3.8-Flash-Next-WV-v5-3.5bpw.tokenwing-00001-of-00002.gguf \
  --quant-plugin libqfn_wv_plugin.so \
  -ngl 99 --moe-stream model --moe-stream-ram 6GiB \
  --moe-stream-overlap 1 --moe-stream-cpu-fuse 1 --moe-stream-cpu-dispatch 1 \
  --moe-stream-cpu-x4 0 --moe-stream-idle-load 2 \
  --moe-stream-fused-route 1 --moe-stream-pred-shared 1 --moe-stream-hc-share 1 \
  --moe-stream-plan 1 --moe-stream-drop-stale 1 --moe-stream-needed-early 1 \
  --moe-stream-lookahead 2 --moe-stream-guess-combined 1 \
  --moe-stream-protect-named 1 --moe-stream-policy decay --moe-stream-promote 1 \
  --moe-stream-rs-window 1 --moe-stream-ple-late 1 --moe-stream-hc-silu 1 \
  --moe-stream-gpu-share-q8 1 --moe-stream-hc-q8 1 --moe-stream-rs-inplace 1 -t 8 -c 8192

# Full-VRAM Workstation GPU (~67.4 tok/s completion decode):
llama-cli -m Qwen3.8-Flash-Next-WV-v5-3.5bpw.tokenwing-00001-of-00002.gguf \
  --quant-plugin libqfn_wv_plugin.so -ngl 99 -c 32768
```
