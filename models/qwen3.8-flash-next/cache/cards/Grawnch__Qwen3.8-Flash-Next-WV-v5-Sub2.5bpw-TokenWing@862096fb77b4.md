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

# Qwen3.8-Flash-Next-WV-v5-Sub2.5bpw-TokenWing (`wv-plugin-v1.2`, ABI v2)

Zero-loss native 2-shard **TokenWing (`wv-plugin-v1.2`, `GGML_QUANT_PLUGIN_ABI_VERSION = 2`, type IDs `112..117`)** GGUF checkpoint of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` total parameters, `48` hybrid `GDN` + sparse-`QSA` trunk layers, `512` routed experts top-`10`) exported directly from `WV-v5-Sub2.5bpw` (`2.46 bpw` effective routed precision inside `2.50 bpw` `EXT2` `512,000 B` `O_DIRECT`-aligned slices) + lossless `WV-v5-3.5bpw` dense spine (`Q6_K`/`Q8_0`/`BF16`/`F32`), built for [`runmyai/tokenwing`](https://github.com/runmyai/tokenwing) (`dev` `>= 6a3e018f`, Linux & Windows).

## Architecture & Lossless Fidelity Highlights
- **TokenWing ABI v2 Out-of-the-Box (`112..117` + `tokenwing.quant_plugin.types`):** Includes `tokenwing.quant_plugin.types = ["", "", "wv_ext2", "", "", ""]`, embedded `tokenizer.chat_template`, `tokenizer.ggml.eos_token_id = 248046` (`<|im_end|>`), and `tokenizer.ggml.add_eos_token = False`.
- **100% Lossless Dense Spine & Co-Distilled Shared Experts:** Preserves the exact `Q6_K`/`Q8_0`/`BF16`/`F32` trunk spine, lossless unswizzled `Q8_0` `output.weight`, and all 48 layers of BF16-co-distilled `Q8_0` shared experts (`ffn_{gate,up,down}_shexp.weight`).
- **Native `VQ4_G32 + SWZ2_G64` `EXT2` (`114`) MoE + Rank-8 SVD:** Encodes custom `[4, 256, 4]` `int8` VQ4 codebooks (`vq4_gu` / `vq4_dn`) and `28/48` layers of `SWZ2_G64` `down_exps` alongside `1,152` co-distilled balanced-FP16 Rank-8 residual SVD records (`QFA1` `quant_plugin_aux`), plus embedded `H=96` submodular `vram_experts`, `usage_seed`, and `layer_order` blobs.
- **Verified Logit Parity & Perplexity:**
  - Full-model 48-layer CPU (`-ngl 0`) vs CUDA (`-ngl 99`) probability cosine: **`1.000000`**
  - WikiText-2 PPL (`c=512`): **`3.6113`** (vs `3.1299` on `WV-v5-3.5bpw`; Median KLD = `0.07073`, Mean KLD = `0.40344`)
  - Math & Systems Code PPL (`c=512`): **`2.5550`** (vs `2.4875` on `WV-v5-3.5bpw`; Median KLD = `0.01165`, Mean KLD = `0.16549`)

## Measured Throughput (`wv-plugin-v1.2` on `runmyai/tokenwing` `dev`)
- **Shard Sizes:** Shard 1 = **`39.48 GiB`** (`35.16 GiB` routed MoE + `4.23 GiB` dense spine + embedded `QFA1`/`H=96` blobs); Shard 2 = **`26.82 GiB`** (Layer-1 PLE `Q4_0` table).
- **8-Thread `AVX-512 VNNI` CPU Expert Kernel (`10` active experts):** **`0.2157 ms/matmul`** (`0.4314 ms/layer` with on-load VNNI slot repacking + 4-row `vec_dot_x4`).
- **Low-VRAM Cold Direct-I/O NVMe Streaming (`<12 GB` VRAM + `6 GiB` RAM, `-t 8`):**
  - Baseline cold `H=96` streaming (`plain_cold`): **`8.17 tok/s`** (`130.17 MiB/tok` NVMe reads)
  - Exact top-10 (`vnni_repack_x4_h96`): **`23.10 tok/s`** (technical) / **`22.79 tok/s`** (code), `129.89 MiB/tok` NVMe reads (`91.92%` cache hit)
- **Full-VRAM GPU (`-ngl 99`, `RTX PRO 6000 Blackwell`):** **`68.35 tok/s`** (`llama-completion` 256-tok decode) | **`56.58 tok/s`** (`llama-bench tg128`) | **`116.11 tok/s`** (`pp512`).

## Runtime Usage (`wv-plugin-v1.2`, Linux `.so` & Windows `.dll`)
Download `libqfn_wv_plugin.so` (Linux x86_64, `GLIBC <= 2.34`) or `qfn_wv_plugin.dll` (Windows x86_64) from [`grawnchtoader/tokenwing-assets` (`wv-plugin-v1.2`)](https://github.com/grawnchtoader/tokenwing-assets/releases/tag/wv-plugin-v1.2):

```bash
# 1a) Peak Decode (<12 GB GPU + 16 GB RAM, prompts < 512 tokens, ~23.1 tok/s):
#     (On Windows PowerShell: $env:WV_PLUGIN_CPU_REPACK_MODE="1"; use --quant-plugin qfn_wv_plugin.dll)
WV_PLUGIN_CPU_REPACK_MODE=1 llama-cli -m Qwen3.8-Flash-Next-WV-v5-Sub2.5bpw.tokenwing-00001-of-00002.gguf \
  --quant-plugin libqfn_wv_plugin.so \
  -ngl 99 --moe-stream model --moe-stream-ram 6GiB \
  --moe-stream-overlap 1 --moe-stream-cpu-fuse 1 --moe-stream-cpu-dispatch 1 \
  --moe-stream-cpu-repack 1 --moe-stream-cpu-x4 2 --moe-stream-idle-load 2 \
  --moe-stream-fused-route 1 --moe-stream-pred-shared 1 --moe-stream-hc-share 1 \
  --moe-stream-plan 1 --moe-stream-drop-stale 1 --moe-stream-needed-early 1 \
  --moe-stream-lookahead 2 --moe-stream-guess-combined 1 --moe-stream-proj-ready 1 \
  --moe-stream-protect-named 1 --moe-stream-policy decay --moe-stream-promote 1 \
  --moe-stream-rs-window 1 --moe-stream-ple-late 1 --moe-stream-hc-silu 1 \
  --moe-stream-gpu-share-q8 1 --moe-stream-hc-q8 1 --moe-stream-rs-inplace 1 -t 8 -c 8192

# 1b) All-Prompt-Length (<12 GB GPU + 16 GB RAM, multi-wave prompts >= 512 tokens, ~21.7 tok/s, 94.9% hit):
llama-cli -m Qwen3.8-Flash-Next-WV-v5-Sub2.5bpw.tokenwing-00001-of-00002.gguf \
  --quant-plugin libqfn_wv_plugin.so \
  -ngl 99 --moe-stream model --moe-stream-ram 6GiB \
  --moe-stream-overlap 1 --moe-stream-cpu-fuse 1 --moe-stream-cpu-dispatch 1 \
  --moe-stream-cpu-x4 2 --moe-stream-idle-load 2 \
  --moe-stream-fused-route 1 --moe-stream-pred-shared 1 --moe-stream-hc-share 1 \
  --moe-stream-plan 1 --moe-stream-drop-stale 1 --moe-stream-needed-early 1 \
  --moe-stream-lookahead 2 --moe-stream-guess-combined 1 \
  --moe-stream-protect-named 1 --moe-stream-policy decay --moe-stream-promote 1 \
  --moe-stream-rs-window 1 --moe-stream-ple-late 1 --moe-stream-hc-silu 1 \
  --moe-stream-gpu-share-q8 1 --moe-stream-hc-q8 1 --moe-stream-rs-inplace 1 -t 8 -c 8192

# 2) Full-VRAM Workstation GPU (~75.0 tok/s completion decode):
llama-cli -m Qwen3.8-Flash-Next-WV-v5-Sub2.5bpw.tokenwing-00001-of-00002.gguf \
  --quant-plugin libqfn_wv_plugin.so -ngl 99 -c 32768
```
