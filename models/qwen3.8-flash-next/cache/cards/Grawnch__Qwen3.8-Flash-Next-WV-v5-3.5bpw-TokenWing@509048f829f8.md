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

# Qwen3.8-Flash-Next-WV-v5-3.5bpw-TokenWing (`wv-plugin-v1.2`, ABI v2)

Upgraded lossless 2-shard **TokenWing (`wv-plugin-v1.2`, `GGML_QUANT_PLUGIN_ABI_VERSION = 2`, type IDs `112..117`)** GGUF checkpoint of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` total parameters, `48` hybrid `GDN` + sparse-`QSA` trunk layers, `512` routed experts top-`10`) at full `WV-v5` `3.5 bpw` reference precision, built for [`runmyai/tokenwing`](https://github.com/runmyai/tokenwing) (`dev` `>= 6a3e018f`, Linux & Windows).

## What's New in the `v1.2` ABI v2 Build
- **TokenWing ABI v2 Out-of-the-Box (`112..117` + `tokenwing.quant_plugin.types`):** Includes `tokenwing.quant_plugin.types = ["wv_ext0", "wv_ext1", "wv_ext2", "", "", ""]`, embedded `tokenizer.chat_template`, `tokenizer.ggml.eos_token_id = 248046` (`<|im_end|>`), and `tokenizer.ggml.add_eos_token = False`.
- **Lossless `WV-v5` Non-Uniform Codebooks + 1,152 Rank-8 SVD Records (`53.48 GiB` Shard 1):** Replaces the previous `51.41 GiB` lossy stock `Q3_K`/`Q4_0`/`IQ2_S` conversion with 100% lossless `EXT0`/`EXT1`/`EXT2` (`112`/`113`/`114`) `WV-v5` expert tensors, co-distilled `Q8_0` shared experts, and `1,152` Rank-8 residual SVD records.
- **`~3.78 GB` `PAIRED_Q3` Override Bloat Eliminated + Unused `blk.48` Dropped:** Weighted-L2 6-bit scale folding (`WV_OVERRIDE_EXT4`) on minority `n_g=20` 3-bit experts inside `EXT1` `down_exps` keeps `quant_plugin_aux` at just `211.9 MiB` (`53.48 GiB` Shard 1 vs `~59.0 GiB` unoptimized `v1.0` plugin output).
- **Embedded `H=96` Submodular Cache Profiles + 4-Row `vec_dot_x4` Dispatch:** Pre-embeds `H=96` `vram_experts`, `usage_seed`, and `layer_order` blobs directly in Shard 1 and unlocks register-blocked 4-row `vec_dot_x4` across `EXT0`, `EXT1`, `EXT2`, and `EXT4`.
- **Verified Logit Parity & Perplexity:**
  - Full-model 48-layer CPU (`-ngl 0`) vs CUDA (`-ngl 99`) probability cosine: **`1.000000`**
  - WikiText-2 PPL (`c=512`): **`3.1299`** | Math & Systems Code PPL (`c=512`): **`2.4875`**

## Measured Throughput (`wv-plugin-v1.2` on `runmyai/tokenwing` `dev`)
- **Shard Sizes:** Shard 1 = **`53.48 GiB`**; Shard 2 = **`26.82 GiB`** (Layer-1 PLE `Q4_0` table).
- **8-Thread `AVX-512 VNNI` CPU Expert Kernel (`10` active experts, `vec_dot_x4`):** **`0.3814 ms/matmul`** (`EXT0 + EXT2` layers, `0.7628 ms/layer`) | **`0.3820 ms/matmul`** (`EXT0 + EXT1` layers, `0.7641 ms/layer`).
- **Low-VRAM Cold Direct-I/O NVMe Streaming (`<12 GB` VRAM + `6 GiB` RAM, `-t 8`, Exact Top-10 `H=96`, `qwen4exp_la2_h96`):**
  - Technical prompt: **`19.51 tok/s`** (`285.52 MiB/tok` NVMe reads, `92.31%` cache hit)
  - Systems/CUDA code prompt: **`19.42 tok/s`** (`272.02 MiB/tok` NVMe reads, `92.10%` cache hit)
- **Full-VRAM GPU (`-ngl 99`, `RTX PRO 6000 Blackwell`):** **`67.40 tok/s`** (`llama-completion` 256-tok decode) | **`52.82 tok/s`** (`llama-bench tg128`) | **`107.30 tok/s`** (`pp512`).

## Runtime Usage (`wv-plugin-v1.2`, Linux `.so` & Windows `.dll`)
Download `libqfn_wv_plugin.so` (Linux x86_64, `GLIBC <= 2.34`) or `qfn_wv_plugin.dll` (Windows x86_64) from [`grawnchtoader/tokenwing-assets` (`wv-plugin-v1.2`)](https://github.com/grawnchtoader/tokenwing-assets/releases/tag/wv-plugin-v1.2):

```bash
# Low-VRAM (<12 GB GPU + 16 GB RAM) Direct-I/O NVMe/RAM MoE Streaming (~19.5 tok/s, 92.3% hit; use qfn_wv_plugin.dll on Windows):
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
