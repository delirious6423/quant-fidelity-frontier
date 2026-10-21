---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- deepseek
- imatrix
- experimental
---


> [!NOTE]
> **Fixed on 2026-10-19. If you downloaded before that, re-download shards 1–7.** The first upload (2026-10-18)
> loaded fine but generated garbage (repetition loops; wikitext-2 perplexity 16.5). The routed-expert shards
> `00001`–`00007` have been requantized. Shards `00008`–`00010`, `imatrix.gguf`, `run.sh` and `bin/` did not
> change, so running the same `hf download` again fetches only the 7 new shards.
> Cause and fix: [below](#2026-10-19-fix-why-the-first-upload-produced-garbage).

# DeepSeek-V4.1-Flash-1017 · UD-IQ2XS GGUF (experimental)

> [!WARNING]
> **Experimental.** Tested on one machine (2× RTX PRO 6000 Blackwell Max-Q) with the bundled build: wikitext-2
> perplexity, a chat sanity check (arithmetic, Chinese, code, thinking on), and speed up to 14K context. No
> benchmark suite (MMLU, GSM8K, ...) has been run on this quant yet. Please report what you see.

GGUF quants of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
This is a dynamic mix in the style of "UD" quants: routed experts at about 2.3–2.5 bits, everything that is small
and sensitive at 6–32 bits. The aim is for all weights except the engram tables to fit on **two 96 GB GPUs**
with room left for a long context, so that no expert has to run on the CPU.

**Stock llama.cpp cannot load this model** (architecture `deepseek41`). You need a build of
[ggml-org/llama.cpp#28696](https://github.com/ggml-org/llama.cpp/pull/28696) with the fixes from
[cliecy/llama.cpp `dsv41-1017-longctx`](https://github.com/cliecy/llama.cpp/tree/dsv41-1017-longctx).
A prebuilt Linux/CUDA build of that branch is included in `bin/`.

## Quick start

```sh
hf download cliecy/DeepSeek-V4.1-Flash-1017-UD-IQ2XS-GGUF --local-dir DeepSeek-V4.1-Flash-1017-UD-IQ2XS
bash DeepSeek-V4.1-Flash-1017-UD-IQ2XS/run.sh
# -> OpenAI-compatible API at http://127.0.0.1:8080/v1
```

`run.sh` unpacks `bin/llama.cpp-dsv41-1017-linux-x64-cuda12.9.tar.zst` the first time it runs (this needs `zstd`),
then starts `llama-server`. You can override these settings through environment variables:

| variable | default | meaning |
|---|---|---|
| `HOST`, `PORT` | `127.0.0.1`, `8080` | listen address |
| `CTX` | `131072` | context length (single slot) |
| `API_KEY` | (none) | require this key |
| `GPUS` | `0,1` | `CUDA_VISIBLE_DEVICES` |
| `TS` | `48,52` | tensor split, see below |
| `THREADS` | physical cores, max 16 | CPU threads (the CPU only does the engram lookups) |
| `LOW_RAM` | `0` | `1` = keep the engram tables on disk and read rows on demand (mmap). About 10 GiB RAM instead of about 115 GiB; fine on NVMe, very slow on HDD |
| `EXTRA_ARGS` | (none) | appended to the command line, e.g. `-ctk q8_0 -ctv q8_0` |

The command `run.sh` runs:

```sh
llama-server -m DeepSeek-V4.1-Flash-1017-UD-IQ2XS-00001-of-00010.gguf \
  -ngl 999 -fa on -sm layer -ts 48,52 -ot "engram_embd.weight=CPU" -lm none -lzm off \
  -c 131072 -np 1 --jinja --chat-template-file deepseek-ai-DeepSeek-V4.1-Flash.jinja
```

### Requirements

- **GPUs:** two NVIDIA GPUs with ≥ 96 GB each, e.g. 2× RTX PRO 6000 Blackwell. About 158 GiB of weights go on
  the GPUs, roughly 79 GiB per card. The prebuilt binaries cover sm_120 (Blackwell workstation), sm_90
  (Hopper) and sm_89 (Ada).
- **Driver:** a driver that supports CUDA 12.9 (≥ 575 recommended; Blackwell needs ≥ 570).
- **RAM:** about 115 GiB free. The two engram tables (103 GiB) stay resident in RAM by default. Use `LOW_RAM=1` otherwise.
- **Disk:** 281 GB. The first load reads all of it, so it takes several minutes on NVMe and more than 30 minutes on a hard disk.
- **OS:** Linux x86_64 with glibc ≥ 2.28. CPU with AVX2.

### Why `-ts 48,52`

V4.1 compresses its KV cache on a few source layers (2, 8, 14, 20), and the layers after each source read the
source's rows. With the default two-GPU split, the boundary lands at layer 21. Layer 20's compressed KV then sits
on GPU 0 while its 19 readers sit on GPU 1, and every generated token copies the whole compressed cache through
the host. That was about 340 MB per token in each direction at 16K context, and it grows with the context.
`-ts 48,52` puts the boundary at layer 20 instead. On Q2_K this took generation at 16K from 23 to 34 tok/s, before
the other fixes.

### Chat template

The template embedded in the GGUF is the DeepSeek-V4 one; tool calls need the V4.1 tag names. `run.sh` passes the
V4.1 template that ships in `bin/`. With thinking on, greedy decoding tends to re-check its answer until it runs
out of tokens. Use `temperature 0.6 / top_p 0.95`, or turn thinking off with
`"chat_template_kwargs": {"enable_thinking": false}`.

## What is in the file

Source: the Q8_0 GGUF of [mrfakename/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF).
In that file the routed experts are native MXFP4 and the small tensors are F32/BF16. It was requantized with
`llama-quantize --allow-requantize --imatrix imatrix.gguf --exclude-weights attn_output_a --tensor-type-file ...`,
with the zero-sign dithering described [below](#2026-10-19-fix-why-the-first-upload-produced-garbage)
(`LLAMA_QUANT_DITHER_ZEROS=1`) for the IQ2 expert tensors.

| tensors | type | size |
|---|---|---|
| routed experts `ffn_gate_exps`, `ffn_up_exps` (40 layers × 384 experts) | IQ2_XS | 2 × 48.8 GiB |
| routed experts `ffn_down_exps` | IQ2_S | 54.1 GiB |
| attention `attn_q_a/q_b/kv/output_a/output_b`, indexer `attn_q_b` | Q6_K | 3.9 GiB |
| shared expert `ffn_{gate,up,down}_shexp` | Q6_K | 1.1 GiB |
| `output` | Q6_K | 0.5 GiB |
| `token_embd`, `engram_wkv` | Q8_0 | 1.0 GiB |
| router `ffn_gate_inp`, compressor `attn_compressor_{kv,gate}`, indexer `proj/attn_k`, `engram_q/k` | BF16 | 0.2 GiB |
| hyper-connection mixers `hc_attn_fn`, `hc_ffn_fn` | **F32** | 0.14 GiB |
| norms, biases, sinks, hc base/scale | F32 | < 0.01 GiB |
| engram tables `blk.{1,14}.engram_embd` (kept on the CPU) | Q4_K | 2 × 51.5 GiB |
| **total** | | **261.4 GiB** |

- **Why `hc_*_fn` is F32.** These matrices produce the coefficients that mix the four residual streams around
  every attention and FFN sublayer, so quantization error compounds across all 40 layers. The reference keeps
  them in FP32, and [measurements in the PR thread](https://github.com/ggml-org/llama.cpp/pull/28696) show BF16
  already misses by 1e-2 while F32 matches to 1e-6. The published 1017 Q2_K files have them at Q2_K.
- **Why engram is Q4_K on the CPU.** The tables are 384M rows × 256 and are read 48 rows per token, so they want
  host memory rather than VRAM. Q4_K keeps them accurate at 103 GiB of RAM.
- **Importance matrix.** `imatrix.gguf` is included. It was computed with the Q2_K model on 194K tokens: a
  general English calibration set, Chinese Wikipedia, wikitext-2 and source code. Its PPL during collection was
  6.53 (ctx 512). `attn_output_a` (the grouped `wo_a`) was quantized without the imatrix, because its 8 × 4096
  entry does not match the 4096-column tensor in this llama.cpp version.
- **Metadata.** `general.file_type` reads Q8_0 because the base type passed to `llama-quantize` was Q8_0. The
  per-tensor types above are what is actually in the file.

## The llama.cpp build in `bin/`

`bin/llama.cpp-dsv41-1017-linux-x64-cuda12.9.tar.zst` contains `llama-server`, `llama-cli`, `llama-bench`,
`llama-perplexity`, `llama-imatrix`, `llama-quantize` and `llama-gguf-split`, plus the V4.1 chat template. It was
built from [cliecy/llama.cpp `dsv41-1017-longctx`](https://github.com/cliecy/llama.cpp/tree/dsv41-1017-longctx)
(commit `1f91bd6`) with CUDA 12.9 for sm_89 / sm_90 / sm_120a and an AVX2 CPU baseline. The libraries are found
through `$ORIGIN`, and the CUDA runtime and cuBLAS are bundled. See `BUILD_INFO.txt` and `THIRD_PARTY_NOTICES.md`
inside. The branch is PR #28696 at `2076412` plus 7 commits:

1. CUDA MMQ `MUL_MAT_ID` padding fix: a backport of master #29953, against illegal-memory-access crashes on large prompt batches.
2. Reader layers reuse the previous index source's top-k (fix by @nicholasshirley).
3. The two-level candidate mask of the reference (needed beyond 16K positions).
4. The fused lightning indexer for V4.1.
5. Gathering only the selected compressed rows at decode.
6. Fused hyper-connection pre op kept across a GPU split (@midagedev, vcruz305/llama.cpp#2).
7. An `llama-imatrix` crash fix for V4.1 graphs.

To build it yourself:

```sh
git clone -b dsv41-1017-longctx https://github.com/cliecy/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-server
```

## Measured with this file

On 2× RTX PRO 6000 Blackwell Max-Q (PCIe), with the bundled build and the `run.sh` settings:

| | this file (fixed) | first upload (2026-10-18) | Q2_K 1017 |
|---|---|---|---|
| wikitext-2 perplexity, 2 × 2048 | **1.62** | 16.54 | 4.69 (same build) |
| wikitext-2 perplexity, 6 × 2048 | **2.044 ± 0.045** | – | 4.749 ± 0.137 (earlier run, see below) |

- Decode: 74–75 tok/s on short chats, 71 tok/s after a 14K-token prompt. Prompt processing: 678 tok/s at 14K.
- Memory with the `run.sh` defaults (`CTX=131072`): 81.0 GiB and 81.2 GiB VRAM on the two cards, about 105 GiB
  RAM for the engram tables. Loading took 30 minutes from a hard disk.
- `LOW_RAM=1` gives the same perplexity as the default mode.

The perplexity of this file is much lower than Q2_K's. The difference comes from the tensors outside the routed
experts (attention, shared expert, hyper-connections, compressor, indexer), which are kept at higher precision
here. The IQ2 experts themselves are slightly worse than Q2_K's: on the first 2048-token chunk this file measures
1.79, and 1.72 with the Q2_K experts swapped in.

## 2026-10-19 fix: why the first upload produced garbage

- The routed experts of the source are MXFP4, and about 11.6 % of their weights are exactly 0.
- IQ2_XS and IQ2_S have no zero level in their codebooks, and `llama-quantize` gives a weight of exactly 0 a
  positive sign. So every zero became the smallest positive grid value.
- Per weight, the error was normal: relative RMS error 0.31 for `gate`/`up`, the same as Q2_K. Within each row,
  though, the zeros all erred the same way, so the errors added up along every row instead of cancelling. Summed
  over a row, the error came out at +3.6 standard deviations for `gate`/`up` and +8 for `down`, positive in 100 % of
  rows. Independent errors would give about ±0.8, with half the rows positive.
- The damage sits in the deeper layers. Swapping expert shards against the Q2_K file showed that IQ2 experts in
  layers 0–23 cost about 3 % perplexity, while IQ2 experts in layers 24–39 raised it from 1.72 to 6.46 (first
  2048-token chunk of wikitext-2).
- **Fix.** Before quantizing, every exact zero gets a tiny pseudo-random sign. The tensor types, file size and
  per-weight error stay the same; only the row bias is gone. The runtime and `bin/` did not change.
- The fix is a small patch to `llama-quant.cpp`, included here as
  [`llama-quant-dither-zeros.patch`](llama-quant-dither-zeros.patch). It is off unless `LLAMA_QUANT_DITHER_ZEROS=1`
  is set. The environment variable is not part of upstream llama.cpp, and the `llama-quantize` in `bin/` does not
  include the patch, so requantizing with that binary reproduces the problem.
- Ruled out along the way:
  - the CUDA kernels: `test-backend-ops` passes, including `MUL_MAT_ID` with this model's matrix shapes and the
    fused MoE paths
  - the engram tables, and the `LOW_RAM` path
  - the GGUF metadata and the imatrix weighting
  - the bundled binary: Q2_K runs correctly with it
- Measured here for IQ2_XS and IQ2_S only. Other IQ types without a zero level (IQ2_XXS, IQ3_XXS, IQ3_S) are
  probably affected the same way but were not checked. When requantizing MXFP4 weights (DeepSeek V4, gpt-oss) to
  such types, check the per-row error sum, not only the RMS error.

## Reference numbers (different quant: Q2_K, not this file)

These were measured with the published **Q2_K** 1017 GGUF, which needs some experts on the CPU (`-ncmoe 3`, or
4 CPU expert layers with `-ts 48,52`), on 2× RTX PRO 6000 Blackwell Max-Q over PCIe. They show what the runtime
fixes do; they are **not** a measurement of this UD-IQ2XS quant.

Generation speed (tok/s) by context depth:

| context | stock branch, default split | `-ts 48,52` | `-ts 48,52` + all fixes |
|---|---|---|---|
| 512 | 44.9 | 40.8 | 42.2 |
| 4K | 36.5 | 39.8 | 47.3 |
| 16K | 23.1 | 34.2 | 45.3 |
| 32K | – | 28.6 | 47.0 |
| 64K | – | 21.7 | 27.9 |
| 120K | – | – | 18.8 |

- Prompt processing: 430 tok/s at 4K, 275 at 64K, 200 at 120K.
- wikitext-2 perplexity (6 × 2048): 4.749 ± 0.137.
- Needle in a haystack at 48.5K tokens, needle at 25 % and at 60 %: both correct.
- The drop at ≥ 64K comes from `top_k` over the compressed positions. With CUDA 12.x it falls back to a full sort;
  a CUDA 13 build (CCCL ≥ 3.2) should avoid that but has not been measured.

## Credits

- [deepseek-ai](https://huggingface.co/deepseek-ai) for the model.
- [@mrfakename](https://huggingface.co/mrfakename) for the 1017 GGUF conversion used as the source.
- [@vcruz305](https://github.com/vcruz305) for the V4.1 conversion and runtime in ggml-org/llama.cpp#28696.
- [@midagedev](https://github.com/midagedev) for the DSpark/HC-pre fixes and the V4.1 chat template.
- [@nicholasshirley](https://github.com/nicholasshirley) for the reused top-k fix.
- Mermiges for the MMQ out-of-bounds analysis.

Quantized and packaged with the help of an AI coding assistant (Claude).

## The llama.cpp fixes in bin/

The bundled llama.cpp is DeepSeek V4.1 support from ggml-org/llama.cpp#28696 (not yet merged) plus 7 patches for long context and multi-GPU (two-level candidate mask, fused lightning indexer, decode-time gather of the selected compressed KV rows, top-k reuse on reader layers, fused HC pre across a GPU split, an imatrix guard, an MMQ padding backport). On a 2-GPU Q2_K run they raise single-stream decode at 16K context from 23 to 45 tok/s. Patch files, build steps and measurements: [LLAMACPP-FIXES.md](https://github.com/cliecy/DeepSeek-V4.1-Flash-1017-2xRTXPRO6000/blob/main/docs/LLAMACPP-FIXES.md) ([中文](https://github.com/cliecy/DeepSeek-V4.1-Flash-1017-2xRTXPRO6000/blob/main/docs/LLAMACPP-FIXES.zh-CN.md)); source branch: [cliecy/llama.cpp@dsv41-1017-longctx](https://github.com/cliecy/llama.cpp/tree/dsv41-1017-longctx).
