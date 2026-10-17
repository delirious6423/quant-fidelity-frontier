---
license: other
license_name: qwen-community-1.0
base_model:
  - orcarouter/Qwen3.8-Flash-Next-Uncensored
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - rocmfp4
  - llama.cpp
  - strix-halo
  - gfx1151
  - rocm
  - amd
  - ryzen-ai-max
  - uncensored
  - research
---

> ### 🔧 Runtime: build the ROCmFPX fork below
> Stock `llama.cpp` will not load this file. You need **both** the **`qwen4exp`** architecture
> **and** the ROCmFP4 tensor types in one tree. Upstream
> [`charlie12345/ROCmFPX`](https://github.com/charlie12345/ROCmFPX) has the ROCmFP4 types but
> not `qwen4exp`. Our fork has both:
>
> **[`kingjones30/ROCmFPX`](https://github.com/kingjones30/ROCmFPX)** — a fork of `charlie12345/ROCmFPX`, branch `main`.
>
> ```bash
> git clone https://github.com/kingjones30/ROCmFPX.git
> cd ROCmFPX
> cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
> cmake --build build --target llama-server llama-quantize -j$(nproc)
> ```
>
> Verified 2026-08-27 on gfx1151: clean clone → **0 build errors** → `llama-server` loads a
> `qwen4exp` ROCmFP4 GGUF from this family and generates coherent text.

# Qwen3.8-Flash-Next-Uncensored — ROCmFP4 FAST GGUF — AMD Ryzen AI Max+ 395 / gfx1151

⚠️ **Research artifact.** Refusal behaviour has been removed. This does not add capability — it
removes guardrails. Use it deliberately, in a context where that is appropriate, and own the output.

> ### ⛔ ngram-mod at ≥64K can wedge the GPU
>
> `--spec-type ngram-mod` is the right spec-decode for this family on ROCm, and it is fine at 32K.
> At 64K and above, a context-checkpoint restore desyncs the QSA indexer cache from the KV
> caches. The next kernel then wedges the SDMA queue — `Ring sdma0 reset failure`, D-state
> llama-server, `kill -9` useless, hard power-off to recover. Field report by
> [@liusecret](https://huggingface.co/liusecret) on
> [`Qwen3.8-Flash-Next-ROCmFP4-STRIX-GGUF#6`](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-ROCmFP4-STRIX-GGUF/discussions/6).
>
> **Workaround (no new binary required):**
> ```
> -ctxcp 0 -cpent -1
> ```
> `--ctx-checkpoints 0` plus `--checkpoint-every-n-tokens -1`. If no checkpoints are created,
> the restore path never runs. Keep ngram-mod on ROCm; do not use it on Vulkan/gfx1151
> (acceptance collapses to 0). A fork that serialises `mem_idx` is the real fix — until that
> binary is in your hands, use the flags.


Quantized from the BF16 weights published by
**[orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored)**
— the abliteration work here is theirs, not mine. Go star their repo.

FAST is the smallest tier and the one to take if you are disk-constrained: the
`Q4_0_ROCMFP4_FAST` recipe — attention, experts, token embeddings and the PLE table all ROCmFP4,
with only the output head lifted to Q6_K.
Converted to BF16 GGUF and quantized by me from their release. **4.27 bpw, 87.94 GiB.**

| tensor group | type |
|---|---|
| MoE expert weights (`ffn_*_exps`) | `TYPE_101` (ROCmFP4, 4.251 bpw) |
| shared expert (`ffn_*_shexp`) | `TYPE_101` |
| attention (`attn_*`) | all `TYPE_101` |
| `per_layer_token_embd.weight` (PLE, 51.2B params) | `TYPE_101` |
| `token_embd.weight` | `TYPE_101` |
| `output.weight` (lm head) | `Q6_K` |

The size matches my aligned build of the same tier to **0.01 GiB** — the abliterated checkpoint is
structurally identical, so the quant recipe transfers exactly.

## The Q6_K head

`output.weight` is Q6_K, never 4-bit. Every sampled token passes through the lm head, so its
quantization error lands directly in the argmax. Verified by **exact tensor name** after both
quantize and split — `output.weight` is a substring of `attn_output.weight`, so a loose check
reports success on a 4-bit head.

## Building a runtime that loads these files

Needs **two** things in one tree: the `qwen4exp` architecture and the ROCmFP4 tensor types.
`charlie12345/ROCmFPX` has the ROCmFP4 types but not `qwen4exp`; the upstream qwen4exp work has no
ROCmFP4. **The patch combining them ships in this repo:**
[`qwen4exp-on-rocmfpx-d3ca537.patch`](./qwen4exp-on-rocmfpx-d3ca537.patch) (156 KB, 25 files).

```bash
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX && git checkout d3ca537
curl -LO https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-STRIX_LEAN-GGUF/resolve/main/qwen4exp-on-rocmfpx-d3ca537.patch
git apply qwen4exp-on-rocmfpx-d3ca537.patch
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-quantize -j$(nproc)
```

Verified from a clean clone: applies without conflicts, compiles with zero errors, and the built
`llama-server` loads these GGUFs and generates. The patch's new files —
`src/llama-memory-hybrid-idx.{cpp,h}` (the QSA indexer's own memory class),
`src/models/qwen4exp.cpp`, `conversion/qwen4exp.py` — are the pieces hand-copying misses.

## Measured — Ryzen AI MAX+ 395, gfx1151, ROCm 7.2.4, full 49/49 offload

- generation: **22.75 tok/s**
- prompt processing: **387.3 tok/s**
- GPU memory: **63.3 GiB resident** — identical to the aligned build

GPU-only, full offload. I do not publish partial-offload speeds.

*Measured with one fixed 6,963-token prompt reused across samples (`cache_prompt: false`), run 1
discarded as warm-up, median of the 4 settled samples — spread 1.6 tok/s. An earlier figure of
222 tok/s came from a flawed method that used a different corpus slice per sample; that injected
slice-to-slice variance straight into the number. Same file, same GTT (63.6 GiB) — only the
measurement changed.*

### Long context

This model's native max is **262,144**, and it runs there on a 128 GB box:

| context | prompt | pp tok/s | gen tok/s | GTT |
|---|---|---|---|---|
| 131,072 | 111,411 | 196 | 15.22 | 69.1 GiB |
| 262,144 | 8,000 | 307 | 22.48 | 72.0 GiB |
| 262,144 | 200,000 | 128 | 10.46 | 74.9 GiB |

The context *window* is nearly free — GTT grows only ~4 GiB from 8k to 128k, because Qwen Sparse
Attention caps KV. What you pay for is **depth**: a 200k-token prompt halves generation. It
degrades smoothly rather than falling off a cliff.

## Refusal / quality (counts only)

Aligned build vs this one, same prompts, greedy, same harness:

| split | aligned | this build |
|---|---|---|
| Harmful (24) | 0 comply | **22 comply** |
| Harmless (12) | 10 ok | **11 ok** |
| Quality (8) | 6/8 | **6/8 — same two failures** |

Quality is unchanged to the specific failing question, which is the point: the abliteration
flipped refusal without the quant damaging the model. Prompts and completions are not published.

## Files

Sharded to stay under HF's 50 GB limit. Point `--model` at the **first** shard.

| file | size |
|---|---|
| `Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-FAST-00001-of-00003.gguf` | 41.63 GiB |
| `Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-FAST-00002-of-00003.gguf` | 41.60 GiB |
| `Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-FAST-00003-of-00003.gguf` | 4.71 GiB |
| `mmproj-Qwen3.8-Flash-Next-Uncensored-BF16.gguf` | 0.85 GiB (vision tower) |

## Usage

```
llama-server \
  --model Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-FAST-00001-of-00003.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-Uncensored-BF16.gguf \
  --host 127.0.0.1 --port 8080 \
  --n-gpu-layers 999 --flash-attn on --fit off \
  --ctx-size 131072 --threads 16 --jinja
```

**Do not use `--no-mmap`.** The PLE table is streamed from the file through the page cache; forcing
it into anonymous memory gets the process OOM-killed with nothing in the server log.

<!-- CREDITS:START -->

## Acknowledgements

**[charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)** — defines the ROCmFP4 tensor
formats. Every file here was produced with its `llama-quantize` and runs on its runtime. MIT, based
on upstream llama.cpp. *The `qwen4exp` architecture is not part of that fork* — it comes from
upstream llama.cpp work and is applied on top via
[`qwen4exp-on-rocmfpx-d3ca537.patch`](./qwen4exp-on-rocmfpx-d3ca537.patch) in this repo.

**[llama.cpp](https://github.com/ggml-org/llama.cpp) — ggml-org and contributors** — the engine,
GGUF format and conversion tooling this is built on.

**AMD ROCm** — the compute platform targeted here (ROCm 7.2.4, gfx1151).

**[orcarouter](https://huggingface.co/orcarouter)** — published the uncensored BF16 checkpoint
this is built from. The abliteration is their engineering; I only converted and quantized it.

**Qwen team** — the original base model. See `base_model`; license qwen-community-1.0.

<!-- CREDITS:END -->
