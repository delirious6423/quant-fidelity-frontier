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
> ⚠️ **Apply the bundled fix patches before `cmake`:** `qwen4exp-qsa-checkpoint-fix.patch`
> always, plus `qwen4exp-mtp-graph-fork.patch` if you want `--spec-type draft-mtp` on this
> clone. Full steps further down.
>
> Verified 2026-08-27 on gfx1151: clean clone → **0 build errors** → `llama-server` loads a
> `qwen4exp` ROCmFP4 GGUF from this family and generates coherent text.

# Qwen3.8-Flash-Next-Uncensored — ROCmFP4 FAST GGUF — AMD Ryzen AI Max+ 395 / gfx1151

## ⚡ Speculative decoding (MTP) now works — measured +27.7% at short context

The `qwen4exp` MTP graph shipped with a broken combiner (it mean-pooled the hyper-connection
streams), so `--spec-type draft-mtp` acceptance sat near **0.36** and gave no real speedup. That is
now **fixed** — this repo ships [`qwen4exp-mtp-graph.patch`](./qwen4exp-mtp-graph.patch); apply it to the tree the build steps below produce and rebuild
(`git apply qwen4exp-mtp-graph.patch` before `cmake --build`).

Pair the model with a Flash-Next MTP head from
[`kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF`](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF).
**Measured** on the Uncensored FAST (imatrix) build with the **Q8_0 head** (`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`) at
**short context (`-c 2048`)**: **acceptance 0.94**, **31.80 tok/s** vs **24.9 tok/s** no-draft
(**+27.7%**), warm 160-token completion, `cache_prompt:false`. The graph fix and the heads are shared
across the Flash-Next family, but this tier's own MTP speed has not been measured, and the Q6_K / Q4
heads were not benchmarked. The head only *proposes* draft tokens; the main model verifies every one,
so your output is unchanged.

```
llama-server -m <the first shard in this repo>.gguf \
  -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp \
  --spec-draft-n-min 0 --spec-draft-n-max 1 --n-gpu-layers-draft 99 \
  -ngl 999 -fa on -np 1 -c 32768 --jinja
```
`-np 1` is required with `draft-mtp`.

⚠️ **Updated 2026-09-17 — re-download if you pulled it earlier.** `qwen4exp-mtp-graph.patch` now
carries the `models.h` and `llama-model.cpp` hunks it needs. The previous version applied cleanly but
**failed to compile** (`'graph_mtp' was not declared in this scope`). The bundled patch matches the
build steps on this card; for the other build path use `qwen4exp-mtp-graph-fork.patch` (if you build from a `kingjones30/ROCmFPX` clone), also bundled here.

**Measured plain vs `draft-mtp`** — median of 3 per cell, one binary, greedy, `cache_prompt:false`,
256 generated tokens, `-c 2048`, Q8_0 head, Uncensored STRIX_LEAN-imatrix weights, gfx1151 / ROCm 7.2.4
(2026-09-17):

| workload | plain | `--spec-draft-n-max 4` | `--spec-draft-n-max 1` |
|---|---|---|---|
| reasoning | 23.91 | 30.94 (+29%, acc 0.680) | **31.94 (+34%, acc 0.945)** |
| JSON output | 23.99 | 28.31 (+18%, acc 0.597) | 27.24 (+14%, acc 0.758) |
| code | 24.09 | 21.56 (**−10%**, acc 0.422) | **26.80 (+11%, acc 0.711)** |
| long-document summary | 23.80 | 20.36 (**−14%**, acc 0.352) | **24.14 (+1%, acc 0.641)** |

⭐ **Use `--spec-draft-n-max 1`.** It did not lose a single workload here, and it wins most where the
next token is predictable. `n-max 4` pays for four draft forward passes per step, so it only wins when
acceptance is high (reasoning, JSON) and is a genuine loss on code and long-document work. MTP also
costs prefill speed, because the draft head processes the prompt too. The older **+27.7%** figure came
from one reasoning-shaped prompt — it holds for that shape, not universally, so measure your own.

> ✅ **Depth:** with the bundled checkpoint fix applied, `draft-mtp` is verified from 2K to
> 128K — see the box further down for what was measured and what is still open.

⚠️ **Research artifact.** Refusal behaviour has been removed. This does not add capability — it
removes guardrails. Use it deliberately, in a context where that is appropriate, and own the output.

> ### ✅ Depth: `draft-mtp` is fixed and measured (2026-09-17)
>
> The ≥64K wedge came from context-checkpoint restores leaving the QSA indexer cache (`mem_idx`) out
> of the checkpoint. The fix ships here as
> [`qwen4exp-qsa-checkpoint-fix.patch`](./qwen4exp-qsa-checkpoint-fix.patch) — it overrides
> `state_write` / `state_read` on `llama_memory_hybrid_idx`. Apply it with the build steps on this
> card **even if you never use speculative decoding**.
>
> With it applied, `--spec-type draft-mtp` ran clean from **2K to 128K** on gfx1151: 8 depth rungs,
> **864 context-checkpoint restores** (2 of them prompt-cache rollbacks at 64K), **0 GPU faults**,
> coherent output at every depth. Measured 2026-09-17 on Ryzen AI MAX+ 395 / ROCm 7.2.4 with the
> Uncensored STRIX_LEAN-imatrix weights + `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, `-c 262144`,
> `--spec-draft-n-max 4`, default context checkpoints. That 128K run used my own fork tree; the exact
> build steps on this card were verified to 16K.
>
> ⚠️ **Still open:** `--spec-type ngram-mod` at ≥64K has **not** been retested with the patch — the
> original field report ([`…-STRIX-GGUF#6`](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-ROCmFP4-STRIX-GGUF/discussions/6), thanks
> [@liusecret](https://huggingface.co/liusecret)) was ngram-mod, so keep `-ctxcp 0 -cpent -1` when you
> use it. And do not use speculative decoding of any kind on Vulkan/gfx1151 — acceptance collapses to 0.
>
> A `speculative replay stalled` warning on ~2% of restores is expected and harmless: that is the
> server's livelock guard dropping one draft and decoding that token normally.


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
# both fixes ship in this repo — apply them before configuring:
curl -LO https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-GGUF/resolve/main/qwen4exp-qsa-checkpoint-fix.patch
git apply qwen4exp-qsa-checkpoint-fix.patch      # checkpoint safety at >=64K: apply this always
curl -LO https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-GGUF/resolve/main/qwen4exp-mtp-graph.patch
git apply qwen4exp-mtp-graph.patch               # only if you want --spec-type draft-mtp
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
