---
license: other
license_name: qwen-community-1.0
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
base_model_relation: quantized
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
  - long-context
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

# Qwen3.8-Flash-Next — ROCmFP4 STRIX\_LEAN GGUF — AMD Ryzen AI Max+ 395 / Strix Halo / gfx1151

## ⚡ Speculative decoding (MTP) now works — measured +27.7% at short context

The `qwen4exp` MTP graph shipped with a broken combiner (it mean-pooled the hyper-connection
streams), so `--spec-type draft-mtp` acceptance sat near **0.36** and gave no real speedup. That is
now **fixed** — this repo ships [`qwen4exp-mtp-graph.patch`](./qwen4exp-mtp-graph.patch); apply it to
the [`kingjones30/ROCmFPX`](https://github.com/kingjones30/ROCmFPX) fork and rebuild
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
build steps on this card; for the other build path use `qwen4exp-mtp-graph-d3ca537.patch` (if you build `charlie12345/ROCmFPX` @ `d3ca537` + the arch patch), also bundled here.

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


> ### ⚠️ Read this before comparing any number here to a discrete GPU
>
> Every measurement on this card is from an **AMD Ryzen AI MAX+ 395 "Strix Halo"** — an
> **integrated GPU with unified memory**. There is no discrete VRAM. The Radeon 8060S addresses
> ordinary system RAM through the **GTT aperture**, and the box's 128 GB is **shared between CPU
> and GPU**.
>
> That is the entire reason a 98 GiB model runs here at all: the GPU can reach system memory, so
> capacity is enormous. The trade is **bandwidth** — roughly 215 GB/s measured, against ~1 TB/s on
> a high-end discrete card. So expect big-model *capacity* that a 24 GB dGPU cannot touch, and
> per-token *speed* well below one.
>
> This is also why the numbers below are reported as **GTT resident**, not "VRAM used" — on this
> hardware those are the same pool, and `nvidia-smi`-style VRAM intuitions do not transfer.

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

STRIX\_LEAN is my default publish tier for Strix Halo: ROCmFP4 experts, half the attention tensors on the higher-quality `TYPE_100` ROCmFP4 variant, Q5_1 on the PLE table, Q5_K token embeddings, Q6_K output head. Quantized directly from my own BF16 conversion of the release weights.
`general.file_type` = **106** (`Q4_0_ROCMFP4_STRIX_LEAN`). **4.78 bpw, 98.49 GiB.**

Read straight out of the GGUF headers of the files in this repo:

| tensor group | type |
|---|---|
| MoE expert weights (`ffn_*_exps`, 144) | `TYPE_101` (ROCmFP4, 4.251 bpw) |
| shared expert (`ffn_*_shexp`, 144) | `TYPE_101` |
| attention (`attn_*`, 120) | half `TYPE_100`, half `TYPE_101` |
| `per_layer_token_embd.weight` (PLE, 51.2B params) | `Q5_1` — 35.76 GiB |
| `token_embd.weight` | `Q5_K` |
| `output.weight` (lm head) | `Q6_K` |
| norms / biases | `F32` |

## The Q6_K head

`output.weight` is Q6_K in every tier, never 4-bit. An unprotected head ruins a 4-bit build:
every token you sample passes through the lm head, so its quantization error lands directly in
the argmax. On a sparse-MoE model the head is also one of the few dense matrices left, which
makes its error stand out more. It is 0.3% of the master weights — pinning it to Q6_K costs
under half a GiB and removes that whole error class.

## Building a runtime that loads these files

You need **two** things in one tree: the `qwen4exp` architecture and the ROCmFP4 tensor types.
Neither side has both — `charlie12345/ROCmFPX` has the ROCmFP4 types but no `qwen4exp`, and the
upstream qwen4exp work has no ROCmFP4. **The patch that combines them ships in this repo:**
[`qwen4exp-on-rocmfpx-d3ca537.patch`](./qwen4exp-on-rocmfpx-d3ca537.patch) (156 KB, 25 files).

```bash
git clone https://github.com/kingjones30/ROCmFPX.git
cd ROCmFPX
# both fixes ship in this repo — apply them before configuring:
curl -LO https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-ROCmFP4-STRIX_LEAN-GGUF/resolve/main/qwen4exp-qsa-checkpoint-fix.patch
git apply qwen4exp-qsa-checkpoint-fix.patch      # checkpoint safety at >=64K: apply this always
curl -LO https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-ROCmFP4-STRIX_LEAN-GGUF/resolve/main/qwen4exp-mtp-graph.patch
git apply qwen4exp-mtp-graph.patch               # only if you want --spec-type draft-mtp
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-quantize -j$(nproc)
```

Verified from a clean clone: applies without conflicts, compiles with zero errors, and the
resulting `llama-server` loads these GGUFs and generates.

**What the patch adds** — the pieces people miss when copying files by hand:

| file | why |
|---|---|
| `src/llama-memory-hybrid-idx.{cpp,h}` | **new** — the Qwen Sparse Attention indexer needs its own memory class |
| `src/models/qwen4exp.cpp` | **new** — the arch implementation (picked up by the `models/*.cpp` GLOB) |
| `conversion/qwen4exp.py` | **new** — conversion support |
| `src/llama-{arch,batch,hparams,kv-cache,model,model-loader,model-saver,quant}.{cpp,h}` | interdependent — cherry-picking individual files will not work |

Conversion note: the entry point is the **`conversion/`** package, not `convert_hf_to_gguf.py`,
which rejects `Qwen4ExpForConditionalGeneration`. The 51B-parameter PLE table will also OOM a
naive conversion that builds an F32 intermediate — cast each shard straight to BF16 and write
positionally.

## Where the file actually lives — measured, not inferred

177B parameters, of which **51.2B are `per_layer_token_embd`** — the PLE n-gram lookup table.
That table is the only real difference between these three tiers, and it behaves nothing like
the rest of the weights.

Solving the three published files for the two ROCmFP4 tensor types (three files, two unknowns,
**exact fit — residual 0.000 GiB on all three**) gives `TYPE_101` = **4.251 bpw** and
`TYPE_100` = **4.506 bpw**. With those, the mass splits:

| tier | file | PLE table | everything else |
|---|---|---|---|
| FAST | 87.94 GiB | 25.34 GiB | **62.60 GiB** |
| STRIX\_LEAN | 98.49 GiB | 35.76 GiB | **62.73 GiB** |
| STRIX | 113.47 GiB | 50.66 GiB | **62.81 GiB** |

**Everything that is not the PLE table is the same size in all three tiers — within 0.21 GiB.**
These are one model plus a differently-quantized lookup table.

That predicts identical GPU memory, and measurement confirms it. GTT resident, **identical across
all three tiers at every depth**:

| context | GTT at load | GTT after prompt |
|---|---|---|
| 8,192 | 63.3 GiB | 63.6 GiB |
| 32,768 | 64.1 GiB | 64.8 GiB |
| 65,536 | 65.1 GiB | 66.2 GiB |
| 131,072 | 67.2 GiB | 69.1 GiB |

~63 GiB is the non-PLE mass plus compute buffers. **The PLE table never enters GPU memory.**

**It is streamed off the SSD, through the OS page cache — automatically, with no flags.** Measured directly with `mincore(2)`
against the live model file during a 131,072-token run: **40.13 of 113.47 GiB resident in page
cache**, with system-wide `Cached` at 40.4 GiB — the page cache *was* this file. Sampled every
30 s under sustained load it holds **32–35 GiB**, rising and falling as the kernel reclaims.
Those pages are file-backed and reclaimable, which is exactly why they never cost you real memory.

Three consequences that matter in practice:

1. **`--no-mmap` will get you OOM-killed.** It forces the table into anonymous memory, which is
   neither file-backed nor reclaimable. The cgroup killer takes the process **with nothing in the
   server log** — the only evidence is `dmesg`.
2. **Don't force the table to CPU with `-ot`.** I tried `-ot "per_layer_token_embd|ple\.=CPU"`:
   identical GTT, and generation fell from ~23 to 13.4 tok/s. The kernel already streams it better.
3. **A bigger tier costs disk, not GPU memory.** If you have the storage, take STRIX.

## Long context — the full ladder

> **Prompt-processing figures were corrected 2026-08-27.** The original ladder used a *different*
> corpus slice for each sample, which injected slice-to-slice variance straight into the pp number
> — it read 220 tok/s at 8k where the hardware actually does 385. Every pp/gen value below is now
> measured with **one fixed prompt reused across samples** (`cache_prompt: false`), run 1 discarded
> as warm-up, median of the 4 settled samples. Spread at 65,536 is **0.4 tok/s across 4 samples**.
> GTT is unchanged at every depth, confirming the workload is identical — only the method changed.
>
> Also measured and rejected: `-t 32` vs `-t 16` is **+0.9%** (noise, confirmed with an A/B/A drift
> check), and `-ub` 1024/2048/4096 show **no effect** — llama.cpp's default `-ub 512` is already
> right on this hardware. There was no tuning win here; there was a measurement error.

Four depths on one Ryzen AI MAX+ 395 (gfx1151, ROCm 7.2.4), full 49/49 offload. Each prompt is
a unique, non-overlapping slice of a real 10.2 MB source-code corpus (61,469 distinct words),
sized exactly with `/tokenize` → slice → `/detokenize`, `cache_prompt: false`, and `prompt_n`
verified every run.

| context | prompt tokens | GTT at load | GTT after prompt | pp tok/s | gen tok/s |
|---|---|---|---|---|---|
| 8,192 | 6,963 | 63.3 GiB | 63.6 GiB | 385 | **22.87** |
| 32,768 | 27,852 | 64.1 GiB | 64.8 GiB | 313 | **19.46** |
| 65,536 | 55,705 | 65.1 GiB | 66.2 GiB | 261 | **18.54** |
| 131,072 | 111,411 | 67.2 GiB | 69.1 GiB | 196 | **15.22** |
| 262,144 | 8,000 | 71.7 GiB | 72.0 GiB | 307 | **22.48** |
| 262,144 | 200,000 | 71.7 GiB | 74.9 GiB | 128 | **10.46** |

**The full native 262,144-token context runs on a 128 GB Strix Halo.** Not 131,072 — that was
simply where I stopped the first ladder, and people rightly asked. At the top rung the box sits at
74.9 GiB GTT with a 200,000-token prompt loaded, leaving real headroom.

**The context *window* is nearly free; depth is what costs.** GTT at load grows only ~3.9 GiB from
8k to 128k, and ~8 GiB all the way to the full 256k window — Qwen Sparse Attention's
512-block / 2048-token budget caps KV, where a conventional model would spend tens of GiB. What you
actually pay for is how much you put *in* that window: at 262,144 a short prompt generates at
**22.48 tok/s**, a 200,000-token prompt at **10.46**. It degrades smoothly, no cliff.

*(262,144 rows measured on STRIX\_LEAN. All three tiers showed identical GTT at every lower rung and
agreed on generation to 0.01 tok/s at 131,072, so this behaviour is the model's, not the tier's.)*

⚠ Honest limit: the corpus is one source tree. A workload spanning many languages and repositories
will touch more of the n-gram table before it saturates. What these numbers do show is that the
table does not grow without bound with depth.

For short prompts (~3,300 tokens) this tier measured **345 tok/s prompt processing, 22.6 tok/s
generation** — median of 3, a different prompt each run.

## Every tier, every depth

Generation tok/s. Same box, same flags, prompt sized to 85% of each context window:

| context | FAST | STRIX_LEAN | STRIX |
|---|---|---|---|
| 8,192 | 21.72 | 22.16 | 22.02 |
| 32,768 | 21.31 | 20.44 | 20.52 |
| 65,536 | 18.26 | 18.15 | 18.23 |
| 131,072 | 15.34 | 15.33 | 15.34 |

The tiers are within noise of each other at every depth — at 131,072 they agree to **0.01 tok/s**.
Tier choice changes file size and (presumably) quality. It does not change speed and it does not
change memory. Quality is the one thing I have not measured, so I won't claim it.

*(The 65,536 FAST figure is the median of 3 re-runs, 18.21–18.34. A single earlier run read 16.34
and did not reproduce — reported here rather than quietly dropped.)*

## Files

Sharded so no file exceeds HF's 50 GB limit. Point `--model` at the **first** shard;
llama.cpp loads the rest automatically.

| file | size |
|---|---|
| `Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX_LEAN-00001-of-00003.gguf` | 41.86 GiB |
| `Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX_LEAN-00002-of-00003.gguf` | 41.62 GiB |
| `Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX_LEAN-00003-of-00003.gguf` | 15.01 GiB |

## Usage

```
llama-server \
  --model Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX_LEAN-00001-of-00003.gguf \
  --host 127.0.0.1 --port 8080 \
  --n-gpu-layers 999 --flash-attn on --fit off \
  --ctx-size 131072 --threads 16 --jinja
```

Leave mmap alone — see the consequences above.

## Memory — it's shared, not VRAM

**"GPU memory" on this box means GTT** — the aperture through which the integrated Radeon
8060S addresses ordinary system RAM. There is no separate VRAM pool, so every GiB the model
takes is a GiB the OS no longer has. `nvidia-smi`-style intuitions do not transfer here.

The file is 98.49 GiB but only ~63–75 GiB is ever resident, depending on context depth. It fits
a 128 GB Strix Halo at the **full native 262,144 context** with room to spare. Load it before
anything else has taken UMA, and budget from `MemAvailable` in `/proc/meminfo` — **never from
GTT free**, which lies on unified-memory parts.

<!-- CREDITS:START -->

## Acknowledgements

This build would not exist without the work below. Please star and follow these
projects — the quantisation format used here is their engineering, not mine.

**[charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)** — the fork that defines the
ROCmFP4 / ROCmFPX tensor formats. Every file in this repository was produced with its
`llama-quantize` and runs on its runtime. Licensed MIT, based on upstream llama.cpp.
*The `qwen4exp` architecture is not part of that fork* — it comes from upstream llama.cpp work and
is applied on top via [`qwen4exp-on-rocmfpx-d3ca537.patch`](./qwen4exp-on-rocmfpx-d3ca537.patch)
in this repo. See the build section above.

**[llama.cpp](https://github.com/ggml-org/llama.cpp) — ggml-org and contributors**
The inference engine, GGUF format and conversion tooling everything here is built on.

**AMD ROCm** — the compute platform these builds target (ROCm 7.2.4 on gfx1151).

**Qwen team** — the base model. See `base_model` for the source release; license is
qwen-community-1.0.

<!-- CREDITS:END -->
