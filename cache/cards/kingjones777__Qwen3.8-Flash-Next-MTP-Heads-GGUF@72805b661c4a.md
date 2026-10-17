---
license: other
license_name: qwen-community-1.0
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - mtp
  - speculative-decoding
  - qwen4exp
  - rocmfp4
  - strix-halo
  - gfx1151
  - amd
  - research
  - not-working
---

# Qwen3.8-Flash-Next — MTP draft heads (GGUF)

## ⚡ These heads now deliver +27.7% at short context — the MTP graph is fixed

Until now the `qwen4exp` MTP graph had a broken combiner (it mean-pooled the hyper-connection streams),
so these heads accepted only **~0.36** of drafts and gave no speedup. The fix ships here as
[`qwen4exp-mtp-graph.patch`](./qwen4exp-mtp-graph.patch) — apply it to the
[`kingjones30/ROCmFPX`](https://github.com/kingjones30/ROCmFPX) fork and rebuild. With it, **measured
with `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (added 2026-09-17) on the Flash-Next Uncensored FAST (imatrix) build at short context
(`-c 2048`)**: **acceptance 0.94**, **31.80 tok/s** vs **24.9 tok/s** no-draft (**+27.7%**), warm
160-token completion, `cache_prompt:false`. The other heads were not benchmarked with the fixed graph.
A head only proposes drafts; the main model verifies every token, so output is unchanged.

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

> ⛔ **Keep `draft-mtp` at ≤32K context for now.** It restores context checkpoints whenever drafts are
> rejected — the same path as the ≥64K GPU wedge field-reported in [`Qwen3.8-Flash-Next-ROCmFP4-STRIX-GGUF#6`](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-ROCmFP4-STRIX-GGUF/discussions/6) — and it has only been
> tested at short context.

> ## ⛔ These files do not load. Read this before downloading anything.
>
> Every head in this repository fails at load time on every public build I know of, including my
> own fork:
>
> ```
> llama_model_load: error loading model: missing tensor 'blk.0.hc_attn_norm.weight'
> ```
>
> **There is no speed to be had here today.** I am publishing them because several people have
> asked for Flash-Next MTP, because the heads themselves look complete, and because the remaining
> gap is small enough and specific enough that somebody other than me may well close it in an
> afternoon. This is a research artifact, not a release.

---

## What these are

`Qwen/Qwen3.8-Flash-Next` ships a Multi-Token-Prediction block. Our converter used to drop it
outright — that is what I said in
[discussion #5](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-ROCmFP4-STRIX-GGUF/discussions/5).
**That part is fixed.** These five files are the extracted MTP block, converted and quantised. They
are what a `--spec-type draft-mtp -md …` run would consume if the loader accepted them.

MTP is worth having on this family. On Qwen3.8-27B, MTP on ROCm measured **2.03–2.46×** on
reasoning workloads (13.46 → 33.06 tok/s). Every published speed number on the Flash-Next cards is
**non-MTP**, so this is a real gap, not a rounding error.

## Files

| File | ftype | Size | Bytes |
|---|---:|---:|---:|
| `mtp-Qwen3.8-Flash-Next-BF16.gguf` | 32 | 7.24 GiB | 7,770,801,344 |
| `mtp-Qwen3.8-Flash-Next-Q4_0_ROCMFP4_FAST.gguf` | 103 | 1.94 GiB | 2,083,139,232 |
| `mtp-Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX.gguf` | 105 | 2.11 GiB | 2,266,977,952 |
| `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` | 15 | 2.59 GiB | 2,786,245,792 |
| `mtp-Qwen3.8-Flash-Next-Q6_K.gguf` | 18 | 3.17 GiB | 3,404,798,752 |
| `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | 7 | 3.85 GiB | 4,135,934,112 |

The BF16 head is the master — it is the one to work from, and the only one stock `gguf-py` can
rewrite (see below). The two ROCmFP4 tiers match the FAST and STRIX main-model tiers.

## The actual problem, precisely

The tensors are **not** missing. Each head carries all 35 tensors, including the complete
hyper-connection stack the error complains about:

```
blk.48.hc_attn_norm.weight     [10240]        <-- the tensor reported "missing"
blk.48.hc_attn_up.weight       [320, 10240]
blk.48.hc_attn_down.weight     [10240, 320]
blk.48.hc_attn_inject.weight   [10240, 4]
blk.48.hc_ffn_{norm,up,down,inject}.weight
output_hc_{norm,up,down}.weight
blk.48.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight
blk.48.indexer.{q_proj,k_proj,q_norm,k_norm}.weight
```

The mismatch is the **layer index**. The converter emits the MTP block at its absolute position in
the parent model — `blk.48` — while the head's own metadata declares:

```
qwen4exp.block_count = 1
```

so the loader builds a single layer at index **0** and looks for `blk.0.*`. Hence
`missing tensor 'blk.0.hc_attn_norm.weight'` while `blk.48.hc_attn_norm.weight` sits right there in
the file.

## What I have NOT proven

The obvious hypothesis is that renumbering `blk.48.*` → `blk.0.*` is sufficient. **I have not
demonstrated that**, and you should not assume it. Renaming fixes the lookup, but the `qwen4exp`
graph still has to actually build the hyper-connection and `nextn` ops for a draft model, and I
have not verified that it does.

Two things blocked me from testing it quickly, both tooling rather than model:

- Stock `gguf-py` cannot rewrite the ROCmFP4 tiers — it does not know the block geometry:
  `Quantized tensor bytes per row (2560) is not a multiple of Q4_0_ROCMFP4_FAST type size (17)`.
  A repack needs the C++ side or a `gguf-py` patch.
- My hand-rolled BF16 repack produced a file of the right size that the loader rejected with
  `gguf_init_from_file_ptr: failed to read tensor data`. That is my rewrite being wrong, not a
  statement about the head.

If you get a renumbered head to load, please open a discussion — that is the single most useful
thing anyone could post here.

## Runtime

Same fork as the main models. Stock `llama.cpp` has neither `qwen4exp` nor the ROCmFP4 types:

```bash
git clone https://github.com/kingjones30/ROCmFPX.git
cd ROCmFPX
# both fixes ship in this repo — apply them before configuring:
curl -LO https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF/resolve/main/qwen4exp-qsa-checkpoint-fix.patch
git apply qwen4exp-qsa-checkpoint-fix.patch      # checkpoint safety at >=64K: apply this always
curl -LO https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF/resolve/main/qwen4exp-mtp-graph.patch
git apply qwen4exp-mtp-graph.patch               # only if you want --spec-type draft-mtp
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-quantize -j$(nproc)
```

Reproduce the failure in one line:

```bash
./build/bin/llama-server -m mtp-Qwen3.8-Flash-Next-Q4_0_ROCMFP4_FAST.gguf -ngl 0 -c 512 --no-warmup
```

## What to run in the meantime

`--spec-type ngram-mod` works on Flash-Next today and is worth having — **but read the depth
warning first.** On ROCm I measure acceptance of **0.750** on code and **1.000** on prose at 32K.

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

⛔ Also do not run ngram-mod on the Vulkan backend on gfx1151 — same config, acceptance collapses
to **0.000 / 0.188**. Speculative decoding of every kind is currently broken on Vulkan/gfx1151.

## Main models

- [`Qwen3.8-Flash-Next-ROCmFP4-STRIX-GGUF`](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-ROCmFP4-STRIX-GGUF)
- [`Qwen3.8-Flash-Next-ROCmFP4-STRIX_LEAN-GGUF`](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-ROCmFP4-STRIX_LEAN-GGUF)
- [`Qwen3.8-Flash-Next-ROCmFP4-FAST-GGUF`](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-ROCmFP4-FAST-GGUF)
- [`Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-GGUF`](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-GGUF)
