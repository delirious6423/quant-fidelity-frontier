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
  - imatrix
  - qwen4exp
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

# Qwen3.8-Flash-Next-Uncensored — ROCmFP4 FAST **imatrix** GGUF — AMD Ryzen AI Max+ 395 / gfx1151

The **importance-matrix-calibrated** FAST build. Same recipe, same 4-bit size, same speed as the
plain [FAST build](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-GGUF)
— **measurably lower perplexity** because quantization was weighted by a calibrated importance matrix.

⚠️ **Research artifact.** Refusal behaviour has been removed. This does not add capability — it
removes guardrails. Use it deliberately, in a context where that is appropriate, and own the output.

## Measured quality — held-out WikiText-2 raw, `-c 512`

| build | PPL | Δ |
|---|---|---|
| plain FAST (no imatrix) — [sibling repo](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-GGUF) | 5.3465 ± 0.034 | — |
| **this — FAST imatrix** | **5.0337 ± 0.031** | **−5.9%** |

Both files are the same 4-bit recipe at **87.9 GiB / 4.27 bpw** — the only difference is that this
one's quantization was importance-weighted. imatrix moves **quality, not speed**: decode t/s is
identical (same bits per weight), so this is graded on perplexity, not tok/s.

- **Calibration corpus:** bartowski
  [`calibration_datav3`](https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8).
- ⚠️ The imatrix was computed on the **4-bit** model, not BF16 — the 51.2B PLE table plus the
  128 GB GTT ceiling blocks a BF16 forward pass on Strix Halo. A higher-precision imatrix source
  would likely gain a little more. It is still a real, measured −5.9%.

*For reference: dropping the protected Q6_K head to a fully 4-bit head costs only **+1.5% PPL**
(measured 5.1075 with imatrix) — small, now that imatrix calibration absorbs most of the head
damage. This build keeps the protected Q6_K head.*

## Speculative decoding (MTP) — now working on this arch

The `qwen4exp` MTP graph shipped with a broken combiner (it mean-pooled the hyper-connection
streams); `draft-mtp` acceptance sat near 0.36. This repo ships the fix as
[`qwen4exp-mtp-graph.patch`](./qwen4exp-mtp-graph.patch) — apply it to the fork above and rebuild.
With it, plus the stock Flash-Next MTP head from
[`kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF`](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF):

- **acceptance 0.94**, **31.80 tok/s** with `--spec-type draft-mtp` vs **24.9 tok/s** no-draft
  (**+27.7%**) — warm 160-token completion at **short context (`-c 2048`)**, neutral prompt,
  `cache_prompt:false`, same box/binary, drafting with the **Q8_0** head
  (`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, in the heads repo). The Q6_K / Q4 heads were not benchmarked
  here and may land somewhat differently.

The bundled MTP head is *stock* Flash-Next (not uncensored). It only proposes draft tokens; the
main model verifies every one, so it never alters this model's output — it just makes generation
faster on content it can predict. Without the patch, plain decode is unaffected; only `draft-mtp`
needs it.

```
llama-server -m Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-FAST-imatrix-00001-of-00003.gguf \
  -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp \
  --spec-draft-n-min 0 --spec-draft-n-max 1 --n-gpu-layers-draft 99 \
  -ngl 999 -fa on -np 1 -c 32768 --jinja
```
With the bundled checkpoint fix applied this is verified to 128K (see below); raise `-c` to suit
your context. Without that patch, keep speculative decoding at ≤32K.

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

## Recipe

Quantized from the BF16 weights published by
**[orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored)**
— the abliteration work is theirs, not mine. Go star their repo.

| tensor group | type |
|---|---|
| MoE expert weights (`ffn_*_exps`) | `TYPE_101` (ROCmFP4, 4.251 bpw) |
| shared expert (`ffn_*_shexp`) | `TYPE_101` |
| attention (`attn_*`) | all `TYPE_101` |
| `per_layer_token_embd.weight` (PLE, 51.2B params) | `TYPE_101` |
| `token_embd.weight` | `TYPE_101` |
| `output.weight` (lm head) | **Q6_K** (protected) |

**The Q6_K head:** `output.weight` is Q6_K, never 4-bit. Every sampled token passes through the lm
head, so its quantization error lands directly in the argmax. Verified by **exact tensor name**
after both quantize and split. With imatrix, a fully 4-bit head costs only +1.5% PPL (measured);
the Q6_K head here keeps that last 1.5%.

## Building a runtime that loads these files

Two patches, both in this repo: `qwen4exp-on-rocmfpx-d3ca537.patch` (arch enablement, 156 KB) and
`qwen4exp-mtp-graph.patch` (draft-mtp fix).

```bash
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX && git checkout d3ca537
curl -LO https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-imatrix-GGUF/resolve/main/qwen4exp-on-rocmfpx-d3ca537.patch
git apply qwen4exp-on-rocmfpx-d3ca537.patch
curl -LO https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-imatrix-GGUF/resolve/main/qwen4exp-mtp-graph.patch   # optional, for draft-mtp
git apply qwen4exp-mtp-graph.patch
# the checkpoint fix also ships in this repo — apply it before configuring:
curl -LO https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-imatrix-GGUF/resolve/main/qwen4exp-qsa-checkpoint-fix.patch
git apply qwen4exp-qsa-checkpoint-fix.patch      # checkpoint safety at >=64K: apply this always
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-quantize -j$(nproc)
```

## Speed — Ryzen AI MAX+ 395, gfx1151, ROCm 7.2.4, full offload

Decode speed is **identical** to the plain FAST build (same bits, same layout) — measured there at
**22.75 tok/s** gen / **387.3 tok/s** prompt / 63.3 GiB GTT (one fixed 6,963-token prompt,
`cache_prompt:false`, median of 4 settled samples). This model's native max context is **262,144**
and it runs there on a 128 GB box; the window is cheap (QSA caps KV), depth is what costs.

## Files

Sharded to stay under HF's 50 GB limit. Point `--model` at the **first** shard.

| file | size |
|---|---|
| `Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-FAST-imatrix-00001-of-00003.gguf` | 44.70 GB |
| `Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-FAST-imatrix-00002-of-00003.gguf` | 44.67 GB |
| `Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-FAST-imatrix-00003-of-00003.gguf` | 5.06 GB |
| `mmproj-Qwen3.8-Flash-Next-Uncensored-BF16.gguf` | 0.91 GB (vision tower) |
| `qwen4exp-on-rocmfpx-d3ca537.patch` | arch enablement |
| `qwen4exp-mtp-graph.patch` | draft-mtp graph fix |

## Usage

```
llama-server \
  --model Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-FAST-imatrix-00001-of-00003.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-Uncensored-BF16.gguf \
  --host 127.0.0.1 --port 8080 \
  --n-gpu-layers 999 --flash-attn on --fit off \
  --ctx-size 131072 --threads 16 --jinja
```

**Do not use `--no-mmap` (and do not use `-dio`).** The PLE table is streamed from the file through
the page cache; forcing it into anonymous memory gets the process OOM-killed with nothing in the
server log.

## Reproduction

```
imatrix : llama-imatrix -f calibration_datav3.txt -o unc.imatrix --output-format dat -ngl 999 -c 512 -b 512 -fa on -dev ROCm0
quantize: llama-quantize --imatrix unc.imatrix --output-tensor-type q6_K <BF16> <out> Q4_0_ROCMFP4_FAST 16
ppl     : llama-perplexity -m <this> -f wiki.test.raw -ngl 999 -fa on -dev ROCm0 -c 512   (NO -dio)
```
A number without its binary is a rumour — every figure above is measured on the runtime above.

<!-- CREDITS:START -->
## Acknowledgements

**[charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)** — defines the ROCmFP4 tensor
formats; every file here was produced with its `llama-quantize` and runs on its runtime. MIT.
*The `qwen4exp` architecture is applied on top via the bundled patches.*
**[llama.cpp](https://github.com/ggml-org/llama.cpp)** — engine, GGUF format, conversion tooling.
**AMD ROCm** — the compute platform (ROCm 7.2.4, gfx1151).
**[orcarouter](https://huggingface.co/orcarouter)** — published the uncensored BF16 checkpoint this
is built from; the abliteration is their engineering.
**Qwen team** — the original base model. License qwen-community-1.0.
<!-- CREDITS:END -->
