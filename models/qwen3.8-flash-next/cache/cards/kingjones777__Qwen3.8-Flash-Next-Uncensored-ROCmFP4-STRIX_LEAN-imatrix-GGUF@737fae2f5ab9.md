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
> **and** the ROCmFP4 tensor types in one tree. Our fork
> **[`kingjones30/ROCmFPX`](https://github.com/kingjones30/ROCmFPX)** (fork of
> [`charlie12345/ROCmFPX`](https://github.com/charlie12345/ROCmFPX), branch `main`) has both.
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

# Qwen3.8-Flash-Next-Uncensored — ROCmFP4 STRIX\_LEAN **imatrix** GGUF — AMD Ryzen AI Max+ 395 / gfx1151

## New file (2026-10-25): `Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-STRIX_LEAN-imatrix-hcq8-ssmf32-00001-of-00003.gguf (+ 00002, 00003)`

The files already in this repository are the imatrix-weighted STRIX_LEAN build in three shards. These three shards are the same
build with three groups of tensors copied from the stock Unsloth UD-Q4_K_XL file: the `hc_attn` / `hc_ffn` down and up
matrices at Q8_0 (194 tensors), and the `hc` inject tensors plus `ssm_alpha` / `ssm_beta` at F32. Every other tensor is unchanged.

**How it was built:**
- Tensor splice into `Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-STRIX_LEAN-imatrix`: `hc_attn` / `hc_ffn` down and up from the
  stock file at Q8_0, `hc` inject and `ssm_alpha` / `ssm_beta` from the stock file at F32.
- Resulting 2-D tensor types: 266 at type 101, 60 at type 100, 194 at Q8_0, 252 F32 (the base has 628 at type 101).
- Split into three shards of at most 40 GiB with gguf-py; every tensor in the shards was compared byte for byte with the unsplit
  file before upload.

**Measured on Super ROC** (AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4; one request at a time, temperature 0, cold prompt cache):

| Measurement | Result |
|---|---:|
| Prompt processing, 1594-token prompt | 1261.2 tok/s |
| Generation after that prompt (up to 256 tokens) | 56.1 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 56.8 tok/s (53.8-60.2) |
| Drafted tokens accepted on those prompts | 1031/1342 |
| Tool calls correct, tool_choice required / auto | 7/7 / 6/7 |

**Quality:** perplexity on a 6K-token long-text corpus (context 4096), measured with Super ROC: 5.97374 for these shards against
6.18385 for the imatrix STRIX_LEAN files already in this repository (3.4 % lower).

**Run it with Super ROC** (https://github.com/kingjones30/superroc):
```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-STRIX_LEAN-imatrix-hcq8-ssmf32-00001-of-00003.gguf --served-model-name qwen3.8-flash-next --context 32768 --think on --reasoning-effort low --speculative mtp --mtp-model /path/to/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --draft-tokens 3
```


The **importance-matrix-calibrated** STRIX\_LEAN build — the **highest-quality tier** in this
family. STRIX\_LEAN already spends more bits than FAST (Q5 token embeddings and PLE, half the
attention at higher precision); importance-weighted quantization on top gives it the **lowest
perplexity of any tier here.**

⚠️ **Research artifact.** Refusal behaviour has been removed. This does not add capability — it
removes guardrails. Use it deliberately, in a context where that is appropriate, and own the output.

## Measured quality — held-out WikiText-2 raw, `-c 512`

| build | PPL |
|---|---|
| plain FAST (no imatrix) — [repo](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-GGUF) | 5.3465 ± 0.034 |
| FAST imatrix — [repo](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-imatrix-GGUF) | 5.0337 ± 0.031 |
| **this — STRIX\_LEAN imatrix** | **4.9865 ± 0.031** |

4.9865 is the lowest perplexity across the whole family — **−6.7% vs plain FAST**, and −0.9% below
the FAST imatrix tier at the same calibration. That gap over FAST is the richer STRIX\_LEAN recipe
(higher-precision embeddings/PLE) *plus* the imatrix weighting.

> **Honesty note:** I do not publish a "plain STRIX\_LEAN vs imatrix STRIX\_LEAN" number, because
> the source BF16 was reclaimed after the build and I can't re-quantize a non-imatrix STRIX\_LEAN
> for a clean same-recipe delta. The clean isolated imatrix effect (same recipe, imatrix on/off) is
> the FAST tier's **−5.9%**; expect STRIX\_LEAN's isolated imatrix gain to be in the same range.
> imatrix moves **quality, not speed** — decode t/s is unchanged.

- **Calibration corpus:** bartowski
  [`calibration_datav3`](https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8).
- ⚠️ imatrix computed on the **4-bit** model (the 51.2B PLE + 128 GB GTT ceiling blocks a BF16
  forward pass on Strix Halo).

## Speculative decoding (MTP)

`draft-mtp` works on this arch once you apply [`qwen4exp-mtp-graph.patch`](./qwen4exp-mtp-graph.patch)
(bundled) — it fixes the graph combiner that otherwise held acceptance near 0.36. Pair with the
stock Flash-Next MTP head from
[`kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF`](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF).
Measured on the **FAST** tier with the fixed graph: acceptance **0.94** (+27.7% tok/s), at **short
context (`-c 2048`)** with the **Q8_0** head (`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, in the heads repo). STRIX\_LEAN
uses the same graph, but its own MTP speed has **not** been measured, and the published heads (Q6_K,
Q4) were not benchmarked. The head is *stock* Flash-Next — it only proposes drafts, the main model
verifies every token, so it never alters this model's output.

```
llama-server -m Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-STRIX_LEAN-imatrix-00001-of-00003.gguf \
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
— the abliteration is theirs. **4.78 bpw, 98.5 GiB.**

| tensor group | type |
|---|---|
| MoE expert weights (`ffn_*_exps`) | `TYPE_101` (ROCmFP4) |
| shared expert (`ffn_*_shexp`) | `TYPE_101` |
| attention (`attn_*`) | half `TYPE_100`, half `TYPE_101` |
| `per_layer_token_embd.weight` (PLE, 51.2B params) | `Q5_1` |
| `token_embd.weight` | `Q5_K` |
| `output.weight` (lm head) | **Q6_K** (protected) |

## Building the runtime

Two patches, both bundled: `qwen4exp-on-rocmfpx-d3ca537.patch` (arch, 156 KB) and
`qwen4exp-mtp-graph.patch` (draft-mtp fix).

```bash
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX && git checkout d3ca537
curl -LO https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-STRIX_LEAN-imatrix-GGUF/resolve/main/qwen4exp-on-rocmfpx-d3ca537.patch
git apply qwen4exp-on-rocmfpx-d3ca537.patch
curl -LO https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-STRIX_LEAN-imatrix-GGUF/resolve/main/qwen4exp-mtp-graph.patch   # optional
git apply qwen4exp-mtp-graph.patch
# the checkpoint fix also ships in this repo — apply it before configuring:
curl -LO https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-STRIX_LEAN-imatrix-GGUF/resolve/main/qwen4exp-qsa-checkpoint-fix.patch
git apply qwen4exp-qsa-checkpoint-fix.patch      # checkpoint safety at >=64K: apply this always
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-quantize -j$(nproc)
```

## Speed — Ryzen AI MAX+ 395, gfx1151, ROCm 7.2.4, full offload

Decode speed is the same as the plain STRIX\_LEAN build (same layout): **23.20 tok/s** gen /
**377.8 tok/s** prompt / 63.3 GiB GTT, one fixed 6,963-token prompt, `cache_prompt:false`, median
of 4. Native max context **262,144** on a 128 GB box.

## Files

| file | size |
|---|---|
| `Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-STRIX_LEAN-imatrix-00001-of-00003.gguf` | ~44.9 GB |
| `Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-STRIX_LEAN-imatrix-00002-of-00003.gguf` | ~44.7 GB |
| `Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-STRIX_LEAN-imatrix-00003-of-00003.gguf` | ~16.1 GB |
| `mmproj-Qwen3.8-Flash-Next-Uncensored-BF16.gguf` | 0.91 GB (vision tower) |
| `qwen4exp-on-rocmfpx-d3ca537.patch` | arch enablement |
| `qwen4exp-mtp-graph.patch` | draft-mtp graph fix |

## Usage

```
llama-server \
  --model Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-STRIX_LEAN-imatrix-00001-of-00003.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-Uncensored-BF16.gguf \
  --host 127.0.0.1 --port 8080 \
  --n-gpu-layers 999 --flash-attn on --fit off \
  --ctx-size 131072 --threads 16 --jinja
```

**Do not use `--no-mmap` (and do not use `-dio`).** The PLE table streams from the file through the
page cache; forcing it into anonymous memory gets the process OOM-killed with nothing in the log.

## Reproduction

```
quantize: llama-quantize --imatrix unc.imatrix <BF16> <out> Q4_0_ROCMFP4_STRIX_LEAN 16
ppl     : llama-perplexity -m <this> -f wiki.test.raw -ngl 999 -fa on -dev ROCm0 -c 512   (NO -dio)
```
A number without its binary is a rumour — every figure above is measured on the fork runtime above.

<!-- CREDITS:START -->
## Acknowledgements
**[charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)** — ROCmFP4 tensor formats (MIT).
**[llama.cpp](https://github.com/ggml-org/llama.cpp)** — engine, GGUF, conversion tooling.
**AMD ROCm** — ROCm 7.2.4, gfx1151. **[orcarouter](https://huggingface.co/orcarouter)** — the
uncensored BF16 checkpoint. **Qwen team** — the base model. License qwen-community-1.0.
<!-- CREDITS:END -->
