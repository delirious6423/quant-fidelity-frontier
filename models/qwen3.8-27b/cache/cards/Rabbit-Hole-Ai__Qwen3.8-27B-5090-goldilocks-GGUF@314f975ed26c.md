---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- qwen3.8
- custom-quant
- mixed-precision
- rtx-5090
- windows
---

# Qwen3.8-27B — "Goldilocks" 7.36 bpw GGUF for a single RTX 5090 (Windows 11, 131k context)

> **⚠️ September 2, 2026 — two corrections, both below, both worth reading before you copy anything.**
> **(1)** The serving block in this card pairs a **quantized KV cache with speculative decoding**.
> That pairing is now measured as a **kernel-dispatch cliff**, not a cheap VRAM saving — see
> [the correction](#-september-2-2026--correction-do-not-pair--ctk-q8_0--ctv-q8_0-with---spec-type).
> **(2)** The rig this cook was solved for **no longer serves it** — see
> [what we run today](#september-2-2026--what-we-run-today).
> Neither correction changes the recipe, the solver, or the fidelity tables. Both change what you
> should type.

A single-file, mixed-precision GGUF of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), size-solved
to exactly fill a 32 GB RTX 5090 running under a live Windows 11 desktop — VRAM full at **131k context (text)** or
**110k context (vision)** with measured headroom.

- **23,990 MiB** weights / **7.36 bpw** effective (stock Q6_K-class quants land ~22.9–23.8 GiB; this spends
  the leftover VRAM on precision instead of leaving it idle)
- Tensor mix: **339 × Q8_0, 167 × Q6_K, 360 × F32** across all 866 tensors
- The **MTP layer (blk.64) is preserved**, so llama.cpp's `--spec-type draft-mtp` speculative decoding
  works — and see [the `blk.64` warning](#a-warning-for-anyone-generating-per-tensor-recipes-check-for-blk64)
  for why that is *not* automatic if you generate recipes from published sensitivity data
- Mean KL divergence vs the BF16 original: **0.001800** (RTN build) / **0.001678** (imatrix build, see the
  August 21 update below) — full benchmark table below, including the stock
  quants that tie or beat it
- **Generation speed — corrected September 2, 2026.** The original "~100 tok/s" bullet here was a
  shallow-depth figure and is superseded. Re-measured with as-served sampling over 512-token samples,
  median of three passes: **72.6 tok/s at 8k depth / 64.0 tok/s at 98k depth** for the imatrix file in
  this repo with the vision projector loaded at 110k; **96.6 / 54.4 tok/s** at 8k / 128k for the text
  config at 131k. (That text pair was measured on a same-size, same-recipe sibling recook rather than
  byte-for-byte on the file here — said plainly so you can discount it accordingly.) The fall-off with
  depth is exactly what the KV correction below is about.

## ⚠️ September 2, 2026 — correction: do NOT pair `-ctk q8_0 -ctv q8_0` with `--spec-type`

The serving block further down originally recommended a **q8_0 KV cache together with**
`--spec-type draft-mtp,ngram-mod,ngram-map-k4v` **and** `--spec-draft-n-max 3`. Those two flags fight
each other on CUDA, and the cost is far larger than the VRAM the quantized cache hands back. The
recommended cache dtype is now **`-fa on -ctk f16 -ctv f16`** — with the caveat, stated honestly in
[the trade](#what-that-means-for-this-file-specifically), that f16 KV does **not** fit at 131k
alongside 23.4 GiB of weights on a 32 GB card.

### The mechanism, read from the engine source

llama.cpp picks its flash-attention kernel per ubatch in `ggml/src/ggml-cuda/fattn.cu`, function
**`ggml_cuda_get_best_fattn_kernel`** (`fattn.cu:358`). Line numbers below are from **build 10437,
commit `16d222fc5`** — the exact build every number in this card was measured on. (Re-checked against a
later mainline checkout, commit `ca3d5a3e`, 28 August 2026: identical code at identical line numbers.)

- With **quantized** K/V, the branch at **`fattn.cu:467-471`** keeps the cheap **VEC** kernel only while
  **`Q->ne[1] <= 2`** (`:469`). `Q->ne[1]` is the number of query tokens in the ubatch.
- Anything larger falls straight through to **`return BEST_FATTN_KERNEL_MMA_F16;`** at **`:482`**.
- `ggml_cuda_flash_attn_ext_get_alloc_size` then sets **`need_f16_K = need_f16_V = true`** for
  `BEST_FATTN_KERNEL_MMA_F16` (**`:551-555`**). The MMA path *requires* f16 K and V — so with a
  quantized cache, **the whole KV cache is converted every decode step.**

At ~123k context that is on the order of 12 GB of extra memory traffic per generated token. It is a
dispatch cliff, not a gradual bandwidth trade.

### The `n_max` arithmetic — the part that is easy to get wrong

A speculative verification batch is **`1 sampled token + n_draft`**. So:

| `--spec-draft-n-max` | verify batch (`Q->ne[1]`) | kernel with quantized KV |
|---:|---:|---|
| 3 (this card's original recommendation) | 4 | MMA_F16 — **pays the full tax** |
| 2 | 3 | MMA_F16 — **still pays the full tax** |
| 1 | 2 | VEC — under the ceiling |
| 0 (speculation off) | 1 | VEC |

An earlier version of our own notes said "q8_0 KV is fine as long as you keep `n_max` below 3."
**That was wrong**, and this is the correction: only **`--spec-draft-n-max <= 1`** stays under the
ceiling.

### The obvious workaround was measured, and it is the worst of the three

"Keep q8_0 KV and just shrink the draft to `n_max 1`" costs no VRAM at all, so it looks free.
Measured (median of 3 passes, 512 generated tokens, `ignore_eos`, as-served sampling):

| arm | TG @ 8,192 depth | TG @ ~123k depth |
|---|---:|---:|
| q8_0 KV + `--spec-draft-n-max 1` — 7.36 bpw cook, `-c 131000 -ub 384` | **74.9** | **57.9** |
| f16 KV + `--spec-draft-n-max 3` — lighter 19.4 GiB weights, `-c 131072 -ub 512` | **113.7** | **93.7** |

Read that honestly: those two arms differ in **weights and `-ub`** as well as the KV flag, so it is a
*config* comparison, not a single-variable KV test. What it does settle is the practical question —
throwing away draft depth to keep a quantized cache does not recover the loss.

### The single-variable evidence

Same weights, same 96k context, same drafter stack, **only `-ctk`/`-ctv` changed** — measured on a
5.79 bpw Unsloth build of this same model, same rig, same engine build (median of 3, 512 tokens,
as-served, VRAM sampled *during* generation):

| arm | TG @ 8,192 | TG @ 87,552 | free VRAM during generation |
|---|---:|---:|---:|
| q8_0 KV | 120.1 | **69.0** | 6,716 MiB |
| f16 KV | 108.5 | **93.3** | 4,280 MiB |

**+35.2% at depth**, bought with 2,436 MiB of free VRAM. A second signature confirms the *mechanism*
and not just the outcome: **f16 flattens the depth curve rather than merely raising it** — the f16 arm
loses 14% going 8,192 → 87,552 where the q8_0 arm loses 43%. The tax scales with cache size, which is
exactly what "convert the whole cache every step" predicts.

(Ignore the shallow column when judging any of this. At 8,192 tokens this drafter stack produced
spreads up to 70% across three passes; the deep rows behaved far better, 12–25%.)

### What that means for this file specifically

**This cook cannot hold f16 KV at 131k on a 32 GB card.** The weights are 23.4 GiB, and f16 KV
measured **~26–30 KiB per token** on this architecture — from two independent measurements, at 96k and
at 64k, which disagree by 13%, so treat it as a range and not a constant. At 131k that is roughly
3.3–3.8 GiB more cache than q8_0, and there is no room for it. Three real options, and which is right
depends on your card:

1. **f16 KV at a reduced context.** Measured on this cook's uncensored sibling (same recipe, same
   7.36 bpw class, same drafter stack): q8_0 at 108k gave 100.9 / 79.6 / 70.6 tok/s at 8,192 / 55,808 /
   99,328 depth; f16 only fit at **64k**, where it gave 95.0 / 92.6 at 8,192 / 55,808. At the matched
   55,808 depth that is **+16.3%** — bought with 44k of surrendered context. Two variables changed (KV
   dtype *and* context), so quote it as a config delta, never as "the flag is worth 16%".
2. **q8_0 KV at full 131k, and accept the tax.** Still a working, coherent config — it is what this
   card originally shipped and what produced the fidelity tables below. It is simply slower at depth
   than it looks.
3. **Lighter weights + f16 KV at full context.** That is what we moved to; see the next section.

**Vision is affected identically.** The mmproj path runs the same attention kernels, so the same cliff
applies with `--mmproj` loaded — the 110k A/B in the next section is a vision measurement.

## September 2, 2026 — what we run today

Full disclosure, because the serving envelope is the part of this card people copy: **as of
2026-09-02, no alias on the rig this cook was solved for serves this cook any more.**

| role | was | is now |
|---|---|---|
| general text, 131k | this cook, 7.36 bpw, q8_0 KV | Unsloth **UD-Q5_K_XL** (19.4 GiB), **f16 KV**, `-c 131072` |
| vision, 110k | this cook + BF16 mmproj, q8_0 KV | same UD-Q5_K_XL + the same BF16 mmproj, **f16 KV** |
| document-agent lane | — | Unsloth **UD-Q4_K_XL**, f16 KV, 79k |
| document-agent lane, "smarter" | — | Unsloth **UD-Q6_K**, q8_0 KV, 64k — f16 measured at only **+7.1%** here, not worth the context it costs |

### Why — with the numbers

The vision entry is the cleanest illustration, because both arms were measured with the projector
loaded at the same 110k context (median of 3 passes, 512 tokens, as-served, VRAM sampled **during**
generation rather than after load):

| arm | TG @ 8,192 | TG @ 98,304 | peak VRAM | min free |
|---|---:|---:|---:|---:|
| **this cook**, 7.36 bpw + q8_0 KV | 72.6 | 64.0 | 31,347 MiB | **840 MiB** |
| Unsloth UD-Q5_K_XL + f16 KV | **110.4** | **83.4** | 30,175 MiB | **2,012 MiB** |

The arithmetic behind it: swapping 23.4 GiB of weights for 19.4 GiB hands back about **4.1 GiB**, and
f16 KV at 110k costs roughly **2.8–3.2 GiB** of that. So the swap bought the whole KV-tax win **without
giving up any context**, and still landed ~1.1 GiB ahead on VRAM — which is why free memory went from
840 MiB (under our 1,536 MiB desktop floor) to 2,012 MiB (clear of it).

**The envelope moved; that is the whole story.** Two things changed on the rig that have nothing to do
with the recipe: the desktop's own idle VRAM footprint grew over three weeks, and a GPU-hungry
document-extraction service now has to sit beside the model. Both consume exactly the headroom this
cook was solved to fill. A quant solved to fit a 32 GB card to the millimetre is, by construction, the
first thing to break when the spare millimetres disappear.

### And the honest cost of that swap

It is a real fidelity loss, measured the same way as the tables below (`llama-perplexity
--kl-divergence` against cached BF16 base logits; a control arm reproduced this cook's published figure
to six decimals, so these numbers join the same chain):

| build | bpw | corpus A mean KLD | same-top-token | vs this cook |
|---|---:|---:|---:|---:|
| this cook (7.36 bpw class) | 7.3654 | 0.001606 | 98.250% | 1.00× |
| Unsloth UD-Q6_K | 6.4372 | 0.003486 | 97.173% | 2.17× |
| Unsloth UD-Q5_K_XL *(now serving)* | 6.1132 | 0.005169 | 96.756% | 3.22× |
| Unsloth UD-Q5_K_M | 5.7895 | 0.006734 | 96.202% | 4.19× |

So this is **not** a retraction. The recipe still wins on fidelity per byte in its size class; we
traded fidelity for headroom because headroom became the binding constraint, and we measured the price
instead of hand-waving it. **The recipe and the solver are what this repo is for** — re-solve
`build_recipe.py` against *your* card's measured budget and you get a file sized for your envelope,
not ours.

### Two practical consequences for you

- **The original RTN file in this repo is now the only copy that exists.** The local original was
  deleted from disk on 2026-08-21 when the imatrix build replaced it, and this repo has been its
  rollback copy ever since. It is not going anywhere.
- **Treat the serving block as a dated snapshot, not living documentation.** On our rig alone this
  model family was repointed three times in three weeks, every time for a VRAM or kernel-dispatch
  reason that is a property of *the card and the engine build*, not of the weights.

## August 21, 2026 update: new imatrix build

There are now **two GGUFs** in this repo:

| File | What it is |
|---|---|
| `Qwen3.8-27B-5090-goldilocks-imatrix.gguf` | **Recommended default.** Same recipe, same 866-tensor layout, same VRAM envelope, MTP block intact -- the only change is that the 167 Q6_K tensors were quantized against Unsloth's published importance matrix instead of round-to-nearest. |
| `Qwen3.8-27B-5090-goldilocks.gguf` | The original RTN cook. Kept for reproducibility and for anyone who has already pinned it by commit SHA. Identical size to within 352 bytes of GGUF header metadata -- same tensor names, same quant types, same speed. |

### Measured fidelity -- imatrix vs. the original RTN build, vs. true BF16

KL-divergence sweep, two independent corpora (literary prose and GPQA-style science prose), same
methodology as the table above (`llama-perplexity --kl-divergence`, llama.cpp build 10437):

| Build | Mean KLD | Median KLD | 99th pct KLD | Same-top-token |
|---|---|---|---|---|
| RTN (original) | 0.001800 | 0.000827 | 0.015519 | 98.050% |
| Imatrix (new) | **0.001678** | **0.000745** | **0.014266** | **98.278%** |

Read this honestly: the improvement is **real but small**. Median KLD dropped ~10% on two unrelated
corpora (literary prose and science prose independently), and every one of six KLD readouts measured
(mean / median / 99th-percentile x 2 corpora) moved the same direction with zero regressions -- that
consistency is why we're confident it's real and not corpus-sampling noise. In behavioural terms it's
close to invisible: top-token agreement with BF16 moved +0.23 percentage points on one corpus and not
at all on the other. Nobody will feel this in a chat session. It's worth taking because it's completely
free -- same file size, same tensor layout, same speed, same VRAM.

### How it was made

Same recipe as the original cook (see "How it was built" below) -- the only change is pointing
`llama-quantize` at Unsloth's published `imatrix_unsloth.gguf` for Qwen3.8-27B (496 calibration entries,
1,251 chunks, ~10.2M tokens) when quantizing the Q6_K tensors. Q6_K is about 52% of the file's bytes,
and the imatrix covers 92% of those tensors; `token_embd.weight` is deliberately exempt -- llama.cpp
reads it by lookup rather than matrix-multiply, so an importance matrix doesn't apply to it.

Quantized with the **same llama.cpp build as the original cook -- build 10431 (commit `1692f9e50`)** --
so this is a clean single-variable comparison: nothing about the binary, recipe, or tensor census
changed between the two files, only the imatrix.

### A caveat worth stating plainly

This is a Q6_K-range imatrix application, and as far as we've found, nobody has published a controlled
Q6_K with-vs-without-imatrix comparison before now. Treat the table above as **one data point**, not a
claim that imatrix-on-Q6_K beats other quantizers in general -- importance matrices are best-established
at lower bit-depths (Q2-Q4), where round-to-nearest has more to lose. Ours may be the first measured
data point at this bit-depth; take it as exactly that.

## Why "Goldilocks"

Stock quants are made for everyone, so on any given card they're either too big (don't fit with full
context) or too small (leave VRAM idle). This one is solved for a specific, very common target: **one
RTX 5090 on Windows 11**, where the desktop/compositor already holds a slice of the 32 GB. The recipe
starts from a per-tensor sensitivity map built with [Thireus](https://huggingface.co/Thireus)' GGUF Tool
Suite and promotes tensors Q6_K → Q8_0 in a principled order until a *measured* VRAM budget is exactly filled. Not extrapolated —
every number below was measured on the card, post-request, with the desktop running.

| Config | Context | mmproj | VRAM used / total | Free |
|---|---|---|---|---|
| Text | 131,000 | none | 30,715 / 32,607 MiB | 1,892 MiB |
| Vision | 110,000 | BF16 | 30,949 / 32,607 MiB | 1,658 MiB |

Context is cheap on this architecture (only 16 of 64 layers are full-attention; the rest are linear/SSM
with fixed-size state): ~43 MiB per 1k tokens at q8_0 KV, compute buffers included. Vision at 118k was
measured and **rejected** (breaks a 1.5 GiB safety floor) — don't push past 110k with the mmproj loaded.

> **Both rows above are q8_0 KV, measured 2026-08-14.** They remain the right numbers for the flags
> they were taken with — but see the September 2 correction: f16 KV, now the recommended cache dtype,
> does not fit inside either row. Re-measure on your own card before trusting any of it, and sample
> VRAM *during* a deep request rather than after load. Our own idle desktop baseline drifted upward by
> roughly 477 MiB between mid-August and early September, which is by itself enough to invalidate a
> margin this thin.

## Measured fidelity — read it honestly

![KL divergence vs BF16 — 9 Qwen3.8-27B quants](kld-vs-bf16.png)

`llama-perplexity --kl-divergence` against a true BF16 reference GGUF (PPL 8.1090), 143 chunks of a mixed
corpus at n_ctx 512, llama.cpp build 10437. The highlighted row is this cook's imatrix build (working name
`thireus-7.4bpw-imatrix`): **mean KLD 0.001678 at 23.4 GiB**, 98.3% same-top-token as BF16. The original
RTN build (`thireus-7.4bpw`, mean KLD 0.001800) is also in the table for comparison.

To be clear about what this is *not*: it is not a fidelity win over the field.
[AtomicChat's AD-Q6_K](https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF) lands statistically identical
fidelity at nearly the same size, and [Unsloth's UD-Q6_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
is slightly closer to BF16 for ~0.7 GiB more. What this repo adds is the **recipe + solver** (re-solve for
your own card's budget) and the **measured single-5090 serving envelope** below.

## A warning for anyone generating per-tensor recipes: check for `blk.64`

This applies to anyone building a Qwen3.8-27B quant from published per-tensor sensitivity data, not
only to this recipe. It cost us a build to find, so it is written down here.

**Qwen3.8-27B has 866 tensors, and 15 of them are `blk.64.*` — the MTP / `nextn` head.** That block is
what `--spec-type draft-mtp` speculative decoding runs on. A GGUF without it loads fine, serves fine and
produces perfectly good text; it just silently has no MTP drafter.

**The trap:** the published Qwen3.8-27B per-tensor sensitivity data in the GGUF Tool Suite ecosystem
has **no `blk.64` columns at all**. Feed that to an automatic recipe generator and `blk.64` looks like
it has *zero* sensitivity — so a greedy solver either demotes it to the cheapest available type or
drops it entirely and spends the freed budget elsewhere. It is not an error and nothing warns you.

**We measured it.** A web-generated 7.4726 bpw recipe (`--use-auto-quant-assign --with-imatrix`, 23 GB
all-GPU target) produced a file with **851 tensors and `blk.64` absent** — the missing 15 tensors are
exactly the MTP head, and that is where the budget for its extra Q8_0 promotions came from. Diffed
against this cook across the 851 shared tensors, 80 assignments differ: it buys FFN precision
(`ffn_down` ×18, `ffn_up` ×16, `ffn_gate` ×12 promoted to Q8_0) and sells attention precision
(`attn_q` ×12, `attn_qkv` ×9 demoted to Q6_K).

**And here is the honest part: that file is more faithful than ours.** On the same two-corpus KLD
sweep it scored mean KLD **0.001474** on corpus A against this cook's 0.001678 — the best of the three
builds tested. We rejected it anyway, because on this rig the MTP drafter is worth more than 0.0002 of
KL divergence. That is a **trade**, and you should make it deliberately rather than have a solver make
it for you silently.

**How this cook avoids it.** The published recipe pins the head explicitly, with **anchored** regexes
(lines 520–527 of `tensortypes_qwen3.8-27b_goldilocks.txt`):

```
^blk\.64\.attn_(k|q|v|output)\.weight$=q8_0
^blk\.64\.ffn_(down|gate|up)\.weight$=q8_0
^blk\.64\.nextn\.eh_proj\.weight$=q8_0
```

Anchoring matters for a second reason too: `llama-quantize --tensor-type` compiles each entry as a
`std::regex` and applies `regex_search`, i.e. **unanchored** — so a bare `attn_q` also matches
`attn_qkv`. All 554 assignments in this recipe are `^…$`-anchored, and all 554 were verified present in
the finished GGUF.

**Verify by census, not by trust:** after quantizing, count tensors. **866, with `blk.64` present.**
851 means the head is gone.

Two related things worth knowing when shopping for someone else's Qwen3.8-27B GGUF:

- Some publishers deliberately store the MTP layers at **Q4_0** even inside otherwise high-bit quants,
  on the argument that a faster drafter repays a lower acceptance rate. That is a defensible trade and
  it is *not* the silent-drop failure above — but it is a choice, and worth knowing which one you
  downloaded. This cook keeps the head at Q8_0.
- Speculative decoding is **lossless** — the target model verifies every drafted token — so a coarser
  or missing drafter cannot make output *worse*, only slower. The whole risk is throughput.

**How much throughput?** Measured as-served on this rig (3 passes, 512 tokens, same engine build,
q8_0 KV, ~123k context), the MTP + ngram draft stack was worth roughly **1.3–1.5×**: 46.5 tok/s with no
drafter against deep medians of 59.5–66.3 tok/s with it.

## Files

| File | What it is |
|---|---|
| `Qwen3.8-27B-5090-goldilocks-imatrix.gguf` | **recommended default** — the imatrix build (23,990 MiB), see the August 21 update above |
| `Qwen3.8-27B-5090-goldilocks.gguf` | the original RTN build (23,990 MiB), kept for reproducibility / SHA-pinning |
| `qwen3.8_chat_template_froggeric6level.jinja` | recommended chat template (see below) |
| `tensortypes_qwen3.8-27b_goldilocks.txt` | the exact per-tensor recipe (554 assignments, `llama-quantize --tensor-type-file` format) |
| `build_recipe.py` | the solver that produced the recipe — reproduce or re-solve for your own VRAM budget |
| `kld-vs-bf16.png` | the fidelity benchmark table above |

**Vision:** the mmproj is deliberately not duplicated here — grab `mmproj-BF16.gguf` from
[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) and pass it with `--mmproj`.
(Verified September 2026: the projector is **quant-independent** — it pairs correctly with any
Qwen3.8-27B text weights, including stock Unsloth ones. There is no separate "matching" projector to
hunt for and none is needed.)

## Serving (llama.cpp, Windows 11)

**No environment variables are needed** — no `GGML_*`, no `CUDA_*`; everything is CLI flags. Built and
served with llama.cpp mainline (build 10437 or later recommended), CUDA 13.3, `sm_120a`.

> **⚠️ Read the [September 2 correction](#-september-2-2026--correction-do-not-pair--ctk-q8_0--ctv-q8_0-with---spec-type)
> before running this.** `-ctk f16 -ctv f16` below is the corrected recommendation, and **f16 KV does
> not fit at 131k with these 23.4 GiB weights on a 32 GB card.** Pick your context by measuring, not by
> copying: budget ~26–30 KiB per token of context for f16 KV (against ~14–17 KiB at q8_0), plus compute
> buffers. If you want this card's original 131k, you must either keep `-ctk q8_0 -ctv q8_0` and accept
> the depth tax, or use lighter weights.

Text — corrected flags, with the context left as `<N>` deliberately:

```bat
llama-server -m Qwen3.8-27B-5090-goldilocks-imatrix.gguf --no-mmproj ^
  --fit-target 512 --load-mode dio ^
  --spec-type draft-mtp,ngram-mod,ngram-map-k4v ^
  --spec-draft-n-max 3 --spec-draft-p-min 0.0 ^
  --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64 ^
  --spec-ngram-map-k4v-size-n 16 --spec-ngram-map-k4v-size-m 96 --spec-ngram-map-k4v-min-hits 1 ^
  -ngl 999 -c <N> -b 512 -ub 512 -np 1 ^
  -fa on -ctk f16 -ctv f16 ^
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00 --presence-penalty 0.0 ^
  --jinja --chat-template-file qwen3.8_chat_template_froggeric6level.jinja ^
  --chat-template-kwargs "{\"reasoning_effort\":\"xhigh\",\"preserve_thinking\":true}" ^
  --reasoning-budget -1
```

On a 32 GB card with these weights, the measured analogue of this config (same recipe class, same
drafter stack) fit f16 KV at **64k**, not 131k. Start there, measure, then move up.

Vision: same flags, two changes — `--mmproj mmproj-BF16.gguf` instead of `--no-mmproj`, and a smaller
`-c` again for the same reason.

**If you would rather keep the full 131k text / 110k vision envelope this card was solved for**, change
exactly two things: `-ctk q8_0 -ctv q8_0`, and `-c 131000` (or `-c 110000` with the mmproj). That is the
original, still-working configuration — you are choosing context over depth throughput, knowingly, which
is a perfectly reasonable choice.

Notes from measuring on this hardware:

- `--load-mode dio` loads in ~5.5 s and was **2× faster than mmap** on this rig (measured, not assumed —
  mmap's warm-reload advantage never materialized; page-fault copy into the CUDA staging buffer is the
  bottleneck).
- Leave the host prompt cache at its default (don't pass `--cache-ram 0`): revisiting a long prompt drops
  from ~6.5 s of re-prefill to ~0.1 s. Host RAM only, zero VRAM impact.
- Sampling values are [Unsloth's published *thinking mode* recommendations](https://unsloth.ai/docs/models/qwen3.8)
  for Qwen3.8 (temp 1.0, top-p 0.95, top-k 20, min-p 0).
- **Benchmark honesty, learned the hard way.** With this drafter stack, greedy sampling lets the ngram
  drafter replay long context verbatim and fakes enormous throughput — we logged a "475 tok/s" sample
  that was pure replay, and greedy token rates that *rose* with depth, which is impossible on fixed
  hardware. Measure with as-served sampling, over at least 512 generated tokens with `ignore_eos`, take
  the **median of three passes**, and sample VRAM *during* generation rather than after load.

## Chat template

The bundled template is [froggeric's Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
v22 with the reasoning-effort block extended from 3 to a 6-level ladder
(`max / xhigh / high / medium / low / minimal`), selectable per-request via
`--chat-template-kwargs` or the `reasoning_effort` chat kwarg. Everything else — thinking on/off
sentinels, tool-call formats, `preserve_thinking` — is untouched froggeric v22. The GGUF's embedded
template also works if you'd rather not use it.

## How it was built (and verified)

1. `convert_hf_to_gguf.py` on the official Qwen/Qwen3.8-27B BF16 safetensors → 54.66 GB BF16 GGUF.
2. `build_recipe.py` takes the proven Thireus-style Qwen3.6-27B q6_0-7bpw per-tensor map (Qwen3.8 has the
   identical 866-tensor shape, so it transfers verbatim), **gates** it (every regex must match exactly one
   real tensor — a `--tensor-type` pattern matching nothing is silently ignored by llama-quantize, so this
   is checked, not assumed), then **solves upward**: promotes Q6_K → Q8_0 in the order the map's own
   structure implies (completing partially-protected groups first: `ssm_out`, `attn_gate`, `attn_q`,
   `attn_qkv`, then the FFN stack, `ffn_down` first, early layers first) until the measured VRAM budget
   is filled.
3. `llama-quantize --tensor-type-file` with the emitted recipe, base type Q6_K.

Verification before shipping: the size model predicted the output within **0.05 MiB** of llama-quantize's
own dry run; all 554 recipe assignments were confirmed present in the final GGUF (zero mismatches); the
tensor census came out at **866 with `blk.64` present** (see the warning above for why that check is not
optional); and KL divergence vs BF16 was measured rather than trusted (table above).

## Credits

This is a remix — the hard parts are other people's work:

- **Qwen team** — the Qwen3.8-27B base model (Apache-2.0).
  <https://huggingface.co/Qwen/Qwen3.8-27B>
- **Thireus** — the [GGUF Tool Suite](https://github.com/Thireus/GGUF-Tool-Suite) and the per-tensor
  sensitivity approach behind it, including the [quant_assign tooling](https://gguf3.thireus.com/) that
  inspired the per-tensor recipe direction here. The q6_0-7bpw Qwen3.6-27B recipe this cook inherits and
  size-solves was produced with his tooling, and a reference quant built the same way validated our size
  model to 0.04%. The `blk.64` warning above is a caveat about *published data coverage* for one model,
  not a criticism of the tooling — his per-tensor sensitivity work is what made this recipe possible at
  all. <https://huggingface.co/Thireus>
- **Unsloth** — the `mmproj-BF16.gguf` this card points at, the published Qwen3.8 sampling
  recommendations, their day-1 GGUFs benchmarked in the table above, the public `imatrix_unsloth.gguf`
  used to build the imatrix GGUF here (August 21 update), and — as of the September 2 update — the
  UD-Q4_K_XL / UD-Q5_K_XL / UD-Q6_K builds now doing the day job on the rig this cook was solved for.
  <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>
- **froggeric** — Qwen-Fixed-Chat-Templates v22, the base of the bundled template.
  <https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates>
- **AtomicChat** — their AD-Q6_K, the honest same-size comparison point in the fidelity table.
  <https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF>
- **llama.cpp** — conversion, quantization and serving; and the CUDA flash-attention dispatch code in
  `ggml/src/ggml-cuda/fattn.cu`, which made the September 2 correction diagnosable from source rather
  than guessable from benchmarks. <https://github.com/ggml-org/llama.cpp>

Both GGUFs in this repo were quantized with the same llama.cpp build — build 10431 (commit `1692f9e50`) —
so the RTN-vs-imatrix comparison above is a clean single-variable experiment. Quantized and measured on a
single RTX 5090 (Windows 11, CUDA 13.3) in a 24-core Threadripper PRO workstation with 255 GB RAM.

---

*Card revisions: **2026-08-14** original · **2026-08-21** imatrix build added · **2026-09-02** KV-cache ×
speculative-decoding correction, "what we run today" disclosure, and the `blk.64` MTP-head warning.
Earlier revisions are in this repo's commit history.*
