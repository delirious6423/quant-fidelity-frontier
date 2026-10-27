---
license: mit
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
tags:
- gguf
- ICE
- ICE-Tiers
- ice-quant
- mtp
- speculative-decoding
- apex
- v2d-lite
- unsloth-dynamic
- moe
- vlm
- qwen35moe
- imatrix
---

# Ornith-1.5-35B-A3B — MTP + UD + APEX-v2D-lite GGUF

[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), twelve
imatrix tiers across three recipe families, **MTP head embedded in every one**.

Ornith 1.5 ships its MTP head natively (753 tensors, `nextn_predict_layers=1`,
20 `blk.40.*` tensors), so nothing here is grafted.

## Tiers

Measured against the bf16 master on wikitext-2-raw, 16 chunks × 2048 ctx
(32,768 tokens). Reference bf16 perplexity: **7.755**. All twelve tiers share one
harness, so the column is directly comparable across families.

| tier | size | mean KLD | 99% KLD | 99.9% KLD | PPL ratio | same top-1 | active bpw | file bpw | **overall** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `MTP-UD-Q6_K` | 30.20 GB | **0.0221** | 0.220 | 0.809 | 0.9957 | 93.85% | 8.063 | 6.804 | **96.6** |
| `MTP-UD-Q5_K_S` | 25.83 GB | **0.0272** | 0.269 | 1.030 | 0.9862 | 93.51% | 7.693 | 5.820 | **96.2** |
| `MTP-25G-ICE` | 24.84 GB | **0.0303** | 0.296 | 1.049 | 0.9814 | 93.16% | 7.686 | 5.597 | **95.9** |
| `MTP-APEX-I-Balanced-v2D-lite` | 26.28 GB | **0.0345** | 0.328 | 1.401 | 0.9907 | 92.55% | 6.913 | 5.922 | **95.4** |
| `MTP-23G-ICE` | 22.83 GB | **0.0361** | 0.332 | 1.272 | 0.9885 | 92.65% | 7.523 | 5.143 | **95.4** |
| `MTP-UD-Q4_K_XL` | 23.21 GB | **0.0380** | 0.389 | 1.405 | 0.9740 | 92.47% | 7.471 | 5.230 | **95.2** |
| `MTP-21G-ICE` | 20.84 GB | **0.0412** | 0.392 | 1.665 | 0.9924 | 92.03% | 7.357 | 4.695 | **94.8** |
| `MTP-APEX-I-Quality-v2D-lite` | 23.84 GB | **0.0415** | 0.424 | 1.386 | 0.9843 | 91.98% | 6.699 | 5.371 | **94.8** |
| `MTP-19G-ICE` | 18.82 GB | **0.0608** | 0.615 | 1.877 | 1.0030 | 90.32% | 7.192 | 4.240 | **93.1** |
| `MTP-UD-IQ4_XS` | 18.68 GB | **0.0723** | 0.758 | 2.464 | 1.0526 | 89.46% | 6.762 | 4.209 | **92.1** |
| `MTP-APEX-I-Compact-v2D-lite` | 17.56 GB | **0.0954** | 0.899 | 2.962 | 1.0101 | 87.83% | 5.228 | 3.956 | **90.3** |
| `MTP-APEX-I-Mini-v2D-lite` | 14.24 GB | **0.2608** | 2.489 | 5.915 | 1.2281 | 80.49% | 4.180 | 3.208 | **79.7** |

> **Read the tail columns with care.** `99.9% KLD` is the ~33rd-worst token out of
> 32,768 — an extreme order statistic with large sampling variance. It inverts between
> adjacent tiers twice here, in **opposite directions** — once favouring an ICE tier,
> once favouring an APEX tier — which is what directionless noise looks like. `99% KLD`
> rests on ~328 tokens and has **zero** inversions across all 12 tiers; `mean KLD` uses
> all 32,768. **Rank on mean KLD**; treat the tail columns as shape, not order.


Sorted best → worst by **overall**, where **bf16 = 100**:

```
overall = 100 × [ 0.70 × 1/(1+meanKLD)   full output distribution
                + 0.30 × sameTop1/100 ]   argmax agreement
```

PPL ratio is shown but deliberately **excluded** from the score: at 32k tokens it
lands under 1.0 for several tiers, which is sample noise, and a term that rewards
"better than bf16" would corrupt the ranking.

**mean KLD** — divergence from bf16 across the full output distribution; lower is
better. **99% / 99.9% KLD** — the tail, i.e. how bad the worst tokens get.
**same top-1** — how often the tier picks the same next token as bf16.

Three things worth reading off the table:

- **File size does not rank these tiers.** `25G-ICE` (24.84 GB) sits 3rd, above
  `APEX-I-Balanced` at 26.28 GB; `23G-ICE` (22.83 GB) sits above `UD-Q4_K_XL` at
  23.21 GB. Rank on the KLD column, not the size column.
- **`23G-ICE` covers two other tiers** — smaller *and* closer to bf16 than
  both `UD-Q4_K_XL` (23.21 GB / 0.0380) and `APEX-I-Quality` (23.84 GB / 0.0415).
- **`APEX-I-Mini` is where the curve turns down.** 81.0 overall, 80.5% top-1, 1.23
  PPL ratio: its expert tensors sit at IQ2_S. Everything from `Compact` up stays
  within ~6 points of bf16.

### active bpw vs file bpw

**active bpw** weights routed experts at 8/256, because only 8 of 256 fire per
token, and excludes `token_embd` (an embedding lookup, not a matmul). It is the
precision the model actually computes with. **file bpw** is just size ÷ params.

On this model, 65.8% of active parameters are non-routed:

| role group | params | % of file | % of active |
|---|---:|---:|---:|
| routed experts | 32.21 B | 92.94% | 34.2% |
| attention | 1.03 B | 2.96% | 34.9% |
| output head | 0.51 B | 1.47% | 17.3% |
| ssm | 0.26 B | 0.74% | 8.7% |
| shared expert | 0.13 B | 0.36% | 4.3% |

Which is why file size misleads here: `UD-Q4_K_XL` is **0.63 GB smaller** than
`APEX-I-Quality-v2D-lite` while computing at 7.471 vs 6.699 active bpw. The UD
tiers pin attention, the shared expert and `token_embd` at Q8_0 at every size.

**But do not rank on active bpw either.** Tested against the measured KLD column,
it misorders **4 of the 8 UD/APEX tiers** (Spearman +0.95), and on a second published
ladder measured on one harness it misorders **10 of 13** (+0.71). It is a *linear*
average of bit-widths while quantization error is not linear in bits: pushing
attention from Q6_K to Q8_0 buys very little real accuracy but moves the metric a
lot, so Q8_0-dense-path designs come out flattered and uniform stock mixes come out
underrated. `APEX-I-Balanced` is the local example — 6.913 active bpw, 3rd on
measured KLD.

Active bpw is useful for explaining *where the bits went*. Mean KLD is what ranks
the files.

## Map sources

- **UD tiers** — replayed 1:1 from
  [`unsloth/Ornith-1.0-35B-GGUF`](https://huggingface.co/unsloth/Ornith-1.0-35B-GGUF)
  (Unsloth Dynamic 2.0). Trunk geometry is identical between Ornith 1.0 and 1.5 —
  40 blocks, 256 experts, 8 active, expert FFN 512, shared FFN 512 — so the map
  transfers by index with no rescaling.
- **APEX tiers** — mudler's published Ornith 1.5 maps, plus the v2D-lite step below.

**v2D-lite** raises precision in two places relative to the base APEX map:

- the key and value projections on the ten full-attention blocks (3, 7, 11 … 39) —
  20 tensors, +5.1 MB on Quality/Balanced, 5.4 MB on Compact, 7.9 MB on Mini;
- `output.weight` by one step, to `Q8_0` — +123 MB, on Compact, Quality and
  Balanced. Mini's output head was already at the target and is unchanged.

Nothing is lowered anywhere, so every APEX tier is strictly at or above the map it
came from. The MTP block is left alone; it is pinned Q8_0 separately.

Every UD and APEX file was verified tensor-by-tensor against its reference map:
**0 deviations across all eight**, with the four UD tiers exact 1:1 replays.

### Which tier to pick

Nine of the twelve are **Pareto-optimal** — nothing else is both smaller *and*
closer to bf16. Pick by VRAM and stop:

| size | mean KLD | tier |
|---:|---:|---|
| 14.24 GB | 0.2608 | `MTP-APEX-I-Mini-v2D-lite` |
| 17.56 GB | 0.0954 | `MTP-APEX-I-Compact-v2D-lite` |
| 18.68 GB | 0.0723 | `MTP-UD-IQ4_XS` |
| 18.82 GB | 0.0608 | `MTP-19G-ICE` |
| 20.84 GB | 0.0412 | `MTP-21G-ICE` |
| 22.83 GB | 0.0361 | `MTP-23G-ICE` |
| 24.84 GB | 0.0303 | `MTP-25G-ICE` |
| 25.83 GB | 0.0272 | `MTP-UD-Q5_K_S` |
| 30.20 GB | 0.0221 | `MTP-UD-Q6_K` |

Three tiers are **already covered** — something smaller is also closer to bf16, so
there is no size budget at which they are the right pick:

| covered tier | covered by |
|---|---|
| `MTP-UD-Q4_K_XL` (23.21 GB / 0.0380) | `MTP-23G-ICE` — 0.38 GB smaller, 5.0% better |
| `MTP-APEX-I-Quality-v2D-lite` (23.84 GB / 0.0415) | `MTP-23G-ICE` — 1.01 GB smaller, 13.0% better |
| `MTP-APEX-I-Balanced-v2D-lite` (26.28 GB / 0.0345) | `MTP-UD-Q5_K_S` — 0.45 GB smaller, 21.1% better |

They are kept for reproducibility and for anyone comparing recipe families.

**Read that table with its context.** ICE was designed *after* these measurements
existed, tuned against this model's own routing histogram. mudler's APEX maps and
Unsloth's UD maps are general-purpose recipes published without access to them — and
they are what ICE was built on top of. Any of these numbers would likely look different
on another model, another corpus or another harness. Four of the twelve tiers here are
mudler's maps and four are Unsloth's; ICE is a modification of that groundwork, not a
replacement for it.

Returns diminish monotonically up the frontier — roughly **−30%** from 18.8 to
20.8 GB, **−19%** from 20.8 to 22.8, **−11%** from 22.8 to 24.8. There is no knee
and no natural stopping point; take the largest that fits comfortably alongside
your context.

## ICE tiers

`ICE` allocates bits by **how far a quantization error travels**, rather than by
activation magnitude alone.

Most of a tensor's error dies with the token that produced it. Three kinds do not:

- **routers** — an error flips a top-8 argmax and a *different* expert runs. Not a
  graded failure.
- **state gates** (`ssm_alpha`, `ssm_beta`) — an error enters a decay and compounds
  along the sequence.
- **`attn_k` / `attn_v`** — an error is written into the KV cache once and re-read
  by every later token.

Those three groups total **47 M parameters, 0.14% of the model**. ICE keeps all of
them exact — routers and state gates at F32, K/V at F16 — which costs **0.15 GB**.

Two further differences from the UD and APEX tiers:

- **`blk.40`'s experts follow the tier instead of being pinned Q8_0.** The MTP block
  is a draft model; its outputs are verified by the target model, so its errors cost
  speed, not correctness. Its projections stay Q8_0. Frees ~0.4 GB per tier.
- **`ffn_down`, `ffn_gate` and `ffn_up` all get the same bit-width.** Bumping
  `ffn_down` is standard practice — llama.cpp's own mixes do it — but on this model
  it is measurably wrong: a uniform control at identical size scored **0.041192
  against 0.046449, an 11.3% improvement**. The ICE tiers ship uniform.

ICE applies **no per-layer depth grading**. Measured on this model's routing
histogram, optimal non-uniform allocation beats uniform by only **+0.139 bpw**,
because quantization error is convex in bit-width — so the ladder is kept flat and
the budget spent elsewhere.

**Measured draft acceptance with the shrunk MTP block: 96.04%** (388/404), so the
smaller draft head still drafts — the size saving is not paid for out of an
unmeasured budget.

**Result** against the UD size/quality curve at equal size: **−13.6%** at 18.82 GB,
**−18.6%** at 20.84 GB, −6.3% at 22.83 GB, −1.9% at 24.84 GB. The advantage grows as
size falls, because at the top of the ladder the calibration floor (~0.0185)
dominates and no allocation choice can move it.

## Lineage — Ornith 1.5 is not Ornith 1.0 continued

Settled by weights with a control, not from any model card:

| tensor | 1.5 vs **1.0** | 1.5 vs **Qwen3.6-35B-A3B** |
|---|---:|---:|
| `norm.weight` | 7.214e-03 | 7.214e-03 |
| `layers.3.self_attn.k_proj` | 3.642e-02 | 3.332e-02 |
| `layers.19.self_attn.v_proj` | 5.857e-02 | 5.720e-02 |
| `layers.39.self_attn.k_proj` | 7.773e-02 | 7.683e-02 |

**Ornith 1.5 is no closer to Ornith 1.0 than to Ornith 1.0's own base** — marginally
further, at every depth. Two independent post-trains from the same region, not a
continuation. Its router moved even more: `mlp.gate.weight` drifted 2.6e-02 at layer
0 rising to **9.7e-02 by layer 39**.

That router drift is why Ornith 1.0's imatrix was not reused here: on a 256-expert
MoE an imatrix is largely a statement about *which experts fire*, and it lands on
`ffn_*_exps` — 93% of the parameters.

## imatrix

`Ornith-1.5-35B-A3B-imatrix.gguf` — **bartowski's**, mirrored unmodified with thanks,
computed on Ornith 1.5's own weights.

| | |
|---|---|
| corpus | `Ornith-1.5-35B-A3B-calibration-v6.txt` (also mirrored) |
| chunks | 573 × 512 = **293,376 tokens** |
| entries | 510 tensors, blocks 0–39 |

calibration-v6 is chat-templated rather than raw text — 782 `<|im_start|>`, 332
`<think>`, 456 `<tool_call>`, 188 code fences, and CJK / Cyrillic / Arabic at
0.55 / 0.33 / 0.21%.

`blk.40` has zero imatrix coverage — llama.cpp never executes the nextn block during
a forward pass, so no imatrix can reach it. The Q8_0 pin covers it.

## MTP / speculative decoding

The head is pinned **Q8_0 in every tier** (~0.90 GB).

**Measured draft acceptance: 92.97%** (397/427), on `APEX-I-Balanced-v2D-lite`,
`--spec-type draft-mtp`, `--spec-draft-n-max 1`, text-only (no `--mmproj`):

| prompt set | accepted |
|---|---:|
| `structured` | 114/117 |
| `code-novel` | 115/123 |
| `copy-edit` | 101/106 |
| `prose-novel` | 67/81 |

Raw run in `gate_spec_bench.json`. Acceptance depends on the prompt mix, so only
compare against numbers taken on the same harness — for reference, the grafted head
on [Ornith 1.0](https://huggingface.co/gbuzhf/Ornith-1.0-35B-MTP-UD-APEX-GGUF) scored
87.4% on this same bench. The t/s figures in that JSON are CPU-only build numbers
and say nothing about your GPU.

Two other Ornith 1.5 GGUF sets handle this block differently, if you are comparing:
`AtomicChat/Ornith-1.5-35B-A3B-GGUF` omits it entirely (733 tensors,
`block_count=40`), and `bartowski/Ornith-1.5-35B-A3B-GGUF` leaves the whole block at
Q4_0.

## Vision

Ornith 1.5 is a VLM. The projector is not re-hosted — use ornith-ai's
[`mmproj-Ornith-1.5-35B-BF16.gguf`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/blob/main/mmproj-Ornith-1.5-35B-BF16.gguf).
Without it the model is blind. Note `--mmproj` force-disables `ctx_shift` and
`cache_reuse`.

## Running it

```bash
llama-server -m Ornith-1.5-35B-MTP-UD-Q5_K_S.gguf \
  --mmproj mmproj-Ornith-1.5-35B-BF16.gguf \
  -c 8192 -fa on --jinja \
  --spec-type draft-mtp,ngram-mod \
  --spec-draft-n-max 1 --spec-draft-n-min 0 --spec-draft-p-min 0.75 \
  --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 24 --spec-ngram-mod-n-match 48
```

## Also included

- `Ornith-1.5-35B-A3B-imatrix.gguf` and `Ornith-1.5-35B-A3B-calibration-v6.txt` —
  bartowski's, mirrored with attribution.
- `KLD_RESULTS.txt` — the raw `llama-perplexity --kl-divergence` output per tier.
- `sha256sums.txt`, `MANIFEST.txt`.

The bf16 master is not re-hosted; ornith-ai already publishes
[`Ornith-1.5-35B-BF16.gguf`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-BF16.gguf)
(71.07 GB, MTP included).

Credit: **bartowski** for the imatrix and for publishing its corpus; **mudler** for
the APEX reference maps; **Unsloth** for the UD 2.0 maps.
