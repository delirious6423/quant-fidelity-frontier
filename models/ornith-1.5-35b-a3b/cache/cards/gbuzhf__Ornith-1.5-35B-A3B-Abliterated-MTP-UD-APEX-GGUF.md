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
- 9-Tier-Standard
- Tiel-Calibrated
- mtp
- speculative-decoding
- apex
- v2d-lite
- unsloth-dynamic
- moe
- vlm
- qwen35moe
- imatrix
- abliterated
- uncensored
---

# Ornith-1.5-35B-A3B Abliterated — MTP + UD + ICE + APEX GGUF

Nine Pareto-optimal tiers of the abliterated model, each with an **MTP head grafted
in** from the original Ornith-1.5 (the abliterated source ships none).

Measured on wikitext-2-raw, 16 chunks x 2048 ctx, against **two** references, so
quantization damage and abliteration damage can be told apart.

## How much did abliteration itself change the model?

| | |
|---|---:|
| **Mean KLD** (abliterated BF16 vs original BF16) | **0.0151** |
| same top-1 token | 95.04% |

For scale, the *best* quantization in this ladder costs ~0.022 KLD. **Abliteration is
a smaller perturbation than Q6_K quantization.**

## Tiers  (9/9 measured)

| tier | size | mean KLD | 99% KLD | 99.9% KLD | PPL ratio | same top-1 | active bpw | file bpw | overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `MTP-UD-Q6_K` | 30.21 GB | **0.0222** | 0.226 | 0.695 | 0.9960 | 94.05% | 8.063 | 6.804 | **96.7** |
| `MTP-UD-Q5_K_S` | 25.84 GB | **0.0261** | 0.269 | 0.899 | 0.9871 | 93.51% | 7.693 | 5.820 | **96.3** |
| `TIEL_Calibrated-25G-ICE` | 24.85 GB | **0.0276** | 0.269 | 1.061 | 0.9847 | 93.33% | 7.686 | 5.597 | **96.1** |
| `MTP-25G-ICE` | 24.85 GB | ~~0.0293~~ **0.0281** | ~~0.300~~ 0.279 | ~~1.280~~ 1.097 | ~~0.9856~~ 0.9910 | ~~93.44%~~ 93.40% | 7.686 | 5.597 | ~~96.0~~ **96.1** |
| `TIEL_Calibrated-23G-ICE` | 22.84 GB | **0.0321** | 0.326 | 1.173 | 0.9907 | 92.89% | 7.523 | 5.143 | **95.7** |
| `MTP-23G-ICE` | 22.84 GB | ~~0.0345~~ **0.0329** | ~~0.345~~ 0.339 | ~~1.151~~ 1.240 | ~~0.9902~~ 0.9941 | ~~92.39%~~ 92.83% | 7.523 | 5.143 | ~~95.4~~ **95.6** |
| `TIEL_Calibrated-21G-ICE` | 20.85 GB | **0.0392** | 0.421 | 1.606 | 0.9997 | 92.27% | 7.357 | 4.695 | **95.0** |
| `MTP-21G-ICE` | 20.85 GB | ~~0.0398~~ **0.0395** | ~~0.400~~ 0.412 | ~~1.655~~ 1.726 | ~~0.9943~~ 0.9929 | ~~92.06%~~ 92.12% | 7.357 | 4.695 | ~~94.9~~ **95.0** |
| `TIEL_Calibrated-19G-ICE` | 18.82 GB | **0.0576** | 0.602 | 2.123 | 1.0015 | 90.42% | 7.192 | 4.240 | **93.3** |
| `MTP-19G-ICE` | 18.82 GB | ~~0.0612~~ **0.0585** | ~~0.630~~ 0.598 | ~~2.327~~ 1.955 | ~~1.0025~~ 0.9995 | ~~90.02%~~ 90.21% | 7.192 | 4.240 | ~~93.0~~ **93.2** |
| `MTP-UD-IQ4_XS` | 18.68 GB | **0.0706** | 0.695 | 2.645 | 1.0502 | 89.40% | 6.762 | 4.209 | **92.2** |
| `MTP-APEX-I-Compact-v2D-lite` | 17.57 GB | **0.0925** | 0.893 | 3.141 | 1.0169 | 87.97% | 5.228 | 3.956 | **90.5** |
| `MTP-APEX-I-Mini-v2D-lite` | 14.37 GB | **0.2546** | 2.410 | 5.578 | 1.2129 | 80.70% | 4.180 | 3.208 | **80.0** |

> **The four `ICE` rows were rebuilt on 2026-08-24** and the struck-through values
> are what they replaced. The new figures are measured at **64 chunks**, the rest of
> the table at 16; a higher chunk count tightens the error bar without moving the
> mean. What changed, why, and the full measurement set:
> [`MEASUREMENTS-ICE-rebuild.md`](MEASUREMENTS-ICE-rebuild.md).

> **The four `TIEL_Calibrated` rows were added on 2026-08-30.** Identical builds to
> their `MTP-*-ICE` counterparts — same abliterated trunk, same norm-fixed MTPv2
> head, same ICE-base recipes, same byte budgets — but built with the importance
> matrix and embedded chat template from
> [peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF),
> with credit to that repo for both. Its imatrix is 3000 x 512 = 1,536,000 tokens
> against the 573 x 512 = 293,376 behind the others. Also measured at 64 chunks.

Sorted best -> worst by **overall** (BF16 = 100), the same composite used on the
[non-abliterated card](https://huggingface.co/gbuzhf/Ornith-1.5-35B-MTP-UD-APEX-GGUF):
`0.70/(1+meanKLD) + 0.30*sameTop1`. All KLD columns are measured
against the **abliterated** BF16, i.e. they isolate what the quantization costs.

> **Read the tail columns with care.** `99.9% KLD` is the ~33rd-worst token out of
> 32,768 — an extreme order statistic with large sampling variance, so it inverts
> between adjacent tiers without that meaning anything. `99% KLD` rests on ~328 tokens
> and orders all nine tiers monotonically; `mean KLD` uses all 32,768 and separates the
> closest pair by 4.3 sigma. **Rank on mean KLD**; treat the tail columns as shape,
> not order.

### Abliterated vs non-abliterated, same recipe

Both columns are the original 16-chunk measurements, so the delta isolates
abliteration. The four `ICE` tiers have since been rebuilt; their current numbers
are in the tier table above and in
[`MEASUREMENTS-ICE-rebuild.md`](MEASUREMENTS-ICE-rebuild.md).

| tier | KLD<br>abl | KLD<br>clean | Δ | top-1<br>abl | top-1<br>clean | Δ | PPL ratio<br>abl | PPL ratio<br>clean | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `MTP-UD-Q6_K` | **0.0222** | 0.0221 | +0.6% | 94.05% | 93.85% | +0.20 pp | 0.9960 | 0.9957 | +0.0003 |
| `MTP-UD-Q5_K_S` | **0.0261** | 0.0272 | -4.2% | 93.51% | 93.51% | +0.01 pp | 0.9871 | 0.9862 | +0.0009 |
| `MTP-25G-ICE` | **0.0293** | 0.0303 | -3.4% | 93.44% | 93.16% | +0.28 pp | 0.9856 | 0.9814 | +0.0042 |
| `MTP-23G-ICE` | **0.0345** | 0.0361 | -4.4% | 92.39% | 92.65% | -0.26 pp | 0.9902 | 0.9885 | +0.0018 |
| `MTP-21G-ICE` | **0.0398** | 0.0412 | -3.4% | 92.06% | 92.03% | +0.03 pp | 0.9943 | 0.9924 | +0.0019 |
| `MTP-19G-ICE` | **0.0612** | 0.0608 | +0.5% | 90.02% | 90.32% | -0.30 pp | 1.0025 | 1.0030 | -0.0005 |
| `MTP-UD-IQ4_XS` | **0.0706** | 0.0723 | -2.4% | 89.40% | 89.46% | -0.06 pp | 1.0502 | 1.0526 | -0.0024 |
| `MTP-APEX-I-Compact-v2D-lite` | **0.0925** | 0.0954 | -3.0% | 87.97% | 87.83% | +0.14 pp | 1.0169 | 1.0101 | +0.0068 |
| `MTP-APEX-I-Mini-v2D-lite` | **0.2546** | 0.2608 | -2.4% | 80.70% | 80.49% | +0.21 pp | 1.2129 | 1.2281 | -0.0151 |

Δ is near zero or slightly negative across the ladder: **abliteration does not make
this model harder to quantize**, and in the mid-range it is marginally easier —
plausibly because projecting a direction out of `ffn_down` narrows its dynamic range.

Two further results, measured against the *original* BF16 as well (full numbers in
`KLD_RESULTS.txt`):

- The two damages are strongly **sub-additive** — abliteration and quantization are
  largely orthogonal, so the combined figure sits far below their sum.
- **No tier un-abliterates.** Across the 9 rungs measured, the distance to the original BF16 stays above abliteration's own distance (0.0151), so quantization never pulls the model back toward the refusal behaviour.

### Which tier is which

| family | what it is |
|---|---|
| **UD-*** | Unsloth Dynamic 2.0 maps, replayed 1:1. Pins attention, the shared expert and `token_embd` at Q8_0 at every size and moves only the routed experts. |
| **ICE-*** | Bits allocated by how far a quantization error *travels*, not by activation magnitude. Named by target size. |
| **APEX-I-*-v2D-lite** | mudler's APEX maps plus one extra step on `attn_k`/`attn_v` in the ten full-attention blocks and on the output head. |

Rule of thumb: **Q6_K / Q5_K_S** near-lossless, **25G/23G-ICE** the quality sweet
spot, **21G/19G-ICE** the best small tiers, **Compact/Mini** only if you are tight on
VRAM — Mini drops sharply.

## The ICE tier, and why this is a 9-tier release

**ICE** allocates bits by *error travel distance*: how far a quantization error
propagates before it reaches the output. Tensors writing straight into the residual
stream, the always-on dense path, and the router are protected; the routed expert
stack — 93% of the parameters but only 8-of-256 active per token — is left uniform.
Routers stay F32, and the draft block is un-pinned so its experts follow the tier.

On the non-abliterated ladder, measured on this model and this harness, ICE lands
ahead at matched size:

| comparison | result |
|---|---|
| 23G-ICE vs **UD-Q4_K_XL** (same size) | **-5.0%** KLD |
| 23G-ICE vs **APEX-I-Quality** | **-13.0%** KLD *and* 0.61 GB smaller |
| 25G-ICE vs **APEX-I-Balanced** | **-12.2%** KLD *and* 1.15 GB smaller |

`UD-Q4_K_XL`,`APEX-I-Quality-v2D-lite` and `APEX-I-Balanced-v2D-lite` are each already covered by
an ICE tier that is both smaller and closer to BF16, so rebuilding them would add size
without adding a quality point. Full derivation, the refuted `ffn_down` rule
and the measured convexity bound are in the
[original Ornith-1.5 card](https://huggingface.co/gbuzhf/Ornith-1.5-35B-MTP-UD-APEX-GGUF).

> ### The 9-Tier Standard
> From this release onward these nine recipes are the standard ladder:
> `UD-Q6_K` · `UD-Q5_K_S` · `25G-ICE` · `23G-ICE` · `21G-ICE` · `19G-ICE` ·
> `UD-IQ4_XS` · `APEX-I-Compact-v2D-lite` · `APEX-I-Mini-v2D-lite`
>
> They are the measured Pareto frontier of a 12-tier sweep on this architecture:
> every dropped tier is beaten on both size *and* KLD by one that ships. Reference
> measurements and methodology:
> [Ornith-1.5-35B-MTP-UD-APEX-GGUF](https://huggingface.co/gbuzhf/Ornith-1.5-35B-MTP-UD-APEX-GGUF).

## What was done to the source

`PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-GGUF` is faithful at the tensor level —
every tensor except `ffn_down` is byte-identical to ornith-ai's BF16. Abliteration is
confined to **`ffn_down` (both routed and shared experts), layers 15-39**, at
~1.6-1.9e-02 L1-relative. Routers, attention, `ffn_gate`, `ffn_up` and layers 0-14
are untouched.

Three things were repaired while grafting:

1. **MTP head restored** — 20 `blk.40.*` tensors from ornith-ai's BF16;
   `block_count` 40 -> 41, `nextn_predict_layers` added.
2. **`tokenizer.ggml.add_bos_token` restored to `False`** — the source omits the key
   entirely, so llama.cpp falls back to its own default and tokenises differently
   from the original. Left unfixed this also invalidates any KLD against the original.
3. **`tokenizer.chat_template` restored** — the source ships a 7536-byte copy with
   the multi-system-message merge block removed; the original is 7828 bytes.

## imatrix

bartowski's `Ornith-1.5-35B-A3B-imatrix.gguf`, reused unmodified. Justified by
measurement, not assumption: **the routers are byte-identical between the original
and the abliterated model**, so the same experts fire and the per-channel statistics
still apply.

## Recipes

All nine tensor maps were confirmed byte-exact against the corresponding shipped
non-abliterated tier before this build, so the two ladders are directly comparable:

- **UD** — replayed 1:1 from `unsloth/Ornith-1.0-35B-GGUF` (Unsloth Dynamic 2.0).
- **APEX v2D-lite** — mudler's Ornith-1.5 maps, with `attn_k`/`attn_v` on the ten
  full-attention blocks and the output head each raised one step.
- **ICE** — bits allocated by how far a quantization error travels rather than by
  activation magnitude; the expert stack is uniform and the draft block un-pinned.

## MTP / speculative decoding

Every tier carries the head, pinned Q8_0 (F16 `attn_k`/`attn_v` on the ICE tiers).
It is grafted from the original model, so **the draft head is not abliterated while
the trunk is** — worth knowing if you rely on the refusal behaviour under drafting:
the head only proposes, the abliterated trunk verifies, so accepted tokens are always
the trunk's.

**Measured draft acceptance:** **96.77%** (390/403) on `23G-ICE`, `--spec-type draft-mtp`, text-only.

| prompt set | acceptance |
|---|---:|
| `code-novel` | 98.45% |
| `structured` | 98.29% |
| `copy-edit` | 97.30% |
| `prose-novel` | 91.57% |

Raw run in `gate_spec_bench.json`. Acceptance depends on the prompt mix — compare only against numbers taken on the same harness.

```bash
llama-server -m Ornith-1.5-35B-A3B-Abliterated-MTPv2-21G-ICE.gguf \
  --mmproj mmproj-Ornith-1.5-35B-A3B-Abliterated-F16.gguf \
  -c 8192 -fa on --jinja \
  --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-n-min 0 --spec-draft-p-min 0.75
```

## MTPv2 — updated draft head (2026-08-23)

`ornith-ai` uploaded a newly trained native MTP head to the official
[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) repo, so this
repo is updated with it.

The nine `MTPv2-*` files are the **same nine recipes** as the previous ladder — same
abliterated trunk, same imatrix, same per-tensor rule files, same pinned llama.cpp build.
Only `blk.40` is re-grafted. Every tensor outside `blk.40`, and the whole metadata block,
is byte-identical to its `MTPv1/` counterpart — checked tensor by tensor before anything
was uploaded.

The four `ICE` tiers were later rebuilt again, on 2026-08-24, with a revised expert
placement and a corrected `blk.40` norm conversion. Their rows in the tier table show
both the old and the new numbers.

**What actually changed.** The head in the original checkpoint had initializer-like
weights; the replacement is trained. Read straight off the two published files:

| `blk.40` tensor | MTPv1 | MTPv2 |
|---|---:|---:|
| `nextn.shared_head_norm.weight` | 1.02281 ± 0.00017 | 2.92531 ± 0.29643 |
| `attn_norm.weight` | 1.00006 ± 0.00302 | 0.90492 ± 0.15616 |
| `nextn.hnorm.weight` | 1.01571 ± 0.00558 | 0.49370 ± 0.09479 |
| `attn_q_norm.weight` | 1.00390 ± 0.00680 | 1.76704 ± 0.33864 |

The previous ladder is unchanged and still here, under **`MTPv1/`** — same files, same
digests, nothing deleted.

### Standalone draft head

The new head is also published on its own, for use as a `--model-draft` sidecar instead
of an embedded head:

| file | size |
|---|---:|
| `mtp-Ornith-1.5-35B-A3B-Abliterated-MTPv2-BF16.gguf` | 3.74 GB |
| `mtp-Ornith-1.5-35B-A3B-Abliterated-MTPv2-Q8_0.gguf` | 1.99 GB |
| `mtp-Ornith-1.5-35B-A3B-Abliterated-MTPv2-Q4_K_M.gguf` | 1.26 GB |

The BF16 one is a straight extraction from the same grafted master the ladder was
quantized from, so its head is bit-identical to the head inside the `MTPv2-*` files.

## Vision

Not re-hosted — use the projector from the source repo:
[`mmproj-Ornith-1.5-35B-A3B-Abliterated-F16.gguf`](https://huggingface.co/PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-GGUF/blob/main/mmproj-Ornith-1.5-35B-A3B-Abliterated-F16.gguf)
(0.90 GB). Without it the model is blind. Note `--mmproj` force-disables `ctx_shift`
and `cache_reuse`.

## Also included

`sha256sums-MTPv2.txt` and `MANIFEST-MTPv2.txt` for the current ladder,
`MEASUREMENTS-ICE-rebuild.md` for the 2026-08-24 ICE rebuild, and the MTPv1
originals under `MTPv1/` (`KLD_RESULTS.txt`, `gate_spec_bench.json`,
`sha256sums.txt`, `MANIFEST.txt`).

The BF16 masters are not re-hosted: the abliterated source is at
[PocketAiHub](https://huggingface.co/PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-GGUF)
and the original at [ornith-ai](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF).

Credit: **PocketAiHub** for the abliteration, **bartowski** for the imatrix and for
publishing its corpus, **mudler** for the APEX reference maps, **Unsloth** for the
UD 2.0 maps, **Shisa-AI** for Final 12K KL distill MTP Head (ShisaMTP variants), **Jzinno** for Dflash2 sidecar, **peculiar-ragdoll** for Tiel iMatrix Calibration and Sharp Chat Template.