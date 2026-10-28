---
license: mit
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
language:
- en
- zh
tags:
- gguf
- ICE
- ice-quant
- imatrix
- sangreal
- mtp
- speculative-decoding
- moe
- qwen35moe
- abliterated
- uncensored
---

# Ornith-1.5-35B-A3B — Huihui × Sangreal ICE

Five ICE tiers of **huihui-ai's abliterated Ornith-1.5-35B-A3B**, quantized against
**Sangreal** — a purpose-built 77-bucket calibration corpus — and shipping
**peculiar-ragdoll's Qwen-Sharp chat template**. The abliterated checkpoint already
carries Ornith-1.5's trained **MTPv2** head, so every tier drafts for speculative
decoding out of the box.

**This release introduces a new ICE rung: `15G-ICE`**, the first tier below the
17 GB bound the method was originally scoped to, built for 16 GB cards. The BF16
master all five were cut from is published
[here](https://huggingface.co/gbuzhf/Ornith-1.5-35B-A3B-Abliterated-CyberTiel-Calibrated-MTPv2-ICE-GGUF).

## Best got better — ICE 1.6 (2026-09-25)

Rebuilt tiers vs the files they replace — mean KLD to BF16, paired on the same chunks, one session.

| tier | code KLD (2K) | code @16K | paired code t | |
|---|---|---:|---:|---|
| **15G** | 0.088158 → **0.083796** (−4.9%) | −4.4% | −6.02 | **clear win - replaced** |
| **19G** | 0.036887 → **0.034761** (−5.8%) | −6.7% | −3.07 | **clear win - replaced** |
| **23G** | 0.018988 → **0.018473** (−2.7%) | −4.5% | −1.49 | **replaced** · lower KLD at 2K and 16K; code top-1 −0.06 pp |
| **25G** | 0.015308 → **0.014603** (−4.6%) | −5.2% | −2.61 | **clear win - replaced** |

![Mean KLD vs file size, code at 16K context: Sangreal ICE against the Unsloth Dynamic ladder](assets/code16k_kld_vs_size.svg)

Code at 16K context (`code.test.raw`, n_ctx 16384 × 8), mean KLD to BF16 vs file size; lower-left is better.
Raw numbers in Table 2 below and in [`measurements/2026-09-25/`](measurements/2026-09-25/).

Changed: `output.weight` Q8_0 → Q6_K, MTP head at the floor, freed bytes → routed experts, 36,025-chunk Sangreal imatrix, template v22.5.1.

> ⚠️ **Uncensored.** This is an abliterated checkpoint — refusal behaviour has been
> suppressed in the weights. Sandbox it at the OS level and control its network and
> code-execution access; with no refusal backstop, a prompt injection from a hostile
> page or third-party code has nothing to stop it.

## Measurements

One binary, one reference, one session, 64 chunks at n_ctx 2048. Every file below —
including our previous CyberTiel ladder and the Official CyberTiel's UD ladder — was
re-measured in this same session; cross-session KLD drifts ~0.8% on identical inputs,
which is the same order as the effect being measured.

**2026-09-25:** a second session (same GPU class, binary, reference, corpora) re-measured every shipped
Sangreal tier and every UD row: all 24 values reproduced **exactly**. Replaced rows show ~~then~~ now.
Raw logs: `measurements/2026-09-25/`.

### Table 1 — Code (`code.test.raw`)

PPL(base) **2.194208** · BF16 = 100

| file | size | mean KLD | 99% KLD | 99.9% KLD | PPL ratio | same top-1 | active bpw | file bpw | overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **≈ 25–27 GB** | | | | | | | | | |
| UD-Q5_K_XL | 26.98 GB | 0.011993 | 0.1702 | 0.8170 | 1.0009 | 97.25 % | 7.833 | 6.080 | **98.3** |
| **(New) Sangreal 25G-ICE** | ~~24.85 GB~~<br>24.82 GB | ~~0.015308~~<br>0.014603 | ~~0.2274~~<br>0.2129 | ~~1.0218~~<br>0.9268 | ~~1.0017~~<br>1.0014 | ~~97.00 %~~<br>97.03 % | ~~7.686~~<br>7.385 | ~~5.599~~<br>5.593 | ~~98.0~~<br>**98.1** |
| CyberTiel 25G-ICE | 24.85 GB | 0.015497 | 0.2207 | 1.0820 | 1.0007 | 97.06 % | 7.686 | 5.599 | **98.0** |
| **≈ 22.5–23 GB** | | | | | | | | | |
| **(New) Sangreal 23G-ICE** | ~~22.84 GB~~<br>22.81 GB | ~~0.018988~~<br>0.018473 | ~~0.2831~~<br>0.2767 | ~~1.2650~~<br>1.2717 | ~~1.0023~~<br>1.0026 | ~~96.73 %~~<br>96.67 % | ~~7.523~~<br>7.215 | ~~5.145~~<br>5.139 | **97.7** |
| CyberTiel 23G-ICE | 22.84 GB | 0.019269 | 0.2822 | 1.3248 | 1.0018 | 96.66 % | 7.523 | 5.145 | **97.7** |
| UD-Q4_K_XL | 22.75 GB | 0.019742 | 0.2909 | 1.3855 | 1.0041 | 96.61 % | 7.474 | 5.126 | **97.6** |
| UD-Q4_K_M | 22.52 GB | 0.020440 | 0.2880 | 1.3215 | 1.0040 | 96.59 % | 7.130 | 5.075 | **97.6** |
| **≈ 21 GB** | | | | | | | | | |
| **Sangreal 21G-ICE** | 20.85 GB | 0.023415 | 0.3529 | 1.4214 | 1.0042 | 96.35 % | 7.357 | 4.698 | **97.3** |
| UD-Q4_K_S | 21.28 GB | 0.023484 | 0.3466 | 1.6264 | 1.0028 | 96.31 % | 7.025 | 4.795 | **97.3** |
| CyberTiel 21G-ICE | 20.85 GB | 0.024669 | 0.3787 | 1.6756 | 1.0044 | 96.28 % | 7.357 | 4.698 | **97.2** |
| **≈ 18 GB** | | | | | | | | | |
| **(New) Sangreal 19G-ICE** | ~~18.82 GB~~<br>18.76 GB | ~~0.036887~~<br>0.034761 | ~~0.5614~~<br>0.5241 | ~~2.4828~~<br>2.1468 | ~~1.0159~~<br>1.0145 | ~~95.33 %~~<br>95.35 % | ~~7.192~~<br>6.871 | ~~4.241~~<br>4.228 | ~~96.1~~<br>**96.3** |
| CyberTiel 19G-ICE | 18.82 GB | 0.036910 | 0.5798 | 2.3285 | 1.0154 | 95.24 % | 7.192 | 4.241 | **96.1** |
| UD-IQ4_XS | 18.12 GB | 0.044161 | 0.6689 | 2.8208 | 1.0191 | 94.80 % | 6.757 | 4.083 | **95.5** |
| **≈ 13.5–15 GB** | | | | | | | | | |
| **(New) Sangreal 15G-ICE** | ~~14.83 GB~~<br>14.81 GB | ~~0.088158~~<br>0.083796 | ~~1.4526~~<br>1.3755 | ~~4.7767~~<br>4.6790 | ~~1.0467~~<br>1.0427 | ~~92.91 %~~<br>92.93 % | ~~6.853~~<br>6.536 | ~~3.341~~<br>3.336 | ~~92.2~~<br>**92.5** |
| UD-IQ3_XXS | 13.60 GB | 0.105845 | 1.7903 | 5.2413 | 1.0594 | 91.96 % | 5.489 | 3.064 | **90.9** |

### Table 2 — Code at long context (`code.test.raw`, n_ctx 16384 × 8)

Scored on positions 8,193–16,384 of each chunk (65,536 tokens, as in the 2K tables). Compare rows within
this table only. `(New)` rows: shipped file ~~then~~, ICE 1.6 file now.

PPL(base) **2.121504** · BF16 = 100

| file | size | mean KLD | 99% KLD | 99.9% KLD | PPL ratio | same top-1 | active bpw | file bpw | overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **≈ 25–27 GB** | | | | | | | | | |
| UD-Q5_K_XL | 26.98 GB | 0.037584 | 0.5148 | 7.1463 | 1.0016 | 96.86 % | 7.833 | 6.080 | **96.5** |
| **(New) Sangreal 25G-ICE** | ~~24.85 GB~~<br>24.82 GB | ~~0.041354~~<br>0.039198 | ~~0.5975~~<br>0.5428 | ~~7.2349~~<br>6.8581 | ~~0.9993~~<br>0.9992 | ~~96.60 %~~<br>96.68 % | ~~7.686~~<br>7.385 | ~~5.599~~<br>5.593 | ~~96.2~~<br>**96.4** |
| **≈ 22.5–23 GB** | | | | | | | | | |
| **(New) Sangreal 23G-ICE** | ~~22.84 GB~~<br>22.81 GB | ~~0.045097~~<br>0.043053 | ~~0.7004~~<br>0.6588 | ~~7.2783~~<br>7.0492 | ~~0.9996~~<br>0.9995 | ~~96.39 %~~<br>96.41 % | ~~7.523~~<br>7.215 | ~~5.145~~<br>5.139 | ~~95.9~~<br>**96.0** |
| UD-Q4_K_XL | 22.75 GB | 0.045938 | 0.7000 | 7.5137 | 0.9971 | 96.29 % | 7.474 | 5.126 | **95.8** |
| UD-Q4_K_M | 22.52 GB | 0.046128 | 0.6952 | 7.2629 | 0.9989 | 96.21 % | 7.130 | 5.075 | **95.8** |
| **≈ 21 GB** | | | | | | | | | |
| UD-Q4_K_S | 21.28 GB | 0.051208 | 0.8028 | 8.1942 | 0.9942 | 95.97 % | 7.025 | 4.795 | **95.4** |
| **Sangreal 21G-ICE** | 20.85 GB | 0.051563 | 0.8315 | 7.3234 | 0.9949 | 96.01 % | 7.357 | 4.698 | **95.4** |
| **≈ 18 GB** | | | | | | | | | |
| **(New) Sangreal 19G-ICE** | ~~18.82 GB~~<br>18.76 GB | ~~0.063182~~<br>0.058960 | ~~1.0950~~<br>0.9983 | ~~8.1007~~<br>6.9570 | ~~1.0189~~<br>1.0160 | ~~95.24 %~~<br>95.28 % | ~~7.192~~<br>6.871 | ~~4.241~~<br>4.228 | ~~94.4~~<br>**94.7** |
| UD-IQ4_XS | 18.12 GB | 0.077803 | 1.4135 | 8.5959 | 1.0284 | 94.66 % | 6.757 | 4.083 | **93.3** |
| UD-Q3_K_XL | 17.23 GB | 0.084033 | 1.5944 | 8.1866 | 1.0209 | 94.21 % | — | 3.882 | **92.8** |
| **≈ 13.5–15 GB** | | | | | | | | | |
| **(New) Sangreal 15G-ICE** | ~~14.83 GB~~<br>14.81 GB | ~~0.112369~~<br>0.107390 | ~~2.1470~~<br>2.1159 | ~~9.0985~~<br>8.3963 | ~~1.0393~~<br>1.0353 | ~~93.09 %~~<br>93.31 % | ~~6.853~~<br>6.536 | ~~3.341~~<br>3.336 | ~~90.9~~<br>**91.2** |
| UD-IQ3_XXS | 13.60 GB | 0.137273 | 2.7651 | 9.3111 | 1.0729 | 92.11 % | 5.489 | 3.064 | **89.2** |

### Table 3 — English text (WikiText-2)

PPL(base) **7.574505** · BF16 = 100

| file | size | mean KLD | 99% KLD | 99.9% KLD | PPL ratio | same top-1 | active bpw | file bpw | overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **≈ 25–27 GB** | | | | | | | | | |
| UD-Q5_K_XL | 26.98 GB | 0.024330 | 0.2425 | 1.0036 | 0.9929 | 93.88 % | 7.833 | 6.080 | **96.5** |
| **(New) Sangreal 25G-ICE** | ~~24.85 GB~~<br>24.82 GB | ~~0.029038~~<br>0.027264 | ~~0.2823~~<br>0.2677 | ~~1.2220~~<br>1.1085 | ~~0.9857~~<br>0.9847 | ~~93.27 %~~<br>93.21 % | ~~7.686~~<br>7.385 | ~~5.599~~<br>5.593 | ~~96.0~~<br>**96.1** |
| CyberTiel 25G-ICE | 24.85 GB | 0.027866 | 0.2793 | 1.0757 | 0.9867 | 93.36 % | 7.686 | 5.599 | **96.1** |
| **≈ 22.5–23 GB** | | | | | | | | | |
| **(New) Sangreal 23G-ICE** | ~~22.84 GB~~<br>22.81 GB | ~~0.033585~~<br>0.031606 | ~~0.3299~~<br>0.3093 | ~~1.3758~~<br>1.1956 | ~~0.9870~~<br>0.9907 | ~~92.69 %~~<br>92.78 % | ~~7.523~~<br>7.215 | ~~5.145~~<br>5.139 | ~~95.5~~<br>**95.7** |
| CyberTiel 23G-ICE | 22.84 GB | 0.032992 | 0.3290 | 1.2449 | 0.9899 | 92.75 % | 7.523 | 5.145 | **95.6** |
| UD-Q4_K_M | 22.52 GB | 0.035692 | 0.3576 | 1.3199 | 0.9766 | 92.42 % | 7.130 | 5.075 | **95.3** |
| UD-Q4_K_XL | 22.75 GB | 0.035837 | 0.3753 | 1.3951 | 0.9761 | 92.56 % | 7.474 | 5.126 | **95.3** |
| **≈ 21 GB** | | | | | | | | | |
| CyberTiel 21G-ICE | 20.85 GB | 0.039658 | 0.4136 | 1.4041 | 0.9848 | 92.02 % | 7.357 | 4.698 | **94.9** |
| **Sangreal 21G-ICE** | 20.85 GB | 0.039968 | 0.4156 | 1.5834 | 0.9819 | 92.03 % | 7.357 | 4.698 | **94.9** |
| UD-Q4_K_S | 21.28 GB | 0.040169 | 0.4107 | 1.5040 | 0.9808 | 91.97 % | 7.025 | 4.795 | **94.9** |
| **≈ 18 GB** | | | | | | | | | |
| **(New) Sangreal 19G-ICE** | ~~18.82 GB~~<br>18.76 GB | ~~0.060435~~<br>0.059599 | ~~0.6098~~<br>0.5982 | ~~2.1340~~<br>2.0323 | ~~0.9948~~<br>0.9907 | ~~90.15 %~~<br>90.17 % | ~~7.192~~<br>6.871 | ~~4.241~~<br>4.228 | **93.1** |
| CyberTiel 19G-ICE | 18.82 GB | 0.060573 | 0.6220 | 2.2733 | 0.9951 | 90.15 % | 7.192 | 4.241 | **93.0** |
| UD-IQ4_XS | 18.12 GB | 0.070787 | 0.7346 | 2.6002 | 1.0441 | 89.51 % | 6.757 | 4.083 | **92.2** |
| **≈ 13.5–15 GB** | | | | | | | | | |
| **(New) Sangreal 15G-ICE** | ~~14.83 GB~~<br>14.81 GB | ~~0.126386~~<br>0.120439 | ~~1.3329~~<br>1.2781 | ~~3.9267~~<br>3.7501 | ~~1.0027~~<br>0.9987 | ~~85.89 %~~<br>86.18 % | ~~6.853~~<br>6.536 | ~~3.341~~<br>3.336 | ~~87.9~~<br>**88.3** |
| UD-IQ3_XXS | 13.60 GB | 0.151623 | 1.6338 | 4.3641 | 1.0920 | 84.85 % | 5.489 | 3.064 | **86.2** |

**`overall` = `0.70/(1 + meanKLD) + 0.30 * sameTop1`, ×100. BF16 = 100.** Same
composite the CyberTiel card uses, so the two are directly comparable: 70% on how
close the whole output distribution stays, 30% on agreement about the argmax.

> **Read the tail columns as shape, not order.** `99.9% KLD` is roughly the 33rd-worst
> token of 32,768 — an extreme order statistic with large sampling variance, so it
> inverts between adjacent files without meaning anything. Rank on `mean KLD`.
>
> **Two bpw columns.** Only 8 of 256 experts fire per token, so a bit in `ffn_*_exps`
> is worth ~3% of a bit in attention, the shared expert or the output head. `active
> bpw` weights by that; `file bpw` is just size ÷ parameters. It is why a 22.81 GB
> file computes at ~7.2 bpw.

> Don't compare the tables to each other. Code is more predictable text, so every
> file scores about half the divergence on it. Compare rows **within** a table.
>
> Every row is measured against **this lineage's own BF16 master** — the huihui
> abliterated checkpoint — in one session, one binary, one reference. Quantizations
> cut from a *different* trunk are deliberately absent: scoring them here would
> measure the distance between trunks and call it quantization damage.

## What Sangreal is

The calibration corpus these tiers were quantized against. **Built from primary
sources, not assembled from eaddario's set, bartowski's `calibration_datav5`, or any
other ready-made calibration file.** 77 buckets, 9,533 documents, 47,443,549 tokens,
`sha256 85a6b823a6762bf6…`.

The full render is **92,663 chunks**. 15G, 19G, 23G and 25G use a **36,025-chunk** pass over it —
63x the stock Ornith imatrix (573) and 45x bartowski's `calibration_datav5` (~800). 21G keeps the
earlier 17,225-chunk pass.

| block | buckets | share | documents | synthetic |
|---|--:|--:|--:|--:|
| code + cyber + spec | 36 | 45.10% | 5,858 | 0.1% |
| reasoning | 5 | 13.16% | 553 | 100.0% |
| agentic | 7 | 11.95% | 1,018 | 65.2% |
| science + domain | 11 | 11.24% | 1,183 | 0.3% |
| language | 11 | 9.66% | 596 | 0.0% |
| mathematics | 7 | 8.89% | 325 | 0.0% |
| **total** | **77** | **100%** | **9,533** | **21.7%** |

Split: technical 70.2 · science+domain 20.1 · language 9.7.

**Synthetic is 21.7% by bytes and sits in exactly three buckets** — `teacher:k2-horizon`,
`teacher:deepseek-v4-pro`, `teacher:hermes`. Everything outside `reasoning` and
`agentic` is real text.

### Language — 11 languages, 5 non-Latin scripts

| bucket | share | docs | | bucket | share | docs |
|---|--:|--:|---|---|--:|--:|
| `arabic` | 0.92% | 48 | | `russian` | 0.87% | 58 |
| `chinese` | 0.89% | 31 | | `spanish` | 0.87% | 72 |
| `hindi` | 0.88% | 44 | | `portuguese` | 0.87% | 70 |
| `japanese` | 0.88% | 33 | | `turkish` | 0.87% | 56 |
| `french` | 0.88% | 60 | | `german` | 0.86% | 73 |
| `english` | 0.87% | 51 | | | | |

The spread across the whole block is **0.06 points**, and that is deliberate:
language's job here is **activation, not ranking**. Each bucket sits just above the
saturation floor at 44.9% consumption, so these buckets *select* for the first time —
the legacy v1 spec had consumed 96% of `turkish` and 92% of `english`.

### Mathematics — absent from the pool until this cycle

| bucket | share | docs |
|---|--:|--:|
| `math` (general) | 3.16% | 170 |
| `analysis` | 1.24% | 28 |
| `algebra` | 1.16% | 46 |
| `geometry` | 1.12% | 30 |
| `proof` | 1.08% | 28 |
| `topology` | 1.05% | 20 |
| `number-theory` | 0.06% | 3 |

Sourced from the LaTeX algebraic geometry, mathlib4, UniMath,
math-comp, open-web-math, AutoMathText and proof-pile.

**The imatrix is published here** — `imatrix/sangreal-36025.imatrix.gguf`, the exact file the ICE 1.6
tiers were quantized against. Pass it to `llama-quantize --imatrix` to rebuild them from the BF16 master,
or to cut your own.

## What's inside

**MTPv2 head, native.** huihui's abliterated checkpoint already carries
Ornith-1.5's trained multi-token-prediction head; its `mtp.*` norms are bit-identical
to ornith-ai's on all seven tensors. Every tier ships `blk.40` with `nextn.*` intact,
so llama.cpp can use it as a draft model for speculative decoding. ICE 1.6 tiers carry it at Q2_K experts /
Q4_K matrices; acceptance is unchanged (15G: 94.1% at `--spec-draft-p-min 0.75`).

**Chat template: peculiar-ragdoll's Qwen-Sharp v22.5.0**, embedded in the ICE 1.5 tiers. It is a genuinely
good template. The ICE 1.6 tiers embed **v22.5.1** (even more sharpened): an agent-mode directive when tools
are present (off via `chat_template_kwargs: {"decisive": false}`) and fixed error coaching; plain chat
renders as v22.5.0.

**ICE** = *Isolation of Compounding Error*: allocate bits by how far a quantization
error **travels**, not by how large the activations are. Routers stay F32, SSM decay
gates stay F32, the KV-cached projections stay F16, the always-on dense path stays
Q8_0 — together 0.14% of the model, kept exact for ~152 MB — and the entire budget is
spent on the routed expert bank, which is 93% of the parameters but only 8-of-256
active per token. ICE 1.6 moves the output head to Q6_K: its error stays
flat from 2K to 16K instead of compounding, which is why `active bpw` falls while KLD improves. Method and the cases where it does **not** win:
[gbuzhf/ICE-quantization](https://huggingface.co/gbuzhf/ICE-quantization).

## Files

- **`measurements/`** — raw `llama-perplexity` output for every 2K row (2026-09-19), and
  `measurements/2026-09-25/` for the ICE 1.6 session: all three columns, the re-measured shipped and UD files,
  the two control builds, and the MTP acceptance run.
- **`recipes/`** — the exact `llama-quantize` tensor map for each tier (443 rules): `cfg_<T>-ICE16.txt` for
  15G, 19G, 23G and 25G; `cfg_21G-ICE_ICEbase.txt` for 21G.
- **`imatrix/`** — `sangreal-36025.imatrix.gguf` (ICE 1.6 tiers: 15G, 19G, 23G and 25G; 36,025 chunks × 512).
  1020 tensors. With this plus `recipes/`, every ICE 1.6 tier here is reproducible byte-for-byte from the
  published BF16 master.

## Chat template — optional Qwen-Sharp v22.5.0-arrow4.0

The GGUFs embed Qwen-Sharp v22.5.0, which stays the default. For agent harnesses (tool use, multi-step work), the repo also ships arrow4.0 — opt in by passing the file:

```bash
llama-server -m <model>.gguf --jinja --chat-template-file chat_template_sharp-v22.5.0-arrow4.0.jinja
```

arrow4.0 is our agent layer on Qwen-Sharp v22.5.0. Plain chat stays byte-identical to Sharp; with tools it adds short agent rules, one note per tool result, a STATUS line, thinking only at decision points, `.arrow` project memory and a hand-back report when work stalls.

Full description and switches: [gbuzhf/Qwen-Sharp-Arrow-Chat-Templates](https://huggingface.co/gbuzhf/Qwen-Sharp-Arrow-Chat-Templates).

## Credits

| | |
|---|---|
| base model | [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |
| abliterated checkpoint | [huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated](https://huggingface.co/huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated) |
| chat template | [peculiar-ragdoll/Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates) |
| Official CyberTiel UD ladder | [peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP) |
| our previous ICE ladder | [gbuzhf/…-CyberTiel-Calibrated-MTPv2-ICE-GGUF](https://huggingface.co/gbuzhf/Ornith-1.5-35B-A3B-Abliterated-CyberTiel-Calibrated-MTPv2-ICE-GGUF) |
| ICE method | [gbuzhf/ICE-quantization](https://huggingface.co/gbuzhf/ICE-quantization) |

KLD measures fidelity to this repo's master and nothing else — not reasoning, tool use
or speed. It is also **within-lineage**: every ladder is measured against its own
master, so these values are not comparable to another repo's. Compare across lineages
with PPL.

