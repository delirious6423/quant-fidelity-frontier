---
license: apache-2.0
tags:
- gguf
- text-generation
- magicquant
- conversational
base_model:
- Qwen/Qwen3.8-27B
---

# MagicQuant Hybrids (v2.0) - Qwen3.8-27B

MagicQuant is a benchmark driven GGUF hybrid discovery and validation system focused on finding real, practical GGUF quants specific to each architecture.

Whether it's a pure baseline model built by llama.cpp, learned tensor configurations from Unsloth, or a custom built MagicQuant hybrid, the model table below shows quants that have won dominance checks, survived collapse spaces, and/or were found to be nonlinearly better. Instead of dumping every quant type possible, MagicQuant tests, validates, and brutally murders anything deemed unworthy.

**This combined release utilizes Unsloth Dynamic V3 configurations, while retaining the distinct Dynamic V2 discoveries that still occupy unique positions on the measured size/KLD frontier.**

<details>
<summary>Support MagicQuant</summary>

I’m a solo developer working full time for myself to achieve my dream. I build open source code on the side. If you like any of my work, buying me a coffee is always appreciated. Otherwise, I hope you enjoy, maybe give me a star or something. Or just send me good vibes. Either way, thank you!

[Click here to see ways to support](https://sayou.biz/support) - BTC, Paypal, GitHub sponsors.

</details>

---

## Final survivors

| Name | Provider | KLD | Size (GB) | Download |
|---|---|---:|---:|---|
| ~~LM-Q8_0~~ | ~~llama.cpp~~ | ~~0.000712~~ | ~~29.05~~ | |
| [<u>MQ-Q6_K_1</u>](#winner-notes "Replaced: LM-Q8_0") | MagicQuant | 0.000703 | 29.03 | [Link](./../../resolve/main/Qwen3.8-27B-MQ-Q6_K_1.gguf?download=true) |
| [<u>MQ-Q6_K_2</u>](#winner-notes "Replaced: MQ-Q8_0, MQ-Q8_0, MQ-Q8_0 + 5 more") | MagicQuant | 0.000873 | 27.26 | [Link](./../../resolve/main/Qwen3.8-27B-MQ-Q6_K_2.gguf?download=true) |
| [<u>MQ-Q6_K_3</u>](#winner-notes "Replaced: Unsloth-UD-Q6_K_XL") | MagicQuant | 0.001047 | 25.94 | [Link](./../../resolve/main/Qwen3.8-27B-MQ-Q6_K_3.gguf?download=true) |
| [<u>UD-Unsloth-UD-Q6_K_XL</u>](#winner-notes "Replaced: MQ-Q8_0") | Unsloth | 0.001238 | 25.33 | [Link](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| UD-Unsloth-UD-Q6_K_L | Unsloth | 0.001439 | 24.23 | [Link](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| [<u>MQ-Q6_K_4</u>](#winner-notes "Replaced: Unsloth-Q6_K, MQ-Unsloth-Q6_K, MQ-Q6_K_3 [Dynamic V3 source]") | MagicQuant | 0.001518 | 23.21 | [Link](./../../resolve/main/Qwen3.8-27B-MQ-Q6_K_4.gguf?download=true) |
| [<u>MQ-Q5_K_1</u>](#winner-notes "Replaced: Unsloth-UD-Q6_K") | MagicQuant | 0.002427 | 22.00 | [Link](./../../resolve/main/Qwen3.8-27B-MQ-Q5_K_1.gguf?download=true) |
| [<u>MQ-Q5_K_2</u>](#winner-notes "Replaced: Unsloth-UD-Q5_K_XL") | MagicQuant | 0.003146 | 20.91 | [Link](./../../resolve/main/Qwen3.8-27B-MQ-Q5_K_2.gguf?download=true) |
| [<u>MQ-Q5_K_3</u>](#winner-notes "Replaced: Unsloth-UD-Q5_K_M, MQ-Q5_K_1 [Dynamic V2 source]") | MagicQuant | 0.003562 | 20.10 | [Link](./../../resolve/main/Qwen3.8-27B-MQ-Q5_K_3.gguf?download=true) |
| [<u>MQ-Q5_K_S_1</u>](#winner-notes "Replaced: Unsloth-UD-Q5_K_S") | MagicQuant | 0.004818 | 18.98 | [Link](./../../resolve/main/Qwen3.8-27B-MQ-Q5_K_S_1.gguf?download=true) |
| [<u>MQ-Q4_K_M_1</u>](#winner-notes "Replaced: Unsloth-UD-Q4_K_XL, MQ-Q4_K_M_1 [Dynamic V2 source]") | MagicQuant | 0.007412 | 17.62 | [Link](./../../resolve/main/Qwen3.8-27B-MQ-Q4_K_M_1.gguf?download=true) |
| [<u>UD-Unsloth-UD-Q4_K_M</u>](#winner-notes "Replaced: MQ-IQ4_NL_1 [Dynamic V2 source]") | Unsloth | 0.011205 | 16.50 | [Link](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| [<u>MQ-IQ4_XS_1</u>](#winner-notes "Replaced: Unsloth-IQ4_NL, MQ-Unsloth-IQ4_XS, Unsloth-IQ4_XS") | MagicQuant | 0.013723 | 16.34 | [Link](./../../resolve/main/Qwen3.8-27B-MQ-IQ4_XS_1.gguf?download=true) |
| UD-Unsloth-UD-Q4_K_S | Unsloth | 0.014502 | 15.39 | [Link](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| UD-Unsloth-UD-IQ4_XS | Unsloth | 0.020127 | 14.29 | [Link](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| [<u>UD-Unsloth-UD-Q3_K_XL</u>](#winner-notes "Replaced: UD-Unsloth-UD-Q3_K_XL [Dynamic V2 normalized source]") | Unsloth | 0.030770 | 13.18 | [Link](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| UD-Unsloth-UD-IQ3_S | Unsloth | 0.046499 | 12.08 | [Link](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| [<u>MQ-IQ2_M_1</u>](#winner-notes "Replaced: Unsloth-UD-IQ3_XXS") | MagicQuant | 0.057811 | 11.96 | [Link](./../../resolve/main/Qwen3.8-27B-MQ-IQ2_M_1.gguf?download=true) |
| UD-Unsloth-UD-IQ3_XXS | Unsloth | 0.070084 | 10.97 | [Link](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| [<u>MQ-IQ2_M_2</u>](#winner-notes "Replaced: Unsloth-UD-Q2_K_XL, MQ-Unsloth-UD-IQ2_M, Unsloth-UD-IQ2_M") | MagicQuant | 0.092394 | 10.77 | [Link](./../../resolve/main/Qwen3.8-27B-MQ-IQ2_M_2.gguf?download=true) |
| [<u>UD-Unsloth-UD-Q2_K_XL</u>](#winner-notes "Replaced: Unsloth-UD-IQ2_S") | Unsloth | 0.105510 | 9.86 | [Link](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| UD-Unsloth-UD-IQ2_XXS | Unsloth | 0.182364 | 9.05 | [Link](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| [<u>MQ-IQ2_XXS_1</u>](#winner-notes "Replaced: Unsloth-UD-IQ2_XXS, MQ-Unsloth-UD-IQ1_M, Unsloth-UD-IQ1_M + 2 more") | MagicQuant | 0.270304 | 8.27 | [Link](./../../resolve/main/Qwen3.8-27B-MQ-IQ2_XXS_1.gguf?download=true) |


<details>
<summary>Provider credits</summary>

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Baseline quantization formats and llama.cpp tooling.
- [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) — External learned baseline source (Dynamic V2 and Dynamic V3 configurations).

</details>

<details>
<summary>Warning - Is MagicQuant Better? (hint: how you frame the question matters)</summary>

External/custom baselines are normalized into MagicQuant's controlled comparison flow. MagicQuant rebuilds a learned baseline under native-source / MagicQuant-controlled conditions, including its own imatrix handling, so hybrids or external baselines (like Unsloth) can be judged on a more equal footing. That does **not** mean MagicQuant proved the original upstream artifact or upstream imatrix was worse. These comparisons exist for internal hybrid-search consistency and equal playing field comparisons, not as a universal judgment of the original creator's exact release artifact.

**Easier to digest explanation:**

MagicQuant compares and benchmarks the model's quant-to-tensor configurations, not the original external artifact. Different winners are retained for different reasons, and not every winner is simply “better” in all contexts. The manifests document what the automated system selected and why.

So, MagicQuant can confidently tell you: under the same quantization-to-tensor configurations, identical imatrix, and this benchmark, this candidate earned its place.

</details>

<details>
<summary>Re-Uploading External Provider Baselines</summary>

By default, if an external provider like Unsloth is deemed the winner, the repository links directly to the original provider instead of re-hosting the quant. External GGUFs are normally only re-uploaded when a specific winning variant does not already exist.

</details>

---

## Release metadata

- [Manifest guide](./../../resolve/main/magicquant-manifest/README.md) — combined-file semantics, provenance fields, and archive layout.
- [Final survivor metrics](./../../resolve/main/magicquant-manifest/magicquant.final-survivors.json) — final names, provenance, KLD, PPL, sizes, download targets, and replacement lineage.
- [Hybrid tensor map](./../../resolve/main/magicquant-manifest/magicquant.hybrid-map.json) — tensor-group assignments and effective-state details for every retained MagicQuant hybrid.
- [Clone tensor configs](./../../resolve/main/magicquant-manifest/magicquant.clone-configs.json) — exact per-GGUF tensor quantization maps for the reconciled output list.
- [Isolation samples](./../../resolve/main/magicquant-manifest/magicquant.isolation-samples.json) — V2 and V3 isolated base/group probes, tagged by `sourceRun`.
- [Bad trade details](./../../resolve/main/magicquant-manifest/magicquant.bad-trades.json) — both runs' structured pruning decisions, tagged by run.
- [Replacement details](./../../resolve/main/magicquant-manifest/magicquant.replacements.json) — both original decision sets plus the cross-run strict-dominance reconciliation.
- [Original run manifests](./../../resolve/main/magicquant-manifest/archive/README.md) — untouched Dynamic V2 and Dynamic V3 manifests and source model cards.

<details>
<summary>Replacement reason codes</summary>

- `STRICT_DOMINANCE` — the winner was no larger and had lower real KLD than the removed anchor.
- `NEAR_BASELINE_PREMIUM` — the winner used only the configured near-baseline size premium and beat the real linear KLD trade line.
- `INTERIOR_DISCOVERY` — the winner was selected as a useful interior point inside a size/KLD gap between anchors.
- `SPACING_COLLAPSE` — two candidates were too close in practical output space, so the stronger one was kept.
- `FINAL_DOMINANCE` — a later validated survivor dominated this artifact in final real benchmark comparison.

<a id="winner-notes"></a>
Underlined names in the table replaced or ultimately inherited the replacement of another artifact. Hover the name for a short summary, or inspect `magicquant-manifest/magicquant.replacements.json` for exact deltas.

</details>

---

## Dynamic V2 + V3 combined release

This repository is the reconciled frontier from two complete MagicQuant runs against the same Qwen3.8-27B source model and imatrix identity. The runs used different generations of Unsloth's learned Dynamic tensor configurations, and that difference produced meaningfully different winners—interesting enough that collapsing everything into only the newer run would discard useful measured trade points.

Dynamic V3 contributes the main frontier, including `MQ-Q6_K_1`, `MQ-Q6_K_2`, the Q5 series, `MQ-Q4_K_M_1`, `MQ-IQ2_XXS_1`, and the V3 Unsloth reference points. Dynamic V2 contributes `MQ-Q6_K_3`, `MQ-Q6_K_4`, `MQ-IQ4_XS_1`, `MQ-IQ2_M_1`, `MQ-IQ2_M_2`, and the `UD-Unsloth-UD-IQ2_XXS` reference point.

The final Q6_K suffixes describe fidelity order, not run generation: V3's source `_1` and `_2` remain `_1` and `_2`; V2's source `_1` and `_2` become `_3` and `_4`. V3's source `MQ-Q6_K_3` is excluded because the V2 source `MQ-Q6_K_2` (now `MQ-Q6_K_4`) is both smaller (23.21 vs 23.43 GB) and better (0.001518 vs 0.001761 KLD).

The combined manifests preserve a `sourceRun`, `sourceName`, and `sourceFileName` trail wherever a final artifact was selected or renamed. The original generated manifests remain untouched under `magicquant-manifest/archive/`, so the V2 and V3 decision histories can still be inspected independently.

## Generic vs Unsloth imatrix A/B benchmark

After the combined survivor frontier had already been discovered with the generic MagicQuant imatrix, the same 23 final tensor configurations were cloned, rebuilt, and benchmarked again using the imatrix published in [Unsloth's Qwen3.8-27B GGUF repository](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). This was a fixed-configuration A/B benchmark, not a second full MagicQuant evolution/search.

Because the generic-imatrix search happened first, these results do **not** establish whether beginning the full search with the Unsloth imatrix would have discovered different hybrids. The table reports only what was actually measured: identical survivor tensor configurations rebuilt under the two imatrices and evaluated by the same benchmark flow.

`Δ KLD` is Unsloth-imatrix KLD minus generic-imatrix KLD. Negative values favor the Unsloth imatrix; positive values favor the generic imatrix.

| Quant | Size GB | Generic Imatrix | Unsloth Imatrix | Δ KLD | Change | Published result |
|---|---:|---:|---:|---:|---:|:---|
| MQ-Q6_K_1 | 29.03 | **0.000703** | 0.000705 | +0.000002 | +0.28% | 🟦 Generic |
| MQ-Q6_K_2 | 27.26 | 0.000874 | **0.000873** | -0.000001 | -0.11% | 🟩 Unsloth |
| MQ-Q6_K_3 | 25.94 | 0.001052 | **0.001047** | -0.000005 | -0.48% | 🟩 Unsloth |
| UD-Unsloth-UD-Q6_K_XL | 25.33 | **0.001238** | 0.001264 | +0.000026 | +2.10% | 🟦 Generic |
| UD-Unsloth-UD-Q6_K_L | 24.23 | **0.001439** | 0.001474 | +0.000035 | +2.43% | 🟦 Generic |
| MQ-Q6_K_4 | 23.21 | **0.001518** | 0.001560 | +0.000042 | +2.77% | 🟦 Generic |
| MQ-Q5_K_1 | 22.00 | **0.002427** | 0.002454 | +0.000027 | +1.11% | 🟦 Generic |
| MQ-Q5_K_2 | 20.91 | 0.003212 | **0.003146** | -0.000066 | -2.05% | 🟩 Unsloth |
| MQ-Q5_K_3 | 20.10 | 0.003742 | **0.003562** | -0.000180 | **-4.81%** | 🟩 Unsloth |
| MQ-Q5_K_S_1 | 18.98 | **0.004818** | 0.005053 | +0.000235 | **+4.88%** | 🟦 Generic |
| MQ-Q4_K_M_1 | 17.62 | **0.007412** | 0.008052 | +0.000640 | **+8.63%** | 🟦 Generic |
| UD-Unsloth-UD-Q4_K_M | 16.50 | 0.011351 | **0.011205** | -0.000146 | -1.29% | 🟩 Unsloth |
| MQ-IQ4_XS_1 | 16.34 | 0.013794 | **0.013723** | -0.000071 | -0.51% | 🟩 Unsloth |
| UD-Unsloth-UD-Q4_K_S | 15.39 | **0.014502** | 0.014936 | +0.000434 | +2.99% | 🟦 Generic |
| UD-Unsloth-UD-IQ4_XS | 14.29 | **0.020127** | 0.020214 | +0.000087 | +0.43% | 🟦 Generic |
| UD-Unsloth-UD-Q3_K_XL | 13.18 | 0.031140 | **0.030770** | -0.000370 | -1.19% | 🟩 Unsloth |
| UD-Unsloth-UD-IQ3_S | 12.08 | 0.047485 | **0.046499** | -0.000986 | -2.08% | 🟩 Unsloth |
| MQ-IQ2_M_1 | 11.96 | **0.057811** | 0.058362 | +0.000551 | +0.95% | 🟦 Generic |
| UD-Unsloth-UD-IQ3_XXS | 10.97 | **0.070084** | 0.071898 | +0.001814 | +2.59% | 🟦 Generic |
| MQ-IQ2_M_2 | 10.77 | **0.092394** | 0.094737 | +0.002343 | +2.54% | 🟦 Generic |
| UD-Unsloth-UD-Q2_K_XL | 9.86 | 0.106262 | **0.105510** | -0.000752 | -0.71% | 🟩 Unsloth |
| UD-Unsloth-UD-IQ2_XXS | 9.05 | **0.182364** | 0.183547 | +0.001183 | +0.65% | 🟦 Generic |
| MQ-IQ2_XXS_1 | 8.27 | 0.275224 | **0.270304** | -0.004920 | -1.79% | 🟩 Unsloth |

### What was published

For locally hosted MagicQuant hybrids, the repository now carries the better-scoring rebuild for each configuration. Six files use the Unsloth-provided imatrix: `MQ-Q6_K_2`, `MQ-Q6_K_3`, `MQ-Q5_K_2`, `MQ-Q5_K_3`, `MQ-IQ4_XS_1`, and `MQ-IQ2_XXS_1`. The other seven locally hosted MagicQuant GGUFs retain their generic-imatrix builds.

External Unsloth rows remain direct links to Unsloth rather than locally re-hosted GGUFs. Their selected top-table KLD is still the better controlled rebuild measurement from this A/B test; it is not a benchmark claim about the original upstream artifact.

The combined `magicquant-manifest/` files remain unchanged. They preserve the original search/evolution decisions, generic-imatrix metrics, byte sizes, hybrid recipes, and clone provenance. The README's primary table is the publication view after per-artifact imatrix selection, so its selected KLD values can differ from the archival manifest values. The six Unsloth-imatrix rebuilds are also 64 bytes smaller than their corresponding generic builds due to GGUF metadata differences; this does not affect the displayed two-decimal GB sizes.

The repository-root `imatrix.dat` remains the generic imatrix associated with the original search manifests. The six named replacement GGUFs were built with the Unsloth-provided imatrix.
