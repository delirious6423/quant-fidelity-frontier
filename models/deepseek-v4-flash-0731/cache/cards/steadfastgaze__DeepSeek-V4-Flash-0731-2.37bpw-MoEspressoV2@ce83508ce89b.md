---
license: mit
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: steadfastgaze
tags:
- moespresso
---

# DeepSeek V4 Flash 0731, 2.37 bpw MoEspresso V2 package

This repository holds a MoEspresso package, and MoEspresso is the only engine
that serves it: https://github.com/steadfastgaze/MoEspresso.

This package requires MoEspresso 2. Install it with Homebrew:

```bash
brew install steadfastgaze/tap/moespresso
```

llama.cpp, transformers, and mlx-lm will not load these files.

Full serving instructions live in the MoEspresso repository.

Original model: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731

The package includes the DeepSeek DSpark drafter. The quality measurements
below use the target model without drafting.

## Quality

The first panel compares complete quantized artifacts in their intended
runtimes. Lower NLL and perplexity are better; higher first-token agreement is
better.

| package and runtime | WikiText mean NLL | WikiText perplexity | API-continuation mean NLL | API first-token agreement |
|---|---:|---:|---:|---:|
| MoEspresso 2.37 bpw / MoEspresso | `1.71466` | `5.5548` | `0.39634` | `66/100` |
| antirez IQ2_XXS / DS4 | `1.76824` | `5.8605` | `0.41517` | `54/100` |
| Unsloth UD-IQ2_XXS / llama.cpp | `1.71695` | `5.5675` | `0.36224` | `62/100` |

The WikiText panel uses the first 32 contiguous 2,048-token windows from the
held-out WikiText-2 test split and scores 32,736 targets. All three runtimes
received identical token IDs. The API panel uses the same 100 prompts and
2,313 provider-selected continuation tokens in every local runtime. Each local
model is teacher-forced under that continuation; this is not free-running
generation agreement.

**MoEspresso was not calibrated on this WikiText test panel.** The corpus and
separation checks are described below.

The release package also carries these model-specific checks:

| check | scope | result |
|---|---|---|
| Renderer and tokenizer goldens | 34 fixed cases | `34/34` |
| Greedy selected-token identity | Five fixed prompts and 14 next-token decisions, compared with the official API | `9/14` |
| Official continuation loss | 100 prompts and 2,313 target tokens | Average NLL `0.3963`; first-token agreement `66/100` |
| Long-context fact recall | 16 fixed facts in a 30,000-token-class prompt | `16/16` exact match |
| WikiText all-position release gate | 32 windows and 65,504 targets | Package perplexity `6.4774`; bf16 teacher perplexity `4.8886` |

The comparison panel and the all-position release gate score different target
positions, so their two WikiText perplexity readings are not directly
comparable. Exact artifact identities, corpus hashes, and scoring controls are
recorded in the
[benchmark protocol](https://github.com/steadfastgaze/MoEspresso/blob/main/docs/benchmark_reproduction.md#deepseek-v4-flash-quality-comparison).

## Recipe

The served target model is 2.37 bits per weight. The bundled drafter is
excluded from that ratio.

MoEspresso selected IQ2_KS and IQ2_K assignments for the routed experts from
calibrated output-error evidence. The dense model weights use q6_K. DeepSeek-V4
auxiliary tensors use the package's declared affine and mxfp8 formats, and the
DSpark sidecar uses IQ2_K. The package manifest records every tensor decision
and the SHA-256 identity of every file.

After downloading, `moespresso-verify <package-directory>` checks the complete
package against that manifest.

## Corpus

The calibration training spine joins two published sources:

- [Bartowski's `calibration_datav5`](https://gist.github.com/bartowski1182/82ae9b520227f57d79ba04add13d0d0d),
  assembled from material credited to Dampf and Kalomaze plus edaddario's
  `combined_all_small` dataset. It mixes general and instruction-style prose,
  fenced code, and some multilingual material.
- [exllamav3's `technical.utf8`](https://github.com/turboderp-org/exllamav3/blob/0b9745c526a13d5b30f1b58a864efc1932d3d9eb/conversion/standard_cal_data/technical.utf8),
  which adds technical and scientific prose, including mathematical notation.

The two files are concatenated and tokenized with the DeepSeek-V4-Flash 0731
tokenizer. A deterministic slice is reserved from calibration fitting.
Separate C4 prose and source-code files are held out as validation probes.
exllamav3's WikiText, multilingual, TinyStories, and synthetic random-token
components are not part of the training spine.

The `Salesforce/wikitext` `wikitext-2-raw-v1` test split used for the published
perplexity panel was not used to produce importance matrices, calibration
moments, or optimizer inputs. A full-file audit found no substantive exact
overlap between either calibration component and that test panel or the known
historical WikiText train excerpt. This supports the panel-specific statement
that the package was not calibrated on the benchmark. It does not establish
that every upstream source in the calibration corpus is free of
Wikipedia-derived prose, because the complete source genealogy of
`calibration_datav5` is not known.

## Files

The repository contains the target-model safetensors shards, the bundled
DSpark sidecar and its shards, tokenizer files, and the package-owned manifests,
plans, reports, and runtime sidecars. `package_manifest.json` is the root of the
verification contract.

## Credits

The base model and DSpark drafter were trained and released by
[DeepSeek](https://huggingface.co/deepseek-ai). The IQ_K formats were designed
by [Iwan Kawrakow](https://github.com/ikawrakow) and are served through
[mlx-iqk](https://github.com/steadfastgaze/mlx-iqk). The dense q6_K path uses
[mlx-kquant](https://github.com/steadfastgaze/mlx-kquant). The calibration
corpus combines work published by
[Bartowski](https://github.com/bartowski1182) and
[turboderp-org](https://github.com/turboderp-org). MoEspresso runs on Apple's
[MLX](https://github.com/ml-explore/mlx).

The complete third-party provenance is recorded in the MoEspresso repository's
`THIRD-PARTY-NOTICES` and in the notices distributed by its dependencies.
