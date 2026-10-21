---
license: mit
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: steadfastgaze
tags:
- moespresso
- coding
---

# DeepSeek V4 Flash 0731 Coder, compact MoEspresso package

This is a 56.869 GB coding-specialized MoEspresso package derived from
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
Its model safetensors occupy 56.831 GB, 32.63% less than the 84.355 GB
[MoEspresso 2.37 bpw package](https://huggingface.co/steadfastgaze/DeepSeek-V4-Flash-0731-2.37bpw-MoEspressoV2).
The package removes about 80B parameters, leaving 204B total, with about 13B
active per token.

You can see a screen recording of it writing a small C compiler here:

https://youtu.be/XiwSilmV8B0

The package requires MoEspresso 2.1 or newer. MoEspresso is the only engine
that serves it. llama.cpp, transformers, and mlx-lm will not load these files.

```bash
brew install steadfastgaze/tap/moespresso
moespresso serve /path/to/package
```

This package is intended for coding and coding-agent workloads. It is not a
smaller general-purpose replacement for the 2.37 bpw package. The measurements
below show both the coding retention and the general-text regression.

## Quality

Lower NLL and perplexity are better. The code panel contains 12 fixed
2,048-token windows and 24,564 next-token targets from `code.utf8`.

| package | model safetensors | code mean NLL | code perplexity | selected-token agreement with 2.37 bpw |
|---|---:|---:|---:|---:|
| This Coder package | `56.831 GB` | `1.01758` | `2.7665` | `88.55%` |
| MoEspresso 2.37 bpw | `84.355 GB` | `0.88582` | `2.4250` | reference |

An unpruned all-IQ1 no-REAP research baseline was researched, although not published,
and relative to that baseline, pruning and specialization increased
WikiText mean NLL by `0.25170` while reducing code mean NLL by `0.10250`.
Extreme IQ1 quantization accounts
for most of the broad-text loss from the 2.37 bpw package. The subsequent
specialization trades additional broad-text quality for better code modeling.

The broad-text and official-continuation panels quantify the resulting product
boundary.

| package | WikiText mean NLL | WikiText perplexity | API-continuation mean NLL | API first-token agreement |
|---|---:|---:|---:|---:|
| This Coder package | `2.41630` | `11.2043` | `1.23133` | `48/100` |
| MoEspresso 2.37 bpw | `1.71466` | `5.5548` | `0.39634` | `66/100` |

The WikiText panel uses the first 32 contiguous 2,048-token windows from the
WikiText-2 test split and scores the last 1,023 tokens of each window, for
32,736 targets. The API panel uses 100 prompts and 2,313 provider-selected
continuation tokens from the official DeepSeek 0731 model. Each local package
is teacher-forced under the stored continuation. This is not free-running API
agreement.

The rendered reasoning and tool validation panel uses held-out conversations
converted to DeepSeek's prompt and tool-call format.

| category | positions | Coder NLL | 2.37 bpw NLL | Coder selected-token accuracy | 2.37 bpw accuracy |
|---|---:|---:|---:|---:|---:|
| All rendered targets | `8,188` | `1.33567` | `1.31694` | `75.67%` | `75.95%` |
| Reasoning | `1,806` | `1.46925` | `1.47384` | `66.11%` | `66.06%` |
| Structured tool calls | `1,880` | `0.46838` | `0.39007` | `90.05%` | `90.69%` |
| Tool-result history | `850` | `1.12205` | `1.11188` | `76.59%` | `78.59%` |

The package also passes additional checks, e.g.:

| check | scope | result |
|---|---|---|
| Long-context fact recall | 16 fixed facts in a 30,000-token-class prompt | `16/16` exact match |
| Coding capability | 16 private tests | `16/16` tests |

These panels were used during package development. They characterize the final
artifact but are not untouched post-selection benchmarks.

## Recipe

The three hash-routed layers keep all 256 experts. Each of the 40 learned-router
layers keeps 176 of 256 experts, for 7,808 retained expert rows in total. The
served router still selects six experts per token.

Expert retention and precision were selected together. Of the 129 routed
projection cells, 32 use IQ1_S_R4, 64 use IQ2_KS, and 33 use IQ2_K. The denser
model weights use q6_K. The layer-12 down projection is explicitly promoted;
short closure screens and full autoregressive reasoning runs showed that
promotion count alone did not predict sustained-thinking quality.

The package does not include a drafter. A separately distributed
[DSpark sidecar](https://huggingface.co/steadfastgaze/DeepSeek-V4-Flash-0731-drafter-only-MoEspressoV2)
can be checked and served with MoEspresso 2.1:

```bash
moespresso verify /path/to/package --drafter /path/to/dspark-sidecar
moespresso serve /path/to/package --drafter /path/to/dspark-sidecar
```

## Corpus and specialization

Weight quantization used
[Bartowski's `calibration_datav5`](https://gist.github.com/bartowski1182/82ae9b520227f57d79ba04add13d0d0d)
and [exllamav3's `technical.utf8`](https://github.com/turboderp-org/exllamav3/blob/0b9745c526a13d5b30f1b58a864efc1932d3d9eb/conversion/standard_cal_data/technical.utf8).

Expert retention used general windows from `calibration_datav5`, coding windows
from [exllamav3's `code.utf8`](https://github.com/turboderp-org/exllamav3/blob/0b9745c526a13d5b30f1b58a864efc1932d3d9eb/conversion/standard_cal_data/code.utf8),
unrelated casual chat, and the `conversations` train split of
[Reasoning and Tool Calling](https://huggingface.co/datasets/steadfastgaze/reasoning-tool-calling-v1).

WikiText-2 was used only for evaluation.

## Verification

After downloading, verify the complete directory before serving it:

```bash
moespresso verify /path/to/package
```

## Credits

The base model was trained and released by
[DeepSeek](https://huggingface.co/deepseek-ai). The IQ_K formats were designed
by [Iwan Kawrakow](https://github.com/ikawrakow) and are served through
[mlx-iqk](https://github.com/steadfastgaze/mlx-iqk). The dense q6_K path uses
[mlx-kquant](https://github.com/steadfastgaze/mlx-kquant). Construction data
includes material published by [Bartowski](https://github.com/bartowski1182)
and [turboderp-org](https://github.com/turboderp-org). MoEspresso runs on Apple's
[MLX](https://github.com/ml-explore/mlx).

The complete third-party provenance is recorded in the MoEspresso repository's
`THIRD-PARTY-NOTICES` and in the notices distributed by its dependencies.
