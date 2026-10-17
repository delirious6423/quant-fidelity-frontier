---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
quantized_by: steadfastgaze
tags:
  - qwen
  - qwen3.8-flash-next
  - moespresso
  - mlx
  - apple-silicon
  - mixture-of-experts
---

# Qwen3.8-Flash-Next for MoEspresso

This package brings Qwen3.8-Flash-Next to
[MoEspresso](https://github.com/steadfastgaze/MoEspresso). Qwen3.8-Flash-Next
is a 125B Mixture-of-Experts model with 6B active parameters per token and a
51B-parameter PLE/n-gram memory.

On a 2021 M1 Max with a 24-core GPU and 32 GB of unified memory, MoEspresso
serves this package at roughly **12–15 decode tokens/s**.
The local bounded configuration also scored **84.3%** on a frozen 48-question
generated-answer comparison, ahead of **Claude Opus 4.8 xhigh at 80.3%** on
the same questions.

The complete download occupies 145.75 GB (135.74 GiB), including the original
BF16 PLE tables. MoEspresso keeps only its active working set in unified
memory.

## Run it

The package requires [MoEspresso 3.0.0](https://github.com/steadfastgaze/MoEspresso)
on an arm64 Apple Silicon Mac running macOS 26.2 or later. Its files use the
MoEspresso package format and cannot be loaded directly by Transformers,
mlx-lm, or GGUF runtimes.

```bash
brew install steadfastgaze/tap/moespresso
brew install hf

hf download steadfastgaze/Qwen3.8-Flash-Next-MoEspressoV3 \
  --local-dir ./models/qwen3.8-flash-next

moespresso verify ./models/qwen3.8-flash-next
moespresso serve ./models/qwen3.8-flash-next
```

The server exposes `POST /v1/chat/completions` and `GET /health` at
`http://127.0.0.1:8080`. Thinking is enabled at medium effort by default, and
requests may select `low`, `medium`, or `xhigh`. Ordinary serving uses a 128K
context limit. `--max-context-tokens` can select a supported limit up to the
package's 262,144-token architecture limit, with larger limits reserving more
cache memory and leaving less room for resident experts.

With the server running, measure decode speed in another terminal:

```bash
moespresso speed
```

For one-shot generation:

```bash
moespresso generate ./models/qwen3.8-flash-next \
  --prompt "Write a lock-free bounded queue in C++ and explain its memory ordering." \
  --max-tokens 2048
```

## Quality on the bounded path

The comparison uses a frozen set of 48 generated-answer questions,
with eight questions from each of six public LiveBench categories: coding,
data analysis, instruction following, language, math, and reasoning. A fixed
salted SHA-256 rule derives the selection from pinned datasets, and the
official task scorers grade every complete answer. The
[benchmark reproduction kit](https://github.com/steadfastgaze/MoEspresso/tree/main/docs/benchmark_reproduction)
contains the selection procedure, runner, isolated graders, and reports.

| Model / configuration | Six-category macro |
|---|---:|
| Qwen3.8 Flash, xhigh | 90.7% |
| GPT-6 Sol, medium | 89.2% |
| Claude Opus 5.5, low | 87.6% |
| **MoEspresso, IQ_K routed experts, Cache-Prior 2/2, capacity 223, medium** | **84.3%** |
| GPT-6 Luna, xhigh | 81.4% |
| Claude Opus 4.8, xhigh | 80.3% |

The local row used medium thinking, greedy generation, 223 resident expert
slots per layer, and the default Cache-Prior policy with factor 2 and two
protected routes. Disk KV was disabled. All 48 answers stopped naturally,
with no output-cap hits. All hosted rows ran through OpenRouter. Hosted Qwen
used fixed sampling controls, while the GPT and Claude endpoints used their
provider-native defaults.

This is a custom comparison derived from public LiveBench tasks. It measures
the actual bounded route used for 32 GB serving. The official LiveBench
leaderboard uses a different generation protocol.

## SSD-backed experts and Cache-Prior routing

Dense weights and shared experts remain in unified memory. MoEspresso assigns
the remaining memory budget to a separate routed-expert pool for each layer,
then loads a missing expert row from SSD when the router selects it.

Bounded serving uses a policy inspired by
[Cache-Prior](https://arxiv.org/html/2412.00099v2). During ordinary
single-token decode from layer 2 onward, the router's two strongest original
choices remain protected. Resident candidates receive a factor-two ranking
preference for the remaining positions. Ten routed experts still execute, and
their contribution weights come from the original router probabilities
normalized over the selected set.

Prefill is unbiased. Layers 0 and 1 also retain the model's original routing
during decode, as does fully resident execution. Every expert remains
available, and a selected nonresident expert is loaded from SSD. Because the
resident preference can change the selected expert set and the model's output,
the quality result above was measured with this policy enabled.

The [Cache-Prior documentation](https://github.com/steadfastgaze/MoEspresso/blob/main/docs/cache_prior.md)
explains the full algorithm and its relationship to REAP.

## Package recipe

The model has 48 layers. Each MoE layer retains all 512 routed experts and
selects ten per token alongside one shared expert. The routed experts' gate,
up, and down projections use IQ3_K in layers 0 and 1, covering six projection
groups. The remaining 138 groups use IQ2_K. Dense and non-routed tensors use
Q6_K, Q8_0, or BF16 according to their role.

The original BF16 PLE/n-gram tables stay on SSD, where MoEspresso reads only
the selected rows instead of placing the complete 95.37 GiB payload in unified
memory. The Qwen attention cache stores its older body in KVarN K4/V4 while
keeping the sink and recent suffix exactly in BF16. This text-only package
excludes the vision encoder and MTP sidecar.

| Payload | GB | GiB |
|---|---:|---:|
| Model safetensors | 43.35 | 40.37 |
| Original BF16 PLE/n-gram tables | 102.40 | 95.37 |
| Complete declared payload | 145.75 | 135.74 |

The package was derived from
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.

## Download verification

`moespresso verify` checks the package manifest, declared paths, byte sizes,
SHA-256 digests, tensor keys, and generated compatibility files. The package
manifest identity is
`pkg:e469be1b1a19e966d4f546c8eaa7b5ee980ad19f0e41883d3c25678a4728fef5`.

## License and credits

The model weights are released under the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE),
which includes separate terms for commercial Model-as-a-Service and AI Work
Assistant use.

The Qwen team trained and released the base model. The IQ_K formats were
designed by [Iwan Kawrakow](https://github.com/ikawrakow). MoEspresso executes
the IQ_K tensors through
[mlx-iqk](https://github.com/steadfastgaze/mlx-iqk). The K-quant tensors run
through the project's
[mlx-kquant fork](https://github.com/steadfastgaze/mlx-kquant), which derives
from [Asher Feldman's original integration](https://github.com/asher/mlx-kquant).
MoEspresso runs on Apple's [MLX](https://github.com/ml-explore/mlx).
