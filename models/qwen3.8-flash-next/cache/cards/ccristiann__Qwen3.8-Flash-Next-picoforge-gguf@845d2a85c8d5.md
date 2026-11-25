---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
  - gguf
  - qwen
  - code
  - mixed-precision
  - mixture-of-experts
  - apple-silicon
  - picoforge
---

# Qwen3.8-Flash-Next, code-tuned mixed-precision experts (picoforge)

Two quantizations of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for
[picoforge](https://github.com/CCristiann/picoforge), a small C + Metal inference engine for
Apple Silicon. They are built from the published 2-bit file (Q2), but in each of the 48 layers
the experts that code uses most are switched to their 4-bit versions (Q4):

| file | hot experts per layer | size | KL to Q4 on unseen code (Q2: 0.288) | largest context with MTP on 64 GB |
|---|---|---|---|---|
| `Qwen3.8-Flash-Next-Q2-Code5`  | 26 of 512 (5%)  | 138.5 GiB | 0.217 | 100K |
| `Qwen3.8-Flash-Next-Q2-Code10` | 51 of 512 (10%) | 139.9 GiB | 0.167 | ~74K |

The full files are not stored here. This repository holds a **patch**: the 4-bit experts
themselves (6.6 GB) and the ranking that chooses them. A script then rebuilds either file from the
Q2 you download from antirez in about two minutes, **byte for byte**, checked by sha256 (see
[Build the files](#build-the-files)).

> Only picoforge can load these files. DwarfStar and llama.cpp do not know the mixed-expert layout.

## The idea in one paragraph

Qwen3.8-Flash-Next is a mixture of experts. Each layer has 512 small feed-forward networks (the
experts), and a router picks 10 of them for every token. At 2 bits the whole model fits in 64 GB
of memory, but it drifts from the 4-bit model. Routing is very uneven, though: on code, a few
dozen experts per layer get a large share of the router's weight. Keeping just those experts at
4 bits recovers much of the gap for very little memory. The 4-bit bytes are copied unchanged from
the published Q4 file, so nothing is re-quantized: each expert is exactly as it appears in one of
the two published files. The only new information is *which* experts get the extra bits.

This is frequency-aware bit allocation for MoE models. The general idea is known (MC-MoE, MoPEQ,
EAC-MoE and others); what is specific here is the ranking on code across eight languages, the
held-out check, and the engine support in picoforge.

## How the experts were chosen

- **Ranking corpus**: 8 open-source repositories, one per language, 12,288 tokens each (98,304
  tokens): llama.cpp (C++), flask (Python), react (JavaScript), ripgrep (Rust), gin (Go), gson
  (Java), zod (TypeScript), curl (C).
- **Score**: the router's gate weight summed per expert over all those tokens, read from the 4-bit
  model on every trunk layer. Experts are ordered by that total (`ranking.json`). The top 5% / 10%
  of each layer become 4-bit.
- **Held out**: the evaluation texts are never used for ranking. They are other repositories
  (tokio, cobra, guava), picoforge's own code, and English prose written after the model's release.

How much of the router weight the 10% hot experts carry on held-out text: **48%** on the other
repositories, **51%** on picoforge code, **18%** on prose. Ranking with Unsloth's importance-matrix
counts instead reaches only 8–12%.

## Quality

Measured against the **4-bit model as the reference**. The model reads 2,048-token windows of the
text, and at every position its next-token probabilities are compared with Q4's. KL is how far
those probabilities are from Q4's (0 = identical, lower is better). Top-1 is how often the most
likely next token is the same as Q4's. 95% confidence intervals are in the
[benchmark notes](https://github.com/CCristiann/picoforge/blob/main/bench/quant/m4_20261123/README.md).

| held-out text | Q2 | Code5 | Code10 |
|---|---|---|---|
| picoforge code: KL / top-1 | 0.288 / 84.1% | 0.217 / 85.9% | **0.167 / 88.1%** |
| tokio, cobra, guava: KL / top-1 | 0.083 / 95.9% | 0.034 / 97.6% | **0.024 / 98.0%** |
| English prose: KL / top-1 | 0.513 / 69.9% | 0.485 / 70.7% | **0.455 / 71.5%** |

Code10 removes 42% of the Q2's distance from Q4 on unseen picoforge code, 71% on the other
repositories and 11% on prose. Prose does not get worse; it improves a little.

Not measured yet: coding-task benchmarks such as HumanEval+. The numbers above measure closeness
to the 4-bit model, not task accuracy.

## Speed and memory (MacBook Pro M5 Pro, 64 GB)

| | Q2 | Code5 | Code10 |
|---|---|---|---|
| decode, greedy | 36–37 tok/s | about −4% | −5 to −8% |
| decode with MTP (same tokens as greedy) | 38.4 tok/s | 35.9 tok/s | — |
| prefill of an 8K-token prompt | 31–32 s | — | 32–33 s |
| largest context with MTP | 100K | 100K | ~74K |

These are single guarded runs, not the project's full 3 + 20 protocol. Multi-token prediction
(MTP) on the mixed files produces exactly the tokens of plain greedy decoding.

## Build the files

You need about 140 GiB of free disk for each file, Python 3 with numpy, and the Q2 file at
antirez's pinned revision. The output name of the Q2 must stay `Qwen3.8-Flash-Next-Q2.gguf`,
because it is recorded inside the recipe.

```sh
git clone https://github.com/CCristiann/picoforge && cd picoforge
python3 -m venv .venv && .venv/bin/pip install numpy huggingface_hub

# the published Q2 (137.1 GiB, sha256 b1b93fa6...)
.venv/bin/hf download antirez/qwen3.8-flash-next-gguf Qwen3.8-Flash-Next-Q2.gguf \
  --revision d600fe1a43d2e1cdcadb85144ce3142f66f9eefe --local-dir models/Qwen3.8-Flash-Next-Q2

# this repository (6.6 GB)
.venv/bin/hf download ccristiann/Qwen3.8-Flash-Next-picoforge-gguf --local-dir build/code-experts

# Code5 (use --fraction 0.10 and the Code10 hash for the other file)
.venv/bin/python tools/quant/build_mixed.py build \
  models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf \
  build/code-experts/Qwen3.8-Flash-Next-Code-Q4-experts.gguf \
  build/code-experts/ranking.json \
  models/Qwen3.8-Flash-Next-Q2-Code5/Qwen3.8-Flash-Next-Q2-Code5.gguf \
  --fraction 0.05 --expect dfda107b7fd56ed089bdeb2f840e9a9a98932c78ea171658ad935d0e9c2cfbc2
```

The script checks the output's sha256 and fails if it differs. The expected values are in
`SHA256SUMS`. The `recipes/` folder lists the hot experts of every layer for both files.

## Run

```sh
make                      # in picoforge: builds pf, pf-bench, pf-agent
./pf models               # lists q2, q2-code5, q2-code10
./pf -m code5             # chat
./pf-agent -m code5       # coding agent in the current directory
```

## Files

| file | what |
|---|---|
| `Qwen3.8-Flash-Next-Code-Q4-experts.gguf` | the 51 best-ranked 4-bit experts of each layer (gate/up Q4_K, down MXFP4), copied from the Q4 |
| `ranking.json` | the order of the 512 experts of each layer, with the gate mass behind it |
| `recipes/*.json` | the hot experts of each layer in Code5 and Code10 |
| `SHA256SUMS` | hashes of the files here, of the Q2 input and of both outputs |
| `LICENSE` | the Qwen Community License 1.0 |

## License and credits

- **Model**: Qwen3.8-Flash-Next by Qwen, under the **Qwen Community License 1.0** (`LICENSE`). Its
  copyright notice and permission notice must go with every copy.
  - Commercial products with more than 100 million monthly users, or more than US$20 million
    monthly revenue, must display the model name.
  - A *Model as a Service* or *AI Work Assistant* business (for example, a commercial product
    mainly for AI-assisted coding) needs a separate license from Qwen. Read `LICENSE` before any
    commercial use.
- **Quantizations**: the Q2 and Q4 weights are Ivan Fioravanti's DwarfStar releases, packaged with
  the native n-gram table by Salvatore Sanfilippo (antirez) in
  [antirez/qwen3.8-flash-next-gguf](https://huggingface.co/antirez/qwen3.8-flash-next-gguf).
- **This work**: the expert ranking, the mixed layout and its engine support, by Cristian Cirje, as
  part of picoforge (MIT). Not affiliated with Qwen.
