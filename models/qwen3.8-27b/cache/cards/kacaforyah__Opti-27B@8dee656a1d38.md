---
license: other
license_name: opti-model-license
license_link: LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - opti
  - gguf
  - compression
  - patent-pending
  - qwen3.8
  - vision
  - llama.cpp
---

https://ko-fi.com/kalebautomates - Help me not be GPU poor

# Opti 27B

**Qwen3.8-27B, text and vision, in an 11.8 GB model file plus a 0.9 GB vision projector: 4.6x smaller than the 16-bit original and within 0.5% of its wikitext-2 perplexity. The 4-bit standard needs 16.8 GB to sit at the 16-bit number; this file sits there at 11.8 GB, 3.47 bits per parameter, on a single 16 GB GPU. On other text and on distribution-level metrics the gap to the 4-bit standard is larger; the tables below say how much.**

Opti 27B is Qwen3.8-27B, 27 billion parameters with vision through a projector file, compressed with Opti, a patent-pending compression method. The 16-bit original is 54.7 GB. This file is 11.8 GB, at 3.47 bits per parameter over the whole file, and on wikitext-2 it reads 6.487 against the original's 6.456: 1.005x unpaired, 1.0075x +/- 0.003 paired against the same 16-bit logits on 64 windows. llama.cpp's Q4_K_M, the 4-bit standard, reads the 16-bit number at 16.8 GB; Opti reads within 0.5% of it at 29.5% fewer bytes, and beats both llama.cpp files measured at or below its size on perplexity. In a pre-registered study of an unattended coding agent, Opti's first release delivered complete solutions about four times as often as the best 2-bit llama.cpp file of the same model, and the current release builds working games on its own. It takes image input through the included projector, and it fits a single 16 GB consumer GPU with room for context. Every perplexity row below was measured with the same protocol and build.

## Results

### Perplexity

wikitext-2, `llama-perplexity`, 1,024-token windows, 96 windows. All six rows: same protocol, same build, one GPU per ladder.

| Model file | Perplexity | vs 16-bit | File size | Bits / parameter |
|---|---|---|---|---|
| Qwen3.8-27B, 16-bit original | 6.456 | 1.000x | 54.7 GB | |
| llama.cpp Q4_K_M (the 4-bit standard) | 6.457 | 1.000x | 16.8 GB | 4.92 |
| llama.cpp Q3_K_M | 6.708 | 1.039x | 13.5 GB | 3.95 |
| **Opti 27B (this file)** | **6.487** | **1.005x** | **11.8 GB** | **3.47** |
| llama.cpp IQ3_XXS (closest size) | 6.928 | 1.073x | 11.6 GB | 3.41 |
| llama.cpp IQ2_M | 7.318 | 1.134x | 10.5 GB | 3.06 |

Three readings of the table:

- **Against the 16-bit original:** 6.487 vs 6.456, 0.5% higher, for a 4.6x reduction in size. Each estimate carries a +/- 0.07 standard error, so the two are not statistically separable at this sample size.
- **Against Q4_K_M:** the 4-bit standard reads the 16-bit number (6.457) at 16.8 GB. Opti's paired, window-by-window difference from it is +0.4% +/- 0.5%, i.e. not distinguishable, at 11.8 GB instead of 16.8 GB. (The paired test is run against Q4_K_M because that is the run with per-window data on the same GPU; the 16-bit comparison above is same-protocol, unpaired.)
- **Against IQ3_XXS, the closest llama.cpp file by size:** Opti's perplexity is 6.4% lower at 2% more bytes.

One accounting note, so nobody has to find it: llama.cpp's files carry Qwen's multi-token-prediction block, which llama.cpp never executes (0.45 GB in the IQ files, 0.2 GB in Q4_K_M); Opti-27B.gguf omits it, and every bits-per-parameter figure divides by the full 27.3B count. Like for like, Opti is 5.9% larger than a stripped IQ3_XXS and 28.4% smaller than a stripped Q4_K_M. Perplexity is unaffected either way. The two llama.cpp files at or below Opti's size measured here sit at 1.073x and 1.134x against the original. Opti sits at 1.005x.


### Other text, other answer: the gap depends on the domain

The wikitext-2 number above is the standard test and the best case. Below, both files are paired window by window against
the 16-bit model itself on the same windows (1,024 tokens each; 64 windows on wikitext-2, 32 on the other texts because the
16-bit model runs partly on CPU on a 24 GB card). No stand-ins.

| Text | 16-bit PPL | Opti 27B vs 16-bit, paired | Q4_K_M vs 16-bit, paired |
|---|---|---|---|
| wikitext-2 test (encyclopedic) | 6.59 | +0.75% +/- 0.28% | +0.31% +/- 0.11% |
| C4 validation (web text) | 10.55 | +4.2% +/- 1.6% | +1.0% +/- 0.2% |
| Python standard library source (code) | 1.34 | +21.8% +/- 2.0% | +3.1% +/- 0.5% |
| Pride and Prejudice (Project Gutenberg) | 3.21 | +58.9% +/- 4.2% | +7.9% +/- 0.8% |

Read the last two rows for what they are. A perplexity of 1.34 on the Python standard library and 3.21 on a public-domain
novel means the 16-bit model has those texts close to memorized; those rows measure verbatim recall of training data, and
every compressed file loses some of it, 3.47 bits far more than 4.92. On text the model has not memorized (wikitext-2 test,
C4 web), Opti sits 0.75% to 4% above the 16-bit model where Q4_K_M sits 0.3% to 1% above it. If your use depends on
reproducing memorized text verbatim, use Q4_K_M. Per-window logs for every row are in the repository under `eval/`.

### KL divergence to the 16-bit logits

The perplexity table measures average log-likelihood. KL divergence measures how far the file's whole next-token distribution
sits from the 16-bit model's, token by token. All files below were scored against the same F16 logits on the same 64 windows
of wikitext-2 (`llama-perplexity --kl-divergence-base` / `--kl-divergence`).

| File | Size | PPL ratio vs F16 (paired) | Mean KLD | Median KLD | 99th pct KLD | Same top-1 token as F16 |
|---|---|---|---|---|---|---|
| llama.cpp Q4_K_M | 16.8 GB | 1.0031 +/- 0.0011 | 0.0156 | 0.0065 | 0.146 | 94.7% |
| llama.cpp Q3_K_M [1] | 13.5 GB | 1.0395 +/- 0.0021 | 0.0496 | 0.0221 | 0.504 | 90.4% |
| **Opti 27B** | 11.8 GB | **1.0075 +/- 0.0028** | **0.0935** | **0.043** | **0.876** | **87.2%** |

[1] The Q3_K_M row was measured on an L40S with a Q3_K_M file re-made from the ggml-org BF16 with llama.cpp `6a1a922d` and an importance matrix. The same setup reproduced the Q4_K_M row (mean KLD 0.0152 against 0.0156 above) and the Opti row (0.0933 against 0.0935). Its log is not in `eval/`.

Read the table for what it says: on average log-likelihood the file is within 0.75% of the 16-bit model, but its token
distribution diverges about six times more than Q4_K_M's and about 1.9 times more than Q3_K_M's, a larger file, and it picks the same top token as F16 on 87% of positions where
Q4_K_M does on 95% and Q3_K_M on 90%. Opti recovers the average far better than it recovers the per-token distribution. That is the same gap
the MMLU-Pro row and the memorized-text rows show from other angles, and it is the honest limit of the current file.

### MMLU-Pro

12,032 questions, answers scored by letter log-likelihood, no chain of thought, llama.cpp scorer.

Two scorers were used; rows are only comparable within a scorer.

| Model | MMLU-Pro | Scorer |
|---|---|---|
| Qwen3.8-27B, 16-bit original | 58.0 | A: Python harness, 12,032 items |
| Opti 27B (first release) | 54.6 | A |
| llama.cpp IQ3_XXS | 53.3 | A |
| llama.cpp IQ2_M | 53.2 | A |
| Qwen3.8-27B, 16-bit original | 57.95 +/- 0.45 | B: llama.cpp `--multiple-choice` |
| llama.cpp Q4_K_M | 57.14 +/- 0.45 | B |
| llama.cpp Q3_K_M | 56.50 +/- 0.45 | B |
| Opti 27B (first release) | 54.2 +/- 0.5 | B |
| **Opti 27B (this release)** | **53.9 +/- 0.5** | B |

Within scorer A, paired over all 12,032 items (McNemar): the 16-bit original is 3.3 points ahead of Opti's first release (p < 0.0001); Opti's first release is 1.3 points ahead of IQ3_XXS (p = 0.0005) and 1.5 ahead of IQ2_M (p < 0.0001). Within scorer B, this release and the first release tie, and this release is about 4 points below the 16-bit original, 3.2 below Q4_K_M and 2.6 below Q3_K_M (differences of the table values). A paired McNemar test on the same 12,032 items, run with a runtime-setting variant of this file that scored 54.03, found each of those three gaps significant (p < 1e-10). On scorer A, this file is above every llama.cpp file at or below its size; there is no scorer-B row for IQ3_XXS or IQ2_M. The scorer-B Q4_K_M and Q3_K_M rows use files re-made from the ggml-org BF16 with llama.cpp `6a1a922d` and an importance matrix, run on an L40S; the 16-bit row is the ggml-org BF16 file. Published Qwen numbers for MMLU-Pro use generated chain-of-thought answers and are not comparable to this protocol, so no comparison to them is made here.

### Unattended coding agent, pre-registered

Perplexity measures prediction. The question that matters is whether the model finishes the job. A pre-registered study put Opti's first release against llama.cpp IQ2_M, the best 2-bit llama.cpp file of the same model, each driving an unattended coding agent on the same task: a 430-line Python package with 7 hidden bugs and a missing feature, graded by 37 hidden tests. 100 runs on 10 GPUs, both models on every GPU.

| Outcome | Opti 27B (first release) | llama.cpp IQ2_M | p |
|---|---|---|---|
| Perfect solution (37/37 tests) | 40% | 12% | 0.0026 |
| Passed (34/37 or better) | 72% | 48% | 0.024 |

The within-GPU odds ratio for a perfect solution is 4.1: Opti is about four times as likely to deliver a complete solution. On the same servers, token-level perplexity on 568 tokens of Wikipedia was 6.03 for Opti against 6.29 for IQ2_M. The gap is visible at the token level and decisive at the task level.

### Demos: it ships games on its own

The current release, driving an unattended coding agent with no human in the loop:

- A playable Breakout, built in 27 steps and 6 minutes.
- A five-file Tetris with 7 pieces, rotation, line clears, score, level and line counters, a next-piece preview, pause, game over and a saved high score, built in 46 steps and 18 minutes.

Both were finished by the agent on its own. Demos available on request.

## What Opti is

Opti is a compression method, not a one-off file. It is patent-pending: U.S. Provisional Patent Application No. 64/154,967, filed September 15, 2026. The method and its training procedure are proprietary. This repository ships the model and projector files, plus the source-available runtime patch under `runtime/` (see `runtime/LICENSE`). The method is not documented here, and the license grants no rights to it.

The same method has been applied to language models from 0.5B to 27B parameters, to a mixture-of-experts model, and to a video diffusion model. At 7B it reached 1.07x the 16-bit perplexity at 3 bits per weight and 1.23x at 2 bits. Opti 27B is the flagship: the largest language model the method has been applied to, with vision, at the point where it sits within 0.5% of the 16-bit original at 3.47 bits per parameter.

The file runs on the Opti runtime, a source-available patch on llama.cpp shipped in this repository under `runtime/` (and mirrored on [GitHub](https://github.com/AutomationOptimization/opti-runtime)) that gives you `llama-server`, `llama-cli` and `llama-perplexity` with Opti support. Stock llama.cpp refuses the file. A hosted OpenAI-compatible API is available on request for those who would rather not run it themselves.

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `Opti-27B.gguf` | 11,836,978,976 (11.0 GiB, 11.8 GB) | `bb0a0481a9ba6b32f2eb2e3e36258366865ecd08483b3cc16fef0152ffd0d201` |
| `Opti-27B-mmproj-f16.gguf` | 927,606,944 | `4200f9a0d79e9f946e208a44c4fdd4533034d36fa90a3dc17159fa2a71e70954` |

`Opti-27B.gguf` is the model, 3.47 bits per parameter over the whole file. `Opti-27B-mmproj-f16.gguf` is the vision projector; load both for image input.

## Running it

Both commands need the Opti runtime, which is included in this repository under `runtime/` (the llama.cpp patch, `build.sh`, its own README and license) and mirrored at [github.com/AutomationOptimization/opti-runtime](https://github.com/AutomationOptimization/opti-runtime). Building it is one step:

```
cd runtime && ./build.sh            # Metal on Apple Silicon, CPU elsewhere
cd runtime && ./build.sh cuda 86    # NVIDIA; 86 = RTX 3090, 89 = RTX 4090 / L40S, 90 = H100
```

That clones llama.cpp at the pinned commit, applies the patch and builds `llama-server`, `llama-cli` and `llama-perplexity`.

Serve it, with vision and reasoning on:

```
llama-server -m Opti-27B.gguf --mmproj Opti-27B-mmproj-f16.gguf -ngl 999 -c 16384 --reasoning on
```

Reproduce the Opti row of the perplexity table:

```
llama-perplexity -m Opti-27B.gguf -f wiki.test.raw -c 1024 --chunks 96 -ngl 999
```

Speed and memory, measured on an RTX 3090 (24 GB), same build and flags for every row, single stream:

| File | Decode, 512 tokens | Prompt processing, 7k-token prompt |
|---|---|---|
| **Opti 27B** | **42.3 tok/s** | **1,127 tok/s** |
| llama.cpp Q4_K_M | 42.3 tok/s | 1,288 tok/s |
| llama.cpp IQ3_XXS | 47.5 tok/s | 1,307 tok/s |

Opti decodes at the same speed as the 4-bit standard in 30% fewer bytes; the extra work per block costs about 11% against IQ3_XXS on decode and
about 12% on prompt processing.

| Workload | Memory used |
|---|---|
| Four concurrent 16k-token conversations, model + projector | 17 GB |
| One 65k-token conversation with vision | 15 GB |

The model fits a single 16 GB consumer GPU with room for context. A 24 GB GPU runs four concurrent conversations with vision.

## Hosted API and pricing

If you would rather not run it yourself, the same model is served behind an OpenAI-compatible API. Subscribe, and your API
key and endpoint URL arrive by email within 24 hours. Any OpenAI SDK works with `base_url` set to the endpoint and
`model` set to `opti`. Vision input and reasoning are on.

| Plan | Price | Included tokens per month | Concurrent requests |
|---|---|---|---|
| [Basic](https://buy.stripe.com/eVq28rbZRa4f3vo8MR8k800) | $15 / month | 7,500,000 | 2 |
| [Pro](https://buy.stripe.com/00w8wP1ld0tF3vo1kp8k801) | $60 / month | 30,000,000 | 4 |
| [Business](https://buy.stripe.com/6oU6oH4xp6S37LEfbf8k802) | $1,000 / month | 500,000,000 | 8 |

That is $2 per million tokens on every plan. Want to try it first? Ask for an evaluation key through the Hugging Face
profile that hosts this repository: the first million tokens are free.


## Attribution

Opti 27B is built on [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team at Alibaba Cloud, released under the Apache License 2.0. That license and its notices continue to apply to the original weights, which remain available from their authors. Opti's compressed form of those weights is subject to `LICENSE`.

I'm available for contract work on quantization, llama.cpp/MLX deployment and eval harnesses, email: kcadenhead@automationoptimization.com