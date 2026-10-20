---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
- gguf
- llama.cpp
- imatrix
- lfm2
- moe
- tool-calling
---

# LFM2.5-24B-A2B GGUF

![Top-1 token agreement with BF16 against file size, for these files and the other public GGUFs of this model](chart.webp)

GGUF quantizations of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), from 13.30 GB. The importance matrix is calibrated on **tool-calling conversations rendered through the model's own chat template**, because this model exists to call tools inside agent harnesses. Every file here, and the other public GGUFs of this model, was measured on the same harness: closeness to BF16 on text, and correctness on 300 BFCL function-calling tasks.

This is an unofficial community quantization. It is not affiliated with or endorsed by Liquid AI.

## Files

| File | Size | Bits per weight | KLD vs BF16 ↓ | Top-1 vs BF16 ↑ | BFCL correct ↑ |
|---|---|---|---|---|---|
| `LFM2.5-24B-A2B-Q5_K_M.gguf` | 17.37 GB | 5.80 | 0.039 | 91.3% | 77.0%\* |
| `LFM2.5-24B-A2B-Q4_K_M.gguf` | 14.95 GB | 4.99 | 0.066 | 88.7% | 78.0% |
| `LFM2.5-24B-A2B-IQ4_XS.gguf` | 13.30 GB | 4.44 | 0.074 | 87.8% | 75.7% |
| BF16, for reference | 47.96 GB | 16.0 | 0 | 100% | 72.7% |

Sizes are decimal GB (10^9 bytes), the unit the file listing uses. \*Q5_K_M was scored on 178 of the 300 tasks; on those same 178, BF16 scores 75.3%.

- **`Q4_K_M`** if it fits. It is the balanced choice.
- **`IQ4_XS`** when every gigabyte counts. It is the smallest file here and stays close to BF16.
- **`Q5_K_M`** if you have room; it is the closest to BF16 of any file measured.

`imatrix.gguf` is the importance matrix all three were built with.

## Compared with the other GGUFs of this model

All measured on one harness: same text, same tasks, same GPU, BOS handled identically for every file (see [BOS](#bos)).

| File | Size | KLD ↓ | Top-1 ↑ | Same tool call as BF16 ↑ | BFCL correct ↑ |
|---|---|---|---|---|---|
| **this repo, Q5_K_M** | 17.37 GB | **0.039** | **91.3%** | 82.6%\* | 77.0%\* |
| AtomicChat AD-Q4_K_M | 15.36 GB | 0.059 | 89.4% | 81.0% | 73.7% |
| **this repo, Q4_K_M** | 14.95 GB | 0.066 | 88.7% | 81.0% | 78.0% |
| unsloth UD-Q4_K_XL | 14.88 GB | 0.062 | 89.1% | 82.0% | 73.0% |
| LiquidAI Q4_K_M | 14.53 GB | 0.132 | 84.1% | 77.0% | 77.0% |
| AtomicChat AD-IQ4_XS | 14.50 GB | 0.068 | 88.5% | 80.3% | 73.0% |
| LiquidAI QAD-Q4_0 † | 13.58 GB | 0.213 | 80.2% | 75.3% | 67.0% |
| **this repo, IQ4_XS** | 13.30 GB | 0.074 | 87.8% | 77.0% | 75.7% |
| BF16 | 47.96 GB | 0 | 100% | 100% | 72.7% |

\* 178 of 300 tasks. † QAD-Q4_0 has retrained weights (quantization-aware distillation), so it deliberately differs from BF16: its KLD and top-1 measure that distance rather than quality.

**`IQ4_XS` is 1.2 GB smaller than the official `Q4_K_M` and sits at roughly half its KLD (0.074 against 0.132), with the same tool-call agreement** — both reproduce BF16's call on 77.0% of tasks, and where they differ, each gets 27 that the other misses.

**On tool calls, no standard 4-to-5-bit file loses accuracy against BF16.** The BFCL scores spread from 72.7% to 78.0% with BF16 at the bottom, which is not quantization helping: about 30% of the tasks flip between right and wrong from any small numerical difference, because the model reasons before it calls and greedy decoding follows a different path. Every run, BF16's included, is one draw. Differences between the files in that column are within this run-to-run spread (standard deviation 2.2 points), so read it as "all hold BF16's accuracy". The full per-category breakdown is in `comparison.csv`.

**"Same tool call as BF16" around 80% does not mean 20% wrong calls.** Calls that differ from BF16's are, on balance, as often correct as BF16's own.

## BOS

The current llama.cpp converter writes `tokenizer.ggml.add_bos_token = false` for this model, while llama.cpp's default for its tokenizer is `true`. With `false`, raw-text prompts start without BOS and the model degrades: BF16 measures **12.19 perplexity on wikitext-2 with BOS and 18.43 without**. Chat through the template is unaffected either way, since the template writes its own BOS.

**These files carry `add_bos_token = true`**, set after quantization by changing that one metadata byte. They behave exactly as they were measured.

GGUFs of this model on the Hub currently disagree on this key, some writing `false` and some omitting it, so the same text reaches different files as different tokens. Every measurement above forces BOS on with `--override-kv tokenizer.ggml.add_bos_token=bool:true`. Left unforced, a file that adds BOS compared against a reference that does not shows several times its real KLD.

## Running

```bash
llama-server -hf m9ter1a/LFM2.5-24B-A2B-GGUF:Q4_K_M --jinja -c 32768
```

`--jinja` applies the model's chat template, which tool calling needs. The model reasons inside `<think>` before answering and writes tool calls as `<|tool_call_start|>[name(arg=value)]<|tool_call_end|>`.

With less VRAM than the file, keep some expert layers on the CPU with `--n-cpu-moe N` (or all of them with `--cpu-moe`). Only 2B of the 24B parameters are active per token, which is what makes this split workable.

## How these were made

- **Converted** from the BF16 safetensors with llama.cpp `convert_hf_to_gguf.py` (commit `45451c3`).
- **Imatrix** from 155 tool-calling conversations from [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) (Apache-2.0), converted to structured tool calls and rendered through the LFM chat template so the matrix sees the tokens the model actually emits, plus wikitext-2 train prose: about 260k tokens, 509 chunks of 512. None of it overlaps the BFCL evaluation tasks.
- **Recipe.** 95.7% of the parameters are in the 64 experts per layer; they take the base type. The dense remainder — short-convolution projections, attention, the two dense FFN layers and the token embedding, about 4% of parameters — is held at Q8_0, because every token passes through it while each expert sees only some tokens. The router stays F32.

## How these were measured

- **KLD and top-1:** `llama-perplexity --kl-divergence` on wikitext-2-raw test, 40 chunks of 2048 tokens, against the BF16 GGUF, on one A100 80GB with every layer on the GPU.
- **Tool calls:** 300 single-turn tasks from [BFCL v3](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) (Apache-2.0) — simple 50, multiple 50, parallel 40, parallel-multiple 40, live simple 40, live multiple 40, and 40 irrelevance tasks where the right answer is not to call. Prompts rendered once with the official chat template so every file gets byte-identical input; greedy decoding through `llama-server` with one slot per server. A repeat BF16 run on 30 of the tasks reproduced 100% of the output tokens, so the bench is deterministic.
  - *Same tool call as BF16*: identical function names and arguments, parallel calls compared as a set.
  - *BFCL correct*: against BFCL's ground-truth answers with a simplified re-implementation of its AST checker (names, required and unexpected parameters, accepted values; strings normalised as BFCL does). It is not the official evaluator.

Everything needed to check these numbers is in this repository: `comparison.csv` (every metric, with confidence intervals and per-category scores), the raw outputs in `results/`, and the scripts in `scripts/`.

## Licence and credits

The model is Liquid AI's. These files are quantized derivatives of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B); `NOTICE` lists exactly what was changed.

Released under the **LFM Open License v1.0** (`lfm1.0`), included as `LICENSE`. Commercial use is permitted only below the Threshold in section 5 of that licence: **annual revenue under $10,000,000**. The licence grants no rights to Liquid AI's trademarks; the name is used here only to describe where these files come from.

Quantized with [llama.cpp](https://github.com/ggml-org/llama.cpp).
