---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - moe
  - imatrix
  - llama.cpp
  - lfm2.5
---

# LFM2.5-24B-A2B: imatrices and low-bit GGUF experiments

Calibration assets and low-bit experiments for
[LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B). The model is a 24B-total,
2B-active MoE that runs on stock llama.cpp.

For agentic work, use the largest quant you can comfortably fit or use Liquid's own
[QAD-Q4_0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-QAD-Q4_0.gguf) (13.58 GB) or a [Q8_0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q8_0.gguf). 

The IQ3_XXS here clears the benchmarks, but it needs about 2x the tokens and loops several times per run (see below).
Going below 3 bits made it unusable for agentic work.

## Files

| file | what |
|---|---|
| `LFM2.5-24B-A2B-codeweighted.imatrix` | imatrix from the BF16 weights: 296 chunks of a ≈70% code corpus, `--parse-special`, every expert covered |
| `LFM2.5-24B-A2B-QAD-codeweighted.imatrix` | the same corpus, computed on Liquid's QAD-Q4_0 weights (for requantizing from that source) |
| `LFM2.5-24B-A2B-BF16src-IQ3_XXS.gguf` | stock IQ3_XXS from BF16 with the imatrix above (9.28 GB). **Not recommended for agentic use** |
| `sensitivity/LFM2.5-24B-A2B.sensitivity.json` | measured per-tensor KL sensitivity (GB10). The values are tied to the device that produced them, so only the rankings transfer |

## Quality vs BF16

The code corpus is a mixed code-heavy set (the first 100 chunks; the same tokens for every row). Wikitext is
`wiki.test.raw`.

| quant | size | code KLD vs BF16 | code same-top | code PPL vs BF16 | wikitext PPL |
|---|---|---:|---:|---:|---:|
| [BF16](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) | 47.96 GB | | | | 15.998 |
| [Q8_0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q8_0.gguf) (plain) | 25.49 GB | 0.009 | 97.2% | -0.05% | |
| [QAD-Q4_0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-QAD-Q4_0.gguf) (Liquid's) | 13.58 GB | 0.037 | 94.2% | -1.8% | 14.457 |
| plain Q4_0 (BF16 + imatrix, not published) | 13.62 GB | 0.053 | 93.2% | +1.1% | 16.287 |
| **IQ3_XXS (this repo)** | 9.28 GB | 0.136 | 88.8% | +5.9% | |

The Q8_0 rows are a plain conversion made here; the link goes to Liquid's published Q8_0 of the same weights.

QAD's PPL is lower than BF16's mostly through calibration rather than through different choices: it picks
the same top-1 tokens with less confidence. Mean KLD vs BF16 therefore slightly under-credits it.

## Agentic benchmark (OpenCode 2, 42 tasks)

The harness is quantkit `bench/` with OpenCode 2's real agentic loop, graded by a fresh pytest. It ran on
a DGX Spark (GB10) with llama.cpp `6d0549831`, ctx 131072 and an output cap of 32768.

A full score is the entry bar. Between quants that clear it, the discriminating number is **output tokens on
the tasks every run passed**: a weaker quant burns tokens to reach the same result.

| quant | runs | fully passed | looping runs per rep | output tokens on the 31 tasks all runs passed |
|---|---|---|---|---|
| [Q8_0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q8_0.gguf) | 2 | 39/42, 31/33 | 0, 0 | 104.6k, 89.4k |
| [QAD-Q4_0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-QAD-Q4_0.gguf) | 4 | 40, 39, 40, 37 /42 | 2, 0, 2, 3 | 106.5k, 122.6k, 117.6k, 120.5k |
| **IQ3_XXS (this repo)** | 2 | 36/42, 39/42 | 5, 4 | 199.7k, 216.4k |
| IQ2_XXS (not published) | 1 (stopped) | 0/5 | loops / hit the 32k cap | |

- QAD-Q4_0 spends about 20% more tokens than Q8_0 for the same results, at half the size.
- IQ3_XXS spends about 2.1x Q8_0's tokens and loops 4-5 times per run.
  - It is about 1/3 of Q8_0's size, but it does not behave almost the same. That is the bar for publishing a
    quant here, so this file stays unrecommended.
- IQ2_XS and IQ2_XXS were built and dropped: both loop on agentic tasks, so they are not published.
- Q8_0 rep 2 is missing its frontier suite (a harness bug, since fixed), so the shared set excludes frontier tasks.

## APEX quant tiers (new — 2026-10)

Added four APEX quantized builds of the BF16 source (all F32-pinned routers, `exp_probs_b`,
the Mamba2/SSM alpha-Dt-bias-conv-norm, and every norm/bias), built from the **code-weighted
imatrix** (band-serialized; `serialized_gen_nemotron.py`), not a borrowed corpus. All four
survived the real-generation coherence check and the chained 3-turn tool-call gate (3/3 each).

| tier | size | bpw | KLD (code) | same-top | PPL | use case |
|---|---:|---:|---:|---:|---:|---|
| [`…-APEX-quality.gguf`](https://huggingface.co/Myric/LFM2.5-24B-A2B-APEX-GGUF/blob/main/LFM2.5-24B-A2B-APEX-quality.gguf) | 14.0 GB | ~5.5 | **0.0351** | 94.16% | 16.79 | maximum quality without BF16 |
| [`…-APEX-compact.gguf`](https://huggingface.co/Myric/LFM2.5-24B-A2B-APEX-GGUF/blob/main/LFM2.5-24B-A2B-APEX-compact.gguf) | 12.0 GB | ~4.5 | **0.0427** | 93.76% | 18.06 | balanced tier, daily driver / multi-box |
| [`…-APEX-subcompact.gguf`](https://huggingface.co/Myric/LFM2.5-24B-A2B-APEX-GGUF/blob/main/LFM2.5-24B-A2B-APEX-subcompact.gguf) | 11.0 GB | ~4.0 | **0.0816** | 91.41% | 18.68 | smaller memory pressure; no per-task headline yet (clean-pass rates not measured) |
| [`…-APEX-mini.gguf`](https://huggingface.co/Myric/LFM2.5-24B-A2B-APEX-GGUF/blob/main/LFM2.5-24B-A2B-APEX-mini.gguf) | 8.8 GB | ~3.2 | **0.1205** | 89.68% | 18.13 | edge / phone / laptop low-capacity boxes |

Reference PPL (BF16): 15.998 on the same code-mixed corpus; rows above are the 64/512-chunk
KLD / same-top / PPL measured against KLD reference logits over the code-weighted corpus.

Note this repo previously shipped a single IQ3_XXS BF16-source quant. The APEX ladder does not
replace it — it gives you four iso-modality stops (14 / 12 / 11 / 8.8 GB) derived from the
same BF16 file with the same imatrix, so they differ only in target budget.

### Tool-calling gate (all four pass 3/3)

The example path: serve the GGUS with `--jinja` (the GGUF carries its own template).

```bash
llama-server -m <tier>.gguf --jinja --ctx-size 8192 -fa on --temp 0.6 --top-p 0.9   --repeat-penalty 1.05 --host 127.0.0.1 --port 8092
```

System prompt used:

```
You are a helpful assistant with access to tools. When a tool is needed, call it; when it
returns a result, answer the user directly using that result.
```

All four tiers handled a chained 3-turn tool conversation (`get_user_location` → `get_weather`
with the resulting Tokyo location → plain-language final answer) in 3/3 trials, and never
called the distractor `search_web` tool. See `TOOL_CALLING.md` in the corresponding run dir
for captured traces.



## Attribution

- Base model: **Liquid AI**, [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B),
  under the LFM Open License v1.0.
- Quantization engine: **llama.cpp** ([ggml-org](https://github.com/ggml-org/llama.cpp)).

This is an unofficial community quantization, not affiliated with or endorsed by Liquid AI.
