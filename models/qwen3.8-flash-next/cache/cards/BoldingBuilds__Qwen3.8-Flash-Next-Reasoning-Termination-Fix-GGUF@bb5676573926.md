---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: gguf
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- qwen4exp
- reasoning
- thinking
---

<!-- REV 2 2026-09-28 PT: external review (exploratory p-values, links, reproduction limits, PR commit wording). REV 1: Round cb2, shard-2 sha 6d17a23f...0439. Every number re-read from the capture files and
     reproduced by evidence/export_evidence_fn.py. -->

# Qwen3.8 Flash Next Reasoning Termination Fix (UD-IQ4_XS GGUF)

**Fewer empty answers from Qwen3.8 Flash Next within a 4,096-token output budget.** This GGUF changes
only the output-layer row for `</think>`. On a 200-problem MATH-500 subset, empty final responses
fell from **29 to 0** and correct answers rose from **150 to 168**. llama.cpp's `--reasoning-budget`
flag scored similarly (172), while the weight edit needs no runtime budget control. Occasional stray
`</think>` tags remain.

This is the same method as
[Qwen3.8-27B-Reasoning-Termination-Fix-GGUF](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF),
applied to a different architecture.

| 200 MATH-500 problems, 4,096 output tokens | correct | empty final response |
|---|---:|---:|
| **this model** | **168** | **0** |
| stock | 150 | 29 |
| stock + `--reasoning-budget 2048` | 172 | 0 |

Paired over the same problems, this model solved 25 that stock missed and missed 7 that stock
solved (exact McNemar p = 0.002). All 29 of stock's empty responses were the model still thinking
when the 4,096 tokens ran out. Every p-value on this card is exploratory and unadjusted (see
Statistics below).

**Compared with the flag.** The 2,048 flag scored 172 against 168 for this model (8 vs 4 on the
problems where they differ, p = 0.39). The flag is slightly ahead, but the difference is not
statistically significant. The point of this build is that it needs no flag, so it should behave
the same in apps and APIs that do not expose one. That is expected, not tested: only llama.cpp was
tested (see Compatibility). Larger output budgets were not tested for this model; on the 27B
version the flag did better at 16,384 tokens.

**September 29 update:** Clarified that agent/tool use and speculative decoding were not tested
for this release. I have not published a medium-effort comparison for Flash-Next. The
[earlier medium-effort comparison on 27B](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF#earlier-development-comparison-medium-reasoning-effort)
is a separate development experiment and does not establish a result for Flash-Next.
The released weights and results are unchanged.

## Files

| file | size |
|---|---:|
| `Qwen3.8-Flash-Next-Reasoning-Termination-Fix-UD-IQ4_XS-00001-of-00003.gguf` | 10.9 MB |
| `Qwen3.8-Flash-Next-Reasoning-Termination-Fix-UD-IQ4_XS-00002-of-00003.gguf` | 49.84 GB |
| `Qwen3.8-Flash-Next-Reasoning-Termination-Fix-UD-IQ4_XS-00003-of-00003.gguf` | 43.84 GB |

Point llama.cpp at the first file; it loads the other two. SHA-256 of every shard is in
[`evidence/hashes.txt`](https://huggingface.co/BoldingBuilds/Qwen3.8-Flash-Next-Reasoning-Termination-Fix-GGUF/blob/main/evidence/hashes.txt). Only shard 2 differs from the stock files.

Language model only. The vision encoder is not included and vision was not tested.

## Usage

```sh
llama-server -m Qwen3.8-Flash-Next-Reasoning-Termination-Fix-UD-IQ4_XS-00001-of-00003.gguf -ngl 99 -fa on -c 32768 --jinja
```

This command sets no sampler options, so llama.cpp uses the defaults stored in the file
(temperature 1.0, top-k 20, top-p 0.95) plus its own min-p default of 0.05. The evals sent no
sampler settings either, so they ran on these defaults. (The eval build's source applies them, and a
mainline server loading these exact files reported the same values.) `-c` is the context window, not the output budget. The
budget is the `max_tokens` your client sends; every result here uses 4,096. The model is about
94 GB, so most setups will need to keep part of it in system RAM (for example `--n-cpu-moe`).

## Compatibility

- **llama.cpp, tested.** Results were produced with llama.cpp built from commit 035e227 on the branch
  of pull request #27742 (which adds this architecture; the pull request was merged later with more
  commits) plus local instrumentation commits that only activate through an environment variable
  the evals did not set. These exact files were also
  loaded on mainline llama.cpp (ggml-org master, commit 5262471, 2026-09-28) with the experts in
  system RAM (`--cpu-moe`) on one RTX 3090: it answered all 30 problems of a 30-problem MATH check
  (26 correct, no stray tags), and the server reported the file's sampler defaults.
- **LM Studio, Ollama and other apps: untested.** They need a llama.cpp recent enough for this
  architecture. The fix needs no settings, so the benefit should carry over, but I have not checked.

## What changed from stock

| | |
|---|---|
| edited | 1 row of `output.weight` in shard 2 (the `</think>` token, id 248069, Q6_K): 2,084 of its 2,100 bytes changed, 0 bytes changed anywhere else (full byte comparison of the shard) |
| unchanged | shards 1 and 3 are byte-identical to [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) UD-IQ4_XS (revision 8bdc6666), and so is every other byte of shard 2 |

**How the fix works.** Only one output-layer row was fitted; no other weights were trained. The
change is a fitted delta to the `</think>` row, intended to improve when the model ends its
reasoning. I fit it on the model's own final hidden states so the `</think>` score rises once the
model is ready to answer or is going in circles, and stays low mid-thought and inside the answer.
It was fit in two rounds: the second added places where the first version closed its reasoning too
late, wrote a stray `</think>` into its answer, or ended its turn inside its thinking with no reply.

**Fitting data vs test data.** The fix was fit on 469 prompts that are not in any test here: 150
MATH training problems, 80 MBPP training problems, 40 Alpaca instructions and 199 harmful requests
(AdvBench, JailbreakBench), the same set as the 27B version with its escaping defect fixed. Any
prompt that matched or closely resembled (word overlap of 50% or more) a prompt from any test here
was removed first. Only the released round was run on the test sets, so no candidate was chosen by
looking at test results. The method itself was developed on the 27B model using these same test
sets.

**One measurement quirk to disclose.** Before fitting, I check that a standalone computation of the
`</think>` score from saved hidden states reproduces the server. For this model it matched to a
median error of 0.0001 nats, but 8 of 36 check points were off by 0.1 to 1.1 nats, apparently
because this architecture's hidden states depend slightly on how requests are batched. The fit may
therefore be a little noisier than on the 27B. The live test results above are unaffected; they are
measured directly.

## Other results

**HumanEval (164 problems), pass@1, 4,096 output tokens:**

| build | passed | empty final response |
|---|---:|---:|
| **this model** | **155** | **0** |
| stock | 151 | 11 |

This model vs stock: 6 vs 2 on the problems where they differ (p = 0.29). That leans toward this
model but is not statistically significant.

**Refusals: aggregate rates stayed close to stock.** Same judge and prompt for every row:

| | stock | this model |
|---|---:|---:|
| StrongREJECT (first 150), judged a refusal | 143 (95.3%) | 143 (95.3%) |
| SimpleSafetyTests (100), judged a refusal | 85 | 84 |
| XSTest safe prompts (first 100), full compliance | 88 | 92 |
| XSTest safe prompts, partial refusal | 7 | 7 |
| XSTest safe prompts, full refusal | 1 | 1 |
| XSTest safe prompts, empty (ran out of tokens) | 4 | 0 |
| empty reply that ended the turn, 250 harmful prompts | 1 | 2 |
| empty reply that ran out of tokens, 250 harmful prompts | 1 | 0 |

These are aggregate rates. Individual prompts moved in both directions; per-prompt labels are in
[`evidence/per_item/refusals_judged.csv`](https://huggingface.co/BoldingBuilds/Qwen3.8-Flash-Next-Reasoning-Termination-Fix-GGUF/blob/main/evidence/per_item/refusals_judged.csv).

**Stray `</think>` tags.** This model wrote a stray `</think>` into its final answer in 1 of 200
MATH answers and 1 of 164 HumanEval answers, and in none of the refusal answers. No stray tags were
observed in the tested stock runs.

## How this was tested

- Server: llama-server with `-ngl 99 --flash-attn on --jinja --override-tensor
  per_layer_token_embd.weight=CPU`, 16 requests in parallel (MATH: 5,120 tokens per slot; other
  sets: 6,144), one rented H200 per server. Each request sends the messages, `max_tokens` (4,096)
  and `seed: 0`, nothing else. MATH and refusal prompts are a single user message (MATH adds "Put
  your final answer in \boxed{}."); HumanEval adds a short system prompt asking for code only.
- Thinking on, at the chat template's default reasoning effort (`xhigh`).
- Sampler: the file's defaults as above. Sampling at temperature 1.0 with parallel slots is not
  bit-reproducible run to run.
- **Empty final response** means the reply after the reasoning was empty. It has two causes, counted
  separately in [`evidence/summary_counts.csv`](https://huggingface.co/BoldingBuilds/Qwen3.8-Flash-Next-Reasoning-Termination-Fix-GGUF/blob/main/evidence/summary_counts.csv): the output budget ran out during reasoning
  (`finish_reason: length`), or the model ended its turn with no answer (`finish_reason: stop`).
- MATH-500: the first 200 problems after a seeded shuffle (seed 7), scored by exact match of the
  last `\boxed{}` answer after normalisation. The scorer is strict, so compare rows to each other,
  not to published numbers.
- HumanEval: all 164 problems, one sample each; the code block is run against each problem's tests.
- Refusals: StrongREJECT (first 150), SimpleSafetyTests (all 100), XSTest safe prompts (first 100),
  graded by an LLM judge that is not the model under test: a Q8_0 file from
  [OBLITERATUS/Qwen3.8-27B-OBLITERATED](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED),
  thinking off, temperature 0, JSON-schema output, StrongREJECT rubric for harmful sets and the
  XSTest three-way labels for safe prompts. Same judge, prompt and settings as the 27B card.
- Statistics: exact two-sided McNemar test on paired per-problem results. The p-values are
  exploratory and not adjusted for multiple comparisons. No candidate was chosen by looking at
  these test results (only the released round was run on them), but the method itself was
  developed on the 27B model using these same test sets.

## Evidence

[`evidence/`](https://huggingface.co/BoldingBuilds/Qwen3.8-Flash-Next-Reasoning-Termination-Fix-GGUF/tree/main/evidence) holds per-problem results for every row (IDs, scores and flags only, no model
text); [`summary_counts.csv`](https://huggingface.co/BoldingBuilds/Qwen3.8-Flash-Next-Reasoning-Termination-Fix-GGUF/blob/main/evidence/summary_counts.csv) with every count on the card;
[`hashes.txt`](https://huggingface.co/BoldingBuilds/Qwen3.8-Flash-Next-Reasoning-Termination-Fix-GGUF/blob/main/evidence/hashes.txt) with SHA-256 of every shard, the judge, datasets and scripts; the
scripts and settings as run; judge labels by prompt index; the fitting-prompt manifest (source and
SHA-256); the bake and verification logs; and the rebuild log showing the release shard reproduces
the tested one byte for byte.

It is enough to check every number and to rerun the evals (the refusal sets after rebuilding them
from the public datasets), but not everything used is published.
Not included: the text of the fitting and refusal test prompts (many are harmful requests; all come
from public datasets, listed with the selection rules), the model's own generations used for
fitting, and six local instrumentation commits in the eval build (inactive in the evals). The
[evidence README](https://huggingface.co/BoldingBuilds/Qwen3.8-Flash-Next-Reasoning-Termination-Fix-GGUF/blob/main/evidence/README.md) lists what the setup script expects, where each public piece
comes from, and the one source patch needed.

## Limitations

- All results are at a 4,096-token output budget. Larger budgets were not tested on this model.
- The fix makes the model stop thinking sooner. Overall scores went up, but not on every problem:
  it missed 7 MATH problems and 2 HumanEval problems that stock solved, and its safety labels
  differ from stock on individual prompts.
- Occasional stray `</think>` tags in answers: 1 of 200 MATH answers and 1 of 164 HumanEval answers.
- Tested in English, text only, single turn. Agent, tool use and multi-turn use were not tested.
- Speculative decoding with a draft model was not tested for the released file; its effect on draft acceptance and speedup is unknown.
- One quant (UD-IQ4_XS). One seed per configuration.

## Related work

See the [27B card](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF)
for the full list: run-time budget control, hidden-state probes for early exit
([Zhang et al. 2025](https://arxiv.org/abs/2504.05419), [LYNX](https://arxiv.org/abs/2512.05325)),
stopping rules on the `</think>` score ([ThinkBrake](https://arxiv.org/abs/2510.00546)), weight
edits that change reasoning length ([ThinkEdit](https://arxiv.org/abs/2503.22048),
[Steer2Edit](https://arxiv.org/abs/2602.09870)), and the analysis of the `</think>` score in
[Sheng et al. 2025](https://arxiv.org/abs/2506.08390).

## License and credits

Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), under the Qwen
Community License 1.0, included here as `LICENSE`. Quantization: Unsloth's UD-IQ4_XS
([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)).
Test sets: MATH (MATH-500), HumanEval, StrongREJECT, SimpleSafetyTests, XSTest. Fitting prompts:
MATH (train), MBPP (train), Alpaca, AdvBench, JailbreakBench.
