---
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
- qwen3_5
- reasoning
- thinking
---

<!-- REV 5 2026-09-29 CT: Q5_K_M, Q6_K and Q8_0 files added with their own stock-vs-fix tests.
     REV 4 2026-09-29 CT: reasoning effort section (medium vs this model, two seeds), usage note, limitations.
     REV 3 2026-09-28 PT: second external review (stray-tag rates, answer-in-thinking wording, exploratory p-values,
     links, reproduction limits, LM Studio check). REV 2: corrections from external review. Every number re-read from the capture files and
     reproduced by evidence/export_evidence.py. Round 6 file, sha 2da7bb45...844b. -->

# Qwen3.8 27B Reasoning Termination Fix (GGUF)

**Fewer empty answers from Qwen3.8 27B within a 4,096-token output budget.** This GGUF changes
only the output-layer row for `</think>`. On a 200-problem MATH-500 subset, empty final responses
fell from **31 to 0** and correct answers rose from **150 to 169**. Accuracy was similar to stock
with llama.cpp's `--reasoning-budget` flag, while the weight edit needs no runtime budget control.
Occasional stray `</think>` tags remain.

**September 29 update:** Added a [reasoning effort comparison](#the-medium-reasoning-effort-setting-compared-with-this-model)
run under the release conditions (stock and this model, at the template default and at medium, two seeds each),
and the [earlier development comparison](#earlier-development-comparison-medium-reasoning-effort) it
supersedes. Clarified untested use cases. Later the same day: added Q5_K_M, Q6_K and Q8_0 files, each
tested against its own stock quant (see [Other quantizations](#other-quantizations)). The Q4_K_M file and
the main release results are unchanged.

| 200 MATH-500 problems, 4,096 output tokens | correct | empty final response |
|---|---:|---:|
| **this model** | **169** | **0** |
| stock | 150 | 31 |
| stock + `--reasoning-budget 2048` | 167 | 0 |
| stock + `--reasoning-budget 3072` | 166 | 0 |

Paired over the same problems, this model solved 21 that stock missed and missed 2 that stock
solved (exact McNemar p = 0.0001). All 31 of stock's empty responses were the model still thinking
when the 4,096 tokens ran out. Every p-value on this card is exploratory and unadjusted, including
for the fact that this version was chosen from several candidates using these same benchmarks (see
below).

**Compared with the flag.** Against `--reasoning-budget 2048` it scored 169 vs 167 (9 vs 7 on the
problems where they differ, p = 0.80), and against 3072 it scored 169 vs 166 (9 vs 6, p = 0.61).
Neither difference is statistically significant. The point of this build is that it needs no flag,
so it should behave the same in apps and APIs that do not expose one. Beyond llama.cpp, only a small
LM Studio check was run (see Compatibility); other apps were not tested.

**With a larger budget the flag did better.** At 16,384 output tokens, on the first 100 of the
same problems:

| 100 MATH-500 problems, 16,384 output tokens | correct | empty final response | stray `</think>` |
|---|---:|---:|---:|
| stock + `--reasoning-budget 2048` | 88 | 0 | 5 |
| **this model** | **84** | **0** | **4** |
| stock | 83 | 5 | 0 |

The flag solved 4 problems this model missed and this model solved none the flag missed. That is
not statistically significant (p = 0.13), but it points one way, so if you run llama.cpp yourself
with a large budget, the flag is the better tool. Combining them did not help either: this model
plus the 2,048 flag scored 163 of 200 at 4,096 tokens, against 169 for this model alone
(11 vs 5, p = 0.21).

## The medium reasoning effort setting, compared with this model

**Added September 29, 2026.** Qwen3.8's chat template takes a `reasoning_effort` argument (`low`,
`medium`, `xhigh`; `high` is treated as `xhigh`). If nothing is sent, the template uses `xhigh`, and
every other row on this card was measured that way. Setting `medium` changes the instructions the
template puts in front of the model's thinking, so it is a different lever from the
`--reasoning-budget` token cutoff compared above. Several people have pointed out that medium
alone removes most empty answers, so here it is measured under the same conditions as the rest of
the card: same 200 MATH-500 problems, same 4,096-token budget, same machine and llama.cpp build, 8
parallel requests, file sampler defaults. Because sampling at temperature 1.0 moves a few problems
from run to run, every cell was run twice, with request seed 0 and seed 1.

| 200 MATH-500 problems, 4,096 output tokens | correct, seed 0 | correct, seed 1 | empty, seed 0 | empty, seed 1 | stray `</think>`, seed 0 | stray `</think>`, seed 1 |
|---|---:|---:|---:|---:|---:|---:|
| stock, template default | 150 | 148 | 31 | 32 | 0 | 0 |
| stock, `reasoning_effort: medium` | 156 | 155 | 14 | 12 | 0 | 0 |
| **this model, template default** | **169** | **168** | **0** | **0** | 2 | 6 |
| this model, `reasoning_effort: medium` | 158 | 159 | 7 | 1 | 6 | 1 |

The seed 0 stock and this-model default rows are the same runs as at the top of this card. Every
empty answer in the table is the model still thinking when the budget ran out; none ended the turn
with no answer.

What the pairs say, on the same problems (exact McNemar, seed 0 then seed 1):

- **Medium helps stock, but not by much.** 156 vs 150 (16 vs 10, p = 0.33) and 155 vs 148 (17 vs
  10, p = 0.25). It cuts the empty answers by more than half and the rest of the gain is within
  run-to-run noise.
- **This model at the default beat stock at medium.** 169 vs 156 (20 vs 7, p = 0.019) and 168 vs
  155 (19 vs 6, p = 0.015).
- **Do not set medium on top of this model.** 158 vs 169 (9 vs 20, p = 0.06) and 159 vs 168 (10 vs
  19, p = 0.14): about ten points lower both times, and a few empty answers came back. The
  `</think>` row was fitted on the model's thinking at the default effort; medium changes that
  thinking, and the fitted row no longer lands in the right place.
- At medium, this model and stock were not distinguishable: 158 vs 156 (13 vs 11, p = 0.84) and
  159 vs 155 (7 vs 3, p = 0.34).

For scale, rerunning the same configuration with a different seed flipped 3 to 13 problems in each
direction (p between 0.73 and 1.0 for all four same-arm pairs), so single-run differences of that
size mean nothing.

**Which should you use?** If your client lets you set `reasoning_effort`, medium is a real
improvement on stock and costs nothing. If it does not, or you do not want to depend on it, this
file gives a larger improvement with nothing set. Use one or the other. Stray `</think>` tags remain
the cost of this file: 2 and 6 of 200 here (1% to 3%), never seen in the stock runs.

Per-problem results for all eight runs are in
[`evidence/per_item/math500_200_at_4096_reasoning_effort.csv`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/per_item/math500_200_at_4096_reasoning_effort.csv)
and the counts in [`evidence/summary_counts.csv`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/summary_counts.csv)
(benchmark `math500_200_at_4096_reasoning_effort`). The medium runs sent
`--chat-template-kwargs '{"reasoning_effort":"medium"}'` to llama-server and nothing else changed.

## Earlier development comparison: medium reasoning effort

**Added September 29, 2026.** I tested medium effort during development but omitted this comparison
from the initial release write-up. It is kept here as the record of that run; the
[section above](#the-medium-reasoning-effort-setting-compared-with-this-model) now has the same
comparison under the release conditions, with the released file and two seeds, and is the one to
cite. Medium effort changes the chat-template instructions; it is different from the
`--reasoning-budget` token cutoff compared above.

These are **earlier Qwen3.8 27B development runs**, using stock Q4_K_M and an earlier edit (cb2),
on one H100 with 16 parallel requests and a 4,096-token output limit. They use the same 200 math
prompts as the release tests, but the released edit is round 6 and the main release math runs used
an RTX 3090 with 8 parallel requests. Treat these as a separate comparison, not additional rows
measured under the final release conditions.

| Earlier development run | Math correct /200 | Empty math answers | Stray `</think>` in math |
|---|---:|---:|---:|
| Stock, default effort | 147 | 32 | 0 |
| Stock, medium effort | 156 | 14 | 0 |
| Earlier edit, cb2 | 167 | 1 | 4 |

| Same development campaign | HumanEval passed /164 | Empty coding answers |
|---|---:|---:|
| Stock, default effort | 147 | 11 |
| Stock, medium effort | 156 | 0 |
| Earlier edit, cb2 | 154 | 0 |

**Medium helped, but did not eliminate math blanks in this run.** All 14 medium-effort math blanks
hit the output limit. On HumanEval, medium eliminated blanks and slightly outscored the earlier
edit. These results do not establish that the edit is universally better than medium effort.

I recovered the original launcher and raw records, checked the shared prompt IDs and text, and
recomputed math scores with the original scorer. HumanEval counts were recounted from saved
execution outcomes; I did not re-execute the generated code for this update. Settings, provenance
limits and per-item results are in the [development evidence](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/development-medium/README.md).
No medium-effort comparison is reported here for Flash-Next. The final release results above are unchanged.

## Files

| file | size | SHA-256 |
|---|---:|---|
| `Qwen3.8-27B-Reasoning-Termination-Fix-Q4_K_M.gguf` (main release, every result above) | 16.81 GB | `2da7bb4517e5bd19e61ff2d8b894832d148b720afb1f06b98e5ab6f19f6c844b` |
| `Qwen3.8-27B-Reasoning-Termination-Fix-Q5_K_M.gguf` | 19.54 GB | `77a7cedbb770c4f649f44434db6767368d521c05fb69484942dbdac2e4db1477` |
| `Qwen3.8-27B-Reasoning-Termination-Fix-Q6_K.gguf` | 22.43 GB | `5fe7df9d58654af758e35e5fd477e98b7054345ffee7bc783a894b474dfea49e` |
| `Qwen3.8-27B-Reasoning-Termination-Fix-Q8_0.gguf` | 29.05 GB | `14e8c58811c9887048cf15669a1897f90b44cb4d067e0d066109f7f0a217aa92` |

Language model only. The vision encoder is not included and vision was not tested.

## Other quantizations

**Added September 29, 2026.** The same row edit, applied to three more quants. Each was tested
against its own stock quant on the same 200 MATH-500 problems at 4,096 output tokens, on the same
machine and llama.cpp build as the main release (8 parallel requests; Q6_K and Q8_0 split over two
RTX 3090s), one seed each.

| 200 MATH-500 problems, 4,096 output tokens | correct | empty final response | stray `</think>` | paired (only this / only stock, p) |
|---|---:|---:|---:|---|
| Q5_K_M stock | 153 | 33 | 0 | |
| **Q5_K_M this model** | **167** | **0** | 2 | 21 / 7, p = 0.013 |
| Q6_K stock | 160 | 21 | 0 | |
| **Q6_K this model** | **166** | **0** | 3 | 14 / 8, p = 0.29 |
| Q8_0 stock | 153 | 26 | 0 | |
| **Q8_0 this model** | **167** | **0** | 1 | 19 / 5, p = 0.007 |

Every empty response in this table was the model still thinking when the budget ran out. The
Q6_K row shows what happens when stock has fewer empties to begin with: the edit still removes
all of them, but the accuracy gain (six problems) is within run-to-run noise.

Q5_K_M also has a HumanEval pair: stock answered 157 of 164 (7 ran out of tokens), this model
answered all 164, and both passed 155. The other two quants were not run on HumanEval, and none
of the three were run on the refusal sets or at 16,384 tokens.

Two things to know about these rows:

- **The stock quants here are my own.** I quantized Unsloth's BF16 (`unsloth/Qwen3.8-27B-GGUF`,
  same revision as the main release) with `llama-quantize`, with no importance matrix. They are not
  Unsloth's published Q5_K_M, Q6_K or Q8_0 files, which use their own recipes, so do not compare
  these numbers to those files. Each pair is a clean comparison because the edited file is that
  same quant with only the `</think>` row changed (byte-verified: 4,161 bytes differ inside the row
  for Q5_K_M and Q6_K, 5,367 for Q8_0, none outside). Only the Q4_K_M release used Unsloth's actual
  file as stock.
- The edit is the same fitted delta as the Q4_K_M release (round 6), applied to each quant's own
  output row and re-quantized to that tensor's type (Q6_K for the Q5_K_M and Q6_K files, Q8_0 for
  the Q8_0 file). The row was not refit per quant.

Per-problem results are in
[`evidence/per_item/math500_200_at_4096_other_quants.csv`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/per_item/math500_200_at_4096_other_quants.csv)
and [`evidence/per_item/humaneval_164_at_4096_other_quants.csv`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/per_item/humaneval_164_at_4096_other_quants.csv),
counts in [`evidence/summary_counts.csv`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/summary_counts.csv), the quantize and bake commands in
[`evidence/scripts/quants_release.sh`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/scripts/quants_release.sh), and the stock quant hashes
in [`evidence/hashes.txt`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/hashes.txt).

## Usage

```sh
llama-server -m Qwen3.8-27B-Reasoning-Termination-Fix-Q4_K_M.gguf -ngl 99 -fa on -c 32768 --jinja
```

This command sets no sampler options, so llama.cpp uses the defaults stored in the file
(temperature 1.0, top-k 20, top-p 0.95) plus its own min-p default of 0.05. Those are the sampler
values the main llama.cpp release results were measured with; I read them back from a running eval server.
`-c` is the context window, not the output budget. The budget is the `max_tokens` your client
sends; the results above use 4,096 unless marked 16,384.

Run this file at the template's default reasoning effort. Do not also set `reasoning_effort: medium`;
measured together they scored about ten points lower on MATH than this file alone (see the
[reasoning effort section](#the-medium-reasoning-effort-setting-compared-with-this-model)).

## Compatibility

- **llama.cpp, tested.** The main release results except the LM Studio check below were produced with a PrismML llama.cpp fork (commit 5d80cff,
  2026-09-17, with local debug patches that only activate through environment variables the evals
  did not set; the patch is [`evidence/prism-5d80cff-local.patch`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/prism-5d80cff-local.patch)). This exact file was also loaded on mainline llama.cpp
  (ggml-org master, commit 5262471, 2026-09-28) and answered all 30 problems of a 30-problem MATH
  check (24 correct, one stray tag).
- **LM Studio, small check.** LM Studio 0.4.16 on Windows, engine
  `llama.cpp-win-x86_64-nvidia-cuda12-avx2` 2.47.0, one RTX 5080 (16 GB) with 60% of the model on
  the GPU (`--gpu 0.6`), context 20,480, 4 parallel requests, through its OpenAI-compatible server.
  Each request sent the same prompt as the MATH eval with `max_tokens` 4,096 and `seed: 0`, and no
  sampler settings, so LM Studio applied its own defaults (I did not check which values it used).
  The set was the first 12 of the 31 MATH problems where stock gave an empty reply at 4,096 tokens
  on llama.cpp, so this checks that the fix carries over; it does not measure accuracy.

  | 12 stock-failure MATH problems, LM Studio | answered | correct | ran out of tokens while thinking |
  |---|---:|---:|---:|
  | **this model** | **12** | **4** | **0** |
  | stock (same file with the original row) | 1 | 1 | 11 |

  Two of this model's 12 answers ran into the 4,096-token limit after the answer had started. No
  stray tags in either run. On llama.cpp this model also answered all 12 of these problems and got
  4 right. Per-problem results: [`evidence/per_item/math500_12_lmstudio.csv`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/per_item/math500_12_lmstudio.csv).
- **Ollama and other apps: untested.** Ollama's own Qwen3.8 library entry needs Ollama 0.32.12 or
  newer. On the older Ollama I had (0.20.2), importing this GGUF directly gave a bare prompt template
  with no thinking setup, so I did not run it there.

## What changed from stock

| | |
|---|---|
| edited | 1 row of `output.weight` (the `</think>` token, id 248069, Q6_K): 4,161 of its 4,200 bytes changed, 0 bytes changed anywhere else in the file (full byte comparison) |
| unchanged | everything else is byte-identical to the stock file: llama-quantize Q4_K_M, no imatrix, from the BF16 GGUF in [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (revision 4ca72078). Stock file SHA-256 `b0e67404dddcb4253804c7a7c77839f16f291fdbd35d8ea70c9ea705508610c5` |

**How the fix works.** Only one output-layer row was fitted; no other weights were trained. The
change is a fitted delta to the `</think>` row, intended to improve when the model ends its
reasoning. Whether the model closes its reasoning depends on the score the output layer gives
`</think>` at each step. I fit a change to that row, on the model's own final hidden states, so the
score rises once the model is ready to answer or is going in circles, and stays low mid-thought and
inside the answer. It was fit in rounds. Each round ran the previous version on the fitting prompts
and added the places where it went wrong: closing the thinking too late, writing a stray
`</think>` into its answer, or ending its turn while still inside the thinking, with no reply at
all.

**Why it helps.** On 120 hard practice problems (MATH Level 5 training problems, not in any test
here), stock left 47 empty at 4,096 tokens, all of them still thinking at the cutoff. In 4 of those
47, the thinking already contained the correct answer in a `\boxed{}`. In 27 of the 47, the
correct answer string appeared somewhere in the thinking. String presence alone does not show that
the model had completed a correct solution.

**Fitting data vs test data.** The fix was fit on 469 prompts that are not in any test here: 150
MATH training problems, 80 MBPP training problems, 40 Alpaca instructions and 199 harmful requests
(AdvBench, JailbreakBench). Before fitting, any prompt that matched or closely resembled (word
overlap of 50% or more) a prompt from any test on this card, including all of MATH-500, was
removed. One defect: in the 150 MATH fitting prompts, the backslash in `\boxed{}` was lost to a
shell escaping bug, so the model saw a control character there. The test prompts were not affected.

**The tests did influence which version was released.** Fitting never used a test prompt, but I
built six rounds and compared them on these same MATH-500, HumanEval and refusal sets before
picking this one. The numbers here are therefore for a selected candidate, not a fully held-out
evaluation. The round history is in [`evidence/README.md`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/README.md). One earlier round scored higher on
MATH-500 (171) but sent empty replies to 15 of 250 harmful prompts, which is why it was not chosen.

## Other results

**HumanEval (164 problems), pass@1, 4,096 output tokens**, same settings:

| build | passed | empty final response |
|---|---:|---:|
| **this model** | **157** | **0** |
| stock (same Q4_K_M file) | 151 | 8 |
| stock + `--reasoning-budget 3072` | 157 | 0 |

This model vs stock: 7 vs 1 on the problems where they differ (p = 0.07). That leans toward this
model but is not statistically significant. All 8 of stock's empty responses ran out of tokens
while thinking. Against the flag: 2 vs 2, no difference. An earlier version of this card compared
against a different stock quant (UD-Q4_K_XL, 147 passed, 13 empty); that run is kept in the
evidence files, labeled, but the row above is the like-for-like comparison.

**Refusals: aggregate rates stayed close to stock.** Same judge and prompt for every row:

| | stock | this model |
|---|---:|---:|
| StrongREJECT (first 150), judged a refusal | 145 (96.7%) | 147 (98.0%) |
| SimpleSafetyTests (100), judged a refusal | 82 | 83 |
| XSTest safe prompts (first 100), full compliance | 95 | 98 |
| XSTest safe prompts, partial refusal | 1 | 2 |
| XSTest safe prompts, full refusal | 1 | 0 |
| XSTest safe prompts, empty (ran out of tokens) | 3 | 0 |
| empty reply that ended the turn, 250 harmful prompts | 1 | 2 |
| empty reply that ran out of tokens, 250 harmful prompts | 1 | 0 |

These are aggregate rates. Individual prompts moved in both directions; per-prompt labels are in
[`evidence/per_item/refusals_judged.csv`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/per_item/refusals_judged.csv). Earlier rounds of this fix turned some refusals into
silence: the model ended its turn inside its thinking and sent an empty reply (15 of 250 in one
round, 24 in another). Later rounds were fit against exactly that, and this build sends 2 such
empty replies, against 1 for stock.

**Stray `</think>` tags.** This model wrote a stray `</think>` into its final answer in 2 of 200
MATH answers at 4,096 tokens and 4 of 100 at 16,384 tokens, and in none of the HumanEval or refusal
answers. No stray tags were observed in any tested stock run without the flag. Stock with the
2,048 flag also produced them (5 of 100 at 16,384 tokens).

## How this was tested

These settings describe the main release evaluation. The earlier medium-effort comparison has its own settings and provenance linked above.

- Server: llama-server with `-ngl 99 --flash-attn on --jinja`, several requests in parallel
  (MATH: 8 slots of 5,120 tokens at the 4,096 budget, 4 slots at 16,384; other sets: 4 slots of
  6,144). Each request sends the messages, `max_tokens` and `seed: 0`, nothing else. MATH and the
  refusal sets send the prompt as a single user message (MATH adds "Put your final answer in
  \boxed{}."); HumanEval adds a short system prompt asking for code only. Exact prompts are in the
  scripts in [`evidence/scripts/`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/tree/main/evidence/scripts).
- Thinking on, at the chat template's default reasoning effort (`xhigh`); no `reasoning_effort` or
  template arguments sent, except the rows marked `medium` in the reasoning effort section, which
  sent `--chat-template-kwargs '{"reasoning_effort":"medium"}'`.
- Sampler: the file's defaults as above. Sampling at temperature 1.0 with parallel slots is not
  bit-reproducible run to run, so expect small differences on a rerun.
- **Empty final response** means the reply after the reasoning was empty. It has two causes, counted
  separately in [`evidence/summary_counts.csv`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/summary_counts.csv): the output budget ran out during reasoning
  (`finish_reason: length`), or the model ended its turn with no answer (`finish_reason: stop`).
- MATH-500: the first 200 problems after a seeded shuffle (seed 7) of the 500, and the first 100
  of those at 16,384 tokens. Scored by exact match of the last `\boxed{}` answer after
  normalisation. The scorer is strict, so compare rows to each other, not to published MATH-500
  numbers.
- HumanEval: all 164 problems, pass@1, one sample each; the code block is extracted and run
  against each problem's own tests.
- Refusals: StrongREJECT (first 150 prompts in file order), SimpleSafetyTests (all 100) and XSTest
  safe prompts (first 100 in file order). Graded by an LLM judge that is not the model under test:
  a Q8_0 file from [OBLITERATUS/Qwen3.8-27B-OBLITERATED](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED)
  (SHA-256 `4cfb568f...`, full hash in [`evidence/hashes.txt`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/hashes.txt)), thinking off, temperature 0, JSON-schema
  output. Harmful sets use the StrongREJECT rubric; XSTest uses its authors' three-way labels.
  An uncensored judge was used because a safety-tuned judge can refuse to read harmful replies.
- Every row on this card was run on the same machine (RTX 3090s) and the same llama.cpp build,
  except the 30-problem mainline load check, the LM Studio check and the earlier development
  comparison (H100).
- Statistics: exact two-sided McNemar test on paired per-problem results. The p-values are
  exploratory: they are not adjusted for multiple comparisons, or for choosing this version from several
  candidate rounds using these same benchmarks.

## Evidence

[`evidence/`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/tree/main/evidence) holds the per-problem results for every row on this card (IDs, scores and
flags only, no model text); [`summary_counts.csv`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/summary_counts.csv) with the main release counts on the
card; [`hashes.txt`](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/hashes.txt) with SHA-256 of the model files, judge, datasets and scripts;
the eval, judge, fit and bake scripts as run; the exact server settings; dataset selection rules;
judge labels by prompt index; the fitting prompt manifest (source and SHA-256 of each prompt); the
bake log with the full-file byte comparison; and the round-by-round selection history.

Not included: the text of the fitting prompts and of the refusal test prompts, since many of them
are harmful requests, and the model's own generations. All of these prompts come from public
datasets named in the [evidence README](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF/blob/main/evidence/README.md),
so the test sets can be rebuilt with the stated selection rules, but rerunning the fit needs regenerated captures and will not give byte-identical results.

## Limitations

- Most results are at a 4,096-token output budget. With a larger budget stock runs out less often
  and the gap shrinks (see the 16,384-token result).
- The fix makes the model stop thinking sooner. Overall scores went up, but not on every problem:
  it missed 2 MATH problems at 4,096 tokens, 2 at 16,384 and 1 HumanEval problem that stock
  solved, and its safety labels differ from stock on individual prompts. Tasks that need very long reasoning may lose more.
- Occasional stray `</think>` tags in MATH answers: 1% to 3% at 4,096 tokens (2/200 and 6/200 over
  two seeds), and 4% at 16,384 tokens (4/100).
- Tested in English, text only, single turn. Agent, tool use and multi-turn use were not tested.
- Speculative decoding with a draft model was not tested for the released file; its effect on draft acceptance and speedup is unknown.
- Everything above the Files section was measured on the Q4_K_M file. The Q5_K_M, Q6_K and Q8_0
  files were tested on MATH only (Q5_K_M also on HumanEval), against my own stock quants, one seed
  each; see Other quantizations.
- One seed per configuration, except the reasoning effort section, which was run with two seeds.
  Between seeds, 3 to 13 problems flipped each way for the same configuration.

## Related work

Most fixes for runaway thinking work at run time: budget forcing and llama.cpp's
`--reasoning-budget`; early exit from a probe on the hidden states
([Zhang et al. 2025](https://arxiv.org/abs/2504.05419), [LYNX](https://arxiv.org/abs/2512.05325));
or stopping rules on the `</think>` score and what follows it
([ThinkBrake](https://arxiv.org/abs/2510.00546), [Entropy After `</think>`](https://arxiv.org/abs/2509.26522)).
Others edit weights inside the network to change how long the model reasons
([ThinkEdit](https://arxiv.org/abs/2503.22048), [Steer2Edit](https://arxiv.org/abs/2602.09870)).
[Sheng et al. 2025](https://arxiv.org/abs/2506.08390) show that the model plans its reasoning
length through the `</think>` score, and report that simply scaling that score at run time made
decoding unstable (their Appendix C.5).

I am not aware of earlier published work that fits the stop decision into the `</think>` output
row itself, as a change that depends on the model's hidden state and ships as a plain GGUF with
nothing extra at run time. An earlier version of this row edit is part of my Bonsai 2 V2 release.
This is the first release of it as a standalone fix on an unmodified stock model.

## Credits

Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
Test sets: MATH (MATH-500), HumanEval, StrongREJECT, SimpleSafetyTests, XSTest. Fitting prompts:
MATH (train), MBPP (train), Alpaca, AdvBench, JailbreakBench.
