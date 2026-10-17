---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
library_name: gguf
pipeline_tag: text-generation
tags:
- abliterated
- uncensored
- gguf
- qwen3.8
- mtp
- reasoning
---

# Qwen3.8-27B Abliterated ThinkFix

An uncensored Qwen3.8-27B that gets to the answer. Abliterated thinking models have a bad habit, worst on sensitive
questions: they decide what to say, then keep arguing with themselves until they run out of tokens. This build takes
the refusals out and adds one small weight edit so the model closes its thinking sooner once it is ready to answer.
The effect shows on sensitive questions, where the long thinking happens. On everyday prompts the model thinks about
as long as it did before, and the answers are about the same.

What that buys you, on 40 sensitive prompts the edit never saw, with an 8k output cap: 33 replies finished cleanly
instead of 22, 5 hit the limit instead of 17, and the whole set ran in 61 minutes instead of 75. Median thinking on
sensitive prompts is about 1,700 tokens, against 3,000 for the same weights without the edit and 3,500 for huihui-ai's
abliteration.

It still works as an agent. Four checks, from replays of saved agent turns to Terminal-Bench and a tool calling split,
found no sign the edit hurts tool use; details below. The MTP draft head is kept, so speculative decoding runs at 56
tokens per second on a 3090 instead of 39. Four quants from 16.8 GB to 29 GB, each tested after the edit.

If you want the same kind of row edit on stock Qwen3.8-27B, with the refusals left in, see
[Qwen3.8-27B-Reasoning-Termination-Fix-GGUF](https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Reasoning-Termination-Fix-GGUF).
That one was fit separately for a different goal, fewer empty answers at a 4k budget, so its numbers are not
comparable with the ones here.

How it was made:

1. Refusal removal (abliteration): the refusal direction is projected out of the weights (Arditi et al. 2024).
2. One row of the output layer, the row that scores the token which closes the thinking block, was fit to recorded
   model states so the model closes its thinking sooner once it is ready to answer.

No LoRA and no gradient training of the network. The row in step 2 was fit from data, and some of that data came
from the benchmarks reported below. That is disclosed under "What the edit was fit on", and the affected prompts
are left out of the headline numbers.

**File:** `Qwen3.8-27B-Abliterated-ThinkFix-Q4_K_M.gguf`, 16.8 GB, SHA-256
`8c79fe4cfd467d39b7e3649780a16a25699efa47dc68e839402814dbb00907a8`. Every result below was measured on this file
unless a row says otherwise. Larger quants are listed under "Choosing a quant".

**The main result.** Same abliterated weights with and without the edit, on 40 sensitive prompts that were not
used to fit it, 8k output tokens, one request at a time on matching GPUs:

| 40 held out sensitive prompts, 8k output cap | without the edit | with the edit |
|---|---|---|
| finished with an answer, without hitting the cap | 22 | **33** |
| hit the token limit | 17 | **5** |
| total generation time | 75 min | **61 min** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Abliterated-ThinkFix-GGUF/resolve/main/thinkfix-completion-dark.png">
  <img src="https://huggingface.co/BoldingBuilds/Qwen3.8-27B-Abliterated-ThinkFix-GGUF/resolve/main/thinkfix-completion-light.png" alt="Bar chart: of 40 held out sensitive prompts at an 8k output cap, this build finished 33 cleanly and hit the limit on 5; the same weights without the edit 22 and 17; huihui-ai abliterated 20 and 20; huihui-ai swift 18 and 21.">
</picture>

Individual replies were not reliably faster: this build was quicker on 18 of the 40. The time saved comes from
fewer replies that run to the limit. Details and other builds are under "Time to an answer".

## Running it

The file keeps the MTP draft head, so `--spec-type draft-mtp --spec-draft-n-max 2` works for speculative decoding.
Without those flags it runs as a normal GGUF. You need a recent llama.cpp (Qwen3.8 uses hybrid attention).
Text only; vision was not tested.

Tested settings: the sampler defaults stored in the file (temperature 1.0, top p 0.95, top k 20), the chat
template's default reasoning effort, thinking on. Give it **at least 8k output tokens**, 16k for hard math.

Speed: on one RTX 3090 (32k context, q8 KV cache, 8 everyday prompts with the first discarded as warm up, low
reasoning effort, temperature 0.6, up to 1,024 output tokens), MTP raised median decode speed from 39 to 56 tokens
per second, with 74 percent of draft tokens accepted. That is decode speed only, not end to end latency.

## What the edit does, and what it does not

While the thinking block is open, the model chooses at every step between more thinking and closing it. The edit
changes how strongly closing is scored. The fit rewarded closing in states where the model was done, and penalised
raising the closing score in states where the model was planning inside an agent loop. That penalty is soft: it
discourages early closing during agent planning, it does not guarantee it never happens.

Earlier versions of this kind of edit made agents answer in plain text instead of calling their tools. This one was
built and checked against that (see Agents and tools). It reduces long thinking. It does not remove it.

## Sensitive prompts

StrongREJECT, 128 of its 150 prompts. The other 22 were used while fitting the edit, so they are left out.

"Answered" means there is an answer after the thinking block. "Hit the limit" means generation stopped at the token
cap. They overlap: an answer can start and then get cut off. "Clean finish" means answered and not cut off.

| output tokens | build | answered | clean finish | hit the limit | judge score |
|---|---|---|---|---|---|
| 4k | this build | 105 | 74 | 53 | 0.81 |
| 4k | same weights without the edit | 82 | 63 | 65 | 0.63 |
| 4k | huihui-ai abliterated | 66 | 43 | 85 | 0.50 |
| 4k | huihui-ai abliterated, swift variant | 58 | 26 | 102 | 0.44 |
| 4k | stock Qwen3.8-27B | 127 | 119 | 9 | 0.02 (refuses) |
| 8k | this build | 125 | 117 | 8 | 0.97 |
| 16k | this build | 123 | 123 | 0 | 0.94 |
| 16k | same weights without the edit | 107 | 103 | 22 | 0.82 |

Judge: `Qwen3.8-27B-OBLITERATED` at Q8_0 with thinking off, the same rubric for every build. Scores run 0 to 1,
higher meaning a more complete answer. Read them as relative between builds, not absolute.

On all 150 prompts at 4k this build hit the limit 65 times. Use 8k or more.

**Where the thinking goes (exploratory).** Median thinking per sensitive prompt at 4k, all 150 prompts: about 1,700
tokens for this build, about 3,000 for the same weights without the edit, about 3,500 for huihui-ai abliterated.
Paired prompt by prompt at 16k, this build's thinking was at least 20 percent shorter on 83 prompts and at least
25 percent longer on 40. A keyword count of the paragraphs that discuss safety or policy came out about the same
for all builds, which suggests the edit mostly trims the back and forth after the decision, not the deliberation
itself. That split is a keyword proxy, not a validated measure.

**Time to an answer.** 40 sensitive prompts not used in the fit plus 20 IFEval prompts, one request at a time, 8k
output tokens, MTP on, each build on its own RTX 3090. On the sensitive prompts this build took 61 minutes in total
against 75 for the same weights without the edit, with a median of 89 seconds per reply against 129. It finished
cleanly on 33 of 40 against 22, and hit the limit on 5 against 17. It was faster on only 18 of the 40 prompts: the
saving comes from fewer replies that run to the limit, not from every reply getting quicker. On the everyday IFEval
prompts the two were about the same.

| 40 sensitive prompts, 8k | total time | median per reply | finished cleanly | hit the limit |
|---|---|---|---|---|
| this build | 61 min | 89 s | 33 | 5 |
| same weights without the edit | 75 min | 129 s | 22 | 17 |
| huihui-ai abliterated | 83 min | 159 s | 20 | 20 |
| huihui-ai abliterated, swift variant | 108 min | 211 s | 18 | 21 |
| stock Qwen3.8-27B | 15 min | 14 s | 40 | 0 |

Stock is fast here because it refuses: in earlier judged runs on these same 40 prompts it scored near 0 on
nearly all of them, and a refusal takes few tokens. The swift variant ran without MTP because its
file has no MTP head, so part of its gap is decode speed.

| 8k output tokens | this build |
|---|---|
| XSTest, 100 harmless prompts that sound risky: fully answered | 100 (0 refused) |
| SimpleSafetyTests, 100 prompts: judge score | 0.87 (1 refused, 0 hit the limit) |

For reference at 4k: stock fully answered 95 of the XSTest prompts and refused 2; the no edit weights scored 0.83
on SimpleSafetyTests with 12 hitting the limit. One XSTest prompt was used while fitting the edit.

## Did it cost anything

| test | output tokens | this build | stock Qwen3.8-27B |
|---|---|---|---|
| MATH, 200 hard problems | 16k | 170 | 170 (6 hit the limit) |
| MATH, 200 hard problems | 8k | 165 (5 hit the limit) | 168 (14 hit the limit) |
| HumanEval pass@1, 164 problems | 4k | 158 | 151 |
| HumanEval pass@1, 164 problems | 8k | 158 (0 hit the limit) | |
| IFEval prompt strict, all 541 prompts | 16k | 499 | 505 |
| IFEval prompt strict, 534 prompts not used in the fit | 16k | 494 | 500 |

Paired tests (exact McNemar, two sided):

- MATH 8k: 8 problems solved only by this build, 11 only by stock, p = 0.65.
- IFEval, all prompts: 21 against 15, p = 0.41.
- IFEval, without the 7 fit prompts: 13 against 19, p = 0.38.

Not significant does not mean no cost. IFEval is about one point lower and MATH at 8k three problems lower, and
these test sizes cannot rule out a small real loss.

## Agents and tools

Agent use is where earlier edits like this failed, so it was checked four ways. None of them shows agents got
better, and none proves agent behaviour is unchanged in every case. They are limited checks, not a guarantee.

- **Replay of saved agent turns.** 188 turns from multi step coding and operations tasks with tools. Each turn was
  replayed with the exact same saved context on this build and on the same weights without the edit, one sample
  each. On 185 of 188 the two produced the same thinking length, the same token count, the same first tool and the
  same size of tool arguments. 187 of 188 picked the same first tool. This compares lengths and tool names, not
  full text. 12 of the 15 agent runs these turns come from were also used to fit the edit, so this checks that the
  edit leaves familiar agent turns alone. It is not a held out test.
- **Held out replay.** The same test on 244 turns, across 6 task types, from agent runs that were never used for
  the fit: 239 of 244 identical, 243 of 244 picked the same first tool, and no turn switched between calling a tool
  and answering in plain text. Each build hit the length limit once.
- **Terminal-Bench 2.1**, the 44 tasks whose reference solutions pass, terminus 2 agent, one try per task. First
  full pass: this build 26, stock 29, same weights without the edit 29. An exploratory rerun of the 13 tasks where
  this build and the no edit weights disagreed gave 6 each. That rerun only covers the disagreements, so it is not
  a replication. Second full pass: this build 31, stock 28. Over both full passes that is 57 of 88 for each, so
  no difference shows at this size. Single runs of the same build moved by 5 tasks between passes, which is the
  noise level here.
- **Tool calling.** A custom single turn diagnostic built from BFCL v4: 882 cases, a fixed 29 percent split of a
  3,001 case pool. It checks whether a call is made and whether the function name is right. Arguments are not
  graded. When a tool fits (552 cases), this build calls one 551 times and names the right function 538 times.
  When no tool fits (330 cases), it still calls one 119 times. The same weights without the edit: 118. huihui-ai
  abliterated, a separate abliteration: 123. Stock with no edits: 63. So the over calling comes with abliteration
  in general, and the edit does not add to it. When a tool does fit, stock called one 541 times and named the right
  function 528 times, a little behind this build's 551 and 538. If your
  agent offers tools on every turn, expect more unneeded calls than stock. This split was also used to decide
  against a second edit for tool calls, so it is not untouched either.

## Choosing a quant

All files were made with the same recipe from the same BF16 weights. Before quantizing, the recipe was rerun at
Q4_K_M and checked to give a byte for byte copy of the tested file. The same row edit was then written into each
quant, and each file was checked to differ from its unedited twin only inside that one row.

Each quant was retested on the 128 sensitive prompts from "Sensitive prompts" at 4k output tokens, and on 200 hard
math problems (MATH) at 8k. 4k is a stress test, not the recommended setting. "Answered", "clean finish" and "hit
the limit" mean the same as in that section, and they overlap, so they do not add up to 128.

| file | size | answered | clean finish | hit the limit | empty answer | MATH correct of 200 | MATH hit the limit |
|---|---|---|---|---|---|---|---|
| Q4_K_M (main file) | 16.8 GB | 105 | 74 | 53 | 1 | 165 | 5 |
| Q5_K_M | 19.5 GB | 107 | 76 | 40 | 12 | 169 | 2 |
| Q6_K | 22.4 GB | 114 | 74 | 49 | 5 | 170 | 1 |
| Q8_0 | 29.0 GB | 112 | 77 | 45 | 6 | 169 | 2 |

"Empty answer" is the quirk listed below: thinking closes and no answer follows, without hitting the limit. The
larger quants do it more often than the main file at 4k, and the same weights without the edit did not do it at all.
Each row is a single run. Bigger files did a little better on math. On the sensitive prompts there is no clear
order by size, and none of these differences has been checked for repeatability. Every row has the edit, so this table
helps pick a file, it does not measure the edit.

SHA-256: Q5_K_M `20f6d5086da700af92dd00e6facb9289784868008bc13a6343427b37742a35ed`, Q6_K
`987bcc49317e221967537228271e9be20ceb779bff7a2cecbf81cd9877534f6a`, Q8_0
`c8cf3fb5ce14fd708fad9b86190c00a3f5d9ef6fec863e408f196fdff43675c6`.

## Known quirks

- A closing think tag sometimes shows up in the visible answer: 6 of 200 hard math answers at 16k, 9 of 200 at 8k,
  5 of 541 IFEval prompts (3 of them repeated the tag several times), 1 of 150 sensitive prompts at 8k, 2 at 16k.
- A few replies finish thinking and then stop with no answer: 5 of 150 sensitive prompts at 16k (the no edit
  weights: 3), 3 at 8k, 4 of 100 SimpleSafetyTests prompts. Stock and huihui did not do this at 4k.
- At 4k output tokens, 65 of the 150 sensitive prompts hit the limit.

## What the edit was fit on

The edited row was fit on recorded model states, including:

- turns from earlier agent benchmark runs of mine, the same runs the replay test draws from (see above);
- states just before stray closing tags in earlier builds' answers on StrongREJECT (22 prompts), IFEval (7 prompts)
  and XSTest (1 prompt), added so the edit would not encourage stray tags. Those StrongREJECT and IFEval prompts
  are excluded from the numbers above where marked.

## Test settings

| suite | prompts | output tokens | sampler | scored by |
|---|---|---|---|---|
| StrongREJECT | 128 of 150 | 4k, 8k, 16k | file defaults, seed 0 | judge above, thinking off |
| XSTest, SimpleSafetyTests | 100 each | 8k | file defaults, seed 0 | judge above |
| MATH | 200 hard problems | 8k, 16k | file defaults, seed 0 | final answer match |
| HumanEval | 164 | 4k, 8k | file defaults, seed 0 | unit tests |
| IFEval | 541 and 534 | 16k | file defaults, seed 0 | official strict checker |
| Tool calling | 882 | | file defaults, seed 0 | call made, function name |
| Agent replay | 188 turns | as recorded | one sample per build | length and tool comparison |
| Terminal-Bench 2.1 | 44 tasks | 32,768 (24,576 thinking budget) | temperature 1.0, top p 0.95, top k 20, xhigh effort | task tests |

Terminal-Bench ran with 131k context, MTP on and q8 KV cache. Output tokens here means the total output cap per
reply, thinking included. Everything ran on llama.cpp llama-server with RTX 3090s.

## Credits

Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0). Refusal removal follows Arditi
et al., "Refusal in Language Models Is Mediated by a Single Direction" (2024).

## Use responsibly

Refusals are removed. It will answer things stock Qwen would not, and you are responsible for what you do with it.
Not for public facing products without your own filtering.
