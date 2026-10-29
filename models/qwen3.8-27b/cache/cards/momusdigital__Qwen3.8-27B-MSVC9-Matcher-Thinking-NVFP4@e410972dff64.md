---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - qwen3.8
  - nvfp4
  - quantized
  - thinking
  - reasoning
  - decompilation
  - reverse-engineering
  - x86
  - msvc
  - code-generation
---

# Qwen3.8-27B MSVC9 Matcher: Thinking (NVFP4)

A fine-tune of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) that reconstructs **C++03 source for one x86
function from its disassembly**, aiming for source that Microsoft Visual C++ 2008 SP1 (`15.00.30729.01`) compiles to
**byte-identical** machine code. It is trained to **think first** (`enable_thinking=True`), write the function, read the
compiler/byte-comparison feedback, and revise. This repository holds the **merged, NVFP4-quantized weights only**.

Successor to [Qwen3.8-27B-MSVC9-Matcher-LoRA](https://huggingface.co/momusdigital/Qwen3.8-27B-MSVC9-Matcher-LoRA), which was trained
for thinking-off, short functions. This round targets larger functions (1 KB and up) with reasoning on.

- **Format:** NVFP4 (4-bit, group size 16, FP8 KV cache), LoRA merged into the BF16 weights, then re-quantized with the base
  model's frozen scales. Needs a Blackwell-class GPU (tested on an RTX 5090, 32 GB) with vLLM 0.31.
- **Research release.** Quality is measured below; most large functions are still not solved.

## How it was trained

**Data: community work traces.** A community platform lets contributors (people directing AI coding agents) claim unmatched
functions and iterate against a local compiler plus byte-comparison checker. Each claim records the agent's written **plan**
before it starts, free-form **notes**, and every **attempt** with the code and three short fields written *before* the check
(`change`, `why`, `expect`) plus an `outcome`. About 16,000 such claims were collected.

**Preparation**
1. Claims were tiered: *gold* (verified match, 512 B+, plan written first, no earlier work seen), *silver* (verified, other),
   *repair* (not matched, 2+ attempts).
2. Every attempt was **re-run on our own checker**, so each fix step is labelled with what actually happened (compile error
   to compiles, fewer differing instructions, or match). Only steps that improved were trained on.
3. Reasoning examples use only text written **before** the code it explains (plans, notes, change/why/expect), never
   after-the-fact rationales; the research on answer-conditioned reasoning says those teach asserting instead of deriving.
4. Examples: first attempts with the contributor's plan as the thinking; fix steps with the previous check result in the
   prompt and the contributor's change/why/expect as the thinking; code-only verified answers; hindsight pairs; and ~20% replay
   of earlier rounds (thinking off) so the no-think mode and small-function skill survive.
5. Weights: gold 1.0, silver 0.8, improving fix 0.7 (1.0 if it matched), hindsight 0.5. Any one (account, model) source
   above 30% of the weight was down-weighted.

**Training.** LoRA rank 64 on the official BF16 model, warm-started from the previous round's adapter, 103 optimizer steps,
24K-token sequences, two nodes. 1,662 examples (706 with thinking), about 21M tokens. Loss only on the final assistant turn,
including its thinking. The training data is **not** released.

## Results

Held-out test: 37 functions of 1 to 4 KB that were already matched by contributors and appear in **no** training data
(25 at 1-2 KB, 12 at 2-4 KB). Each function gets up to 12-22 checker rounds. Thinking on, 48K max tokens per reply,
128K context, 8 parallel sessions on one RTX 5090.

| | Matched | 1-2 KB | 2-4 KB | Output tokens / reply | Minutes / function |
|---|---|---|---|---|---|
| **This model** | **7 / 37** | 7 / 25 | 0 / 12 | **~20K** | **~107** |
| Qwen3.8-27B (same base, NVFP4) | 4 / 37 | 4 / 25 | 0 / 12 | ~41K | ~231 |

- Four functions only this model matched, one only the base matched (exact McNemar p = 0.38). **The gap is suggestive, not
  statistically established.** The solid finding is efficiency: about **half the tokens and 2.2x faster** for equal or better results.
- On a harder set of 120 functions over 1 KB from units with no training traces, it matched **0 of the 19** attempted before
  the run was stopped. Neither it nor the base can do these yet.
- Nothing above 2 KB was matched by either model.

## What the fine-tune improved, and where it still fails

Per-reply outcomes on the 37-function held-out test (504 checker rounds for this model, 519 for the base):

| Outcome of a reply | This model | Base model |
|---|---|---|
| Compiles (candidate worth comparing) | **37%** of replies with code | 5% |
| Compile error | 39% | 59% |
| Rejected for a forbidden construct (inline asm, raw offsets, hard-coded addresses) | 6% | 26% |
| No code block in the reply | 25% | 7% |
| Reply hit the token cap | 13% | 48% |

- **What it improved:** it writes code that obeys the task rules and compiles far more often (37% vs 5%), uses about half the
  tokens, and rarely runs out of budget (13% vs 48% of replies cut off). Of the functions it failed, 21 of 30 produced at least
  one compiling candidate; the base managed that on only 7 of 33. The base's wins came from early lucky matches (1 to 5 rounds)
  or none; this model's came from iterating (3 to 12 rounds), which is the behavior it was trained for.
- **Gaps:** its first reply usually fails to compile (27 of 37), so it depends on feedback rounds; it often just stops without code
  (25% of replies, and 35% on 2 to 4 KB functions) or hits the cap while still reasoning; it invents its own declarations
  instead of using the provided ones; and it is not better at the hardest cases.
- **By size:** 25 functions at 1 to 2 KB: 7 matched (base 4), 43% of replies compile. 12 functions at 2 to 4 KB: 0 matched
  (base 0), only 9% of replies compile, 47% are compile errors, and 35% have no code.
- **Floating-point code (x87):** 24 of the 37 functions use legacy x87 floating point. This model matched 3 of them (12%) against
  4 of the 13 others (31%) and made more compile errors on them (41% vs 34%). The base matched 1 of the 24.
- **What got matched:** eight functions in total, 1.1 to 2.0 KB, mostly object-loading routines (3 of the 8) and
  path-search routines (2 of the 8); four other loading routines were missed by both models. Every function
  that either model matched is also one a stronger model had matched earlier, so these are not new results.
- **Caution:** 37 functions is a small sample. Treat the size and x87 patterns as leads, not conclusions.

## How its reasoning changed

- **Shorter, plan-style thinking.** It writes a structured outline (stack frame, locals, calls, code regions) in a few hundred
  to ~20K tokens, where the base often thinks 40K+ tokens and frequently hits the cap. This is the imitation of the training
  notes; it is cheaper but also shallower on very hard functions.
- **It iterates.** Matches typically took ~6 checker rounds; it uses the difference listing to fix compile errors and
  instruction-order mismatches.
- **Known failures:** it sometimes invents its own struct and global declarations instead of using the provided ones (compile
  errors); on hard functions thinking can run to the token cap and return no code (~17% of replies on the hardest set); nothing
  large is solved.

## Usage

Serve with vLLM and send the function's disassembly plus the compiler flags and declarations in the prompt; keep thinking on:

```bash
vllm serve <this-repo> --max-model-len 131072 --kv-cache-dtype fp8 \
  --default-chat-template-kwargs '{"enable_thinking": true}'
```

Send `max_tokens` of 32K-48K, keep the previous two attempts plus the checker feedback in the conversation, and take the last
```` ```cpp ```` block as the answer. Prompt + `max_tokens` must stay under `--max-model-len`; large functions need 28-36K prompt
tokens before any revision. Speculative decoding (MTP) crashed our server on this build; leave it off.

## Credits and license

Base model: Qwen3.8-27B (Apache-2.0). NVFP4 recipe and scales follow the community
`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` export. This model is released under Apache-2.0.
