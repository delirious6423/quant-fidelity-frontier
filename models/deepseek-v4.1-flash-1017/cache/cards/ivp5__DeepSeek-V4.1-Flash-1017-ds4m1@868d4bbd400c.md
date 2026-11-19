---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
  - deepseek
  - moe
  - quantization
  - apple-silicon
  - metal
  - gguf
---

# DeepSeek-V4.1-Flash-1017 — ds4m1: the full model on one 64 GB M1 Max

**41 tok/s decode · 313 tok/s prefill at a 1,045 MHz GPU clock (39 and 299 per GHz) · 595 tokens and 17.9 s per solved coding task** on a single Apple M1 Max
with 64 GB — all 384 routed experts, the Engram tables, thinking and tool calls. The experts stream from the SSD into a 38 GiB cache; everything else stays in memory.
Engine: [ds4-m1](https://github.com/ivp5/ds4-m1) (antirez's [ds4](https://github.com/antirez/ds4) + Metal, Neural Engine and
SSD-streaming patches, and a trained speculative drafter). **At the GPU's full 1,296 MHz clock that is an estimated ~48 tok/s decode and ~378 tok/s prefill** (the measured rates scaled by the clock with the exponents measured inside levels, 0.69 for decode and 0.88 for prefill; a laptop under sustained load settles near 1,000-1,050 MHz, which is where the measured rates were taken).

## Update 2026-11-19: fewer tokens per solved task, and the loop fixes

This update replaces the 2026-11-08 build. It is slower per token and faster per solved task, because it writes fewer tokens:

| | 2026-11-08 | **2026-11-19** |
|---|---:|---:|
| Coding-agent tasks solved (15 tasks, one process) | 15 / 15 | **15 / 15** |
| Tokens per solved task | 822 | **595** (x0.72) |
| Seconds per solved task | 20.1 | **17.9** (x0.89) |
| Decode | 48.4 tok/s at 1,061 MHz (45.6 per GHz) | 41.1 tok/s at 1,045 MHz (39.3 per GHz) |
| Prefill | 342.6 tok/s at 1,061 MHz (322.9 per GHz) | 312.5 tok/s at 1,045 MHz (299.0 per GHz) |

The absolute rates move with the GPU clock, which the machine's heat and power target set; the per-GHz figures are the ones to compare between runs.
On 10 freshly generated bug-fix tasks (three from harder modules) the agent solved 10 of 10 at 108.6 s and 2,369 tokens per solution, with no loop stop.

**What changed, and what it fixes.** The 11-08 build could fall into repeating a line or a tool-call parameter. Three changes address that at its sources:

- **A stale attention window, repaired.** On prefill steps of 130-144 rows (a tool result of that length), the rows past 128 attended over a window shifted back by 128 positions. The first token of the next turn was computed from those rows. This was an engine defect, present since the 256-row prefill steps were introduced.
- **Full-rank dense matrices.** Two families of attention matrices (the query projection's second stage and the attention-output pair) were stored at reduced rank. Measured against the official model, the rank cut — not the 4-bit precision — was what moved the decisions at the rows where a repeat either stops or continues. They are now stored at full rank and 4 bits (5.14 GB against 3.03). On the row of a recorded looping session where the model must choose between stopping and one more duplicate, the margin for stopping went from -3.56 to +3.69 nats (the official model: +4.09).
- **A tool call is held to its schema.** A parameter name already written in a call cannot be written again, and a call whose schema parameters are all written closes. A duplicated parameter is never a valid call, so this removes nothing a correct session writes; it costs no speed (36.7 against 36.2 tok/s in a paired run).

Also since 11-08: the routed experts of 20 layers were retrained against the official model's outputs (the two `*_trained.bin` files), the agent's
thinking is no longer capped by a routing shortcut, and its attention index width is the model's own 512. Together these account for the
lower token count and the lower rate per GHz.

**Same behaviour as the official model on an open-ended repeat.** One case is unchanged by this update, and it is the official model's own
behaviour: when a reply already holds the same line twice and nothing in the request says how many are wanted, the model tends to write it
again. The official weights rank the repeated line first after the second copy in 19 of 20 test documents; this build continued such a seeded
repeat in 5 of 8 sampled draws. A requested count is kept exactly by both (20 of 20 lines, 12 of 12). In agent use the loop detector ends
such a turn. A change to the weights that removes the tendency is being worked on and is not part of this release.

`ds4-server` keeps the 11-08 configuration and files; the two programs read different expert files from the same pack.

The tables below this section that are dated 2026-11-08 describe the 11-08 files, which are still in the pack.

## How it compares (measured 2026-11-08 unless marked)

| | **ds4m1 (this repo)** | vcruz305 SAGE 1.55 bpw (EXL3 / REXL3) | Unsloth UD-IQ2_XXS (GGUF) |
|---|---|---|---|
| Runs on | **M1 Max, 64 GB** | 128 GB-class GPU (108.5 GiB resident) | llama.cpp, 300.5 GB of weights |
| Download | **292 GB** for the agent and CLI (445 GB with the server's files) | 320 GB | 300.5 GB |
| Decode (2026-11-19) | **41.1 tok/s at 1,045 MHz** | — | — |
| Prefill (2026-11-19) | **312.5 tok/s at 1,045 MHz** | — | — |
| Routed experts vs official, same test (8 agent sessions) | **KL 0.047 · top-1 92.8%** | REXL3: KL 0.091 · top-1 89.9% | — |
| wiki1100 vs official, routed experts only (the same 1,099 positions) | **KL 0.324 · top-1 85.1%** | REXL3: KL 0.380 · top-1 83.9% | — |
| wiki1100 vs official, the whole model | KL 0.833 · top-1 73.6% | **EXL3: KL 0.088 · top-1 88.2% (its own slice)** | KL 0.203 · top-1 90.5% (its own slice) |
| KL vs official on its card's own text | — | EXL3: 0.088 wiki1100, 0.098 e2e_eval; REXL3: 0.051 e2e_eval | 0.203 wiki1100, 0.142 long17k |
| Top-1 on its card's own text | — | EXL3: 88.2% wiki1100, 94.5% e2e_eval | 90.5% wiki1100, 91.2% long17k |
| Whole model vs Q8_0 reference (4 agent sessions, 2026-11-19) | KL 0.105 · top-1 90.7% | — | — |
| Coding-agent tasks (2026-11-19) | **15/15 solved, 595 tokens and 17.9 s per solution** | — | — |

— = not stated on that card. The same-test row swaps only the routed experts (attention, dense layers, router and head are
the official weights in both), so it compares the two expert codecs directly.

## Speed (M1 Max, 64 GB)

| Workload | Decode | Prefill | GPU clock |
|---|---:|---:|---:|
| **2026-11-19 build:** 16 coding-agent tasks back to back (one process) | 41.1 tok/s | 312.5 tok/s | 1,045 MHz |
| same, steady tasks 5-15 | 40.7 tok/s | 306.0 tok/s | 1,026 MHz |
| same build with the schema rule on, last eight tasks of a second run | 41.3 tok/s | 312.7 tok/s | 1,043 MHz |
| 2026-11-08 build: 16 coding-agent tasks back to back | 48.4 tok/s | 342.6 tok/s | 1,061 MHz |
| `ds4` one-shot, WikiText prompt of 2,062 tokens + 256 generated (a fresh process) | 20.0 tok/s | 247.7 tok/s | |
| `ds4` one-shot, WikiText prompt of 8,172 tokens + 256 generated (a fresh process) | 19.3 tok/s | 268.1 tok/s | |
| `ds4` one-shot, WikiText prompt of 32,803 tokens + 256 generated (a fresh process) | 22.7 tok/s | 291.0 tok/s | |

The agent rows are agent work (code, tool calls), where the drafter's tokens are accepted 75-80 % of the time; on encyclopedic prose it accepts
25-70 %, so the one-shot rows generate slower. A fresh process also installs its first experts from the SSD (a few seconds).

## Fidelity

**Routed experts against vcruz305's SAGE REXL3 1.55 bpw** — official FP8/FP4 model as teacher, 11,263 positions of 8 agent sessions:

| Routed experts | KL mean | KL p50 | KL p90 | Top-1 |
|---|---:|---:|---:|---:|
| **ds4m1** (2-bit gate/up + 2-bit down) | **0.0467** | 0.00337 | **0.132** | **92.8%** |
| SAGE REXL3 1.55 bpw | 0.0906 | 0.00333 | 0.271 | 89.9% |

ds4m1 wins on all 8 sessions; the medians are equal and the gap is the tail.

**wiki1100 against the official model** — the first 1,100 tokens of the WikiText-2 raw test split (the official tokenizer, no BOS), 1,099 next-token positions; the official log-probabilities computed on the AMD box, the published engine's on this Mac:

| Model | KL mean | KL p50 | Top-1 | Top-5 | Perplexity (official 2.22) |
|---|---:|---:|---:|---:|---:|
| ds4m1, the whole model, shipped configuration | 0.8330 | 0.26375 | 73.6% | 58.1% | 4.47 |
| ds4m1, the whole model, deterministic routing | 0.5227 | 0.06636 | 80.0% | 63.7% | 3.32 |
| ds4m1 routed experts only (official everything else) | 0.3239 | | 85.1% | | |
| SAGE REXL3 1.55 routed experts only | 0.3801 | | 83.9% | | |

Two runs of the shipped configuration differ from each other by KL 0.1116 (mean): the router prefers experts already in the cache. The official model predicts most of this text with near certainty (top-1 probability of 0.99 or more at 603 of 1,099 positions, perplexity 2.22); at those positions the deterministic run reads KL 0.21, and at the positions where the official top-1 probability is under 0.5 it reads 1.31. The gap is not only the start of the text: the second half alone reads KL 0.54 shipped and 0.37 with deterministic routing. The shipped configuration's extra KL over deterministic routing is largest in the first ~330 positions.

vcruz305 and Unsloth do not say which 1,100 tokens of WikiText-2 they used or how they scored them. On this slice, against this reference, SAGE REXL3's routed experts alone read KL 0.380, while vcruz305's card reports 0.088 for its whole EXL3 model on its own slice: numbers from different cards are not on one scale. The routed-experts row compares like with like: both codecs on the same positions against the same official log-probabilities.

**The whole published model** against the Q8_0 GGUF, teacher-forced on four coding-agent sessions (thinking / answer tokens split), measured 2026-11-19 on this build:

| Session | Positions | KL mean | KL p50 | Top-1 | Top-5 | Thinking KL / top-1 | Answer KL / top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| calc | 696 | 0.117 | 0.00099 | 89.5% | 74.3% | 0.182 / 84.7% | 0.042 / 95.0% |
| textutil | 404 | 0.070 | 0.00001 | 95.3% | 69.0% | 0.259 / 82.7% | 0.027 / 98.2% |
| roman (long thinking) | 1,148 | 0.115 | 0.00855 | 90.2% | 77.4% | 0.133 / 88.6% | 0.066 / 94.3% |
| matrix (long thinking) | 668 | 0.097 | 0.00079 | 90.3% | 75.2% | 0.145 / 85.8% | 0.040 / 95.5% |
| **pooled, 2026-11-19** | **2,916** | **0.105** | | **90.7%** | **75.0%** | | |
| pooled, 2026-11-08 | 3,385 | 0.118 | | 90.5% | 69.8% | | |

## Tokens per correct solution

Every generated token of every attempt, over the correct solutions:

| Task set | Solved | Tokens per solution | Seconds per solution |
|---|---:|---:|---:|
| 2026-11-19: 15 coding-agent tasks (Python modules with pytest suites) | 15 / 15 | 595 | 17.9 |
| 2026-11-19: 10 freshly generated bug-fix tasks | 10 / 10 | 2,369 | 108.6 |
| 2026-11-08: the same 15 coding-agent tasks | 15 / 15 | 822 | 20.1 |

DeepSeek's own benchmark suites (HLE, Terminal-Bench, SWE, ...) were not re-run here: each takes days at this rate.

## Files

Six files, 445 GB. The agent and the CLI read four of them (292 GB); `ds4-server` reads the 11-08 expert pair instead of the trained one.

```
# the agent and the CLI (292 GB)
hf download ivp5/DeepSeek-V4.1-Flash-1017-ds4m1 --local-dir ds4m1 --include "DeepSeek-V4.1-Flash-1017-ds4m1.gguf" "experts_*_trained.bin" "engram_iq4nl.bin" "MANIFEST.sha256"
# everything (445 GB)
hf download ivp5/DeepSeek-V4.1-Flash-1017-ds4m1 --local-dir ds4m1
```

Verify with `shasum -a 256 -c MANIFEST.sha256` (it lists all six; a file you did not download is reported missing).

| File | Bytes | Holds | Read by |
|---|---:|---|---|
| `DeepSeek-V4.1-Flash-1017-ds4m1.gguf` | 28,483,206,208 | every non-routed tensor, plus the engine's small parts as I8 tensors: both 4-bit dense tables (full-rank and 11-08), 2-bit head, router look-ahead, the drafter, the Neural Engine programs | all |
| `experts_gate_up_bs64_trained.bin` | 99,090,448,384 | routed gate + up, 40 layers x 384 experts, 2-bit; 20 layers retrained | agent, CLI |
| `experts_down_g32_trained.bin` | 53,791,965,184 | routed down, 2-bit; 20 layers retrained | agent, CLI |
| `engram_iq4nl.bin` | 110,595,319,712 | the two Engram tables, IQ4_NL | all |
| `experts_gate_up_bs64.bin` | 99,090,448,384 | routed gate + up as published 2026-11-08 | server |
| `experts_down_g32.bin` | 53,791,965,184 | routed down as published 2026-11-08 | server |

Use these files with the engine at the commit below. Earlier engine builds are not supported: update both together.

## Run

```
./ds4-agent -m ds4m1/DeepSeek-V4.1-Flash-1017-ds4m1.gguf
```

Engine: [ivp5/ds4-m1](https://github.com/ivp5/ds4-m1) (main, commit [b588b0c](https://github.com/ivp5/ds4-m1/commit/b588b0c): antirez/ds4 b3f5018 merged); build it with
`make -j8 ds4 ds4-agent ds4-server`. The other three files are found beside the GGUF. The first start unpacks 18 GB of small parts into
`~/Library/Caches/ds4m1/` (`DS4M1_CACHE` moves it) and compiles the Neural Engine programs; the first session after that is the one to time.

**Memory limits.** The engine wires about 53 GiB. Every number on this card was measured after raising macOS's wired-memory
limits (they reset at reboot):

```
sudo sysctl -w iogpu.wired_limit_mb=59808 vm.global_user_wire_limit=58497454571 vm.user_wire_limit=58497454571
```

The expert cache (4,100 slots, 38.2 GiB) must fit in the GPU working set, which is `iogpu.wired_limit_mb`, and is locked
with `mlock`, which the two `vm` limits bound. With lower limits the engine shrinks the cache (it prints
`V4.1 SSD cache fitted from 4100 to N experts`): more experts are read from the SSD and decode is slower. Close other large apps.
Sampling: temperature 1.0, top-p 1.0, min-p 0.2. Outputs vary run to run: the router prefers experts already in the cache.

## License

MIT, as [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). Cite DeepSeek for the base model.
