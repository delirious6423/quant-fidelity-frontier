---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - quantized
  - llama.cpp
  - shoehorn
---

# Qwen3.8-27B — shoehorn budget-exact fits

Four GGUF quantizations of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [shoehorn](https://github.com/notactuallytreyanastasio/shoehorn),
plus the accuracy benchmark that checks whether they still think.

A preset quant like `Q8_0` or `IQ4_XS` applies one recipe to every tensor, and
whatever size falls out is the size you get. shoehorn works the other way round:
you give it a VRAM budget, a context length and a KV cache type, and it solves
for a per-tensor mix of quant formats that fills whatever is left after
inference overhead. The point is to land on a specific card at a specific
context, not to hit a particular bits-per-weight number.

So the obvious question is whether solving for a size costs you accuracy. We
benchmarked all four against the two stock quants. The short answer is in the
table below; the long answer is that at 4.9 bits per weight and up, this
benchmark could not measure a difference, and at 2.9 bits it very clearly could.

## The files

| file | bpw | size | solved for | GiB |
|---|---|---|---|---|
| `qwen38-27b-fit-128k-q4kv.gguf` | 6.099 | 20,838,706,944 B | 131072 ctx, q4_0 KV, 23.5 GiB budget | 19.41 |
| `qwen38-27b-fit-192k-q4kv.gguf` | 5.509 | 18,825,459,584 B | 196608 ctx, q4_0 KV, 23.5 GiB budget | 17.53 |
| `qwen38-27b-fit-256k-q4kv.gguf` | 4.920 | 16,812,197,504 B | 262144 ctx, q4_0 KV, 23.5 GiB budget | 15.66 |
| `qwen38-27b-fit-256k-17g.gguf`  | 2.876 |  9,832,877,184 B | 262144 ctx, q4_0 KV, **17 GiB** budget | 9.16 |

Every budget above is a budget for weights **plus** KV cache plus overhead, at
the stated context. That is what "solved for" means here — the 17 GiB file was
built to fit a 24 GB card together with its speculative-decoding draft model
and a full 256K context, and it does: 21.6 GiB, measured in earlier testing of that configuration.

These are text-only. The base model is multimodal; no vision projector
(`mmproj`) is included here and the vision path was neither converted nor
tested.

### Exactly which bytes were benchmarked

Every number in this card was produced by these exact files. The published
weights are byte-identical to the ones that ran the benchmark — the hashes
below were taken on the machine that ran the benchmark, and match what
HuggingFace stores for each of the four published files.

| file | size (bytes) | sha256 |
|---|---|---|
| `qwen38-27b-fit-128k-q4kv.gguf` | 20838706944 | `f30987d23ed027d4ecf5e49aed5a8479162e8d4f796c6e8a541d15e16d08384a` |
| `qwen38-27b-fit-192k-q4kv.gguf` | 18825459584 | `2faadd35212ae9132e147eb22e423a7d708aed19da1bbc5a5492029806ad99f6` |
| `qwen38-27b-fit-256k-q4kv.gguf` | 16812197504 | `b156c0c1aa0a2659e4d0ed203f7e6e7f8ff847a081f77f07aebf683cea80eb7a` |
| `qwen38-27b-fit-256k-17g.gguf` | 9832877184 | `787b2bfafc39177d051b13e5c2359c5950c92496b9ed1fd3f51df750ec78dd6b` |

Verify with `sha256sum <file>` after downloading.

The comparison files, which are **not** in this repo but which the benchmark
numbers depend on:

| file | size (bytes) | sha256 |
|---|---|---|
| `Qwen3.8-27B-Q8_0.gguf` | 29047085024 | `57484e3196aaff8dbbd71c666158b00206baa2bc134e4f16ae90fc8cbdfbeea2` |
| `Qwen3.8-27B-IQ4_XS.gguf` | 15705860064 | `8b29da0e6cbc5032d478cd2333f5071b5195b4fa903c1d51d86704386c18fb29` |
| `Qwen3.8-27B-MTP-ONLY-Q8_0.gguf` | 4503398688 | `674d0fc3b2b09c48cf77fbab0aba39b9c4ee538bd240fa87c1f13044260f7d7b` |

The two stock quants are from
[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF),
and the revision matters. Both files match that repo exactly at any revision
**before** commit `408fcc1807ab` (2026-08-14, "Add Unsloth style chat
template"). That commit embedded a chat template and grew each file by exactly
1024 bytes, so the current `main` copies hash differently. If you download them
today you will get the +1024-byte versions.

We checked whether that mattered, because it is the kind of difference that
quietly breaks a comparison. The fits were built from our own BF16 conversion
and carry a different embedded chat template (9993 characters) than the
baselines (8945). Different prompt formatting between the two groups would
invalidate the whole table.

It does not, for this eval. The harness sends exactly one system message and
one user message and uses no tools. Rendering both templates on that input
produces byte-identical output — 563 characters, same hash, both resolving
`reasoning_effort` to `xhigh`. Every difference between the two templates lives
in a path this benchmark never takes: multi-system-message merging, tool-call
argument handling, and a `high`→`xhigh` alias.

## Benchmark

92 questions, four sources: AIME2025 (25), GPQA Diamond (25), SuperGPQA (25),
COMPSEC (17).

Every leg ran on the same RTX PRO 6000 Blackwell at a 250 W cap, the same
llama.cpp build (22b8e31), the same MTP draft model for speculative decoding,
`q4_0` KV cache, flash attention on, temperature 0, seed 42, 32768 max tokens
per answer.

One caveat you will notice in the published CSVs: 24 rows are tagged
`GPQA Diamond` and one is tagged `GPQA Diamond (modified)`. That marker is
inherited verbatim from the upstream ds4 corpus — it is antirez's annotation,
not a change of ours — and the item is counted within the 25. So a reader
tallying `source == "GPQA Diamond"` gets 24, not 25.

| leg | bpw | ctx | accuracy | pass | wrong | exhausted | median case | median completion tok |
|---|---|---|---|---|---|---|---|---|
| `Qwen3.8-27B-Q8_0` (stock) | 8.500 | 131072 | 83.7% | 77 | 7 | 8 | 51s | 3475 |
| `Qwen3.8-27B-IQ4_XS` (stock) | 4.596 | 131072 | 84.8% | 78 | 5 | 9 | 42s | 3721 |
| fit-128k | 6.099 | 131072 | 88.0% | 81 | 5 | 6 | 53s | 3612 |
| fit-192k | 5.509 | 196608 | 88.0% | 81 | 4 | 7 | 56s | 4178 |
| fit-256k | 4.920 | 262144 | 87.0% | 80 | 5 | 7 | 63s | 4436 |
| fit-17g | 2.876 | 131072 | 67.4% | 62 | 9 | 21 | 64s | 6464 |

By source:

| leg | AIME2025 | COMPSEC | GPQA Diamond | SuperGPQA |
|---|---|---|---|---|
| Q8_0 | 84% (21/25) | 100% (17/17) | 80% (20/25) | 76% (19/25) |
| IQ4_XS | 88% (22/25) | 100% (17/17) | 76% (19/25) | 80% (20/25) |
| fit-128k | 96% (24/25) | 88% (15/17) | 84% (21/25) | 84% (21/25) |
| fit-192k | 92% (23/25) | 94% (16/17) | 84% (21/25) | 84% (21/25) |
| fit-256k | 92% (23/25) | 100% (17/17) | 80% (20/25) | 80% (20/25) |
| fit-17g | 64% (16/25) | 100% (17/17) | 48% (12/25) | 68% (17/25) |

Note that the three 23.5 GiB fits each ran at the context they were solved for,
while the stock quants and fit-17g ran at 131072. That is deliberate — running
a fit at its target context is the thing the fit exists to do.

The comparison worth looking at twice is fit-256k against stock IQ4_XS: 4.920
bpw against 4.596, 15.66 GiB against 14.63, and 87.0% against 84.8% — but
fit-256k carried a 262144-token context while IQ4_XS ran at 131072.

### Read the top five rows as one cluster

The fits come out a few points above Q8_0. **That is not a result.** The
comparison is paired, so only questions where two legs disagree carry
information, and there are very few of them: fit-128k versus Q8_0 is 8
discordant questions out of 92, splitting 6–2 in the fit's favour, McNemar
exact p = 0.29. Every pairwise comparison among those five legs is
non-significant.

Simulating the exact test at the discordance rate these legs actually produce,
the power at 92 questions is **0.17**. Eighty percent power would need about
400 questions. Even a perfect effect — every discordant question falling the
same way — only reaches 0.82 at this sample size.

So the defensible claim is **no measurable accuracy loss against Q8_0 down to
4.920 bpw**, and explicitly *not* that the fits are better than Q8_0. If you
want a ranking among those five, this benchmark cannot give you one.

### What happens at 2.876 bpw

The 2.876 bpw file is the one comparison here that clears the bar. Against
Q8_0: 19-vs-4 discordant, p = 0.003, power 0.88. Against fit-256k: 20-vs-2,
p = 0.0001.

But *how* it degrades is more useful than the headline. Its answered-only
accuracy is 87.3% (62/71) against Q8_0's 91.7% (77/84) — a gap of about 4
points, not 16. The headline gap is mostly **exhaustion**: 21 cases ran out of
the 32768-token budget before committing to an answer, against 6–9 for every
other leg, and its median completion length is 6464 tokens against 3475–4436.

It is not mostly answering wrongly, and it is not uniformly slower to think
either. On the questions it finishes, its median completion is 3818 tokens
against Q8_0's 2956 — about 29% longer, not double. The doubled headline median
is the *upper tail*: its third quartile is 28824 tokens, against 11k–13.5k for
every other leg (exclusive quartiles). Most questions look normal; a subset runs away and hits the
wall.

The collapse is concentrated where the reasoning is longest: GPQA Diamond falls
to 48% while COMPSEC stays at 100%.

A larger token budget would likely recover part of that 16 points. We have not
tested that, and until someone does, treat 67.4% as "this configuration, at
32768 tokens" rather than as the model's ceiling.

### The same questions fail in every leg

Across all six legs, 58 exhaustion events land on only 22 distinct questions,
and 4 questions exhaust in **all six** legs — including Q8_0. Under
independence with the same per-leg counts you would expect 0.000 questions to
exhaust everywhere, and a permutation test over question ids gives p < 5e-6
(200,000 trials).

Some questions are simply long-reasoning traps, and that property survives
across quant families and across a 3x range in bits per weight.

## Running them

```
llama-server \
  -m qwen38-27b-fit-128k-q4kv.gguf \
  -md Qwen3.8-27B-MTP-ONLY-Q8_0.gguf --spec-draft-n-max 3 \
  -c 131072 -ctk q4_0 -ctv q4_0 -fa on \
  -ngl 99 -ngld 99 --host 127.0.0.1 --port 8083
```

Set `-c` to the context the file was solved for. The KV type is not optional:
these were solved against `q4_0` KV, and running f16 KV changes the memory
arithmetic the fit was built around.

`-ngld 99` matters — without it the draft model runs on the CPU and speculative
decoding ends up slower than not doing it at all.

The draft model used here is an MTP-only Q8_0 extraction from the same base.
It is not included in this repo.

## How the fits were made

```
shoehorn --ctx 131072 --kv q4_0 --budget 23.5GiB --calibrate
shoehorn --ctx 196608 --kv q4_0 --budget 23.5GiB --calibrate
shoehorn --ctx 262144 --kv q4_0 --budget 23.5GiB --calibrate --force-calibrate
shoehorn --ctx 262144 --kv q4_0 --budget 17GiB   --calibrate --force-calibrate
```

Starting from a BF16 GGUF and an imatrix generated with `llama-imatrix` on
bartowski's calibration text.

`--calibrate` is the part that does the work. It writes the file, launches
llama.cpp once, reads back the *measured* allocations and re-solves against
them. At 128K the KV estimate was 9.14 GiB and the real allocation was 2.25 GiB
— most layers in a hybrid model don't keep a full-attention cache — and the
re-solve handed 5.75 GiB back to the weights.

`--force-calibrate` is our own patch, on a fork. Three estimate-based gates
bail out before `--calibrate` ever runs, which means the tool can refuse a job
on the strength of an estimate it is about to discover is wrong. The flag turns
those gates into warnings. Both 256K fits need it.

On a multi-GPU box, pass `--budget` explicitly — the NVML probe reads device 0
only.

## What's in this repo

Per-leg results for all six benchmark legs live under `bench/`: the summary
JSON, the per-case CSV (question id, source, domain, pass/fail, token counts,
timings) and a `config-public.txt` recording the exact server command, sampler
settings, build and driver version for that leg.

The evaluation harness itself is ours and is not public, so the eval is not
reproducible from this repo — the inference side is (the server command is
recorded per leg, and the question sources are named above).

The raw model outputs are **not** published. They quote the benchmark questions
verbatim, and 25 of the 92 come from GPQA Diamond, whose authors ask that
examples not be reproduced online. The per-case CSVs carry question ids and
results but no question text.

## Credits and licences

- **Base model**: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
  by the Qwen team, Apache-2.0. These are derived weights and the base model's
  licence terms govern them; the quantization does not change what you may do
  with the model.
- **Quantizer**: [shoehorn](https://github.com/notactuallytreyanastasio/shoehorn),
  MIT. The budget solver, the calibration loop and the per-tensor format
  assignment are all its work. The `--force-calibrate` flag is a local patch on
  a fork, not upstream.
- **Evaluation**: [antirez/ds4](https://github.com/antirez/ds4) ("DwarfStar")
  by Salvatore Sanfilippo, MIT — an inference engine for DeepSeek V4 Flash and
  GLM 5.2 on consumer hardware. Our harness is a Rust port of its `ds4_bench.c`
  and `ds4_eval.c`, and the 92-question selection is antirez's. These numbers
  would not exist without it.
- **Inference**: [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT.

Benchmark questions originate with their respective authors: AIME2025, GPQA
Diamond, SuperGPQA, and a computer-security set. Because the question set is
public, treat these numbers as a *relative* comparison between quantizations
run under identical conditions — not as leaderboard-comparable scores.
