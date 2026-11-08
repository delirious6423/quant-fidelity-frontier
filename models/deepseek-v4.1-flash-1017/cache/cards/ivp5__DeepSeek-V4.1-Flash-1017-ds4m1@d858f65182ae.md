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

**48 tok/s decode · 343 tok/s prefill** on a single Apple M1 Max with 64 GB — all 384 routed experts, the Engram tables,
thinking and tool calls. The experts stream from the SSD into a 38 GiB cache; everything else stays in memory.
Engine: [ds4-m1](https://github.com/ivp5/ds4-m1) (antirez's [ds4](https://github.com/antirez/ds4) + Metal, Neural Engine and
SSD-streaming patches, and a trained speculative drafter).

## How it compares

| | **ds4m1 (this repo)** | vcruz305 SAGE 1.55 bpw (EXL3 / REXL3) | Unsloth UD-IQ2_XXS (GGUF) |
|---|---|---|---|
| Runs on | **M1 Max, 64 GB** | 128 GB-class GPU (108.5 GiB resident) | llama.cpp, 300.5 GB of weights |
| Download | **287 GB** | 320 GB | 300.5 GB |
| Decode | **48.4 tok/s** | — | — |
| Prefill | **342.6 tok/s** | — | — |
| Routed experts vs official, same test (8 agent sessions) | **KL 0.047 · top-1 92.8%** | REXL3: KL 0.091 · top-1 89.9% | — |
| KL vs official on its card's own text | — | EXL3: 0.088 wiki1100, 0.098 e2e_eval; REXL3: 0.051 e2e_eval | 0.203 wiki1100, 0.142 long17k |
| Top-1 on its card's own text | — | EXL3: 88.2% wiki1100, 94.5% e2e_eval | 90.5% wiki1100, 91.2% long17k |
| Whole model vs Q8_0 reference (4 agent sessions) | KL 0.118 · top-1 90.5% | — | — |
| Coding-agent tasks | **15/15 solved, 822 tokens and 20 s per solution** | — | — |

— = not stated on that card. The same-test row swaps only the routed experts (attention, dense layers, router and head are
the official weights in both), so it compares the two expert codecs directly.

## Speed (M1 Max, 64 GB)

| Workload | Decode | Prefill | GPU clock |
|---|---:|---:|---:|
| 16 coding-agent tasks back to back (one process, ~7 min) | 48.4 tok/s | 342.6 tok/s | 1,061 MHz |
| same, steady tasks 5-15, a second run | 48.6 tok/s | 342.2 tok/s | 1,073 MHz |

## Fidelity

**Routed experts against vcruz305's SAGE REXL3 1.55 bpw** — official FP8/FP4 model as teacher, 11,263 positions of 8 agent sessions:

| Routed experts | KL mean | KL p50 | KL p90 | Top-1 |
|---|---:|---:|---:|---:|
| **ds4m1** (2-bit gate/up + 2-bit down) | **0.0467** | 0.00337 | **0.132** | **92.8%** |
| SAGE REXL3 1.55 bpw | 0.0906 | 0.00333 | 0.271 | 89.9% |

ds4m1 wins on all 8 sessions; the medians are equal and the gap is the tail.

**The whole published model** against the Q8_0 GGUF, teacher-forced on four coding-agent sessions (thinking / answer tokens split):

| Session | Positions | KL mean | KL p50 | Top-1 | Top-5 | Thinking KL / top-1 | Answer KL / top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| calc | 818 | 0.118 | 0.00088 | 89.9% | 69.8% | 0.212 / 81.8% | 0.039 / 96.6% |
| textutil | 536 | 0.053 | 0.00006 | 97.4% | 63.8% | 0.255 / 86.7% | 0.020 / 99.1% |
| roman (long thinking) | 1,254 | 0.147 | 0.01430 | 88.4% | 72.2% | 0.181 / 85.3% | 0.082 / 94.6% |
| matrix (long thinking) | 777 | 0.116 | 0.00083 | 89.6% | 70.1% | 0.198 / 82.2% | 0.046 / 95.9% |
| **pooled** | **3,385** | **0.118** | | **90.5%** | **69.8%** | | |

## Tokens per correct solution

Every generated token of every attempt, over the correct solutions:

| Task set | Solved | Tokens per solution | Seconds per solution |
|---|---:|---:|---:|
| 15 coding-agent tasks (Python modules with pytest suites) | 15 / 15 | 822 | 20.1 |

DeepSeek's own benchmark suites (HLE, Terminal-Bench, SWE, ...) were not re-run here: each takes days at 48 tok/s.

## Files

`hf download ivp5/DeepSeek-V4.1-Flash-1017-ds4m1 --local-dir ds4m1` — four files, 287 GB; verify with `shasum -a 256 -c MANIFEST.sha256`.

| File | Bytes | Holds |
|---|---:|---|
| `DeepSeek-V4.1-Flash-1017-ds4m1.gguf` | 23,344,348,224 | every non-routed tensor, plus the engine's small parts as I8 tensors: 4-bit GPTQ dense copies, 2-bit GPTQ head, router look-ahead, the drafter, the Neural Engine programs |
| `experts_gate_up_bs64.bin` | 99,090,448,384 | routed gate + up, 40 layers x 384 experts, 2-bit |
| `experts_down_g32.bin` | 53,791,965,184 | routed down, 2-bit |
| `engram_iq4nl.bin` | 110,595,319,712 | the two Engram tables, IQ4_NL |

## Run

```
./ds4-agent -m ds4m1/DeepSeek-V4.1-Flash-1017-ds4m1.gguf
```

Needs the ds4-m1 release with pack support (being pushed to GitHub; the 10-21 tree there predates it). The other three files are found beside the GGUF. The first start unpacks 13 GB of small parts into
`~/Library/Caches/ds4m1/` (`DS4M1_CACHE` moves it).

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
