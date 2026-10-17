---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - qwen3.8
---

# Qwen3.8-27B — GGUF quants, with the point where they stop driving tools

Quants of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). What this repo adds over
the other GGUF builds of the same model is not the files: it is a measured answer to *how far down
you can quantize before an agent stops working*, and the honesty about how noisy that measurement is.

## 📦 Files

| file | size | bits/weight | agentic verdict |
|---|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | 16.81 GB | — | not measured on the benchmark below |
| `Qwen3.8-27B-IQ4_XS.gguf` | 15.31 GB | — | not measured on the benchmark below |
| `Qwen3.8-27B-Q3_K_M.gguf` | 13.50 GB | 3.40 | 40/40 — works |
| `Qwen3.8-27B-IQ2_M.gguf` | 10.00 GB | 2.93 | **best value — still drives tools** |
| `Qwen3.8-27B-IQ2_XS.gguf` | 9.09 GB | 2.66 | 72/80 — measurably worse, saves little |
| `Qwen3.8-27B-IQ1_M-compact.gguf` | 6.46 GB | 1.94 | fluent text, **not** for agents |

The two top rungs sit above the 3.40-bpw build that scores 40/40, but they were not put through the
benchmark themselves — "not measured" here means exactly that, not "assumed fine".

## 🌐 Running it in a browser

`IQ1_M-compact` is the build that [Elffuss Claw](https://claw.elffuss.com) loads on WebGPU —
one tab, one machine, no install and no sharding across devices — through
[Elffuss Engine](https://github.com/KikoCis/elffuss-engine) (Apache-2.0), which reads a **standard
GGUF over HTTP** by range requests and dequantises inside the shader. It is slow and the first load is
several gigabytes; it answers, it does not feel instant. Note the row above: at 1.94 bpw this build is
for text, not for driving tools.

## 🧠 Why the context is cheap on this model

Of its 65 blocks: **48 are SSM / linear-attention**, **16 are full attention** (every 4th), and
**1 is an MTP head**. Only those 16 layers hold a KV cache that grows with sequence length:

```
16 layers × 4 kv-heads × (256 + 256) × 2 bytes = 64 KB per token
```

A dense model with all 65 blocks attending would pay roughly **256 KB/token** — four times as much.
Full arithmetic and per-context tables in [`CONTEXT.md`](CONTEXT.md).

⚠️ **The flip side, measured**: the same architecture makes *prefill* slow in llama.cpp today. In
multi-turn agent loops, where the conversation is re-processed every turn, this model spends far more
time on prefill than a dense model of the same size. Cheap in memory, expensive in time.

## The agentic cliff — where this model stops working

Every rung below was measured on the same objective tool-use benchmark (40 tasks, checked by running
the produced artifact — no LLM judge), with the sampling profile the model declares in its own
metadata (`temp 1 / top_p 0.95 / top_k 20`).

| bits/weight | file | agentic tasks | verdict |
|---|---|---|---|
| 3.40 | `Q3_K_M` | **40/40 — 100 %** | works |
| **2.93** | **`IQ2_M`** | **79/80 — 98.8 %** | **best value** |
| 2.66 | `IQ2_XS` | 72/80 — 90.0 % | measurably worse, saves nothing |
| 2.23 | `IQ1_M` (uniform) | 60/160 — 37.5 % | **broken for agents** |
| 1.94 | `IQ1_M-compact` | 49/160 — 30.6 % | **broken for agents** |

**The threshold sits between 2.66 and 2.23 bits per weight.** Above it the model completes multi-step
tool tasks essentially perfectly; below it, it solves roughly one task in three — and the dominant
failure is not a wrong answer, it is emitting no tool call at all.

- **Want an agent? Take `IQ2_M`.** It matches the 3.40-bpw build at **28 % less size**.
- `IQ2_XS` is worse on a paired test (6 tasks where `IQ2_M` wins, 0 the other way, *p* = 0.031) while
  being only 9 % smaller.
- **The 1-bit rungs are published for completeness, not for agentic use.**

### Three things worth knowing about how this was measured

**Single runs lie.** Re-running one unchanged build through the same 40 tasks four times gives
17, 12, 16, 15 — a five-task spread, wider than most differences between builds. Every row above
2.23 bpw is at least two rounds; the 1-bit rows are four.

**Fidelity does not predict capability.** Against the BF16 original, the 3.40-bpw build disagrees on
the next-token argmax for 9.5 % of positions and still scores 40/40. Perplexity and KL-divergence are
fidelity data, not capability proxies.

**Every number above is tied to a backend, not just to the file.** All the rows were measured
through Ollama. Re-running `IQ1_M-compact` through llama.cpp's `llama-server` — same file, same
tasks, same sampling profile — scores 19/40 (47.5 %) instead of the 30.6 % listed. That is seven
tasks, outside the five-task noise band above. Treat every figure here as "this build, on this
runtime", and never compare it against a number someone else measured on a different one.

### On `IQ1_M-compact` specifically

The smallest build here, 0.27 GB below the community reference. Measured: **no pairwise difference is
statistically significant** (paired sign test, *p* = 0.473 vs the reference; *p* = 0.424 vs our larger
1-bit build) — but it ranks last on all three central estimates, and this benchmark lacks the
resolution to separate builds at that size. **Absence of a detected difference is not evidence of
equivalence.**

## ❌ What is NOT verified — read before relying on it

- **The 262,144 context is the model's declared native window, not one we verified.** No
  needle-in-a-haystack run was completed.
- **Text only.** The base model is multimodal; this repo ships **no `mmproj`**, so vision inputs will
  not work. Other publishers ship one — see below.
- **No fidelity table per tier.** If you want a release where fidelity *is* measured tier by tier
  against the BF16 reference, see [Qwen3.6-27B-GGUF](https://huggingface.co/KikoCis/Qwen3.6-27B-GGUF).

## 🚀 How to run it

```bash
llama-cli -m Qwen3.8-27B-IQ2_M.gguf -c 32768 -no-cnv -p "your prompt"
```

`-no-cnv` matters: without it, recent llama-cli builds drop into conversation mode and wait on stdin,
which looks exactly like a hang when scripted.

**Sampling** — Qwen documents two profiles, and the choice matters:

| mode | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| **Thinking** (use this for agentic/reasoning work) | 1.0 | 0.95 | 20 | 0.0 |
| Instruct (non-thinking) | 0.7 | 0.80 | 20 | 1.5 |

The Modelfiles in [`configs/`](configs) use the **thinking** profile at 8K / 32K / 128K / 262K.

```bash
ollama create qwen3.8-27b-32k -f configs/Modelfile.32k
```

## 🔁 Provenance

Two lineages live in this repo. Both are faithful quantizations of the same released weights —
neither is a fine-tune:

- `Q3_K_M`, `Q4_K_M`, `IQ4_XS` — quantized from a local conversion of the original safetensors.
- `IQ2_M`, `IQ2_XS`, `IQ1_M-compact` — quantized from a BF16 GGUF conversion published by
  [unsloth](https://huggingface.co/unsloth). Those rungs build on unsloth's conversion work.

Weights are otherwise unmodified. The quantization pipeline itself is not published.

## 🙏 Credit & alternatives

Model, weights and training are **© Qwen** — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
Apache-2.0. This repo adds only the quantization, the context configs and the measurements.

Other GGUF builds, several more complete than this one — they ship the full ladder and the `mmproj`
for vision: [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF),
[bartowski](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF),
[ggml-org](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF),
[lmstudio-community](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF).


<!-- heal:start -->
## 🩹 Daily heal (LoRA)

A LoRA adapter retrained **every day** to make Qwen3.8-27B a steadier coding agent: read before answering, edit only the lines that change, verify the fix, cite real files. It is a separate file — the quant ladder above stays the faithful, unmodified quantization — and it applies to **any** GGUF in this repo:

```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf --lora heal/Qwen3.8-27B-heal-LoRA.gguf --jinja -c 32768
```

| Version | Date | Planted bugs fixed | Valid tool calls | Finished | Avg. steps |
|---|---|---:|---:|---:|---:|
| original Q4_K_M | — | 8 % | 92 % | 17 % | 10.75 |
| heal v1 | 2026-10-13 | 25 % | 92 % | 25 % | 10.75 |

Measured (results appear in the table as each day's evaluation completes) on 12 held-out bug-fix tasks — one-token bugs planted in real functions of public repos never used for training, counted only if the file ends byte-identical to the original — with native tool calling, temperature 1.0 / top_p 0.95, on the Q4_K_M file above plus that day's adapter. Every daily version is a git tag (`heal-<date>`). Built alongside [elffuss.com](https://elffuss.com).
<!-- heal:end -->

## 🗒️ Changelog

- 2026-10-13 — daily heal LoRA (heal v1): 25 % planted bugs fixed on the held-out set.
- **2026-09-20** — `IQ2_M` re-uploaded without the inherited unsloth repo/tag fields.
- **2026-09-20** — added the backend caveat: the scores in this card were measured through Ollama, and the same `IQ1_M-compact` file scores 19/40 through llama-server. Seven tasks apart — the figures describe a build on a runtime, not a file.
- **2026-09-19** — `IQ1_M-compact` re-uploaded without the unsloth repo/tag fields it inherited from the source conversion — they credited the wrong party for this build.
- **2026-09-19** — provenance corrected: the sub-3-bit rungs come from unsloth's BF16 conversion, the others from a local safetensors conversion. The card claimed one lineage for all of them.
- **2026-09-19** — added `IQ2_XS` (9.09 GB) — the rung below `IQ2_M`, already in the cliff table, now downloadable so the threshold claim can be checked.
- **2026-09-16** — larger rungs withdrawn for re-upload with clean metadata; pipeline details removed
  from the card.
- **2026-09-18** — `Q3_K_M`, `IQ4_XS` and `Q4_K_M` back, with metadata carrying no build-machine paths.
- **2026-09-12** — added `IQ2_M`: the smallest rung that still drives tools (79/80).
- **2026-08-16** — first release.
