---
library_name: transformers
license: apache-2.0
language: multilingual
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - qwen3
  - qwen3.8
  - qwen
  - llm
  - text-generation
  - conversational
  - multimodal
  - vision
  - image-text-to-text
  - speculative-decoding
  - dflash
  - dflash2
  - blackwell
  - nvfp4
  - rtx-5090
  - "5090"
  - reasoning
  - coding
  - agent
  - benchmark
  - gsm8k
  - perplexity
---

# Qwen3.8-27B-ColdFusion-GAIN-Blackwell-DFlash2-Ultra-V1.0

**2.4× faster per task · up to 200+ tok/s · HumanEval 84% → 94% · GSM8K 94% → 96%**

A 27B hybrid text + image model set up to run fast on one RTX 5090 (32 GB, `sm_120`) with
llama.cpp CUDA and DFlash2 speculative decoding. 262K context.

> **Update 2026-09-11** — new measured data: **CUDA graphs are worth +33 % and can stay
> enabled** (this page previously advised disabling them, which costs a third of the decode
> throughput), a reasoning-effort sweep on GSM8K, the real NVFP4-vs-Q4_K_M quality/speed
> trade, a full speculative-decoding knob sweep, and the multi-agent cache fix. Everything
> below is measured on the same box, with the conditions stated next to each table.

## The gain, concretely — time and speed

Same machine (RTX 5090 32 GB + 128 GB), same prompts, greedy. Base = `unsloth/Qwen3.8-27B-GGUF` **UD-Q4_K_M** as-is (no speculation).

| Task | Base Q4_K_M | This model | Gain |
|---|---:|---:|---:|
| **HumanEval problem** (code) | 11.3 s | **4.8 s** | **2.4×** |
| **GSM8K problem** (math) | 6.3 s | **2.5 s** | **2.5×** |
| Decode — code | 66.7 tok/s | **125.7 tok/s** | **+88%** |
| Decode — math | 69.3 tok/s | **127.8 tok/s** | **+84%** |
| Decode — copy / edit (peak) | ~70 tok/s | **up to ~210 tok/s** | **~3×** |
| Tokens / problem (code) | 753 | **602** | −20% |
| Tokens / problem (math) | 438 | **323** | −26% |

A full **HumanEval-50 run: 9m25s → 4m00s**. A **GSM8K-50 run: 5m16s → 2m06s**.

## Accuracy — held, not sacrificed

| | Base Q4_K_M | This model |
|---|---:|---:|
| HumanEval-50 pass@1 | 84.0% | **94.0%** |
| GSM8K-50 | 94.0% | **96.0%** |

---

## NEW — Reasoning effort: measured on a stratified GSM8K-100

`reasoning_effort` is a chat-template kwarg. It changes how much thinking the model does
*before* it answers, which dominates wall-clock on real tasks far more than any kernel tuning.

Benchmark: **100 GSM8K problems, 5 slices across the whole 1319-problem test set** (offsets
0/300/600/900/1200, 20 each), `temperature 0`, answers checked numerically, `max_tokens 3072`.
Config: this model + DFlash2, `-np 1 -c 65536 -fa on`, KV q8_0.

| Config | GSM8K | s / problem | tok / problem | tok/s |
|---|---:|---:|---:|---:|
| **`reasoning_effort=medium`** | **98/100 = 98%** | **1.99** | 353 | **177** |
| `reasoning_effort=low` | 96/100 = 96% | 2.12 | 357 | 169 |
| TWIN-TURBO Fable `low` (Q4_K_M, MTP draft) | 98/100 = 98% | 2.46 | **296** | 121 |
| TWIN-TURBO Fable `medium` (Q4_K_M, MTP draft) | 97/100 = 97% | 2.94 | 387 | 132 |

**Read this before picking a mode:**

- **`medium` is strictly better than `low` here**: +2 points *and* 6% faster (it wrote fewer
  tokens). The default of this model is `xhigh`; if you never set the kwarg you are running the
  slowest mode for no measured accuracy benefit on this benchmark.
- Use `--chat-template-kwargs "{\"reasoning_effort\":\"medium\"}"` on the server so clients get
  it by default. A client that sends `reasoning_effort` explicitly overrides it.
- Keep `max_tokens ≥ 4096` on hard tasks in `low`/`medium`; and note that a *low* effort can
  occasionally loop without concluding (observed once on a debugging task: 4096 tokens of
  reasoning, 0 tokens of answer — the same task answered fine in `medium`).
- A 4-task hard bench (temp 0, bug hunt / scheduling / system diagnosis / Bayes) done during
  the same session: GAIN answered **7 of 8 faster** than Fable (e.g. 9.5 s vs 11.8 s on the
  scheduling problem, both finding the optimal 13 h makespan; Bayes posterior 1.94% correct in
  both; the diagnosis task was answered by GAIN and timed out in Fable `low`). Both models got
  the ambiguous diagnosis task *wrong* — this is not a "perfect" model, it is a fast one.

## NEW — NVFP4 vs Q4_K_M: the actual trade

Both builds of the *same* finetune, same drafter, same everything else. Perplexity:
**24 chunks × 2048 ctx**, KV in f16 to isolate the weight effect, llama-perplexity on a mixed
code/markdown corpus.

| Build | Size | Perplexity | Decode (temp 0.7, lmx protocol) |
|---|---:|---:|---:|
| this repo — **NVFP4** + Q8_0 head/embed | 16.9 GB | 2.8059 ± 0.034 | **97.4 tok/s** |
| **Q4_K_M** of the same finetune + DFlash2 | 18.5 GB | **2.7325 ± 0.033** | 84.3 tok/s |

**+15.5% decode for +2.7% perplexity.** Measured twice, interleaved, same window.

Why NVFP4 costs quality at all — it is structural, not a bad quant: on Blackwell the NVFP4
path is **W4A4** (the *activations* are also FP4) while Q4_K_M runs **A16**. See llama.cpp PR
#24364 ("force W4A8 path for NVFP4_W4A16 layers on Blackwell, where NVFP4 normally uses the
native W4A4 path"). So high-quality NVFP4 recipes need either W4A8/A16 dispatch or more
high-precision tensors — which costs back the bytes you saved.

Related measurements from the same session (Fable, requantized to NVFP4 by hand):

| Variant | Perplexity | Notes |
|---|---:|---|
| Q4_K_M (the bar) | 2.7660 | |
| NVFP4 + imatrix scale search | 2.8016 | 16.0 GB — best speed/quality NVFP4 trade |
| NVFP4 legacy fixed `amax/6` | 2.8231 | |
| NVFP4 + FFN/attention in Q6_K | **2.7573** | beats the bar, but 22 GB so it is *slower* |

Quantizer notes worth knowing if you build your own: the imatrix-aware NVFP4 scale search is
upstream PR #25153 (discrete search over ±12 UE4M3 codes around `amax/6`, optionally
importance-weighted). A weighted-LSQ / coordinate-descent scale solve was tried and is
**worse** (+3.0% perplexity here), matching the PR author's own finding. The single best
quality-per-byte move found was putting `output.weight` in Q4_K rather than Q8_0: it removes
3.6 ms from every draft phase (the MTP/DFlash head is read once per draft step) for +0.14%
perplexity.

## NEW — CUDA graphs: +33 %, and they can stay ON

This page used to advise `GGML_CUDA_DISABLE_GRAPHS=1` to dodge a CUDA illegal-memory-access crash
seen on some builds. Measured on this box, that advice costs **a third of the decode throughput**.

Agent profile (`temperature 0`, `reasoning_effort=medium`, 400-token code generation, best of 2
runs, two rounds, repeatability ±0.2 %):

| Setting | tok/s | draft ms | verify ms |
|---|---:|---:|---:|
| **CUDA graphs ON** (default) | **125.70 / 126.43 → 126.1** | 3.91 | **24.14 / 24.15** |
| CUDA graphs OFF (`GGML_CUDA_DISABLE_GRAPHS=1`) | 94.80 / 94.46 → 94.6 | 4.66 | **33.37 / 33.51** |

**+33 % decode**, and the entire gain sits in the *verify* pass: 33.4 → 24.1 ms per step, i.e.
**~9 ms of every forward pass was pure kernel-launch latency** (an eager 27B graph is on the order
of a thousand launches; a captured graph replays them in a single submission). The draft pass
barely moves (4.66 → 3.91 ms) because the DFlash2 drafter is **not** graph-captured — so there is
still ~2–3 ms of launch overhead left on the table if the drafter's graph were captured too.

Stability (the original concern): a sustained run with graphs ON and **2 slots alternating** (the
worst case for graph re-capture) pushed 7200 tokens at 121.1 tok/s with **zero CUDA errors** and a
healthy server, on the *stock* `llama-cuda` build. Together with the rest of this session (200+
timed runs, a 100-problem GSM8K sweep, perplexity runs, multi-slot tests) on a second build,
graphs never caused a failure here.

**Leave graphs enabled** (do not set the variable). If you do hit an illegal memory access on your
build, `GGML_CUDA_DISABLE_GRAPHS=1` is a fallback — you are trading 33 % of throughput for a
build-specific bug, and it is worth reporting upstream instead.


## NEW — Speculative decoding: what actually helps

Agent profile: `temperature 0`, `reasoning_effort=medium`, 400-token code generation, best of
2 runs, two full rounds (repeatability was ±0.2%).

| Setting | tok/s | draft ms | verify ms | Verdict |
|---|---:|---:|---:|---|
| Baseline (`--spec-draft-n-max 7`, KV q8_0) | 126.0 | 4.23 | 24.10 | reference |
| **Draft KV in f16** (`--spec-draft-type-k/-v f16`) | **127.3** | 3.88 | 24.18 | **+1.1%, free → recommended** |
| **Both KVs in f16** (`-ctk f16 -ctv f16`) | **130.9** | 3.90 | 23.50 | **+3.9%**, but KV doubles (see below) |
| Draft model in NVFP4 (`...-draft-DFlash2-NVFP4.gguf`) | 123.3 | **5.08** | 24.40 | **slower than Q4_K_M — use the Q4_K_M drafter** |
| `--spec-type draft-dflash,ngram-simple` | 93.4 | — | — | −7%: n-gram drafts dilute the block drafts |
| `--spec-draft-p-min 0.6` | 66.6 | — | — | **−34%** |
| `--spec-draft-n-max 4` | 111.3 | 6.73 | 20.78 | −12% |
| `--spec-draft-n-max 10 / 13` | 124.5 / 123.2 | 4.29 / 4.36 | — | ≤ baseline |
| **No drafter at all** (plain NVFP4 decode) | 57.4 | — | — | the DFlash2 is worth **2.2×** here |

Two findings that matter more than the flags:

1. **`n_max = 7` is already the optimum** (it is the drafter's block size; higher values are
   clamped, lower values break its block path — n=4 is *slower*).
2. **The verify pass is sub-linear**: ~1.1 ms per *additional* verified token at ~24 ms for the
   first column. Extra drafted columns are nearly free — the ceiling is the **acceptance rate**,
   not the draft cost. Mean accepted length measured: 2.08 tokens/step on code, 2.9–3.6 on
   agent/reasoning prompts. This is why drafter *quality* matters and drafter *speed* barely does.

## NEW — Multi-agent / long sessions (the biggest single win)

If several agents share the server (e.g. an IDE agent plus a CLI agent), the default prompt
cache is too small and the KV of the idle slot is dropped — the agent pays a full re-prefill
when it comes back:

```
slot.prompt_clear: id 1 | erasing 0 tokens     # every turn
```

Fix (host RAM, **zero VRAM cost**):

```
--cache-ram 32768 --no-cache-idle-slots
```

Measured on a 140k-token context pair of agents: **agent return 8.62 s → 0.32 s**, and in a
full run 78 s → ~1 s. `-np 1` is the default recommendation; use `-np 2`/`-np 4` only with the
flag below, since the drafter's context is not unified across slots.

### Two validated configurations (VRAM measured with the NVFP4 model + DFlash2)

| Config | Slots | Context per slot | VRAM | Notes |
|---|---|---:|---:|---|
| Single agent, max context | 1 | **262 144** | 29 011 MiB | `-np 1 -c 262144` |
| Two agents | 2 | **131 072** | 30 109 MiB | `-np 2 --kv-unified --kv-unified-per-slot 131072` |

`--kv-unified-per-slot 131072` is the flag that actually guarantees 131k *per slot*: the server
log then reads `sizing KV pool to n_parallel * kv_unified_per_slot = 2 * 131072 = 262144` and
`n_slots = 2, n_ctx_slot = 131072`. Without it, `-np 2` alone divides the pool.

If you want the +3.9% from an f16 target KV, reduce the context instead: f16 at 131 072 costs
8.4 GB of KV versus 9.1 GB for q8_0 at 262 144 — i.e. **the f16 build at half context uses less
memory than the q8_0 build at full context**, and decodes faster.

## Files

| File | Size | |
|---|---:|---|
| `...-NVFP4.gguf` | 16.9 GB | Model — NVFP4 backbone, Q8_0 `lm_head` + `token_embd`, MTP head included |
| `...-draft-DFlash2-Q4_K_M.gguf` | 1.14 GB | DFlash2 drafter — **the one to use** |
| `...-draft-DFlash2-NVFP4.gguf` | 1.09 GB | NVFP4 drafter — measured *slower* here (per-kernel overhead on a 1 GB model) |
| `...-mmproj-BF16.gguf` | 0.93 GB | Vision projector (optional) |

## Built from

- Base: `Qwen/Qwen3.8-27B`
- Finetune: `DavidAU/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1` (reasoning-oriented)
- NVFP4 quant: `esatapedico` — MID-HIGH tier (NVFP4 backbone, Q8_0 head/embed)
- Drafter: `z-lab/Qwen3.8-27B-DFlash2` — block-diffusion, lossless

DFlash2 accepts ~4–6 tokens per step. It's trained on the base Qwen3.8-27B, so it works with
any quant of it. Speculative decoding is lossless (rejection sampling).

## Run (llama.cpp CUDA)

```bash
# CUDA graphs stay ON (default) -> +33 % decode. Only if your build hits the old
# "CUDA illegal memory access" crash: set GGML_CUDA_DISABLE_GRAPHS=1 (costs ~33 %).
# set GGML_CUDA_DISABLE_GRAPHS=1    # Windows;  export on Linux/macOS

llama-server.exe \
  -m  ...-NVFP4.gguf \
  -md ...-draft-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  -ngl all -np 1 -c 262144 -fa on \
  -ctk q8_0 -ctv q8_0 \
  --spec-draft-type-k f16 --spec-draft-type-v f16 \
  --chat-template-kwargs "{\"reasoning_effort\":\"medium\"}" \
  --cache-ram 32768 --no-cache-idle-slots \
  --alias qwen38-ultra:latest --reasoning-format auto
# add --mmproj ...-mmproj-BF16.gguf for image input (disables prompt-cache reuse)
```

`--spec-draft-n-max 7` is the drafter's maximum (block_size 8); higher values are clamped.
The draft KV is f16 (+1.1% measured); the target KV stays q8_0 to keep 262K context.
Two-slot variant: replace the slot/context flags with
`-np 2 --kv-unified --kv-unified-per-slot 131072` and pick a different `--alias`
(comma-separated aliases are supported: `--alias "qwen38-ultra:latest,my-second-slot"`).

## Notes

- Thinking model — set `max_tokens ≥ 4096` on hard tasks.
- 262K context works; needles checked at 187K.
- CUDA graphs are ON by default here and measured **+33 %** (94.6 → 126.1 tok/s), stable across
  sustained 2-slot runs. `GGML_CUDA_DISABLE_GRAPHS=1` is a last-resort fallback, not a default.
- Ollama: see the provided `Modelfile`.
- Reproducing the numbers above: decode figures come from the `lmx` protocol (temperature 0.7,
  256-token generations) and from agent-profile runs at temperature 0; accuracy figures are
  GSM8K with numeric answer checking at temperature 0. Only interleaved A/B comparisons are
  trusted — separate measurement windows drift by up to 10% on this machine depending on
  background load (browser, torrent client), which is what makes single-run speed claims
  unreliable.
