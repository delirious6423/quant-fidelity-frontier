---
library_name: transformers
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
license: mit
pipeline_tag: text-generation
language:
- en
- ru
- zh
- es
tags:
- deepseek-v4
- mixture-of-experts
- pruning
- ream
---

# DeepSeek V4 Flash — REAM144 (163B)

A slimmed-down DeepSeek-V4-Flash-0731: **144 of the original 256 experts per layer**,
picked and carried over byte-for-byte. Same 43 layers, same shared expert, same top-6
routing — just a smaller crew. The 2-bit build of this checkpoint targets 64 GiB
machines; this repo is the full-precision source of that build.

> [!TIP]
> **The method is a REAP × REAM hybrid — not purely either.** Experts are *pruned*
> REAP-style: ranked by how much they actually contribute on real traffic, with the
> survivors copied over byte-for-byte, untouched. But the keep-list isn't a plain
> top-N — every domain (Russian, code, tool use, math…) gets a protected quota of its
> own specialists. And the few experts that turned out to be near-duplicates were
> *merged* REAM-style instead of dropped. The router is then re-balanced so the
> smaller crew is used the way the original was. All of it in one step from the
> original model — no cascades.

> [!IMPORTANT]
> This checkpoint keeps the original packed FP4/FP8 weight layout and the custom 0731
> architecture. It is **not** loadable with stock `transformers` generation, vLLM or
> SGLang — it exists for the [DS4 fork](https://github.com/iamwavecut/ds4) toolchain
> and for making quantized builds. Want something you can just run? Grab the ready
> 2-bit build: [DeepSeek-V4-Flash-0731-REAM144-163B-DS4-GGUF](https://huggingface.co/WaveCut/DeepSeek-V4-Flash-0731-REAM144-163B-DS4-GGUF).

> [!WARNING]
> Live smoke testing passed 7/10 scenarios on the first run. Independent reruns show the failures (**Tool calling (DSML)**, **Code refactoring**, **Tool call → code chain**) are **intermittent, not absolute** — see the Stability column below for per-scenario pass rates. Multilingual chat, reasoning and long dialogs are consistently healthy.

## What's in the repo

≈96 GB of sharded Safetensors plus config and tokenizer. Everything follows the
original 0731 format, so tooling that understands the base model understands this one.
The config and tokenizer load fine with `AutoConfig` / `AutoTokenizer`; per-layer
pruning choices are recorded in `SELECTION.json`.

## DSpark

The model's built-in speculative decoder (three extra MoE stages that draft tokens
ahead) is preserved untouched under its `mtp.*` tensor namespace. In the GGUF release
it ships as a separate optional file — see the companion repo.

## How it was made

One pruning step, straight from the original — no cascading. Expert importance was
measured by running [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
over a ~5-million-token calibration mix (multi-turn dialogs, thinking and direct modes,
rendered with the model's own chat encoder). The strongest experts of every domain were
protected from pruning, the survivors were carried over **byte-identical**, and the
router was re-balanced to keep the original selection behavior.

| Calibration domain | Share |
| --- | ---: |
| Code | 35% |
| Agentic / tool use | 19% |
| Multilingual chat | 16% |
| Math | 8% |
| General chat | 6% |
| Roleplay | 6% |
| Russian | 5% |
| Long docs | 4% |

This line replaces the earlier cascaded REAM builds (now archived under `-exp` names),
which degraded badly in multi-turn use.

## Smoke results

Every scenario is a live multi-turn conversation, run on the companion 2-bit GGUF build — the only runnable form of this checkpoint. Treat the results as a lower bound for this full-precision source (raw evidence ships in the companion repo's `SMOKE_REPORT.json`).

| Scenario | First run | Stability (reruns) |
| --- | :---: | :---: |
| Russian wordplay, multi-turn | ✅ | — |
| English → Russian code-switching | ✅ | — |
| Code Q&A over a 4k-token file | ✅ | — |
| Tool calling (DSML) | ❌ | 5/10 |
| Russian multi-turn reasoning | ✅ | — |
| Spanish creative writing | ✅ | — |
| Code refactoring | ❌ | 8/10 |
| Chinese summarization | ✅ | — |
| Long-dialog focus (drift check) | ✅ | — |
| Tool call → code chain | ❌ | 5/10 |

Stability = pass rate over independent reruns of the scenarios that failed the first run; passing scenarios were not re-run.

## Limitations

- Runs only through the DS4-fork ecosystem; this repo is the archival/source form.
- Pruning is training-free: rarely-used specialist skills of the removed experts are
  gone by design. See the smoke table for what was verified.
