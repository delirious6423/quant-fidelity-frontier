---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model:
- google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
library_name: gguf
language:
- en
tags:
- gguf
- gemma4
- moe
- roleplay
- abliterated
- uncensored
- rimworld
- not-for-all-audiences
---

# gemma4-rp-maxxed (Q3_K_XL GGUF)

A roleplay build of Google's **Gemma 4 26B-A4B-it** made for
[HermesVoice](https://github.com/CharlesMod/RimWorldLLM), a RimWorld mod that gives colonists
their own voices. It has refusals removed, is pruned to the roleplay domain, and fits a
**16 GB card with three 64k-context slots and no expert offload**.

> **Adult content.** This model is decensored. It will write grim, violent and sexual content
> between adults when asked. It is not a general assistant and not suitable for minors.

| File | Size | sha256 |
|---|---|---|
| `gemma4-rp-maxxed-Q3_K_XL.gguf` | 10,001,779,136 B | `595eddca7b75d56ee4a6fa363deb27f1c70c8792b97508e3336a3a3d3f071b29` |
| `gemma-4-26B-A4B-it-abl-lora.gguf` (refusal-removal edit only, as a LoRA over stock unsloth UD-Q3_K_XL; no prune) | 101,540,384 B | `af207ae96e2e21015a964be1909df743c4f86e94a5e808da86a105d6be97448e` |

- **Base:** google/gemma-4-26B-A4B-it @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`; quantisation is
  unsloth's UD-Q3_K_XL recipe with unsloth's imatrix.
- **Architecture:** gemma4 MoE, 30 layers, **96 experts** (pruned from 128), top-8, sliding window
  1024, vocabulary 158,191 (pruned).
- **Runs on:** llama.cpp (tested on b11279). Other GGUF runtimes that follow llama.cpp's gemma4
  support should work; the pruned expert and vocabulary counts are ordinary metadata.

## Quick start

```bash
llama-server -m gemma4-rp-maxxed-Q3_K_XL.gguf -ngl 999 -np 3 -c 196608 \
  -ctk q4_0 -ctv q4_0 -fa on --jinja
```

On a 16 GB card that also runs a game, add `--n-cpu-moe 3` to move three layers of experts to
system RAM (about 0.7 GB of VRAM back). Thinking is off by default with Gemma 4's chat template;
keep it off for short dialogue lines.

The HermesVoice engine downloads this file, picks the context and offload for your card, and
starts and stops it with the game. You don't need to run it by hand.

## What was changed

1. **Refusal removal (abliteration).** Heretic-style ARA, reimplemented layer by layer so it runs on
   16 GB. It edits layers 11–23: `attn_output`, the dense `ffn_down` and all 128 expert
   `ffn_down_exps` slices. A preserve term on harmless prompts and ordinary roleplay text keeps
   the edit small (largest per-matrix |ΔW|/|W| 0.196). The 36 edited tensors are re-encoded to the
   donor's types (Q8_0, IQ4_NL) with unsloth's imatrix.
2. **Expert prune, 25 %.** Each layer keeps the 96 experts with the highest REAP-style saliency
   (frequency × gate × output norm) measured on RimWorld roleplay text. Router rows, expert
   tensors and per-expert scales are sliced. Top-k stays 8.
3. **Vocabulary prune.** It keeps tokens that are ASCII or contain common typographic
   punctuation and Latin-1 accents, plus every control and special token. Merges are filtered and
   special-token ids remapped. Tokenisation of ASCII text is identical to the stock model
   (400/400 checked, same top-1 on 50/50 snippets).

## Measured

| | stock UD-Q3_K_XL | this model |
|---|---|---|
| Adult roleplay probe, refusals | 95/100 | **0/100** |
| Generic harmful set, strict refusals | 100/100 | 2/100 (about a third still add a disclaimer before complying) |
| Voice distinctness (HermesVoice M0 bench) | 0.926 | **0.944** |
| Intent accuracy with grammar (M0) | 27/30 | **28/30** |
| Grim RimWorld lines: refusal / disclaimer / moralising | 0 / 0 / 0 | 0 / 0 / 0 |
| VRAM, `-np 3 -c 196608`, q4_0 KV | 13,110 MiB at `--n-cpu-moe 6` | **12,200 MiB at `--n-cpu-moe 0`** |
| Decode speed (RTX 5080) | 77 t/s | **148 t/s** |
| Text KLD vs stock: roleplay / RimWorld / general | 0 | 0.68 / 0.96 / 1.88 |

KLD has a heavy tail on this architecture: llama.cpp itself gives a KLD of 0.26 between two
equivalent runs of the stock model at different `-ub` sizes, so read that row as a ranking. The
general-text drift is the expected price of the expert prune.

## Intended use and limits

- **Intended:** in-character dialogue and narration for games and interactive fiction, including
  grim and adult themes between consenting adult characters.
- **Not intended:** general assistance, factual questions, code, or languages other than
  English. Pruning removed experts used by general text. Text outside ASCII and common
  punctuation (other scripts, emoji) falls back to bytes and is worse.
- **Prohibited:** any sexual content involving minors. The build's probes, prompts and
  calibration corpora contain no sexual content involving minors, animals, non-consent or incest.
  You are responsible for how you use the outputs, under the law where you live and under the
  [Gemma terms](https://ai.google.dev/gemma/docs/gemma_4_license).

## License

Apache 2.0, as for the Gemma 4 base model. This is a derivative of Google's Gemma 4; it is not
made or endorsed by Google.
