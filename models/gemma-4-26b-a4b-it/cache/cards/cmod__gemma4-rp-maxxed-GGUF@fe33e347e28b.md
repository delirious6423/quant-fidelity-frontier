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
- roleplay-model
- rp
- creative-writing
- dialogue
- npc
- game
- llama.cpp
- 16gb
- gemma
---

# gemma4-rp-maxxed (Q3_K_XL GGUF)

**An uncensored roleplay and character-dialogue model that runs fully on one 16 GB GPU.** It is
Google's **Gemma 4 26B-A4B-it** (a mixture-of-experts model: 26B parameters, about 4B active per
token, so it is fast) with refusals removed, pruned to the roleplay domain, and quantised to fit
**three 64k-context conversations at once on a 16 GB card** with nothing offloaded.

Made for [HermesVoice](https://github.com/CharlesMod/RimWorldLLM), a RimWorld mod that gives every
colonist their own voice, but it is a general-purpose RP model too: short, in-character, grounded
lines; grim and adult themes between adults without lectures or disclaimers.

**At a glance**
- 10.0 GB file; 12.1 GB VRAM with 3 × 64k context (q4_0 KV); 105 tok/s per slot with 3 busy, more alone (RTX 5080).
- 0/100 refusals on an adult roleplay probe (stock: 95/100); better character distinctness than stock.
- English only. Not a general assistant: no code, facts or other languages (see limits).

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

## The HermesVoice model family

Three rungs, picked automatically by the [HermesVoice engine](https://github.com/CharlesMod/easyEngine) from your VRAM.
Each is a normal GGUF and works on its own in any llama.cpp-based front end.

| Model | File size | VRAM (3 × 64k ctx, q4_0 KV) | For |
|---|---|---|---|
| [**gemma4-rp-maxxed**](https://huggingface.co/cmod/gemma4-rp-maxxed-GGUF) (26B MoE, 4B active) | 10.0 GB | 12.1 GB | 16 GB cards (12 GB with expert offload): the best voices |
| [**gemma4-e4b-rp**](https://huggingface.co/cmod/gemma4-e4b-rp-GGUF) (4B dense, QAT) | 3.65 GB | 4.75 GB | 6-12 GB cards, laptops, Apple Silicon |
| stock [Gemma 4 E2B Q4_K_M](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) | 3.1 GB | 2.6 GB | fallback for very small machines |

## Quick start

```bash
llama-server -m gemma4-rp-maxxed-Q3_K_XL.gguf -ngl 999 -np 3 -c 196608 \
  -ctk q4_0 -ctv q4_0 -fa on --jinja
```

On a 16 GB card that also runs a game, add `--n-cpu-moe 3` to move three layers of experts to
system RAM (about 0.7 GB of VRAM back). On a **12 GB card**, raise `--n-cpu-moe` until it fits
(speed drops as more experts sit in RAM) or use fewer slots / less context (`-np 1 -c 65536`).
On a 24 GB Tesla P40 it runs fully on the card at about 46-48 tok/s. Thinking is off by default with Gemma 4's chat template;
keep it off for short dialogue lines.

The HermesVoice engine downloads this file, picks the context and offload for your card, and
starts and stops it with the game. You don't need to run it by hand.

## Using it outside HermesVoice

It is an ordinary Gemma 4 chat model in GGUF. It is tested on llama.cpp's `llama-server`; front
ends on top of it (SillyTavern, or anything that speaks the OpenAI API) work as usual. Other runtimes
built on llama.cpp with Gemma 4 support (KoboldCpp, LM Studio) should load it, but were not tested.

- **Chat template:** Gemma 4. With llama-server use `--jinja`, which reads the template from the GGUF.
- **Thinking:** off. Gemma 4's template has thinking off by default; keep it that way for dialogue
  (`--reasoning off` on llama-server). Thinking on makes short replies slow and sometimes empty.
- **Samplers** (Google's own values for Gemma 4; an A/B test on this build found nothing better):
  `temperature 1.0, top_k 64, top_p 0.95, min_p 0`, repetition penalty off (1.0), DRY off.
  For JSON or grammar-constrained output (classifiers, tool calls) use greedy (`temperature 0`).
- **Character cards:** it keeps a voice well from a short card (a few lines of personality plus two
  or three example lines). It was tuned on short spoken turns, so ask for "one or two lines" if replies run long.

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

### In HermesVoice (the game's own bench, 2026-11-25, RTX 5080)

| | this model |
|---|---|
| Lines passing every quality check (grounded, in character, no refusal, right length) | 99.7 % |
| Voice: which colonist said it (pawn ID accuracy) | 0.85 |
| Conversation loops (of 30 talks) | 0 |
| Intent accuracy with grammar (greedy) | 96.7 % |
| Planted instructions in game text followed: unprotected / with the mod's guards | 62-75 % / 0 % |

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
