---
license: apache-2.0
base_model:
- guideboardlabs/Qwen3.8-27B-Q3-DOWN-XS-GGUF
- Qwen/Qwen3.8-27B
language:
- en
tags:
- gguf
- qwen3.5
- llama.cpp
- quantization
- vocabulary-pruned
- tokenizer
- amd
- vulkan
- rdna1
- local-llm
---

<p align="center">
  <img src="monoglot-logo.png" alt="Monoglot" width="300">
</p>

# Monoglot: Qwen3.8-27B with its vocabulary cut to what English actually uses

**One language, on purpose.** This is my go at hacking a quant apart: I took a GGUF I had already built,
deleted 118,300 of its 248,320 vocabulary rows, and measured what that actually buys on a card that is
already full. All the numbers below are measured on my own machine, and everything I did not measure is
listed as not measured at the end.

It is a vocabulary prune, not a retrain and not a re-quant. Every surviving weight row is bit-identical to
the source file.

## Credit, in order of who did what

- **The idea and the first public demonstration: [fakezeta](https://huggingface.co/fakezeta).** The release
  [`fakezeta/Qwen3.8-27B-UD-Q3_K_XL-Condensed-P2`](https://huggingface.co/fakezeta/Qwen3.8-27B-UD-Q3_K_XL-Condensed-P2)
  pruned an Unsloth `UD-Q3_K_XL` Qwen3.8-27B and shipped the converter changes needed to run a
  condensed vocabulary at all. I read that release, not a summary of it.
- **The tool: [`bsaleh03/ASCII-Condensed-prune-tools`](https://github.com/bsaleh03/ASCII-Condensed-prune-tools).**
  The policies, the keep-set construction and the in-quantised-space row gather are all his work. That
  repository carries **no licence**, so none of its code is redistributed here; this model is a product of
  the tool, and the link is the credit.
- **[@TeksEdge](https://x.com/teksedge/status/2122118832441708600)**, whose post is how I found out this was
  happening.
- **Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** (Apache-2.0), quantised to
  [`guideboardlabs/Qwen3.8-27B-Q3-DOWN-XS-GGUF`](https://huggingface.co/guideboardlabs/Qwen3.8-27B-Q3-DOWN-XS-GGUF)
  from [an Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) with an imatrix built here. That
  quant is the file this one was carved out of.

What is mine is the application to this particular quant, the measurements, and the decision to publish
them with the parts that did not work.

## What was cut

Policy **P1G**. ASCII and the accented Latin range would have been P2; P1G keeps ASCII, math and typography
symbols, Greek, box-drawing and status glyphs, units and currency instead of the accented Latin range. The
policies are per-character and closed under substring, so every merge that produces a kept token has kept
parents and BPE reachability survives.

| Policy | Rows kept | Cut | Keeps |
|---|---:|---:|---|
| P1 | 129,006 | 48.05% | ASCII only |
| P2 | 141,141 | 43.16% | ASCII + Latin-1 / Latin-Ext-A |
| P1M | 129,272 | 47.94% | ASCII + math and typography |
| **P1G (this file)** | **130,020** | **47.64%** | P1M + Greek, box-drawing, units, currency |

I picked P1G because I use this model for code, logs and technical prose, where `π`, `≥`, `├──` and `✓`
turn up constantly and cost 1,014 rows over P1M.

Specials, all 256 byte-fallback tokens and partial-UTF-8 fragments always survive, so any text is still
representable. Dropped characters cost one token per UTF-8 byte instead of one per character; the measured
cost is below.

## What it saved

The source quant puts most of a gigabyte into each vocabulary tensor, which is the whole reason this was
worth doing on an 8 GB card:

| Tensor | Type | Before | After |
|---|---|---:|---:|
| `output.weight` | Q5_K | 874 MB | 456 MB |
| `token_embd.weight` | Q2_K | 417 MB | 218 MB |
| **File** | mixed IQ2/IQ3/Q5_K | **8,480,272,384 B** | **7,869,784,224 B** |

610 MB smaller, 7.2%. The two vocabulary tensors were 15.2% of the source file and are 8.6% of this one.

For context, the rest of the file is `ffn_gate` and `ffn_down` at IQ2_XS, `ffn_up` at IQ2_XXS, `attn_qkv` at
IQ2_XS, `attn_output` at IQ3_S, `attn_k`/`attn_v` at IQ4_XS. So this is a very low-bit FFN with a
deliberately expensive output head, which is exactly the shape where vocabulary pruning pays best.

## What it bought, measured

`llama.cpp` `3ba1ce0` Vulkan, RX 5700 XT (8 GB, RDNA1), `-ngl 999 -ctk q8_0 -ctv q8_0 -fa on -t 6
--cpu-mask 0x3f`, `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`. Three reps, and the card empty before each run.

| llama-bench | source | this file | |
|---|---:|---:|---|
| pp128 | 169.2 | 169.5 | flat |
| pp256 | 174.8 | 175.5 | flat |
| pp384 | 172.6 | 173.1 | flat |
| **pp512** | **28.0** | **173.2** | **6.2x** |
| tg128 | 23.81 ± 0.05 | 24.46 ± 0.10 | +2.7% |

Prompt processing falls off a cliff at exactly 512 tokens in the source file and stays fallen. 512 is
`llama-server`'s default ubatch, so in a server that is every prompt chunk, not an edge case.

Through `llama-server` at a 16K window, 499-token prompt, two requests each:

| | source | this file |
|---|---:|---:|
| prefill | 27.2 / 26.5 t/s | **156.0 / 153.6 t/s** |
| decode | 8.9 / 9.8 t/s | **23.0 / 23.6 t/s** |
| VRAM | 7.95 GiB of 7.98 | 7.87 GiB of 7.98 |
| GTT spill | 1.27 GiB | 0.56 GiB |

At 32K context this file still reads 156.6 / 155.7 t/s prefill and 22.8 / 23.1 t/s decode.

**Why the gain is a step and not a slope:** at 16K the source file no longer fits. About 1.3 GB of its
weights stream out of system memory over PCIe 3.0 at roughly 13 GB/s, which prices decode at
1.27 GB / 13 GB/s of about 98 ms per token on top of the resident read, and pushes the ubatch compute buffer
over the edge as well. Removing 610 MB puts it back under the wall. The 7.2% is not the win; crossing the
threshold is. On a card with headroom this same file is a 2.7% decode gain and nothing else.

## What it costs

Token counts for the same string, source file versus this file:

| Text | Source | This file | Ratio |
|---|---:|---:|---:|
| English prose | 30 | 30 | 1.00x |
| Python | 41 | 41 | 1.00x |
| Math with `sum, pi, >=, check` glyphs | 27 | 27 | 1.00x |
| French with accents | 18 | 31 | 1.72x |
| Japanese | 15 | 40 | 2.67x |
| Chinese | 15 | 50 | 3.33x |
| Russian | 17 | 93 | 5.47x |

English, code and math tokenise one for one. Anything outside the kept set falls back to bytes, and the
inflation is what you pay in both context window and speed. **If you need multilingual output, use the
source file.**

## Correctness checks that did pass

- `verify_prune.py --policy P1G`: all 864 non-vocabulary tensors byte-identical, sampled gathered rows
  match their source rows, keep-set exactly what the policy specifies, every metadata key preserved.
- Greedy 64-token continuations are byte-identical to the source for an English prompt and for a code
  prompt. Identical greedy text is the strong signal here: it requires both the prompt tokeniser to map
  text to the same pieces and the output-head rows to stay aligned with their token strings.
- A `tools` request returns `finish_reason: tool_calls` with the correct function name and correct JSON
  arguments, so chat templates and tool calling survive the id remap.

## What I did not measure

- **No instruction-following benchmark has been run on this file.** No IFEval, no other scored suite, so
  there is no number here to compare with the IFEval figures quoted around the original release. Correctness
  rests on the three checks above, which prove nothing is misaligned and nothing about instruction-following
  quality.
- **No perplexity or KL gate**, deliberately: those are not comparable across a vocabulary change. The same
  text tokenises differently, the logit spaces differ in size, and a PPL number from this file against the
  source would be measuring the tokeniser, not the model.
- **One machine.** RDNA1, Vulkan, one llama.cpp build, one quant. The 512-token cliff is a property of a card
  this close to full; do not carry these ratios to a 16 GB or 24 GB card, where the source file fits and the
  prune is a few percent.
- **Text only.** No `mmproj` in this repo. The prune is text-vocabulary only and says nothing about a vision
  projector, whose own vocabulary is untouched.

## Using it

```bash
llama-server -m Qwen3.8-27B-Q3-DOWN-XS-Monoglot-P1G-130K.gguf \
  -c 16384 -ngl 999 -ctk q8_0 -ctv q8_0 -fa on
```

The chat template, special tokens and sampler behaviour are carried through the prune and remapped, so it
is a drop-in replacement for the source file on the same port and alias.

Reproducing the prune, with your own policy instead of mine:

```bash
python3 prune_vocab.py SRC.gguf DST.gguf --policy P1G
python3 verify_prune.py SRC.gguf DST.gguf --policy P1G   # must print ALL CHECKS PASSED
```

`verify_prune` with `--policy` is not optional: a prune run with the wrong policy produces a perfectly
consistent file containing the wrong tokens and passes every structural check.

## If you are building the same thing

Three things I would tell myself before starting:

1. Check the two vocabulary tensors' share of the file first. If they are under about 8%, the prune is a
   file-size trick and probably not worth your afternoon.
2. Check whether the card is empty before you believe any benchmark. Two leaked GPU processes held 2.5 GB
   when I started and my first run measured 2.65 t/s on a model that does 23.8. A shared card reads as a slow
   model, never as an error.
3. Measure prefill at the ubatch size your server actually uses, not at a comfortable default. The 6.2x in
   this file is invisible if you only ever look at decode speed.

## Correction, 2026-11-16: their stock 2-bit beats this file

I re-gated Unsloth's current Dynamic v3.1 quants against this one on the same frozen perplexity gate, same
binary, same 8 chunks of 512 from wikitext-2:

| File | Bytes | PPL |
|---|---:|---:|
| **unsloth/Qwen3.8-27B UD-IQ2_S (v3.1)** | **8,422,719,488** | **7.5478** |
| This file (Q3-DOWN-XS pruned) | 7,869,784,224 | 8.2587 |

Their 2-bit wins by 8.7% perplexity and is 68 MB smaller. The "86.5% of control" figure in the tables below
was measured against **Unsloth UD-IQ3_XXS at 11,913,559,104 bytes**, and Unsloth re-cut every quant for
Dynamic v3.1 on 2026-11-11, so that control no longer exists in that form: their UD-IQ3_XXS now serves at
10.93 GB. The ladder was ranking against a reference that had been replaced, and none of our own numbers
moved to reveal it.

The replacement artifact is
[`guideboardlabs/Qwen3.8-27B-UD-IQ2_S-Monoglot-GGUF`](https://huggingface.co/guideboardlabs/Qwen3.8-27B-UD-IQ2_S-Monoglot-GGUF),
which is their weights with the same vocabulary cut applied. This repository stays up as the record of the
first attempt and of the comparison, not as the recommendation.

Two things still hold from this file's measurements, and they are worth keeping:

- The 512-token prefill cliff is real and reproducible: pp384 reads 172.6 t/s and pp512 reads 28.0 on this
  card, and freeing the 610 MB of vocabulary rows is what lifts it.
- The fast path has **no margin**. This file sits about 7 MB below the cliff at ctx 16384 with q8_0 KV, so a
  KV type change, a context bump or a second process on the card flips it to ~10 t/s. Treat it as a
  knife-edge configuration, which is why the replacement card recommends q4_0 KV.

## Files and verification

| File | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3.8-27B-Q3-DOWN-XS-Monoglot-P1G-130K.gguf` | 7,869,784,224 | `25546eb531ecb299846b1c0f7f51b4d0dd913fda5eac3e21185b43e317ee8b4c` |

The tokenizer inside the GGUF is the ground truth for the condensed vocabulary, so the file is
self-contained: no separate `tokenizer.json`, no config surgery, nothing to keep in sync.

## Licence

Apache-2.0, following the base model. Weights derive from Qwen3.8-27B under its licence and from
GuideboardLabs' Q3-DOWN-XS quant. The pruning tool is linked, not included, because it carries no licence.
