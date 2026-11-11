---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- deepseek
- deepseek_v41
- moe
- mxfp4
- nvfp4
- gguf
- mixed-precision
- v100
- volta
- sm70
- vllm
- speculative-decoding
- serving
---

# Volta Max uPLIft FP4 · hot_iq package · DeepSeek-V4.1-Flash-1017 on 8x V100-SXM2 · PP2 x TP4

> **Upload in progress. Files are incomplete. Do not download yet.** (2026-11-10) A dated line here will say when
> the upload is complete.

## Volta Max uPLIft: as much of the routed work at NVIDIA's NVFP4 precision as fits on eight V100s, every routed expert on the cards in decode · 6,087 of 15,360 routed experts at exactly NVFP4 precision · 79 to 85 % of decode routes in the measured sets (in-sample) · 64.7 tok/s legal-style, 83.6 general · one download

[![decode routes at NVFP4 precision: 79-85 %](https://img.shields.io/badge/routes%20at%20NVFP4%20precision-79--85%20%25-76B900?style=for-the-badge)](#how-close-to-nvfp4)
[![legal-style: 64.7 tok/s](https://img.shields.io/badge/legal--style-64.7%20tok%2Fs-1F6FEB?style=for-the-badge)](#measured-results)
[![general: 83.6 tok/s](https://img.shields.io/badge/general-83.6%20tok%2Fs-8250DF?style=for-the-badge)](#measured-results)
[![31K tool loops: 88-91 tok/s](https://img.shields.io/badge/31K%20tool%20loops-88--91%20tok%2Fs-2EA043?style=for-the-badge)](#measured-results)
[![package: 556 GB](https://img.shields.io/badge/package-556%20GB-555555?style=for-the-badge)](#files-and-sizes)
[![speed bench, code 1K to 256K: 104-127 tok/s](https://img.shields.io/badge/speed%20bench%2C%20code%201K%20to%20256K-104--127%20tok%2Fs-0969DA?style=for-the-badge)](#speed-bench-1k-to-256k)

Built by **Nick Mermiges, Esq.**, founder of [PLI Labs](https://proprietarylegal.ai/labs.html) · [proprietarylegal.ai](https://proprietarylegal.ai)

[![GitHub: Volta Max uPLIft FP4](https://img.shields.io/badge/GitHub-Volta%20Max%20uPLIft%20FP4-181717?logo=github)](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4) [![Related build: Volta Full NVFP4](https://img.shields.io/badge/related%20build-Volta%20Full%20NVFP4-555555?logo=github)](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100) [![Earlier build: Volta sPLIt](https://img.shields.io/badge/earlier%20build-Volta%20sPLIt-555555?logo=github)](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4) [![Volta sPLIt on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Volta%20sPLIt-model%20card-FFD21E)](https://huggingface.co/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-GGUF) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Hardware: 8x V100-SXM2-32GB](https://img.shields.io/badge/hardware-8x%20V100--SXM2--32GB-76B900)](#what-this-is)

**Code, engine patches, launcher and every measurement:**
[github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4).
**Our earlier and faster build of the same model on the same cards, with every expert compressed:**
[Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4)
([Hugging Face card](https://huggingface.co/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-GGUF)).
**Every routed expert at exactly NVFP4 precision, the less-used served from host RAM, slower:**
[Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100).

**Volta Max uPLIft is the build that puts as much of the routed work at NVIDIA's NVFP4 precision as fits on eight V100
cards, with every routed expert held on the cards.** This package is its one download, and its configuration key is
`hot_iq`. Every routed expert's compressed IQ2-class record is placed on the cards first. Then experts are promoted to
their NVFP4 weights in order of use per byte, one by one, until the next one does not fit. This maximizes use, not the
count. Packing the cheapest promotions first would fit 6,377 to 6,379 experts at NVFP4 against 6,087, but less-used
ones, so it could not carry more of the work at NVFP4 (to within one expert's share on each pipeline stage,
[How the hot set was chosen](#how-the-hot-set-was-chosen)). Use and work here are the mixed tally's counts of decode
routes, measured in-sample. That puts 6,087 of the 15,360 routed experts at exactly
the precision of NVIDIA's [DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) pack,
unchanged. In the measured sets 79.0 to 85.0 % of decode routes ran on those experts (route counters, in-sample,
[Measured results](#measured-results)). The other 15.0 to 21.0 % ran on ISTA-DASLab's IQ2-class records. In decode
all experts are in VRAM: no routed expert is read over PCIe. Prompt reading is different: it reads the compressed
experts at NVFP4 precision from records staged in host RAM. Speeds follow under the same heading.

<!-- pli:blurb:start -->

> Published by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai). PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work and discuss what your organization needs.

**Engine, patches, launcher and every measurement:** [GitHub: Volta Max uPLIft FP4](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4)

<!-- pli:blurb:end -->

## How close to NVFP4

**The closeness test is designed and its rules are public. Its results are not in yet.** It compares this
build's output with NVIDIA's NVFP4 pack served with every routed expert at its own NVFP4 precision (the reference,
[Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100)), with Volta sPLIt as the yardstick: tokens 2 to 32 of greedy continuations on 320 contexts of five kinds
(same next token, log-probability gap, KL divergence, first difference) and 513 MMLU questions, on the same eight
cards. Its rules and the sha256 of its scripts were fixed on 2026-11-10 at 19:54 UTC, before any capture, in
[PREREGISTRATION.txt](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/quality/closeness-test/PREREGISTRATION.txt). The captures run on 2026-11-10 and 2026-11-11. The results will be added here and to the measurement
log as dated rows, whatever they show.
The scripts, item ids, sources and rules are in the GitHub repository's [quality/closeness-test/](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/tree/main/quality/closeness-test).

## What this is

**A repack of two existing quantizations of DeepSeek-V4.1-Flash-1017, not a new quantization method.** Every routed
expert of every layer is in one of two forms on the cards:

- **hot**: the expert's record from NVIDIA's
  [DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) pack. NVFP4 folds
  exactly to MXFP4 (4.25 bits per weight), the precision of DeepSeek's official expert weights. The fold changes no
  weight value.
- **compressed**: the expert's IQ2-class record from ISTA-DASLab's
  [GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF), unchanged.

The dense tensors and the IQ records come from that GGUF, the Engram tables from its IQ4_NL file, and the config,
tokenizer and DSpark drafter from [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
The package puts all of it in one place, with the file that says which experts are hot (`tier_map.json`) and the
serving profile the measurements below were taken with.

**Who it is for:** eight V100-SXM2-32GB cards in two 4-card NVLink boards, pipeline parallel 2 x tensor parallel 4,
served by the engine of the GitHub repository above. It is not a llama.cpp or stock vLLM model. The MXFP4 files are
cut into the four tensor-parallel slices that engine loads, and the tier map is fitted to that split and to the
cards' 32 GB.

### How the hot set was chosen

With every compressed record on the cards and the profile's reserves (4.5 / 7.5 GiB, one prompt slot), the engine
has room for 6,087 of the 15,360 routed experts at full precision (3,833 of 7,680 on the first pipeline stage, 2,254
of 7,680 on the second). Its planner promotes experts in order of use per byte added and stops at the first
promotion that does not fit. That is not the largest number that fits: promotion costs differ by layer, and
promoting the cheapest first, whatever their use, would fit 6,377 to 6,379 (arithmetic, GitHub README build section).
It is the most routed use that fits: this greedy is within one expert's use of the fractional optimum, which bounds
every packing, so no other choice at these budgets carries more of the tally's routed use at NVFP4 than the planner's
plus one expert's share, on each stage. Which ones was decided by a decode routing tally: counts of how often
each expert was routed to, in four equally weighted kinds of text. One is our own legal work (private). The other three
are the public prompt sets of the GitHub repository: the general short set, the source-code tool loops, and the
legal-style short set with the US Code tool loop. The tally is not published. **This package carries only the result:
per layer, the ids of the hot experts in ascending order.** It carries no count, score or ranking.

Because the public prompt sets were part of the tally, the speeds and route shares measured on them below are
in-sample, the 216K tool loop and its own long thinking chat (chat 0) included. The held-out checks are chats 1 and 2
of the long thinking test, which take later sections of the public source text.

## Measured results

MEASURED 2026-11-10 on the 8-card host, one request at a time, greedy (T=0), card clock ceiling 1,425 MHz, power cap
150 W per card, this package's configuration (one prompt slot, reserves 4.5 / 7.5 GiB):

| Measure | Result |
|---|---|
| Short prompts, general set | 83.6 tok/s (36.25 ms per round) |
| Short prompts, legal-style set | 64.7 tok/s (35.78 ms per round) |
| Tool loop at 31K tokens, source code | 90.9 tok/s (40.09 ms per round) |
| Tool loop at 31K tokens, US Code Title 28 | 87.8 tok/s (38.18 ms per round) |
| Tool loop at 216K tokens, run 1: first read | 216,255 tokens in 168.72 s (about 1,280 tok/s) |
| Tool loop at 216K tokens, run 1: seven follow-ups | 7 of 7 kept the whole previous prompt, first token median 9.36 s, decode mean 90.4 tok/s, final answer ended normally (1,385 tokens) |
| Tool loop at 216K tokens, run 2, on the bench boot | first read 166.22 s, 7 of 7 kept the previous prompt, first token median 9.47 s, decode mean 99.8 tok/s over all seven, 93.9 over the six that ended with a tool call. The final request ran to the 32,000-token limit inside thinking with no final answer: 1 of 2 runs at the limit |
| Decode routes on compressed experts | general 21.0 %, legal-style 16.7 %, source loop 15.0 %, US Code loop 15.7 % |
| Closeness to the all-full-precision build, 12 prompts | PASS (rule fixed before the run: 8 of 12 on both measures) |
| Long thinking chats at 216K, temperature 0 | chats 0, 1 and 2 of the termination test ended normally (0 of 3 at the 32,000-token limit) |

Short-set tok/s is output tokens over the whole request time, the wait for the first token included. Tool-loop tok/s
is decode only (first to last streamed token). A round is one target forward pass that checks the DSpark drafter's
proposals. The route shares are counted by the engine's route counters on a separate boot. The short sets ran
again at the end of the speed bench on another boot of the same configuration: general 87.0 tok/s (34.73 ms per
round), legal-style 67.3 (34.41 ms). The headline carries the first boot's pair, the lower one.

### Speed bench: 1K to 256K

MEASURED 2026-11-10, card clock ceiling 1,425 MHz, 169 requests, the prompts and method of Volta sPLIt's speed bench
(real documents of five kinds cut to exact lengths, greedy, 1,024 tokens out, a random cache salt). Thinking off,
mean of 2 runs. Each cell is this package / Volta sPLIt's published bench of 2026-11-05 at a 1,522 MHz ceiling, so the
clocks differ. Full tables: [GitHub, Tokens per second](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#tokens-per-second).

![Decode speed by prompt length, Volta Max uPLIft against Volta sPLIt's published bench, for code, structured, legal and prose](https://huggingface.co/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4/resolve/main/charts/decode-by-length-light.png)

| prompt | first token | prompt reading | code | legal |
|---|---:|---:|---:|---:|
| 1K | 5.60 s / 1.22 s | 175-247 / 820-948 | 123.3 / 215.0 | 71.9 / 117.6 |
| 32K | 31.4 s / 17.2 s | 1,030-1,055 / 1,894-1,913 | 119.3 / 200.5 | 66.7 / 116.5 |
| 128K | 106.4 s / 59.2 s | 1,221-1,263 / 2,198-2,220 | 115.4 / 182.8 | 67.9 / 110.9 |
| 256K | 209.6 s / 118.7 s | 1,243-1,276 / 2,197-2,222 | 103.8 / 160.0 | 61.8 / 103.1 |

Decode falls slowly with prompt length, and the greedy output of the two runs was byte-identical in 45 of 45 cells.
Prompt reading is slow: it tops out at 1,243 to 1,276 tok/s, so a 256K prompt waits 209.6 s for its first token.
This package is slower than Volta sPLIt in every thinking-off cell, every prompt-reading cell and every short-prompt
row of the bench. With thinking on, code at 4K, 8K, 16K and 256K was 1.9 to 8.2 tok/s faster and the other 41 cells
slower (one run per cell).

**The final request of the 216K tool loop ran to the limit on the bench boot.** The first read took 166.22 s and all
seven follow-ups kept the previous prompt, then the final request ran to the 32,000-token limit inside thinking and
gave no final answer. On the first boot the same loop's final answer ended normally (1,385 tokens). So on this
build that loop's final request has run to the limit in 1 of 2 runs.

**Closeness.** The reference is the same NVFP4 pack with every expert at full precision (published as
[Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100)), and the yardstick is our earlier build, Volta sPLIt. Per prompt, this build's first token that
differs from the reference must come no earlier than Volta sPLIt's, and its mean absolute log-probability difference
must be no larger. Result: first differing token 11 for, 1 against. Mean difference 8 for, 4 against, 0 undecided. The
check runs with thinking off.

**For comparison** (same host, GitHub measurement log): Volta sPLIt decodes 118 to 119 tok/s on the legal-style set and
137 to 138 on the general set (2026-11-09, at the earlier 1,522 MHz clock ceiling). This package is slower than Volta
sPLIt, which runs every routed expert from its IQ2-class record. It is faster than the GitHub repository's three-tier
mode, an earlier step on the way to this build that keeps a tier of full-precision experts in host RAM and decoded 51 to 52 tok/s general and 57 to 58
legal-style (2026-11-09, 1,522 MHz), with compressed experts carrying 3.7 to 9.3 % of routes. It is also faster than
[Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100), which serves every routed expert at NVFP4 precision from the cards and host RAM
([its speeds](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100#tokens-per-second)).

## Files and sizes

| Path | Files | Bytes | Contents |
|---|---:|---:|---|
| `experts-mxfp4/L{00..39}-r{0..3}.bin` | 160 | 299,473,305,600 | per layer and tensor-parallel rank, all 384 routed experts' MXFP4 records in expert-id order, 4,874,240 bytes each |
| `experts-mxfp4/manifest.json` | 1 | 433 | format `nvfp4-8r-mxfp4-v2-id-order` |
| `tier_map.json` | 1 | 23,564 | the hot experts per layer and the tier digest per stage |
| `gguf/` | 4 | 138,238,893,056 | ISTA-DASLab GSQ-RCO IQ2_XXS GGUF, four shards, revision `ad1e8991`, unchanged |
| `engram/` | 1 | 110,595,290,816 | ISTA-DASLab Engram IQ4_NL, revision `ad1e8991`, unchanged |
| `dspark/` | 5 | 7,933,308,215 | DeepSeek DSpark drafter shards 44 to 46, revision `1bae0e10`, unchanged, with a drafter `config.json` and an index of those three shards written by PLI Labs (the index is the base index's 2,401 entries for shards 44 to 46) |
| `base/` | 19 | 13,934,483 | DeepSeek config, tokenizer, encoding and licence, revision `1bae0e10`, unchanged |
| `serve/hot_iq.yaml.template` | 1 | 2,164 | the serving profile, with `${PKG}` for the package path |
| `charts/decode-by-length-light.png`, `charts/decode-by-length-dark.png` | 2 | 523,078 | the speed chart of [Speed bench](#speed-bench-1k-to-256k), light and dark |
| `manifest.json`, `SHA256SUMS` | 2 | | every file with its size and sha256 |

The GGUF and Engram files are ISTA-DASLab's four-shard files of revision `ad1e8991`, from before their repo moved
to llama.cpp's six-shard layout on 2026-10-27. The engine of this release reads them as they are. llama.cpp does not
load them (ISTA-DASLab's card, "Updating an existing download", gives the converter).

Total about 556.3 GB (518.1 GiB). Every expert is in `experts-mxfp4/`, also the compressed ones: the engine reads a
prompt with the full-precision records of all experts, staging the compressed experts' records from host RAM, and
decodes the compressed experts from their IQ records on the cards. This is the configuration that was measured. The
GGUF ships whole and unchanged, so the hot experts' IQ records (52.0 GB) are in it and unused.

**Host RAM:** the compressed experts' MXFP4 records, 180.8 GB, are held in pinned host memory for prompt reading,
plus the Engram tables. The measurements were taken on a host with 629 GiB of RAM.

## How to run it

**When the upload is complete.** Until a dated line at the top of this card says so, the steps below cannot
work: do not download yet. As of 2026-11-10 the public launcher and the tier-map reader have passed the CPU check (the tier map gives the
measured tiers and digests on all 40 layers, and every expert record of layers 0 and 20 on the first rank is
byte-equal to the measured one) but not yet an eight-card boot. The boot check follows and will be added as a
dated row.

1. When the upload is complete, download: `hf download ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4 --local-dir /path/to/pkg` (556 GB).
2. Check: `cd /path/to/pkg && sha256sum -c SHA256SUMS`.
3. Build the engine and lay out its run directory as the [GitHub repository](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4)
   describes in [From the Hugging Face package](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#from-the-hugging-face-package-hot_iq).
4. Start, with `LANE` and `CUDA_VISIBLE_DEVICES` set as the repository describes:
   `cd "$LANE/run" && bash serve_hot_iq.sh /path/to/pkg 8000`. The launcher writes the serving profile from this
   package's `serve/hot_iq.yaml.template`, sets the load-time allocator setting the measurements ran with, and starts
   the repository's `run_serve.sh` with one prompt slot.
5. At load each rank logs its tiers. The first stage must log 3,833 hot / 3,847 warm experts and tier digest
   `8e7829184090`, the second 2,254 / 5,426 and `5b09c0383a80`.

With a tier map the engine does not plan tiers. If a card has less free memory after the dense weights than the
measured cards had, it refuses to load and names the bytes it is short, rather than choosing a different set.

## Limits

- **Only for this hardware and engine.** The tier map fits PP2 x TP4 on 32 GB cards. Another split is refused.
- **Prompt reading uses the full-precision records**, staged from host RAM. A faster mode that reads prompts with the
  compressed experts' IQ records exists in the GitHub repository's work but is not used here. On the long chat that
  tests it, its first read stopped the host or a card three times out of three. Replays on a single card and on two
  cards completed without a fault. The cause is not yet known.
- **The hot set is fixed.** There is no promotion of experts during a chat. Text unlike the tally's four kinds may route
  more often to compressed experts.
- **Quality checks are narrow:** 12 prompts with thinking off, and three long thinking chats at temperature 0. The
  termination test's other six chats and its run at sampling settings were not run.
- **Compressed records are not full precision.** On synthetic inputs an expert's IQ2-class record is far from its
  official record (relative L2 0.60 to 0.69 in three sampled layers). On real text the closeness check above passed.
- **Speeds are in-sample** on the public prompt sets (see above), and were taken at a 1,425 MHz clock ceiling.

## Credits and licences

- **DeepSeek**: DeepSeek-V4.1-Flash-1017, its official MXFP4 experts, the DSpark drafter and the Engram design. MIT.
- **NVIDIA**: the DeepSeek-V4.1-Flash-1017-NVFP4 pack whose expert records are folded here. MIT.
- **ISTA-DASLab**: the GSQ-RCO GGUF and the Engram IQ4_NL file, shipped unchanged. Its weights inherit the base
  model's MIT licence.
- **ggml**: the GGUF and IQ formats. **1CatAI / 1Cat-vLLM**, **v100-skinny** and **the vLLM project**: the engine
  underneath (see the GitHub repository's CREDITS.md).

Licence texts are in [LICENSES/](LICENSES). The files PLI Labs wrote (tier map, manifests, profile template, this
card) are MIT ([LICENSE](LICENSE)). See [NOTICE](NOTICE).
