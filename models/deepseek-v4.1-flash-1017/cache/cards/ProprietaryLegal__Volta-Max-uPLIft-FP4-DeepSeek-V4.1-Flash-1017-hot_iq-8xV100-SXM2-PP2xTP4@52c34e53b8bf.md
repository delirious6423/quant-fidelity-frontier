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

> **Upload complete (2026-11-11).** Every file of this package is on the Hub. `SHA256SUMS` lists each one, this card
> and its charts included.

**2026-11-12: small-file update ([commit history](https://huggingface.co/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4/commits/main)).** The weights are unchanged. New: one serving template and one
tier map per option in `serve/` and `tier_maps/`, for the engine of the GitHub repository of 2026-11-12, which
serves two prompt slots by default with fast prompt reading. `serve/hot_iq.yaml.template` is now the default
option's (two slots). The first release's `tier_map.json` stays as its record, and that engine refuses it. With three requests running together on the three-slot option the
build delivers 192.3 tok/s in all, 1.62x one stream, and two requests on the two-slot default give 159.0 tok/s
(1.34x), same token ids as each prompt alone (MEASURED 2026-11-12, [table and limits](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#serving-several-requests-at-once-window-18)). Quality not
established for the slot options and fast reading. The closeness figures and the speed tables further down this
card are the first release's (one slot, native reading). The headline and badges give the final build's speeds. On the final build (MEASURED 2026-11-12, two slots, one request at a time)
the short sets ran at 123.2 tok/s general and 97.3 legal-style, and code decoded at 150 to 180 tok/s
from 1K to 256K prompts ([speed bench](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#speed-bench-2026-11-12-final-build)).
The previous public bench, the kernel build of 2026-11-11 (one slot, native reading), gave 101.1 and 81.1 tok/s
on the short sets and a first token 207.9 s after a 256K prompt: build against build, not one change. On the
final build's bench boot the cards of stage 1 went 2 MiB over the build's memory line, with no out-of-memory
line ([Limits](#limits)).
The repository's launcher also exports two request-path switches taken over from Volta sPLIt (window 16): a system
message sent in the middle of a chat no longer makes the server read the chat again (a 30,779-token chat: 30,752
tokens reused and 0.45 s, against 14.84 s), token for token the same output, decode speed no measurable change.
Since window 17 the slot options also read prompts with two bit-equal kernels (a one-launch SwiGLU and the RB 4
expert kernel): same output bits, second reads of a 215,847-token prompt 4.43 s faster, first reads no measurable
change.

## Volta Max uPLIft: as much of the routed work at NVIDIA's NVFP4 precision as fits on eight V100s, every routed expert on the cards in decode · 5,949 of 15,360 routed experts at exactly NVFP4 precision in the two-slot default · about 78 to 85 % of decode routes (predicted for two slots, 80 to 86 % measured on one) · same next token as NVIDIA's NVFP4 pack 95.1 % (Volta sPLIt 92.2 %, first release) · 192.3 tok/s with three requests together on the three-slot option (1.62x one stream) · 97.3 tok/s legal-style, 123.2 general (short sets, one request at a time on the two-slot default) · one download

[![same next token as the NVFP4 pack: 95.1 %, Volta sPLIt 92.2 %](https://img.shields.io/badge/same%20next%20token%20as%20NVFP4-95.1%20%25%20(Volta%20sPLIt%2092.2%20%25)-76B900?style=for-the-badge)](#how-close-to-nvfp4)
[![decode routes at NVFP4 precision: about 78-85 % with two slots (predicted)](https://img.shields.io/badge/routes%20at%20NVFP4%20precision-78--85%20%25%20(2%20slots%2C%20predicted)-76B900?style=for-the-badge)](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#serving-several-requests-at-once-window-18)
[![three requests together on three slots: 192.3 tok/s, 1.62x one stream](https://img.shields.io/badge/3%20requests%20together%2C%203%20slots-192.3%20tok%2Fs%20(1.62x)-1F6FEB?style=for-the-badge)](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#serving-several-requests-at-once-window-18)
[![legal-style short set, one request at a time on the two-slot default: 97.3 tok/s](https://img.shields.io/badge/legal--style%20short%20set%2C%20one%20at%20a%20time-97.3%20tok%2Fs-1F6FEB?style=for-the-badge)](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#speed-bench-2026-11-12-final-build)
[![general short set, one request at a time on the two-slot default: 123.2 tok/s](https://img.shields.io/badge/general%20short%20set%2C%20one%20at%20a%20time-123.2%20tok%2Fs-8250DF?style=for-the-badge)](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#speed-bench-2026-11-12-final-build)
[![speed bench, code 1K to 256K, one request at a time on the two-slot default: 150-180 tok/s](https://img.shields.io/badge/speed%20bench%2C%20code%201K%20to%20256K%2C%20one%20at%20a%20time-150--180%20tok%2Fs-0969DA?style=for-the-badge)](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#speed-bench-2026-11-12-final-build)
[![package: 556 GB](https://img.shields.io/badge/package-556%20GB-555555?style=for-the-badge)](#files-and-sizes)

Built by **Nick Mermiges, Esq.**, founder of [PLI Labs](https://proprietarylegal.ai/labs.html) · [proprietarylegal.ai](https://proprietarylegal.ai)

[![GitHub: Volta Max uPLIft FP4](https://img.shields.io/badge/GitHub-Volta%20Max%20uPLIft%20FP4-181717?logo=github)](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4) [![Related build: Volta Full NVFP4](https://img.shields.io/badge/related%20build-Volta%20Full%20NVFP4-555555?logo=github)](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100) [![Earlier build: Volta sPLIt](https://img.shields.io/badge/earlier%20build-Volta%20sPLIt-555555?logo=github)](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4) [![Volta sPLIt on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Volta%20sPLIt-model%20card-FFD21E)](https://huggingface.co/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-GGUF) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Hardware: 8x V100-SXM2-32GB](https://img.shields.io/badge/hardware-8x%20V100--SXM2--32GB-76B900)](#what-this-is)

**Code, engine patches, launcher and every measurement:**
[github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4).
**Our earlier and faster build of the same model on the same cards, with every expert compressed:**
[Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4)
([Hugging Face card](https://huggingface.co/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-GGUF)).
**Every routed expert at exactly NVFP4 precision, the less-used served from host RAM, slower:**
[Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100).

![Greedy continuations that stay identical to the NVFP4 reference: hot_iq, Volta sPLIt and the reference's own repeat](https://huggingface.co/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4/resolve/main/charts/closeness-survival.png)

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

**How close to NVIDIA's NVFP4 pack** (MEASURED 2026-11-11, 320 contexts and 513 MMLU questions, rules fixed before
any capture, [details](#how-close-to-nvfp4)). `hot_iq` picks the same next token as the NVFP4 reference on 95.1 % of
compared tokens, Volta sPLIt on 92.2 %. Its KL divergence from the reference is 0.079 nats against Volta sPLIt's
0.196, and on the tokens both builds still share with the reference it removes 63 % of Volta sPLIt's KL divergence.
It changes 1.6 % of the multiple-choice answers, Volta sPLIt 15.4 %. 37 of Volta sPLIt's 79 changed answers are on
the 38 questions the reference left unanswered: on the 475 it answered, the changed shares are 1.7 % and 8.8 %
(counted after scoring, outside the preregistered rules).

<!-- pli:blurb:start -->

> Published by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai). PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work and discuss what your organization needs.

**Engine, patches, launcher and every measurement:** [GitHub: Volta Max uPLIft FP4](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4)

<!-- pli:blurb:end -->

## How close to NVFP4

`hot_iq` runs the experts that carry most of the decode work (79 to 85 % of decode routes, in-sample) at exactly
NVIDIA's NVFP4 precision and the rest from smaller IQ2-class records, so every routed expert fits on eight 32 GB
Tesla V100 cards. This section measures how far that moves the model's output from the NVFP4 pack, and compares it
with Volta sPLIt, the build that runs every routed expert from IQ2-class records. The reference throughout is **NVIDIA's NVFP4 pack**
served on the same cards with every routed expert at its own NVFP4 precision. That pack is itself a quantization of
the original model, so these figures say how close `hot_iq` is to the NVFP4 pack, not to the unquantized model.

### What is exact by construction

| Fact | Value | Evidence |
|---|---|---|
| Experts kept at NVFP4 precision (hot) | layers 0 to 19: 3,833 of 7,680. Layers 20 to 39: 2,254 of 7,680 | boot log tier lines on all four ranks of each stage of this package's boot (2026-11-11), tier digests 8e7829184090 and 5b09c0383a80 |
| Hot experts equal NVIDIA's values | the NVFP4 to MXFP4 fold is exact: 44,696,862,720 values bit-equal, relative L2 error 0 | the fold check in [What this is](#what-this-is) |
| Decode routes that land on hot experts (the share of the work done at exactly NVFP4 precision) | general text 79.0%, legal-style 83.3%, source code loop 85.0%, US Code loop 84.3% | route counters while serving the texts the hot experts were chosen from (in-sample) |
| Prompt reading | every expert, hot or not, is computed from its full NVFP4 record | the engine's prompt-reading path |

So the first token of every answer comes from the same weights as the reference. Only later tokens can be touched
by the IQ records, on the 15 to 21% of expert routes that are not hot.

### What was measured

The test, its rules and its scripts are in the GitHub repository's [`quality/closeness-test/`](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/tree/main/quality/closeness-test). The rules were fixed and
their scripts hashed before any capture ([pre-registration](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/quality/closeness-test/PREREGISTRATION.txt)). Every figure
below is reported as measured, with a 95% bootstrap interval. Temperature 0, thinking off, one request at a time.
The receipts (hashes, log-probabilities, counts and timing, no text), the scorer's output and the run decisions (a
scrubbed public copy) are in the same folder.

- **Captures.** MEASURED, all at the 1,475 MHz clock cap. Volta sPLIt on its production boot, 2026-11-10 from
  21:32 UTC. `hot_iq` on a boot of this package, 2026-11-11 from 01:58 UTC. The reference on its own boot,
  2026-11-11 from 02:44 UTC. Two amendments to the capture budgets are recorded before any score: the Volta sPLIt arm
  and the `hot_iq` arm were each captured in two parts on one boot (15 and up to 75 minutes, decided 2026-11-10 at
  21:36 UTC, and 30 and up to 20 minutes, decided 2026-11-11 at 02:05 UTC), each from a timing line only. No script,
  rule or constant changed. The scorer accepted both continuations, which re-captured the first 10 contexts
  identically. The Volta sPLIt arm was captured on 2026-11-10, before three later changes to the Volta sPLIt build.
  This test makes no claim about their effect.

- **320 contexts** of code, prose, legal-style, structured and chat text from
  public sources, none of them used to choose the hot experts ([sources](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/quality/closeness-test/SOURCES.md)). Each build
  writes 32 tokens greedily. Tokens 2 to 32 are compared while the text still equals the reference's.
- **513 MMLU questions**, 9 from each of 57 subjects, answered as `Answer: X`.
- **The noise floor.** The reference was run twice on 50 contexts and 50 questions. It was
  identical on all 50 repeated contexts and all 50 repeated questions, so the noise floor is zero.

### Next-token closeness

| Measure | hot_iq | Volta sPLIt | Reference repeat (noise floor) |
|---|---|---|---|
| Contexts | 320 | 320 | 50 |
| Tokens compared (after the first, up to the first difference) | 4818 | 2455 | 1550 |
| Same next token as the reference | 95.1% (94.4% to 95.7%) | 92.2% (90.7% to 93.4%) | 100.0% (100.0% to 100.0%) |
| Mean log-probability gap on the reference's token (nats) | 0.1163 (0.1009 to 0.1342) | 0.2451 (0.2053 to 0.2959) | 0.0000 (0.0000 to 0.0000) |
| Mean KL divergence from the reference, top 20 (nats, lower bound) | 0.0794 (0.0687 to 0.0915) | 0.1958 (0.1648 to 0.2329) | 0.0000 (0.0000 to 0.0000) |
| All 32 tokens identical | 25.6% (20.9% to 30.0%) | 10.9% (7.8% to 14.4%) | 100.0% (100.0% to 100.0%) |
| First token identical (prompt reading, control) | 100.0% (100.0% to 100.0%) | 70.6% (65.9% to 75.6%) | 100.0% (100.0% to 100.0%) |

On the tokens where both builds still matched the reference, measured on exactly the same tokens:

| Measure (same tokens for both builds) | hot_iq | Volta sPLIt | Volta sPLIt minus hot_iq | Verdict |
|---|---|---|---|---|
| Next token differs from the reference | 4.3% (3.4% to 5.4%) | 7.8% (6.5% to 9.4%) | 3.5% (2.4% to 4.7%) | HOT_IQ CLOSER |
| Mean log-probability gap (nats) | 0.1035 (0.0803 to 0.1324) | 0.2351 (0.1942 to 0.2869) | 0.1316 (0.0992 to 0.1690) | HOT_IQ CLOSER |
| Mean KL divergence, top 20 (nats, lower bound) | 0.0693 (0.0536 to 0.0873) | 0.1887 (0.1570 to 0.2274) | 0.1194 (0.0921 to 0.1521) | HOT_IQ CLOSER |

`hot_iq` removes 63.3% (95% interval 55.0% to 70.5%) of Volta sPLIt's KL divergence from the NVFP4 reference, and
56.0% (95% interval 47.3% to 63.8%) of its log-probability gap.

By text kind, on the same shared tokens:

| Text kind | Contexts | Tokens compared | hot_iq KL | Volta sPLIt KL | hot_iq same token | Volta sPLIt same token |
|---|---|---|---|---|---|---|
| code | 64 | 626 | 0.0219 (0.0125 to 0.0359) | 0.0675 (0.0434 to 0.1071) | 97.3% (95.5% to 98.5%) | 96.2% (93.9% to 97.6%) |
| prose | 64 | 223 | 0.2967 (0.1717 to 0.4755) | 0.6661 (0.4558 to 0.9473) | 89.2% (83.4% to 93.3%) | 79.8% (71.6% to 86.1%) |
| legal | 64 | 368 | 0.0631 (0.0402 to 0.0955) | 0.2082 (0.1448 to 0.3017) | 94.0% (90.2% to 96.3%) | 89.9% (84.4% to 93.4%) |
| structured | 64 | 591 | 0.0395 (0.0226 to 0.0625) | 0.0865 (0.0586 to 0.1285) | 97.5% (95.6% to 98.6%) | 94.9% (92.0% to 96.7%) |
| chat | 64 | 359 | 0.0664 (0.0430 to 0.0974) | 0.2516 (0.1623 to 0.3773) | 95.8% (93.5% to 97.7%) | 91.1% (86.8% to 93.9%) |

Chart: [distance from the NVFP4 reference by text kind](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/quality/closeness-test/charts/kinds.svg).

### How long the output stays identical

Each build's 32 greedy tokens were compared with the reference's until the first token that differs. `hot_iq`
stayed identical through all 32 tokens on 25.6% (95% interval 20.9% to 30.0%) of contexts, Volta sPLIt on
10.9% (95% interval 7.8% to 14.4%). The median first differing token was 13.0 (95% interval 10.0 to 15.5)
for `hot_iq` and 4.0 (95% interval 3.0 to 4.0) for Volta sPLIt. Context by context,
hot_iq stayed identical longer on 200 contexts, Volta sPLIt on 31, and the two diverged at the same token on 89.

![Greedy continuations that stay identical to the NVFP4 reference: hot_iq, Volta sPLIt and the reference's own repeat](https://huggingface.co/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4/resolve/main/charts/closeness-survival.png)

### Multiple choice

| Build | Accuracy | Minus the reference | Answer differs from the reference | No answer |
|---|---|---|---|---|
| NVFP4 reference | 83.8% (80.7% to 86.7%) |  |  | 38 |
| hot_iq | 83.6% (80.5% to 86.5%) | -0.2% (-1.4% to 0.8%) | 1.6% (0.6% to 2.7%) | 38 |
| Volta sPLIt | 84.4% (81.3% to 87.5%) | 0.6% (-2.3% to 3.5%) | 15.4% (12.3% to 18.5%) | 1 |

The reference answered 83.8% (95% interval 80.7% to 86.7%) correctly. `hot_iq`'s answer differed from the
reference's on 1.6% (95% interval 0.6% to 2.7%) of the questions, Volta sPLIt's on
15.4% (95% interval 12.3% to 18.5%). Against the reference, `hot_iq`'s accuracy is
NOT DISTINGUISHABLE FROM THE REFERENCE AT THIS SAMPLE SIZE, and Volta sPLIt's is NOT DISTINGUISHABLE FROM THE REFERENCE AT THIS SAMPLE SIZE.

**Counted after scoring, outside the rules fixed in advance** (answers only, from the receipts,
[script](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/quality/closeness-test/after-scoring/mc_no_answer_counts.py)). The reference and `hot_iq` gave
no answer on the same 38 questions. 37 of Volta sPLIt's 79 changed answers fall on those 38 questions, where Volta
sPLIt answered and the reference did not. On the 475 questions the reference answered, `hot_iq` changed 8 answers
(1.7 %) and Volta sPLIt 42 (8.8 %).

Chart: [multiple-choice accuracy by build](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/quality/closeness-test/charts/mc.svg).

### Verdicts, by the rules fixed in advance

| Question | Verdict |
|---|---|
| Is `hot_iq` closer to the NVFP4 reference than Volta sPLIt (KL divergence, shared tokens)? | HOT_IQ CLOSER |
| The same, by log-probability gap | HOT_IQ CLOSER |
| The same, by next-token disagreement | HOT_IQ CLOSER |
| Does `hot_iq` change fewer multiple-choice answers than Volta sPLIt? | HOT_IQ CLOSER |
| `hot_iq` against the reference's own repeat (KL divergence) | DIFFERS FROM THE REFERENCE |
| Volta sPLIt against the reference's own repeat (KL divergence) | DIFFERS FROM THE REFERENCE |

"Indistinguishable at this sample size" means the 95% interval reaches the noise floor. It does not mean equal. A
difference smaller than this test can resolve may still exist.

### What these numbers show and what they do not

They show how often `hot_iq` picks the same next token as NVIDIA's NVFP4 pack on these cards, how far its
probabilities move, how long its greedy output stays the same, and how many multiple-choice answers change, each
next to the same figures for Volta sPLIt and next to the reference's own repeatability.

They do not show closeness to the unquantized model (the reference is NVIDIA's quantization). They do not cover
sampling at a temperature above 0, thinking, or long answers, because every test here is greedy, without thinking,
and at most 32 tokens. The KL divergence is computed over each build's top 20 tokens, so it is a lower bound of the
full value. The prompts are short (about 100 to 250 tokens). The figures come from one hardware layout, eight V100
cards in two pipeline stages, where the reference's NVFP4 weights are computed by this engine, not by NVIDIA's own
hardware path. With 513 questions, an accuracy difference of less than about 4 points
cannot be resolved, which is why the share of changed answers is reported beside accuracy.

The earlier check on the replaced 12-prompt set, which this build passed, is in [Measured results](#measured-results).

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
| Closeness test, 320 contexts and 513 MMLU questions (2026-11-11, 1,475 MHz) | same next token 95.1 % (Volta sPLIt 92.2 %), KL divergence 0.0794 nats (0.1958), changed answers 1.6 % (15.4 %). HOT_IQ CLOSER on all four comparisons fixed in advance ([How close to NVFP4](#how-close-to-nvfp4)) |
| Boot of this package on eight cards (2026-11-11, 1,475 MHz) | ready after 263 s, no traceback, tiers and both tier digests equal the measured boot's on all eight ranks. Greedy output on 12 + 5 check prompts token for token identical to the measured build's (1,024 tokens each) |
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
slower (one run per cell). The final build of 2026-11-12, at 1,522 MHz, gives its first token sooner than Volta sPLIt's bench at every length from 1K to 256K (108.4 s against 118.7 s at 256K) and still decodes slower in every code and legal cell and every short-prompt row
([speed bench](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#speed-bench-2026-11-12-final-build)).

![Decode speed by prompt length on the final build, one request at a time on the two-slot default, against Volta sPLIt's published bench](https://huggingface.co/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4/resolve/main/charts/decode-by-length-final-light.png)

The chart above is the final build of 2026-11-12: two-slot default, one request at a time, 175 W and 1,522 MHz,
the clock ceiling of Volta sPLIt's bench. Against the previous public bench, the kernel build of 2026-11-11, the
general short set went from 101.1 to 123.2 tok/s, the legal-style one from 81.1 to 97.3, and the first token
after a 256K prompt from 207.9 to 108.4 s, build against build ([before and after](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#before-and-after)).

**The final request of the 216K tool loop ran to the limit on the bench boot.** The first read took 166.22 s and all
seven follow-ups kept the previous prompt, then the final request ran to the 32,000-token limit inside thinking and
gave no final answer. On the first boot the same loop's final answer ended normally (1,385 tokens). So on this
build that loop's final request has run to the limit in 1 of 2 runs. It ran to the limit again on the final build's bench boot of 2026-11-12: 2 of 3 runs with the group-plan kernel, 4 of 7 on every build so far.

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
| `tier_map.json` | 1 | 23,564 | the first release: the hot experts per layer and the tier digest per stage |
| `tier_maps/{slots1,slots2,slots3,native-slots1}.json` | 4 | 93,406 | since 2026-11-12: one tier map per serving option, same format |
| `gguf/` | 4 | 138,238,893,056 | ISTA-DASLab GSQ-RCO IQ2_XXS GGUF, four shards, revision `ad1e8991`, unchanged |
| `engram/` | 1 | 110,595,290,816 | ISTA-DASLab Engram IQ4_NL, revision `ad1e8991`, unchanged |
| `dspark/` | 5 | 7,933,308,215 | DeepSeek DSpark drafter shards 44 to 46, revision `1bae0e10`, unchanged, with a drafter `config.json` and an index of those three shards written by PLI Labs (the index is the base index's 2,401 entries for shards 44 to 46) |
| `base/` | 19 | 13,934,483 | DeepSeek config, tokenizer, encoding and licence, revision `1bae0e10`, unchanged |
| `serve/hot_iq.yaml.template` | 1 | 2,377 | the serving profile of the default option (two slots), with `${PKG}` for the package path. In the first release: one slot, 2,164 bytes |
| `serve/hot_iq-{slots1,slots2,slots3,native-slots1}.yaml.template` | 4 | 9,528 | since 2026-11-12: one serving profile per option |
| `charts/decode-by-length-light.png`, `charts/decode-by-length-dark.png`, `charts/decode-by-length-final-light.png`, `charts/decode-by-length-final-dark.png`, `charts/closeness-survival.png` | 5 | 1,142,507 | the speed charts of [Speed bench](#speed-bench-1k-to-256k), first release and final build, light and dark, and the closeness chart of [How close to NVFP4](#how-close-to-nvfp4) |
| `manifest.json`, `SHA256SUMS` | 2 | | every file with its size and sha256 |

The GGUF and Engram files are ISTA-DASLab's four-shard files of revision `ad1e8991`, from before their repo moved
to llama.cpp's six-shard layout on 2026-10-27. The engine of this release reads them as they are. llama.cpp does not
load them (ISTA-DASLab's card, "Updating an existing download", gives the converter).

Total about 556.3 GB (518.1 GiB). Every expert is in `experts-mxfp4/`, also the compressed ones: the engine reads a
prompt with the full-precision records of all experts, staging the compressed experts' records from host RAM, and
decodes the compressed experts from their IQ records on the cards. This is the configuration that was measured. The
GGUF ships whole and unchanged, so the hot experts' IQ records (52.0 GB) are in it and unused.

**Host RAM:** with native reading (the first release and the `native-slots1` option) the compressed experts' MXFP4
records, 180.8 GB, are held in pinned host memory for prompt reading, plus the Engram tables. The measurements were
taken on a host with 629 GiB of RAM. The default option reads prompts from the IQ2-class records on the cards
instead. Its host RAM use was not measured on its own, and its measurements of 2026-11-12 ran on the same host.

## How to run it

The tier-map reader passed the CPU check on 2026-11-10 (the tier map gives the measured
tiers and digests on all 40 layers, and every expert record of layers 0 and 20 on the first rank is byte-equal to
the measured one). On 2026-11-11 this package booted on eight cards with those tiers and digests and gave the
measured build's output token for token ([Measured results](#measured-results)).

1. Download: `hf download ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4 --local-dir /path/to/pkg` (556 GB).
2. Check: `cd /path/to/pkg && sha256sum -c SHA256SUMS`.
3. Build the engine and lay out its run directory as the [GitHub repository](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4)
   describes in [From the Hugging Face package](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#from-the-hugging-face-package-hot_iq).
4. Start, with `LANE` and `CUDA_VISIBLE_DEVICES` set as the repository describes:
   `cd "$LANE/run" && bash serve_hot_iq.sh /path/to/pkg 8000`. `NVFP4_OPTION` picks the option: `slots2` (the
   default), `slots1`, `slots3` (fast prompt reading) or `native-slots1` (one slot, native reading, the
   reading of the closeness figures). The launcher writes the serving profile from this package's
   `serve/hot_iq-OPTION.yaml.template` with `tier_maps/OPTION.json` and starts the repository's `run_serve.sh`
   with that option's slot count, which sets the load-time allocator setting the measurements ran with. The fast-reading options name the RB 4
   prompt-reading library, which the repository builds from source
   ([engine/kernels/prefill-rb4](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/tree/main/engine/kernels/prefill-rb4)). Without it, change their line
   `nvfp4_8r_prefill_rb: rb4` to `rb2`: same output bits, without the second-read gain.
5. At load each rank logs its tiers, hot / compressed experts and tier digest per stage:

| Option | Stage 0 | Stage 1 |
|---|---|---|
| `slots1` | 3,833 / 3,847, `8e7829184090` | 2,382 / 5,298, `7f442956514b` |
| `slots2` (default) | 3,691 / 3,989, `0b6440ce205f` | 2,258 / 5,422, `f67e73174ded` |
| `slots3` | 3,531 / 4,149, `a5d4b7a514d1` | 2,132 / 5,548, `7cad74ab15b9` |
| `native-slots1` | 3,833 / 3,847, `8e7829184090` | 2,459 / 5,221, `9473486077c6` |
| first release (`tier_map.json`, its engine only) | 3,833 / 3,847, `8e7829184090` | 2,254 / 5,426, `5b09c0383a80` |

With a tier map the engine does not plan tiers. If a card has less free memory after the dense weights than the
measured cards had, it refuses to load and names the bytes it is short, rather than choosing a different set.

## Limits

- **Only for this hardware and engine.** The tier map fits PP2 x TP4 on 32 GB cards. Another split is refused.
- **Prompt reading.** The first release and the `native-slots1` option read prompts with the full-precision records,
  staged from host RAM. Since 2026-11-12 the default reads them the fast way, with the compressed experts' IQ
  records. Its first read of a long chat had stopped the host or a card three times out of three. After a change to
  the host's power delivery it read nine of nine long chats with no fault, 86 to 119 s near 200K tokens against 155
  to 165 s. Quality not established: it changes the output from the first token, no closeness test ran with it, and
  its window failed its memory check by 64 MiB on stage 1, which the slot options answer with 192 MiB more reserve.
- **Slots and a long read.** With two or three slots a short request sent during a 209K-token read starts in about
  7 s but decodes at about 9.6 tok/s until the read ends, and the read takes about 35 s longer.
- **Memory on the final bench boot.** On the bench boot of 2026-11-12 (two-slot default, one request at a time)
  the cards of stage 1 reached 32,000 MiB against the build's own 31,998 MiB line (the card's 32,510 MiB less 512),
  2 MiB over it and 510 MiB below the card's capacity, in 38 of 359 samples, all during prompts of 8K to 256K
  tokens. No out-of-memory line in any of the 169 requests. The one-, two- and three-slot boots of the same
  window stayed under the line ([speed bench](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#speed-bench-2026-11-12-final-build)).
- **The hot set is fixed.** There is no promotion of experts during a chat. Text unlike the tally's four kinds may route
  more often to compressed experts.
- **Quality checks are narrow:** the closeness test is greedy, thinking off and at most 32 tokens per context, plus
  12 prompts with thinking off and three long thinking chats at temperature 0. The termination test's other six
  chats and its run at sampling settings were not run.
- **Compressed records are not full precision.** On synthetic inputs an expert's IQ2-class record is far from its
  official record (relative L2 0.60 to 0.69 in three sampled layers). On real text, how far that moves the output is
  measured in [How close to NVFP4](#how-close-to-nvfp4).
- **Speeds are in-sample** on the public prompt sets (see above). The first bench ran at a 1,425 MHz clock ceiling,
  the final bench of 2026-11-12 at 1,522 MHz.

## Credits and licences

- **DeepSeek**: DeepSeek-V4.1-Flash-1017, its official MXFP4 experts, the DSpark drafter and the Engram design. MIT.
- **NVIDIA**: the DeepSeek-V4.1-Flash-1017-NVFP4 pack whose expert records are folded here. MIT.
- **ISTA-DASLab**: the GSQ-RCO GGUF and the Engram IQ4_NL file, shipped unchanged. Its weights inherit the base
  model's MIT licence.
- **ggml**: the GGUF and IQ formats. **1CatAI / 1Cat-vLLM**, **v100-skinny** and **the vLLM project**: the engine
  underneath (see the GitHub repository's CREDITS.md).

Licence texts are in [LICENSES/](LICENSES). The files PLI Labs wrote (tier map, manifests, profile template, this
card) are MIT ([LICENSE](LICENSE)). See [NOTICE](NOTICE).
