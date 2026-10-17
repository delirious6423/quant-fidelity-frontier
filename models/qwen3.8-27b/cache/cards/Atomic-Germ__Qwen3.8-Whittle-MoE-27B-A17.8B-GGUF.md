---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags: [moe, mixture-of-experts, research-preview, whittle, qwen3.5, gguf, router-healing]
---

> ## v2.2.1 fixes the reasoning abort (29 Aug 2026)
>
> **Fixed in the safetensors weights at the root of this repo.** Served with
> reasoning enabled, v2.2 could end its turn inside its own chain of thought
> instead of answering. The stop gate had been trained on sequences where the chat
> template opened a `<think>` block that the training data never closed, so the
> most heavily weighted signal in the set landed on a terminator *inside* the
> thinking block.
>
> v2.2.1 replaces the 64 `shared_expert_gate` tensors with a gate trained on real
> chain-of-thought traces from the parent, where the block is closed by the
> teacher's own words. On a 24-prompt structural probe with reasoning enabled,
> turns that ended without ever emitting `</think>` (or closed it with an empty
> answer) fell from **21/24 to 3/24**. Zeroing the gate entirely scores 11/24, so
> the new gate is doing real work rather than just undoing the damage.
>
> With reasoning off, the held-out counted set (n=36) is unharmed and modestly
> better. Measured like-for-like against v2.2 on the same prompts: turns that stop
> cleanly 32/36 -> 34/36, distinct-item exact counts 20/36 -> 24/36, and mean
> overrun 1.69x -> 1.14x. Mean repeated-4-gram is **unchanged at 0.141** — the new
> gate does not reduce repetition, and 0.141 is still above our 0.05 target. Long
> lists of 45+ items remain weak (4/12 and 5/12 respectively). The reason to take
> v2.2.1 is the reasoning abort, not the counting.
>
> **The GGUF files have not been rebuilt yet and still contain the v2.2 gate.**
> If you serve a quantised tier, keep reasoning off until those are republished.
>
> Nothing else changed: v2.2.1 is v2.2 with 64 tensors (0.66 MB) swapped. The v2.2
> gate is archived at `v2.2/shared_expert_gate.safetensors` (root weights plus
> those tensors reconstructs v2.2 exactly), and v2.1 remains at `v2.1/`.


**Funding status (20 Aug 2026):** this is a self funded research preview and the compute budget is now exhausted. Further work on this model depends on donations: https://ko-fi.com/davida81328

<p align="center">
  <img src="whittle.svg" width="640" alt="Whittle">
</p>

# Whittle MoE 27B (A18B): a mixture of experts rescued by its routers

**A post hoc mixture of experts carved from Qwen3.8-27B, then taught when to stop talking. As of v2.2 the always-on shared-expert gate carries a trained stop signal: list repetition is down 80 percent against v2.1 and conversational turns terminate 85 percent of the time (was 56). This is the model this repo was always trying to build.**

The architecture is a partition, not a rebuild. All 64 layers keep the parent's attention side untouched: the 3 to 1 hybrid of gated deltanet layers and full attention, 16 attention layers in all, hidden size 5120. The surgery is in the feed forward. Each layer's dense FFN, 17408 wide in the parent, was cut into 64 routed expert slivers of width 192 plus one always on shared expert of width 5120. The arithmetic is exact: 64 x 192 + 5120 = 17408, the original width to the neuron, with zero new FFN weights invented. A small router per layer picks 16 of the 64 slivers per token, so each token runs 8192 of the original 17408 FFN width: 17.8B active parameters of 27B total. Freshly carved, the model was gibberish (4 of 39 on our knowledge battery). Training only the 64 routers with every expert frozen recovered 27 of 39: the knowledge never left the slivers, the routers just had to learn which ones to fire. Multi turn SFT and the complete answer distillation rounds did the rest.

Download the v2.2 Q4_K_M, serve it with llama.cpp, and talk to it. Loop rates on a fixed harness, same prompts and seeds since the first release: 8 percent on long single answers (down from 69), 7 percent across conversations, 22 percent on structured output (down from 75), zero silent or truncated answers, knowledge battery 28 of 39. Every number, including the bad ones, is in the table below with its full history.

It is not the parent. Hard structured asks still fail about one time in five, and arithmetic inherits the family weakness. What it is: a working demonstration that a starved post hoc MoE can be brought back by retraining the smallest possible thing first, with the whole method, dataset, and failure log public. Runs on 24GB of VRAM quantized, splits across two 12GB cards.


## How to run it

```
/home/support/llama.cpp/build/bin/llama-server -m Whittle-MoE-27B-A18B-v2.2-Q4_K_M.gguf --host 0.0.0.0 --port 8090 -ngl 99 -c 8192 -fa on --jinja
```
### Honest limitations, read before using

- **Structured output is still weak.** 39 percent of SQL, HTML and markdown
  table generations degenerate. Better than 75 percent, still not good.
  This is the target of the next round.
- **The lineage has a token level wobble this round did not fix**:
  occasional made up words and scuffed numbers, inherited from the MoE
  compression itself. The knowledge battery moved one fact net, inside the
  measured noise floor, but one clear regression exists (capital of Egypt)
  alongside three clear recoveries (longest river, tallest mountain, first
  Moon walker).
- Evaluated by one person on a small harness. Treat every number as a
  workshop measurement, not a benchmark.


## Training data

Router distillation used logits from Qwen3.8-27B over this project's heal corpus.
Multi turn SFT used ultrachat_200k (MIT), tulu-3-sft-mixture (ODC-BY),
CodeFeedback-Filtered-Instruction (Apache-2.0) and the project's own corpus.

## v2: the anti-loop round (22 Aug 2026)

Five earlier training attempts established that repetition and stopping are
one behaviour: every fix for looping also taught the model to truncate,
because the training data never contained an answer that ends because it is
finished. This round trained on exactly that: 245 complete answers written
by the parent model with its top 32 token probabilities at every position,
spanning 7 token factual answers to 4,000 token structured documents and
multi turn conversations (dataset: logic65/whittle-teacher32-complete-answers).
A capacity sweep published in WHITTLE_FINDINGS.md ruled out the main
alternative first: more active experts change nothing, the fix had to be
learned.

### v2 files

- `Whittle-MoE-27B-A18B-v2-antiloop-Q8_0.gguf` and `-Q4_K_M.gguf`
- v2 bf16 safetensors under `v2/`
- the training adapter under `v2-adapter/`, trainer and gate scripts in the
  findings, training data in the dataset repo

## v2.1: the balancing round (22 Aug 2026)

The current release, and the recommended download. One further training epoch on the v2 lineage: the anchor slice was broadened (short answers, medium answers, code, and one seed of each conversation restored), structured examples kept, extras down-weighted to 1.5x, learning rate halved.

Both GGUFs are verified against stock llama.cpp: correct expert routing (top 16 of 64
reported by the runtime), EOS 248046, chat template embedded, multi turn tested.

### Measured results, same harness and seeds as the baseline

| measure | previous release | v2 | v2.1 |
|---|---|---|---|
| single turn loop rate (12 prompts x 3 seeds) | 69% | 11% | 8% |
| multi turn loop rate (2 conversations x 2 seeds, all turns) | 64% | 7% | 7% |
| late turn loop rate (5th turn onward) | 56% | 17% | 8% |
| structured output loop rate (SQL, HTML, tables) | ~75% | 39% | 22% |
| median answer length | 268 words | 386 words | 388 words |
| silent or truncated answers | not measured | 0 | 0 |
| knowledge battery | 28/39 | 27/39 | 28/39 |




## v2.2: the stop gate (28 Aug 2026)

The current release. The change is 0.33M parameters - 0.001 percent of the
model: the 64 per-layer `shared_expert_gate` tensors, which ship at random
init in every Qwen3.5-MoE-style checkpoint and multiply the always-on shared
expert per token. Trained with distillation against the parent plus explicit
supervision on the position that predicts the terminator, they become a stop
signal that is active on every token. Stock llama.cpp already applies this
tensor, so the GGUFs work unmodified.

Two failed attempts preceded it, and the root cause is worth recording: the
first distillation capture stopped one position short of the terminator, so
the most heavily weighted training position was the one where the teacher
predicts a newline (120 of 164 answers, p = 0.75) - the objective taught the
model to continue, 32x amplified. Overrun scaled with trainable capacity
(2.85x untrained, 3.62x at 0.33M, 6.39x at 147M) until the capture was fixed.
If you train stop behaviour from distilled logits, assert that your
highest-weighted positions actually carry the terminator label.

### Measured results - 36 held-out counted prompts + 8 held-out conversations, sampled (temp 0.7, top_p 0.8, top_k 20)

| measure | v2.1 | v2.2 |
|---|---|---|
| counted lists: answers that terminate | 18/36 | **27/36** |
| counted lists: worst-window repetition | 0.237 | **0.048** |
| counted lists: mean overrun vs asked count | 2.85x | **2.57x** |
| counted lists: exact item count | **8/36** | 2/36 |
| conversation turns that terminate (48 turns) | 27/48 | **41/48** |
| conversation repetition, mean / late turns | 0.031 / 0.036 | 0.035 / 0.052 |
| turns over the 0.15 loop threshold | 1/48 | 2/48 |

The regression is stated plainly: v2.2 reliably ends a list but not at the
requested count. Termination and counting are different skills. Counting
accuracy above roughly 30 items appears to be an architectural depth limit of
single-pass transformer generation rather than anything this training did -
every checkpoint tested, including the untrained carve, is near zero on exact
counts at 45+ items. If your use case is "give me exactly N items", pin v2.1
(revision `3ae1efa7`); for everything else v2.2 is the better model.

For scale: a 125.8M shared-expert LoRA trained on the same corrected objective
reached 25/36 termination at 3.24x overrun - the 0.33M gate beats 380x more
parameters. The stop lives in the always-active path.

The legacy loop-rate table above still shows v2.1; the v2.2 column on that
harness is being re-measured on the released Q4 and will be added.

### v2.2 files

- root safetensors are v2.2 (v2.1 is git revision `3ae1efa7` of this repo)
- GGUFs: Q8_0, Q6_K, DQ5_K_XL, DQ4_K_XL, Q4_K_M, DQ3_K_XL in the
  [GGUF repo](https://huggingface.co/logic65/Qwen3.8-Whittle-MoE-27B-A17.8B-GGUF)
- trained gate + trainer + eval harness + the corrected capture scripts:
  [whittle-stop-kd](https://huggingface.co/datasets/logic65/whittle-stop-kd)

## A faster sibling: Whittle-Next-27B-A3B

This model keeps **17.8B parameters active** per token: it is the stronger and slower of the two.

If you want the same 27B total at a fraction of the compute, [**Whittle-Next-27B-A3B**](https://huggingface.co/logic65/Whittle-Next-27B-A3B)
([GGUF](https://huggingface.co/logic65/Whittle-Next-27B-A3B-GGUF)) activates only **~3B** parameters per token, so it generates at the speed of a
3B model, and adds hyper-connection residual streams plus a 2B-row hashed n-gram memory carried over from Qwen3.8-Flash-Next. It runs on stock
llama.cpp with no patches.

Be warned, it is a **research preview and a weaker model overall**: it is noticeably worse at long-context reading (60k+ tokens) than this one,
and the numbers on its card are honest about that. Pick this model for quality, pick Whittle-Next for speed and memory footprint, or use both.

## Support this work

Everything here runs on personal hardware and paid Colab time, funded out of pocket.
If a 27B MoE you can run at home is useful to you, or you just want to see where
router healing goes next, donations directly buy the GPU hours:

**[ko-fi.com/davida81328](https://ko-fi.com/davida81328)**


## Distillation data

All teacher data behind this model is public:
https://huggingface.co/datasets/logic65/whittle-teacher32-complete-answers
(main set: 245 complete answers with top 32 logprobs; r2-structured split:
34 long structured exemplars up to 4.9k tokens). Generators and trainer
scripts ship with the data and in WHITTLE_FINDINGS.md.
