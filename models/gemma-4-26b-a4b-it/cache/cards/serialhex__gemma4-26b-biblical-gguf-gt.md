---
base_model: google/gemma-4-26B-A4B-it
tags:
- gguf
- theology
- christianity
- reasoning
license: cc0-1.0
---

# An AI that reads the whole passage before it talks

> **This is the Grug-Think + GGUF edition of the big model** — the fast
> reasoning variant of the 26B-A4B mixture-of-experts fine-tune, quantized
> to run with [llama.cpp](https://github.com/ggml-org/llama.cpp) or
> [Ollama](https://ollama.com). The MoE architecture means most of those 26B
> parameters sit idle per token (~4B active), so this is more capable than
> the 12B line at a smaller real inference cost than its total size suggests.

Ask a general-purpose chatbot a real theology question — "why does God let
bad things happen," "did Paul really mean that," "is this verse about me or
about Israel" — and you get one of two failure modes: a confident answer built
on a single verse yanked out of its chapter, or three paragraphs of hedging
that never actually engages the question. Neither helps you think.

This is a QLoRA fine-tune of [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it)
trained specifically *not* to do that. Every answer is built from reading the
full pericope — the passage, the argument, the narrated scene — not an
isolated verse. When the Bible genuinely holds something in tension (grace
and real human response, sovereignty and freedom), the model is trained to
hold it too, instead of flattening it into whichever half is easier to state.
And where Christians have historically disagreed for real reasons, it says so
and shows you the shape of the disagreement instead of picking a side and
pretending it was never contested.

**You don't need to be a seminarian for this to be useful.** If you've ever
had a question about a Bible passage and gotten either a Sunday-school
one-liner or an answer that clearly didn't read past the one verse you
quoted — that's the gap this is trying to close.

## What's actually different about it

- **Reads at the passage level, not the verse level.** Context (who's
  speaking, to whom, what comes before and after) does real work in every
  answer, not just as a disclaimer at the end.
- **Shows its reasoning.** Every answer carries a `think` trace — the actual
  chain of "here's what the text says, here's what that rules out, here's
  the conclusion" — so you can check *why*, not just read a verdict.
- **Doesn't dodge the hard, genuinely contested questions.** Predestination,
  the nature of grace, sacramental theology — where real, informed Christians
  have disagreed across history, the model engages the disagreement instead
  of silently picking a denomination's answer or refusing to engage.
- **Built on sources you can go check yourself.** Training data is drawn from
  the public-domain World English Bible, the early creeds, the Didache,
  patristic writings, and classical logic/philosophy texts — not a black box.
  The dataset is [CC0](https://huggingface.co/datasets/serialhex/Open-Biblical-Dataset)
  and public.

## Why Grug-Think: speed, not readability

Nobody reads the `think` trace — it's internal scratch space, not a blog
post for a human audience. So the base model narrating it in full sentences
("Let me consider the context... First, I'll note that...") is pure waste:
tokens generated and waited on that do nothing for output quality.
Grug-Think runs every training trace through a rewrite pass targeting short,
dense, declarative reasoning — the same hypothesis-and-check, the same
conclusion, filler banned — so the model learns to *think in way fewer
tokens* for identical output quality. On the biggest model in this family,
that token savings matters even more than on the 12B line — this is the
combination to reach for when you want the stronger model without paying for
it in latency.

## Pair it with retrieval for real gains

This model was trained to *reason well over a passage once it has one in
front of it* — it was not trained as a Bible-memorization engine, and a base
LLM's recall of exact wording and cross-references degrades the way any
LLM's does. You will get noticeably better, more precise answers if you put
it behind even a simple RAG setup: grep/embed the relevant chapter(s) for the
user's question and inject the actual verse text into context before the
model answers, rather than relying on it to recall the passage from training
alone. Retrieval does the remembering; the model does the reading and the
reasoning — that split is where this model is strongest.

## Quick start

```bash
# llama.cpp
llama-cli -hf serialhex/gemma4-26b-biblical-gguf-gt -p "Did Paul mean 'all have sinned' literally, or is Romans 3 making a narrower argument?"

# or with Ollama
ollama run hf.co/serialhex/gemma4-26b-biblical-gguf-gt
```

## The family

| Model | What it is |
|---|---|
| [gemma4-12b-biblical-qlora](https://huggingface.co/serialhex/gemma4-12b-biblical-qlora) | Smaller base model (Gemma 4 12B) |
| [gemma4-12b-biblical-gguf](https://huggingface.co/serialhex/gemma4-12b-biblical-gguf) | 12B, quantized |
| [gemma4-12b-biblical-qlora-gt](https://huggingface.co/serialhex/gemma4-12b-biblical-qlora-gt) | 12B, Grug-Think reasoning |
| [gemma4-12b-biblical-gguf-gt](https://huggingface.co/serialhex/gemma4-12b-biblical-gguf-gt) | 12B, Grug-Think, quantized |
| [gemma4-26b-biblical-qlora](https://huggingface.co/serialhex/gemma4-26b-biblical-qlora) | Gemma 4 26B-A4B (MoE) fine-tune |
| [gemma4-26b-biblical-qlora-gt](https://huggingface.co/serialhex/gemma4-26b-biblical-qlora-gt) | 26B, Grug-Think reasoning |
| [gemma4-26b-biblical-gguf-gt](https://huggingface.co/serialhex/gemma4-26b-biblical-gguf-gt) | **This model** — 26B, Grug-Think, quantized |

## Known limitations

- This is a LoRA fine-tune on top of a general instruction model, not a
  theologian. It will be wrong sometimes. Verify anything load-bearing
  against the sources it cites.
- Training data skews toward Protestant evangelical framing in places even
  where it tries to represent other traditions fairly — spot-check before
  treating it as neutral across traditions.
- It is not a substitute for a pastor, a priest, a rabbi, or a therapist.

## Quantization

Q4_K_M, converted from the merged full-precision Grug-Think model via
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `convert_hf_to_gguf.py`
and `llama-quantize`. The merge step used [Unsloth](https://github.com/unslothai/unsloth)
directly, since this model's MoE LoRA wrapper modules aren't recognized by
plain `peft` merge.
