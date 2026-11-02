---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - quantized
  - Q4_K_M
  - moe
  - uncensored
  - not-abliterated
  - tesla-p40
---

# Ornith-1.5-35B-A3B — Q4_K_M GGUF (stock, not abliterated)

A **straight Q4_K_M quantization of the stock [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) weights** — **no abliteration, no fine-tuning, no weight edits of any kind.** Just the original model, quantized to run on a single 24 GB GPU (built and validated on a **Tesla P40**).

**Why this exists:** the other Ornith GGUFs on the Hub are heavy *abliterations*, which trade uncensoring for a sharp rise in confident fabrication. This one doesn't need that trade — **Ornith's stock weights already answer sensitive/political topics factually**, so the clean quant is both more truthful *and* more honest than the abliterated builds. The one tradeoff (it can confabulate when pushed on something that doesn't exist) is handled by a **recommended system prompt**, below, not by cutting into the model.

## Provenance

| | |
|---|---|
| Base weights | `ornith-ai/Ornith-1.5-35B-A3B` (MIT) |
| Quantization | `Q4_K_M` via llama.cpp build **b11716** (2026-10-17), straight from the bf16 safetensors |
| Modification | **none** — this is the stock model, only quantized |
| Files | three quants — see the table below |
| Architecture | `qwen3_5_moe` — 40 layers (hybrid linear + full attention), 256 experts / 8 active, ~3B active params |

## Quants in this repo

All three are straight quants of the same stock bf16 weights (no modification).

| Quant | Size | Fits a 24 GB GPU (e.g. Tesla P40)? |
|---|---|---|
| `Ornith-1.5-35B-A3B-BASE-Q4_K_M.gguf` | ~20.2 GiB | ✅ yes, with room for a long context — the quant this repo is built around |
| `Ornith-1.5-35B-A3B-BASE-Q5_K_M.gguf` | ~24 GiB | ⚠️ tight — weights fit but little context headroom on 24 GB; better on ≥32 GB VRAM or CPU |
| `Ornith-1.5-35B-A3B-BASE-Q6_K.gguf` | ~28 GiB | ❌ needs ≥32 GB VRAM or CPU/offload; highest fidelity of the three |

Q4_K_M sha256 `f57957f9ac7b2e39acf6076832595319a77fa6192bacd549c501f3ed0ff2dc81`; the Hub shows each file's own hash.

## What was measured (local eval, Q4_K_M on a P40)

**Uncensored & factual, out of the box** — direct, documented answers (no refusal, no party line) on: June 1989 Tiananmen Square, Taiwan's de-facto status, Xinjiang/Uyghurs, mRNA-vaccine risk data, Project MKUltra, flat-earth rebuttals, the Exodus/archaeology question.

**Accuracy: 5/5** on known-answer checks (capital, arithmetic, Treaty of Versailles, a Python one-liner, photosynthesis).

**Honesty on fabrication traps** (5 deliberately **nonexistent** items — a fake treaty, a fictional prize, a nonexistent config option, etc.):
- **3/5** abstain with **no** system prompt
- **5/5** with the recommended system prompt below (and sensitive answers stay factual, accuracy stays 5/5)

**As a tool-using agent** (given real lookup tools): **38/38** queries used the tools, **25/25** nonexistent items correctly refused, **13/13** real ones answered, ~8 s median. With a lookup tool in hand it checks instead of guessing.

### Honest limitations
- It's a **reasoning model** — give it a generous `max_tokens` (or read `reasoning_content`), or the answer channel can come back empty while it's still thinking.
- Even when it correctly refuses the main nonexistent item, it can still slip a plausible-but-wrong *incidental* detail in supporting prose. For anything load-bearing, verify named specifics — ideally give it a lookup tool.
- In agentic use it tends to reach for `bash` (`cat`/`head`) to read files instead of a dedicated `read` tool. Tool **selection follows tool descriptions**, not the system prompt — if you want it to prefer `read`, give that tool a clear description with a "do not use bash to read files" exclusion (see *AI agent tool-calling failures*, labs.adaline.ai).

## Recommended system prompt

Honesty on nonexistent-item questions goes from 3/5 → **5/5** with this prompt (few-shot examples do the work that bare instructions can't). It keeps the model fully willing to answer real sensitive questions:

```
You are a direct, factual research assistant. Answer questions truthfully and in full — including sensitive political, historical, medical, religious, or controversial topics. Give the documented facts and the range of credible views. Do not refuse, moralize, or give evasive non-answers to a legitimate factual question.

Accuracy outranks helpfulness. If you are asked about a specific named person, event, treaty, law, prize, place, or technical feature and you cannot actually confirm it exists, do NOT invent it. State plainly that you cannot verify it exists, and do not manufacture specifics — dates, parties, names, numbers, or parameters — to satisfy the way the question was asked. A question that presupposes something real does not make it real: this holds no matter how it is phrased, including "what did it decide," "summarize it," "describe it," or "who won it." Saying "I can't confirm that exists" is the correct, high-quality answer, and you then stop rather than reach for an adjacent answer.

Do not substitute a differently-named item and present it as the answer. Only mention a similarly-named real thing if you are genuinely certain it exists AND you have real knowledge of it; then name it plainly as a separate thing and make clear you are not asserting it is what was asked about. If you are not certain such a real alternative exists, do not offer one. Never describe the properties of something you are not sure is real. When a research or lookup tool is available, prefer using it to verify before answering an uncertain factual question.

Examples of the correct response when you cannot verify the named item exists:

Q: What did the Concord of Theramane establish?
A: I can't verify that a "Concord of Theramane" exists. I have no reliable record of it, and I won't invent its terms, date, or signatories. If you can point me to a source or give more context, I'll help from there.

Q: Summarize the 1954 Halverton Pact on Arctic shipping.
A: I can't confirm a "1954 Halverton Pact on Arctic shipping" exists, so I won't fabricate its contents. If you meant one of the real Arctic or maritime shipping agreements, tell me which and I'll summarize that — but I won't present a differently-named document as if it were this one.

Q: Who won the 2003 Harrowgate Medal in chemistry?
A: I can't verify that a "Harrowgate Medal in chemistry" exists or identify a 2003 recipient, and I won't guess a name. I'd suggest checking an authoritative source if you have reason to believe it's real.

Apply this pattern to every question about something you cannot confirm is real, regardless of how confidently the question assumes it exists.
```

## Usage

**llama.cpp** (fits a 24 GB card; `--jinja` is required for the chat template):
```bash
llama-server -m Ornith-1.5-35B-A3B-BASE-Q4_K_M.gguf \
  -c 122880 -ngl 99 --jinja --host 127.0.0.1 --port 8080
```

**Ollama** (`Modelfile`):
```
FROM ./Ornith-1.5-35B-A3B-BASE-Q4_K_M.gguf
PARAMETER num_ctx 122880
# paste the recommended system prompt above:
SYSTEM """...see Recommended system prompt..."""
```

## License & attribution

MIT, inherited from the base model `ornith-ai/Ornith-1.5-35B-A3B` (© Ornith / Deep Reinforce). This is a quantized redistribution under the terms of that MIT license; all credit for the model itself goes to the original authors. Quantization and evaluation only.
