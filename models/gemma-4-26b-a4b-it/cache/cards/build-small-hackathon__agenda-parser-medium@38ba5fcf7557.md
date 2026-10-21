---
license: gemma
base_model: google/gemma-4-26B-A4B-it
datasets:
- build-small-hackathon/agenda-parser-tool-traces
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- agent
- agentic
- tool-use
- function-calling
- react
- local-government
- agenda-parser
---

# agenda-parser-medium

**A Gemma 4 26B-A4B (MoE) fine-tune that drives the [Agenda Parser](https://huggingface.co/rdubwiley) agents' tool-calling loop** — quantized to **Q4_K_M** GGUF for [llama.cpp](https://github.com/ggml-org/llama.cpp).

This is the **medium** member of a three-model family (26B total / ~4B active params) fine-tuned to follow a strict ReAct *single-JSON-action* protocol over public-meeting agenda packets and local-government legal questions. It is **not** a general chat assistant.

| | |
|---|---|
| **Base model** | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (26B total / ~4B active) |
| **Method** | LoRA SFT → merged → Q4_K_M GGUF |
| **Training data** | [`build-small-hackathon/agenda-parser-tool-traces`](https://huggingface.co/datasets/build-small-hackathon/agenda-parser-tool-traces) (`filtered` config) |
| **LoRA adapter** | [`build-small-hackathon/agenda-parser-medium-lora`](https://huggingface.co/build-small-hackathon/agenda-parser-medium-lora) |
| **License** | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |

## What it does — the agent protocol

The model is trained to act as a ReAct agent that calls one tool at a time. Each step it must emit **a single JSON object and nothing else**:

```json
{"thought": "<one short sentence>", "tool": "<tool name>", "args": { ... }}
```

It reads the tool's result, then emits the next action, until it calls `final_answer`. It is trained on **two toolkits**:

- **Agenda packet research** — `list_agenda_items`, `get_item_text`, `search_packet` (semantic), `find_text` (exact), `summarize`, `report`, `final_answer`. Answers questions about an uploaded agenda packet (what an item approves, costs, dates, which items mention X, briefings).
- **Cornell LII legal research** (scoped to **local-government** law) — `search_regulations`, `resolve_cfr`/`resolve_usc`, `mcl_find`/`mcl_search`/`mcl_text`/`mcl_outline`/`mcl_lookup`, etc. Answers questions on Open Meetings Act, FOIA, municipal budgeting/taxation, zoning, ethics, and the Michigan statutes governing local governments — citing CFR/USC and reading Michigan MCL text.

## How it was trained

1. **Teacher traces.** Two strong teacher models — **Kimi k2.6** and **DeepSeek 4 pro** (via [OpenCode Go](https://opencode.ai)) — drove the *real* agent loop over 11 public agenda packets and a set of local-government legal questions. Tools executed live, so every observation is grounded.
2. **Judge filtering.** Each completed trace's final answer was scored for **faithfulness** against the text the agent actually retrieved (fast OpenCode-Go judge); only high-faithfulness traces were kept. One accepted agent step = one training example.
3. **SFT.** LoRA on the base's attention projections (q/k/v/o), 4 epochs over **974 examples** (held-out packet excluded — see Evaluation), full-sequence loss (the Gemma chat template lacks `{% generation %}` markers for assistant-only loss), bf16 + gradient checkpointing, then **merged** and converted to GGUF.

| hyperparameter | value |
|---|---|
| LoRA rank / α / dropout | 32 / 64 / 0.05 |
| target modules | attention + MLP `q,k,v,o,gate,up,down_proj` (auto-detected real `nn.Linear`) |
| epochs | 4 |
| learning rate | 1e-4 (cosine, 3% warmup) |
| batch × grad-accum | 1 × 16 |
| max sequence length | 4096 |
| precision / GPU | bf16 / H100 |
| final in-training token accuracy | ~0.96 |

The full training/generation pipeline (trace capture, judge, LoRA, merge, GGUF) is reproducible from the dataset card.

## Training data & provenance

Built from [`build-small-hackathon/agenda-parser-tool-traces`](https://huggingface.co/datasets/build-small-hackathon/agenda-parser-tool-traces): per-step `{system, user, assistant}` chat examples whose `system` message is the deployed agent's exact tool catalog + protocol. The source agenda packets are published in that dataset's [`source_packets/`](https://huggingface.co/datasets/build-small-hackathon/agenda-parser-tool-traces/tree/main/source_packets) folder; each trace row links to its source by `meta.unit_id`. Distilled from third-party teacher models (their terms may apply to generated text); source PDFs are public meeting records.

## Sibling models

| model | base | quant | this card |
|---|---|---|:---:|
| [`agenda-parser-lite`](https://huggingface.co/build-small-hackathon/agenda-parser-lite) | Gemma 4 E4B | Q8_0 | |
| [`agenda-parser-medium`](https://huggingface.co/build-small-hackathon/agenda-parser-medium) | Gemma 4 26B-A4B (MoE) | Q4_K_M | ✅ |
| [`agenda-parser-high`](https://huggingface.co/build-small-hackathon/agenda-parser-high) | Gemma 4 26B-A4B (MoE) | Q8_0 | |

(`lite` = fast/small; `medium` = balanced; `high` = best quality. `medium`/`high` share the 26B-A4B base, fine-tuned independently and shipped at different quants.)

## Evaluation

One agenda packet (`oakland-1570`) and a held-out task seed are **excluded from training** and reserved for a base-vs-fine-tuned A/B benchmark (objective protocol metrics — valid-JSON-action rate, clean-`final_answer` rate, tool-error rate — plus an LLM-judge of answer faithfulness, absolute and pairwise). See the project repo's `sft/eval.py`.

## Run

```bash
huggingface-cli download build-small-hackathon/agenda-parser-medium agenda-parser-medium-Q4_K_M.gguf
# --jinja loads the embedded chat/tool template
llama-server -m agenda-parser-medium-Q4_K_M.gguf --jinja
```

The model expects the agent's system prompt (tool catalog + protocol) and replies with one JSON action per turn.

## Intended use & limitations

- **Intended:** the in-process llama.cpp backend for the Agenda Parser agents over uploaded agenda PDFs and local-government legal lookups.
- **Out of scope:** general-purpose chat; non-tool-calling use; legal/financial advice. Always verify answers against the cited source packet / statute.
- Inherits the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and use restrictions.
