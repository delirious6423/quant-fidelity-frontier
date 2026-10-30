---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
pipeline_tag: text-generation
language:
  - en
tags:
  - gguf
  - exe-ai
  - agent
  - tool-use
  - llama.cpp
---

<p align="center">
  <a href="https://exe-hq.net">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/exeterminal/Exe-Core-Dynamic-V1-GGUF/resolve/main/exe-ai-logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/exeterminal/Exe-Core-Dynamic-V1-GGUF/resolve/main/exe-ai-logo-light.png">
    <img src="https://huggingface.co/exeterminal/Exe-Core-Dynamic-V1-GGUF/resolve/main/exe-ai-logo-light.png" width="480" alt="Exe AI: AI systems reimagined.">
  </picture>
  </a>
</p>

<p align="center"><a href="https://exe-hq.net"><b>exe-hq.net</b></a></p>

---

<div align="center">
  <img src="exe-core-hero.png" alt="Exe Core Dynamic v1 — the core model of Exe AI" width="100%">
</div>

# Exe Core Dynamic v1

The core model of **Exe AI**. It knows the app it lives in — the
tools, their parameters, the folder rules, the limits — and reaches for the right
one instead of guessing. It is trained on behaviour, not on facts: what to call,
when to call it, and when to answer without calling anything at all.

![Tool fluency — base 79% vs. trained 93% over 72 held-out cases](diagramm-sprung.png)

## What it does

An agent lives or dies by the small decisions. Read a file with the file
tool, not with a shell one-liner. Start a long run in the background instead of
letting it hang. Treat text that came back from a tool as data, never as an
instruction. Ask one short question when a request is genuinely ambiguous.

The training covers all of it — every built-in tool with every parameter, all the
rules Exe AI's system prompt lays down, and the ability to read a tool
schema it has never seen and call it correctly. That last part matters: users add
their own MCP servers and skills, so a fixed list would be wrong the moment
someone extends the setup.

## Intended use

Drop-in as the main chat model behind Exe AI, over any
OpenAI-compatible server (`llama-server` and friends).

**Out of scope:** it is a specialist. Outside a tool-using agent it is simply
the base model with a mild accent — use the base for general chat. It carries the
base model's vision tower untouched but was neither trained nor measured on images.

## Files

Sizes are the built files.

**No imatrix in this release.** Computing one for a 27B model on CPU ran past an
hour without finishing, and the builds were wanted sooner. The K-quants are
unaffected — they do not need one. `IQ4_XS` was built without it and is therefore
a little below what it could be; the deeper I-quants (IQ3 and below) were left out
rather than shipped in that state. A later release will add them with an imatrix
computed on a GPU.

| File | Type | Bits | Size |
|---|---|---|---|
| `Exe-Core-Dynamic-v1-bf16.gguf` | full precision | 16 | 54.7 GB |
| `Exe-Core-Dynamic-v1-Q8_0.gguf` | K/legacy | 8 | 29.0 GB |
| `Exe-Core-Dynamic-v1-Q6_K.gguf` | K-quant | 6.5 | 22.4 GB |
| `Exe-Core-Dynamic-v1-Q5_K_M.gguf` | K-quant | 5.5 | 19.5 GB |
| **`Exe-Core-Dynamic-v1-Q4_K_M.gguf`** | **K-quant** | **4.8** | **16.8 GB** |
| `Exe-Core-Dynamic-v1-Q4_K_S.gguf` | K-quant | 4.5 | 15.8 GB |
| `Exe-Core-Dynamic-v1-IQ4_XS.gguf` | I-quant | 4.25 | 15.4 GB |
| `Exe-Core-Dynamic-v1-Q3_K_L.gguf` | K-quant | 4.0 | 14.6 GB |
| `Exe-Core-Dynamic-v1-Q3_K_M.gguf` | K-quant | 3.9 | 13.5 GB |
| `Exe-Core-Dynamic-v1-Q2_K.gguf` | K-quant | 3.0 | 10.9 GB |

`Q4_K_M` is the recommended build: the usual sweet spot, and at 16.8 GB it fits a
24 GB card. `Q2_K` at 10.9 GB is the smallest here — usable, but expect it to slip
on the harder tool decisions.

## Prompt and sampling

Exe AI's own system prompt and the tool schemas ride along with every
request — the model is trained to read them, not to recite them. `temperature 0.1`
for tool work. Context up to 262k from the base.

## Base model and license

- **Base:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- **License:** Apache-2.0 (base and this derivative). You may use, modify, rebrand
  and redistribute; the origin of the base model must be named — it is, here.

## Training

A LoRA adapter (rank 16) on the full bf16 base, two epochs over 1241 examples
across 19 behaviour groups, with the prompt masked out of the loss so the model
learns the behaviour rather than the prompt. Held-out validation loss fell
monotonically to 0.0107 with no turn upward. The adapter was then fused back into
the bf16 base, and every build here comes from that fused model.

## Evaluation

On 72 held-out Exe AI cases at `temperature 0.1`, the untrained base solves
**57 / 72 (79%)** and Exe Core Dynamic v1 solves **67 / 72 (93%)** — the same
cases, the same model, the same settings, with the training as the only
difference. **No behaviour group went backwards.**

The largest gains: file tools 0/4 → 4/4, background runs 2/4 → 4/4.

**Honest limits:** two known weaknesses did not improve — naming the project's own
Python environment, and asking one short question instead of looking around first.
Both share a root the training reduced but did not remove: the model still prefers
to inspect before it acts. Measured on text cases only; the vision tower was frozen
and is untested here.

## Transparency

This is a fine-tuned derivative of an openly licensed base model, released with its
provenance, intended use, limits and evaluation stated above, in line with
transparency expectations for shared models (incl. the EU AI Act).
