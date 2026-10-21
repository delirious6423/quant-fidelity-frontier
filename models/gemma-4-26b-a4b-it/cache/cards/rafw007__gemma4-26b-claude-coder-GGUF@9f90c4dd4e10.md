---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- gemma4
- claude-code
- codex
- opencode
- coding
- agent
- tool-calling
- ollama
language:
- en
- pl
---

<!-- meta-description: Gemma 4 26B coding agent for Claude Code / Codex / opencode, 64K+ context, native tool-calling, Q5_K_M GGUF, runs fully local on 32GB Apple Silicon. -->

# Gemma 4 26B Claude Coder — local coding agent

A custom model built on **Gemma 4 26B** (dense, ~25.8B params), tuned to act as an
**autonomous coding and administration agent**. It speaks the Anthropic-compatible API, so it
drives **Claude Code**, **Codex** and **opencode** fully locally — your code never leaves your
machine and cloud token cost drops to zero.

This is the **32 GB-class** big sibling of the Gemma 4 Claude Coder family (E2B / E4B). It ships on
a **Q5_K_M** GGUF quantization, deliberately chosen over Q4_K_M: the smaller Q4_K_M build injected
token corruption into long code generations (broken tags, glued digit-letter tokens), and Q5_K_M
fixes it — long files come out clean. The system prompt focuses on real work inside a codebase:
use tools instead of guessing, write files instead of pasting, ground every answer in real tool
output (never fabricate results), stay in one language, and always finish the file you start.
No-think mode is wired into the system prompt for fast, direct answers.

## Models in the family

| Model | Base | Context | Purpose |
|---|---|---|---|
| **gemma4-26b-claude-coder** | Gemma 4 26B (dense ~25.8B, Q5_K_M) | 64K (native 256K) | Strongest member — heavier reasoning and clean long-code generation on 32 GB hardware. |
| gemma4-e4b-claude-coder | Gemma 4 E4B (eff. 4B / 8B w/ embeddings) | 64K | Stronger 16 GB coder — reasoning and tool use on larger tasks. |
| gemma4-e2b-claude-coder | Gemma 4 E2B (eff. 2B / 5.1B w/ embeddings) | 64K | Fast everyday 16 GB coder — edits, autocomplete, short agent loops. |

## What it's for

- Driving **Claude Code / Codex / opencode** locally (`ollama launch claude --model rafw007/gemma4-26b-claude-coder`).
- Agentic code writing and editing with native **function calling / tool use**.
- Administration and devops tasks on a server — real `nmap`, `df`, `du` with no hallucinated output.
- Full privacy and offline operation — no code sent to the cloud.

## Measured behavior (June 2026 tests)

- **Tool-calling without hallucination** — real `message.tool_calls`, and admin tasks (`df`/`du`,
  full `/24` nmap scans with host tables) report the *actual* output rather than inventing it.
- **Clean long code (the headline fix)** — a full pygame Tetris generated complete, runnable and
  syntactically valid (200+ lines), with **zero corruption signatures** and zero language drift on
  a task that broke under the Q4_K_M build.
- **Guardrails intact** — this is the **non-abliterated** base, so it refuses to write malware.
- **No-think holds on the direct path** — empty thinking field, content is clean.

## Context

- **64K tokens** configured — matching Claude Code's recommendation (64K minimum).
- Base Gemma 4 26B natively supports **256K**, and on 32 GB Apple Silicon the engine serves the
  full native window **100% on GPU** (sliding-window attention keeps the KV cache small), so
  context never becomes the bottleneck.

## Test hardware

The model was built and tested on:

- **Mac Studio M2, 32 GB-class** — Ollama 0.30, GPU (Metal) inference
- **Mac Mini M4, 32 GB RAM, macOS** — Ollama 0.30, GPU (Metal) inference (32 GB-class target)
- Quantization: **Q5_K_M** (~21 GB) and **Q6_K** (~23 GB) GGUF builds available

### Measured performance

| Placement | Hardware | Speed | Tool calling |
|---|---|---|---|
| 100% GPU, native ctx, CONTEXT 65536 | Mac Studio M2 | ~52-56 tok/s | native, real `message.tool_calls` |

The model loads **entirely on the GPU with no CPU spill** (verified via `ollama ps`: 100% GPU,
CONTEXT 65536). The only real cost is a one-time cold load of the ~21 GB weights, not a per-turn
cost; warm generation runs ~52-56 tok/s on the Studio. The Mac Mini M4 (32 GB) is the same 32 GB
target class — bounded by memory bandwidth rather than the model.

## No-think mode

The whole Gemma 4 family has thinking baked into the weights. The system prompt ships with
`/nothink` + an anti-reasoning instruction, which works on the direct API path and under
opencode/codex. Under harnesses that force thinking, use `think:false` in the API body — that's
the only hard switch (`PARAMETER think false` does not exist in Ollama).

## Note on long code generation

Q5_K_M removes the bulk corruption seen on the smaller Q4_K_M build — in testing, long single-pass
generations came out clean (zero `.`->`-` glitches, zero language drift). If you generate files for
production, a quick corruption scan before use is still good practice, but the Q5_K_M build tested
clean on the long-code task that previously failed.

## How it was made

Designed, built and tested with the help of **Claude Opus 4.8** — the best coding model in the
world. Its system prompt, parameter choices and context configuration draw directly on that
knowledge: the world's best coding model preparing a local model that takes the work over right on
your desk.

## Available files

| File | Quant | Size | Notes |
|---|---|---|---|
| `gemma4-26b-claude-coder-Q5_K_M.gguf` | Q5_K_M | ~21 GB | Recommended balance of quality/size; fits 32 GB with full 64K ctx. |
| `gemma4-26b-claude-coder-Q6_K.gguf` | Q6_K | ~23 GB | Closer to the original weights; slightly larger/slower. |

Both are derived from the same `google/gemma-4-26B-A4B-it` base and carry the identical Claude Coder
system prompt and parameters (see `Modelfile`).

## License

Apache 2.0 (inherited from the base Gemma 4).
