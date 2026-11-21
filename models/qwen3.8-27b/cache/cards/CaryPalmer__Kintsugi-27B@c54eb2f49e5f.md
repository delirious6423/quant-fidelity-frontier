---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
- 2-bit
- long-context
---

# Kintsugi-27B v1

![Kintsugi-27B stat card](kintsugi-card.png)

Kintsugi-27B is Qwen3.8-27B at 2.47 bits per weight. It runs on one 12 GB NVIDIA card (RTX 20, 30 or 40 series) with the
full 262,144-token context. Free to use for anything, under Apache-2.0.

**Run it:** the Windows bundle (engine, launcher, saved-chat layer), setup steps, licences and full credits are at
**https://github.com/professorpalmer/Kintsugi-27B**. Download `Kintsugi-27B-12G.gguf` from this page into the bundle's
`models` folder. Issues and pull requests are welcome there.

**Coding harnesses:** tested with Claude Code, OpenAI Codex, Pi and OpenCode (5 of 6 tasks in each, no malformed tool
calls). Use bundle v1.0.2 or later; the settings are in `HARNESSES.md` in the GitHub repo.

| Test | Result |
|---|---|
| HumanEval (medium thinking) | 161 / 164 |
| AppWorld test_normal (agent tasks, 2 seeds) | 151 / 168 on each seed |
| Needles + exact code copies at 245k tokens | 24 / 24 and 20 / 20 |
| Fidelity to BF16 (public quant-challenge golden set) | top-1 90.9 %, mean KL 0.056, PPL ratio 1.033 |
| Decode on an RTX 4070 at 64k / 117k / 200k / 245k | 64 / 60 / 44 / 30 tokens per second |
| Restore a saved 245k-token conversation | 2.5 s |

The engine is a llama.cpp fork with EXL3 trellis kernels; the file needs that engine (it is in the bundle).

File: `Kintsugi-27B-12G.gguf`, 10,546,687,168 bytes, sha256 `0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74`.

Built on the work of the Qwen team, turboderp (exllamav3), the QTIP authors, Escha Labs, YourHighnessLA (L0xRE),
StillDeadcode (quant challenge), Ajay9o9, the llama.cpp / ggml contributors and many more: see CREDITS.md in the
GitHub repo.
