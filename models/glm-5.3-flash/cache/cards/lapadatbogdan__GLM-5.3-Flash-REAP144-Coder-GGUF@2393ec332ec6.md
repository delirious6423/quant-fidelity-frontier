---
license: mit
base_model: zai-org/GLM-5.3-Flash
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - moe
  - expert-pruning
  - reap
  - code
  - agentic
  - llama.cpp
---

# GLM-5.3-Flash-REAP144-Coder-GGUF

> **Known issue (15 Oct 2026):** this release was calibrated on Python agentic coding sessions only. Outside that
> domain it loses general reasoning: it can get simple arithmetic wrong (for example 17 x 23) and often loops on other
> programming languages and on general questions, at every quant including Q8. A new release, calibrated on a mix of
> agentic sessions, code in 21 languages and math, is being validated and will replace these files. Until then, use
> this model for Python agentic coding only.

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (321B MoE) with **144 of 288 experts per layer**, chosen
on real agentic coding sessions, quantized with an importance matrix from the same sessions. Built to run agentic
coding on a home PC. **321B-A18B becomes 161B-A17B.**

![Solved agentic coding tasks per variant](https://raw.githubusercontent.com/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder/main/assets/results.svg)

Recipe, masks, patches and evaluation harness: [github.com/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder](https://github.com/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder)

## Files

| Quant | Size | Status | Fits (GPU VRAM + RAM) | Solved (same 48 attempts) | Turns vs full |
|---|---|---|---|---|---|
| Q4_K_M | 97.6 GB (3 parts) | stable | 24 + 96, 32 + 96, 128 GB unified | 48/48 | x1.58 |
| Q3_K | 71.4 GB (2 parts) | stable | 24 + 64, 32 + 64 | 48/48 | x1.80 |
| IQ2_S | 54.8 GB (2 parts) | stable | 32 + 32 at the limit, 24 + 48+ | 45/48 | x1.83 |
| IQ2_XXS | 45.3 GB | **experimental** | 24 + 32 | 39/48 | x2.77 |
| *full GLM-5.3-Flash, Q8* | *341 GB* | | | *48/48* | *x1.00* |
| MTP draft block (optional) | 2.6 GB | for 2+ GPUs with Maya | see below | | |

Tasks: 30 real Python repository tasks (ansible, openlibrary, qutebrowser), 2 attempts each, tests must pass,
clean room, 200 turns, 384K context. All rows on the same 48 attempts (4 tasks left out: 3 no model solves, 1 flaky; 4 attempts that hit our 12-hour limit per attempt on some model are left out for all). "Turns vs full" is the extra work (and time) relative to the full model on the
same solved attempts. Numbers as of 13 Oct 2026.

**IQ2_XXS is experimental.** It is the only version that fits 24 GB VRAM + 32 GB RAM, and it solves most tasks. Its
weak spot is how it fails: it never edits, it loops (same command over and over) or keeps exploring until the
200-turn limit. It also needs about 2.8x the turns. An anti-loop guard in the agent
did not help (27/37 vs 30/37 on the same attempts). If you have 24 + 48 GB or more, use IQ2_S. An improved 2 bit
version, trained on short successful sessions, is in progress.

Measured speed, IQ2_XXS with [Maya](https://github.com/mw00/project-maya) on RX 7900 XTX 24 GB + 32 GB DDR4-2133,
i7-7700K, PCIe 3.0: **~16 tokens/s generation, ~950 tokens/s prompt processing**.

Measured by a community tester: RTX 4090 (24 GB) + 64 GB DDR5-4800, i9-14900K, PCIe 4.0, Windows 11, Maya v1.1.9
after `--calibrate`, context 32K:

| File | Generation | Prompt processing (2K / 8K tokens) |
|---|---|---|
| IQ2_XXS | **35.9 tokens/s** | 1441 / 2564 tokens/s |
| IQ2_S | **29.9 tokens/s** | 1092 / 2382 tokens/s |
| Q3_K | **20.4 tokens/s** | 792 / 2184 tokens/s |

Calibration added 16% (IQ2_XXS) to 50% (Q3_K) generation speed on that PC.

## Perplexity

Perplexity on ~22K tokens of unseen code (11 chunks of 2048), llama.cpp `llama-perplexity`, same file for all:

| Model | PPL | vs full |
|---|---|---|
| GLM-5.3-Flash, full (Q8) | 1.681 | - |
| REAP144 Q8 | 2.053 | +22.1% |
| REAP144 Q4_K_M | 2.062 | +22.7% |
| REAP144 Q3_K | 2.091 | +24.4% |
| REAP144 IQ2_S | 2.174 | +29.3% |
| REAP144 IQ2_XXS | 2.303 | +37.0% |

We choose variants by results on agentic tasks; perplexity is shown for reference.

## Run

Split files load directly; point the runtime at the first part.

```bash
pip install -U huggingface_hub
hf download lapadatbogdan/GLM-5.3-Flash-REAP144-Coder-GGUF --include "*IQ2_S*" --local-dir glm53-iq2s
```

- **Maya** (made for GLM-5.3-Flash, puts the hottest experts on the GPU and the rest in RAM):
  `git clone https://github.com/mw00/project-maya && cd project-maya && ./maya.sh --gguf-dir ../glm53-iq2s --no-vision`
  (add `--backend hip` on AMD; we measured with `--backend hip --gpu 0 --context 32768`)
- **llama.cpp** (tested at `79e2e74`): `llama-server -m glm53-iq2s/GLM-5.3-Flash-REAP144-Coder-IQ2_S-00001-of-00002.gguf --jinja -c 131072`
  with expert tensors offloaded to CPU as needed (`--n-cpu-moe`).

### MTP draft block (2+ GPUs, Maya)

`GLM-5.3-Flash-REAP144-Coder-MTP.gguf` (2.6 GB) is GLM-5.3-Flash's own multi-token-prediction block (all 288 draft
experts, Q2_K; the rest Q8_0), taken from the full model. Maya uses it for speculative decoding on two or more GPUs:
one card drafts the next token while the others finish the current one. Output is unchanged; only speed changes.

```bash
hf download lapadatbogdan/GLM-5.3-Flash-REAP144-Coder-GGUF GLM-5.3-Flash-REAP144-Coder-MTP.gguf --local-dir mtp
./maya.sh --gguf-dir ../glm53-iq2s --no-vision --gpus 0,1 --env STRATA_GLM_MTP_GGUF=$PWD/../mtp/GLM-5.3-Flash-REAP144-Coder-MTP.gguf
```

- **Measured:** the block's draft matches this model's next token **~70%** of the time (568 positions of a real
  agentic session, IQ2_XXS trunk; Q2_K to Q6_K draft experts all gave 68-71%).
- **Not measured by us:** the speedup. Maya reports, for the full model, decode 14.1 -> 21.0 tok/s on 4x Tesla T4
  from drafting (+49%) and ~76% accepted drafts on two AMD cards; with ~70% we expect somewhat less, and the gain
  depends a lot on the cards. If you run it on 2+ GPUs, please share your tok/s.
- **One GPU:** Maya does not draft on a single GPU yet, so this file does nothing there.

## How it was made

1. Agentic coding sessions written by the full model, on tasks outside the evaluation set.
2. REAP-style score per expert, summed over the sessions: `sum(router weight x ||expert output||)`.
3. Keep the top 144 experts per layer. Held-out KL to the full model: 0.73 (frequency mask: 0.94).
4. imatrix on the same sessions, then Q4_K_M / Q3_K / IQ2_S / IQ2_XXS (KLD to Q8: 0.051 / 0.084 / 0.147 / 0.208).

At the same 144 experts, a frequency mask needed x2.21 turns and a public REAP mask calibrated on seven domains x2.29;
the code-scored mask needed x1.21 (Q8). Which experts you keep matters more than how many.

## What did not work

120 and 96 experts, a frequency mask, a generic REAP 50% mask, an optimization-based expert search (RCO),
LoRA healing and an anti-loop agent for IQ2_XXS were all tried and measured; see the table in the GitHub repo so you do not have to repeat them.

## Limits

Python only, three repositories, 2 attempts per task: differences of 1-3 tasks are noise. Pruned models lose
knowledge outside the calibration domain; this one is for coding. Text only: no vision projector (mmproj) is included.

## Acknowledgements

- REAP (Lasby et al., 2025) for the expert-saliency score our mask builds on.
- This project was built with the help of Claude (Anthropic) as a coding and research assistant.
  The decisions are the author's.

## License and credits

MIT, like the base model. Base model by Z.ai. REAP: Lasby et al., 2025. Runtimes: llama.cpp, Maya.

Contact: lapadat.bogdanv@gmail.com or [LinkedIn](https://www.linkedin.com/in/bogdan-lapadat-57baa08/)
