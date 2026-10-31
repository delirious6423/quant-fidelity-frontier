---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - qwen3.5
  - heretic
  - ara
  - writing
---

# Qwen3.8-27b-noslop-v2 (GGUF)

A **mild** "less slop" edit of Qwen3.8-27B, made with [Heretic](https://github.com/p-e-w/heretic) (Arbitrary-Rank Ablation) using the upstream `config.noslop.toml` recipe unchanged (trial 18 of 60). This is **not** a "no slop" model and it does not remove em dashes. It is published because, unlike our first release, it stays close to the original model in coding-agent runs.

Only one file is here for now: **Q2_K_XL** (9.83 GB). It is the same per-tensor layout as Unsloth's current UD-Q2_K_XL, built from the edited weights with our own importance matrix. This is the exact file that was tested below.

## What was measured

All numbers are for this Q2_K_XL file against a base Q2_K_XL built the same way (same layout, same importance matrix).

| Check | Original | This model |
|---|---|---|
| Replies with fiction slop words (100 story prompts, greedy) | 86 / 100 | 65 / 100 |
| First-token KL divergence vs original | 0 | 0.0075 |
| Coding-agent runs passed (pi and Claude Code, 30 tasks x 2 rounds = 120 runs) | 110 / 120 | 106 / 120 |
| Runs stopped by a loop or no-progress detector | 2 / 120 | 6 / 120 |
| Average time per run | 34 s | 38 s |
| Average tool calls per run | 7.9 | 9.8 |
| Final agent message contains an em/en dash | 100 / 120 | 92 / 120 |

Read this honestly:

- The slop reduction is small (86 to 65 of 100, on fiction word lists only).
- It still loops a bit more than the original (6 vs 2 flagged runs of 120) and passes slightly fewer tasks. It does **not** meet a strict "never loops more than the original" rule.
- Dashes in agent final messages barely change (92 vs 100 of 120). Do not expect dash-free output.
- Agent test details: sandboxed tasks with hidden tests, 600 s wall limit, 100 tool-call cap, loop detectors. The test is small and the original itself fails or loops occasionally, so differences of a few runs are within noise.
- Only the Q2_K_XL quantisation was tested. Other quantisations are not provided and were not evaluated.

If you need the strongest dash reduction in plain chat, our first release (`RefinedNeuro/Qwen3.8-27b-noslop-gguf`) does more, but it loops much more in coding agents (20 / 120 flagged runs in the same test). V2 trades effect for safety.

## Usage

```bash
llama-server -m Qwen3.8-27b-noslop-v2-Q2_K_XL.gguf --jinja -c 32768
```

Works with the same chat template and sampling settings as Qwen3.8-27B. The multi-token-prediction head is retained.

## Method

Heretic master with the unmodified upstream `config.noslop.toml` (KeywordRate over fiction slop words, first-token KL, ARA rank 50), 60 trials, TPE after 20 random starts, 4-bit search. Trial 18 was picked as the low-divergence candidate. The adapter was merged into the original bf16 weights and quantised. No tokens are banned and no output filtering is applied.

Related: other candidates from the same search were rejected after the agent test (for example trial 25, which scored well in the search but passed only 5 of 42 agent runs).
