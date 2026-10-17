---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
library_name: mlx
tags:
- mlx
- apple-silicon
- mtplx
- speculative-decoding
- multi-token-prediction
- qwen3_5
- merge
- task-arithmetic
---

# Qwopus3.8-27B · MTPLX 8-bit Quality

The near-lossless MTPLX build of **Qwopus3.8-27B** — a task-vector transplant that carries the Qwopus reasoning distillation onto the **Qwen3.8-27B** base. Qwopus3.8 keeps Qwen3.8's architecture, context length (262,144) and native MTP speculative head, and layers the Opus-style distilled reasoning trait on top: **materially fewer tokens for the same result**, and a stronger non-thinking (instruct) mode. Speed-focused sibling: [Qwopus3.8-27B-MTPLX-4bit-Speed](https://huggingface.co/nom666/Qwopus3.8-27B-MTPLX-4bit-Speed).

## How it was made

Qwen3.6-27B and Qwen3.8-27B share an identical `qwen3_5` skeleton (64 layers, 5120 hidden, 248K vocab, hybrid GDN + full attention, 1-layer MTP head). That makes cross-generation task arithmetic shape-exact:

```
Qwopus3.8 = Qwen3.8 + (Qwopus3.6-27B-v2 − Qwen3.6)
```

- All **1,199 tensors** matched and merged (fp32 math, bf16 out) — including the `mtp.*` speculative head and the vision tower; zero skipped, zero shape mismatches.
- Forged with MTPLX: body 8-bit (group 64, affine); **MTP head kept bf16**.
- The delta-merged MTP head verifies *above* the donor artifacts: acceptance **96.0% / 90.7% / 82.7%** at draft depths 1–3.

## Results (vs Qwen3.8-27B 8-bit, same harness, Apple M5 Max)

**Non-thinking mode:**

| | Qwopus3.8 8-bit (this build) | Qwen3.8 8-bit |
|---|---|---|
| 38-task instruct/code/math suite | **36/38** | 35/38 |
| Instruction-following subset | **3/5** | 2/5 |
| Hard-task suite (executed code + math + logic) | 8/13 @ **2,889 tok** | 8/13 @ 3,384 tok |
| Decode @ d0 (turbo) | 49.6 tok/s | ~49 tok/s |

Non-thinking Qwopus3.8 answers with ~15% fewer tokens at equal hard-task success, and holds instruction-following better than the base model does with thinking disabled. The 4-bit sibling scored **37/38 (instruction-following 4/5)** on the same non-thinking suite.

**Thinking mode — token efficiency:** at high reasoning effort this build matches the base on a 13-task hard suite (13/13 both) while spending **10,050 vs 18,455 reasoning tokens — 45% fewer**. The efficiency trait holds on every jointly-solved task of a harder adversarial set (e.g. 11.6K vs 18.0K tokens on a combinatorial coding task). Long-haul agentic tool-chains at 30K/60K/100K-token context ran clean (goal reached, zero repeated calls, zero drift).

## Usage

```bash
brew install youssofal/mtplx/mtplx
mtplx pull nom666/Qwopus3.8-27B-MTPLX-8bit-Quality
mtplx quickstart --model nom666/Qwopus3.8-27B-MTPLX-8bit-Quality \
  --reasoning off --profile turbo \
  --temperature 0.3 --top-p 0.9 --top-k 40
```

For thinking mode, use `--reasoning on --reasoning-effort low` (or `medium`) with a generous `max_tokens`. **Do not greedy-decode in thinking mode** (a known Qwen-family failure pattern; temperature 0 makes reasoning termination deterministic-worst-case).

## Notes

- Recommended for **instruct-style serving and agentic/coding loops with reasoning off**, or thinking-mode use where reasoning-token economy matters.
- With thinking enabled, Qwopus3.8 solves hard tasks with ~45% fewer reasoning tokens, though base Qwen3.8 remains the stronger choice for heavy thinking-mode workloads — it is more consistent at concluding very long reasoning inside tight token budgets (Qwopus3.8 can occasionally deliberate past a tight budget on some prompts) and keeps an edge on the hardest reasoning tasks.
- This is a weight-space merge, not a fine-tune: no gradient training was performed on Qwen3.8.
- All credit for the Qwopus distillation to **Jackrong** ([Qwopus3.6-27B-v2](https://huggingface.co/Jackrong/Qwopus3.6-27B-v2)); base model **Qwen3.8-27B** by the Qwen team; MTPLX runtime by **Youssofal**.
