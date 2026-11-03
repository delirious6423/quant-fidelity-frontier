---
license: mit
base_model:
  - ornith-ai/Ornith-1.5-35B-A3B
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - weight-space-programming
  - coding-agent
  - qwen3_5_moe
  - mtp
---

# Cyber-Tiel-Coder-35B-A3B-LoopFix-GGUF

An experimental **Weight-Space Programmed (WSP)** variant of `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf` built by having an LLM agent inspect failure traces, extract contrastive late-layer directions, and directly edit **64 `Q8_0` columns** in two late shared-expert MLP down-projection tensors (`blk.26` and `blk.34`) inside the quantized GGUF—with **no fine-tuning, no LoRA, and no gradients**.

Inspired by MIT CSAIL's [**Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights**](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi et al., 2026 — specifically Experiment 8 on removing reasoning loops in Qwen3.5).

---

## What Problem Does This Fix?

When running `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL` as a local coding agent (in **T3 Code** + **OMP / `oh-my-pi`** and local `codex` subagents), four failure modes repeatedly stalled multi-step runs:

1. **Going in circles inside `<think>`**: After reading 2–3 files, the model would burn 4,000+ reasoning tokens second-guessing a conclusion it had already proved in paragraph 2.
2. **Repeating rejected tool calls**: When a hashline edit failed, it would repeat the exact same broken syntax across 4–6 consecutive turns instead of adapting to the tool error message.
3. **Asking instead of acting**: Stopping to ask clarifying questions on unambiguous cleanup tasks instead of taking reasonable, conservative action.
4. **Subagent JSON-counting stalls**: Spending 4+ minutes (`6,000+` tokens) manually tallying raw JSON task arrays inside `<think>` during codebase scouting instead of yielding file paths and CLI entrypoints.

---

## How the Edit Works (100% Harness-Agnostic GGUF)

Rather than overfitting the GGUF to a single agent harness, the model intervention is strictly **harness-agnostic** (while optional `omp`-specific rules are provided separately in [`omp-harness/`](./omp-harness)):

### 1. Prefix-Aligned Suffix-Mean Direction Extraction ([`reproduce/cvector-generator-v2.cpp`](./reproduce/cvector-generator-v2.cpp))
- Standard activation-steering tools right-pad shorter prompts in a contrastive pair with `" "` tokens before mean-pooling, diluting the behavioral signal when looping prompts are longer than decisive prompts.
- `cvector-generator-v2.cpp` computes the exact token-level `common_prefix_len` per pair and averages hidden states strictly over each prompt's unpadded suffix tokens ($\mu_{\text{pos}} - \mu_{\text{neg}}$) across paired scenarios ([`reproduce/contrast-looping-v3.txt`](./reproduce/contrast-looping-v3.txt) vs. [`reproduce/contrast-productive-v3.txt`](./reproduce/contrast-productive-v3.txt)).

### 2. Rank-2 Late-Layer Shared-Expert Projection ([`reproduce/apply_loopfix.py`](./reproduce/apply_loopfix.py))
- Projects out the orthogonal basis $\{\hat{u}_1, \hat{u}_2^{\perp}\}$ ($\alpha_1 = 0.30, \alpha_2 = 0.15$) from the top-2 `Q8_0` column blocks (64 columns out of 512) of two late shared-expert down-projections:
  - **`blk.26.ffn_down_shexp.weight`**: blocks `[15, 4]` (columns `128..159, 480..511`), relative $L_2 = 0.641\%$ (`49,809` bytes changed).
  - **`blk.34.ffn_down_shexp.weight`**: blocks `[2, 12]` (columns `64..95, 384..415`), relative $L_2 = 0.582\%$ (`39,321` bytes changed).
- Leaving early/mid layers (`blk.15`) untouched preserves exact line-number and syntax binding while cutting late-layer `<think>` over-deliberation.
- Only those 4 `Q8_0` column blocks are re-encoded; every other tensor and quantization block in the 22 GB GGUF is bit-for-bit identical to the source checkpoint.

### 3. In-Place GGUF Chat-Template Fix ([`reproduce/patched-chat-template.jinja`](./reproduce/patched-chat-template.jinja))
- `llama-server` automatically injects `preserve_reasoning = true` whenever a template references `preserve_reasoning`, causing stale `<think>` blocks from prior turns to accumulate in context.
- The patched template defaults `_preserve_thinking = false`, bounds intra-turn `<think>` retention in multi-step tool loops, and replaces the upstream template's `"ask a sharp question, don't guess"` / `"keep caveats"` clauses with concise, harness-agnostic act-over-ask and raw-output discipline—padded in Jinja comments to the exact same UTF-8 byte length (`30,529` bytes).

---

## Benchmark Results (27-Case Suite)

Evaluated on `llama-server` (`--reasoning-effort low --reasoning-budget 4096 --temp 0.6 --top-p 0.95 --top-k 20`, `seed=42`) across:
- [`reproduce/suite.json`](./reproduce/suite.json) (12 cases: recovery from repeated errors, git worktree failure recovery, state ledgers, and 6 Python algorithm tasks)
- [`reproduce/holdout.json`](./reproduce/holdout.json) (8 unseen coding & agentic holdout tasks)
- `replay.json` & `session_probes.json` (7 multi-turn session replays from real failed coding/scouting runs)

| Configuration | All 27 Cases | General Non-OMP Cases (19) | `suite.json` (12) | `holdout.json` (8) | `replay.json` (3) | `session_probes.json` (4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** (unmodified GGUF) | 16/23 (10,997 tok) | 16/19 (84.2%) | 10/12 (6,702 tok) | 6/8 (2,085 tok) | 0/3 (2,210 tok) | — |
| **`v1` (`alpha03-recovery`)** | 20/23 (8,120 tok) | 18/19 (94.7%) | 10/12 (5,929 tok) | **8/8** (1,521 tok) | 2/3 (670 tok) | — |
| **`v3-agnostic`** (GGUF only, 0 harness rules) | 21/27 (11,151 tok) | **19/19 (100%)** | 11/12 (5,135 tok) | **8/8 (1,341 tok)** | 1/3 (867 tok) | 1/4 (3,808 tok) |
| **`v3-final`** (GGUF + [`omp-harness/`](./omp-harness) rules) | **25/27 (12,945 tok)** | **19/19 (100%)** | 11/12 (5,452 tok) | **8/8 (1,358 tok)** | 2/3 (414 tok) | **4/4 (5,721 tok)** |

---

## Quick Start

### Option A: Download the Ready-to-Run GGUF
Download `Cyber-Tiel-Coder-35B-A3B-LoopFix-MTP-UD-Q4_K_XL.gguf` and serve with `llama-server`:
```bash
llama-server \
  --model Cyber-Tiel-Coder-35B-A3B-LoopFix-MTP-UD-Q4_K_XL.gguf \
  --ctx-size 131072 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --reasoning-effort low --reasoning-budget 4096 \
  --temp 0.6 --top-p 0.95 --top-k 20
```

### Option B: Apply the Patch in 5 Seconds to Your Existing GGUF
If you already have `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf` locally, you only need the ~1 MB [`reproduce/`](./reproduce) folder (`pip install gguf numpy`):
```bash
cp --reflink=auto Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf Cyber-Tiel-Coder-35B-A3B-LoopFix-MTP-UD-Q4_K_XL.gguf
python3 reproduce/apply_loopfix.py \
  --source Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf \
  --target Cyber-Tiel-Coder-35B-A3B-LoopFix-MTP-UD-Q4_K_XL.gguf \
  --layers1 26,34 --layers2 26,34 --alpha1 0.30 --alpha2 0.15 --patch-template
```

---

## Credits & References
- Base model: [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (MIT License) & `Cyber-Tiel-Coder-35B-A3B`.
- Methodology inspired by: Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, [*"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, November 2026).
