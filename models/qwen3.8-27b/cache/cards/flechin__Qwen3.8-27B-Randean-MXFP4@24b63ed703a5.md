---
base_model:
  - Qwen/Qwen3.8-27B
  - amd/Qwen3.8-27B-Quark-AWQ-MXFP4
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
thumbnail: https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/assets/randean_logo.jpg
tags:
  - qwen3_5
  - mxfp4
  - awq
  - quark
  - orpo
  - qlora
  - qat
  - reasoning
  - epistemology
  - reality-aligned
  - de-hedged
  - anti-sycophancy
  - causal-grounding
  - agentic
  - rocm
model-index:
  - name: Qwen3.8-27B-Randean-MXFP4
    results:
      - task:
          type: text-generation
          name: Terminal-Bench 2.1 (89 Docker CLI Tasks)
        dataset:
          name: Terminal-Bench 2.1
          type: terminal-bench/terminal-bench-2-1
        metrics:
          - type: resolved_rate
            value: 56.2
            name: Tasks Resolved (%)
      - task:
          type: text-generation
          name: LiveCodeBench v6 (175 Problems)
        dataset:
          name: livecodebench/code_generation_lite
          type: livecodebench/code_generation_lite
        metrics:
          - type: pass@1
            value: 72.00
            name: pass@1 (%)
          - type: hard_pass@1
            value: 48.75
            name: Hard Problems pass@1 (%)
      - task:
          type: text-generation
          name: BullshitBench v2 (100 Traps)
        dataset:
          name: BullshitBench v2
          type: bullshit-benchmark/questions.v2
        metrics:
          - type: mean_score
            value: 1.9300
            name: Mean Score (out of 2.0, Global Rank #1 / 171)
          - type: clear_pushback_rate
            value: 96.0
            name: Green Pushback Rate (%)
      - task:
          type: text-generation
          name: Randean Precision Tools Hard (24 Agentic Traps, T=0.2)
        dataset:
          name: Randean Precision Tool Suite
          type: randean/precision_tools_hard
        metrics:
          - type: score_pct
            value: 91.67
            name: Hard Tool Precision (%)
      - task:
          type: text-generation
          name: Randean Active Grounding Suite (H5, 45 Prompts)
        dataset:
          name: Randean H5 Active Grounding
          type: randean/h5_active_grounding
        metrics:
          - type: accuracy
            value: 82.2
            name: 3-Way Active Grounding Accuracy (%)
      - task:
          type: text-generation
          name: GPQA-Diamond (198 PhD STEM Questions)
        dataset:
          name: Idavidrein/gpqa
          type: Idavidrein/gpqa
        metrics:
          - type: accuracy
            value: 78.79
            name: Accuracy (%)
      - task:
          type: text-generation
          name: AIME 2024-2025 (60 Olympiad Problems)
        dataset:
          name: AIME 2024-2025
          type: math-ai/aime24-25
        metrics:
          - type: accuracy
            value: 78.33
            name: Overall AIME 2024-2025 Accuracy (%)
      - task:
          type: text-generation
          name: Google IFEval Full (541 Prompts)
        dataset:
          name: google/IFEval
          type: google/IFEval
        metrics:
          - type: prompt_level_strict_acc
            value: 86.88
            name: Prompt-Level Strict Accuracy (%)
          - type: inst_level_strict_acc
            value: 89.93
            name: Instruction-Level Strict Accuracy (%)
      - task:
          type: text-generation
          name: Anthropic SycophancyEval (400 Prompts)
        dataset:
          name: Anthropic/sycophancy-eval
          type: Anthropic/sycophancy-eval
        metrics:
          - type: post_challenge_accuracy
            value: 93.00
            name: 2-Turn Post-Challenge Accuracy (%)
      - task:
          type: text-generation
          name: OpenAI HumanEval
        dataset:
          name: openai_humaneval
          type: openai/openai_humaneval
        metrics:
          - type: pass@1
            value: 97.5
            name: pass@1 (%)
      - task:
          type: text-generation
          name: OpenAI GSM8K
        dataset:
          name: gsm8k
          type: openai/gsm8k
        metrics:
          - type: exact_match
            value: 97.0
            name: Exact Match (%)
      - task:
          type: text-generation
          name: Randean 84-Question Epistemological Grounding Suite
        dataset:
          name: Randean Epistemology Bench
          type: randean/epistemology_bench
        metrics:
          - type: score_pct
            value: 83.33
            name: Overall Score (%)
          - type: short_conversational_pct
            value: 80.95
            name: Short Conversational Score (%)
          - type: adversarial_score_pct
            value: 85.71
            name: Long Adversarial Trap Score (%)
---

<p align="center">
  <img src="https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/assets/randean_logo.jpg" alt="Randean Logo" width="680" style="border-radius: 12px;" />
</p>

# Qwen3.8-27B-Randean-MXFP4 (`v3.0`)

**`Qwen3.8-27B-Randean-MXFP4` (`v3.0`)** is a **19 GB 4-bit (`MXFP4`)** version of `Qwen3.8-27B` tuned to give **direct, reality-grounded answers** instead of hedging, playing along with broken premises, or padding recommendations with buzzwords — while running on a **single 32 GB GPU** and **beating stock `Qwen3.8-27B` on hard coding and real-world terminal engineering**.

- **GGUF Quantizations (`llama.cpp` / Ollama / LM Studio + Vision `mmproj`)**: [`flechin/Qwen3.8-27B-Randean-GGUF`](https://huggingface.co/flechin/Qwen3.8-27B-Randean-GGUF) (`Q4_K_M` & `Q6_K` imatrix, `Q8_0`)
- **Interactive Demo Space (ZeroGPU)**: [`https://huggingface.co/spaces/flechin/randean`](https://huggingface.co/spaces/flechin/randean)
- **Research Paper (10-Page PDF)**: [`paper/randean_paper.pdf`](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/paper/randean_paper.pdf)

---

## Why `v3.0` Beats `v2.0` and Competing Models

Compared directly against **`Randean v2.0`**, **Stock `Qwen3.8-27B`**, **`Qwen3.8-27B-Pi` (8-bit FP8)**, **`Swift-1.5-27B` (8-bit FP8)**, **`Inkling-2-Mini` (26B MoE)**, and **`Qwen3.8-Flash-Next` (177B MoE)**:

- **Stronger hard coding & real terminal engineering**:
  - **Terminal-Bench 2.1 (`89` Docker CLI tasks)**: **`56.2%` (`50/89`)** — our highest score yet (`+2.3%` above both Stock and `v1.0` on TB 2.0) while thinking **`37%` less per turn** and finishing **`151s` faster per task**.
  - **LiveCodeBench v6 (`175` problems)**: **`72.0%`** overall (**`+3.4%`** over `v2.0`, **`+2.9%`** over Stock) and **`48.8%` on Hard problems** (**`+7.5%`** over `v2.0`, **`+10.0%`** over Stock) with fewer timeouts and `27%` shorter reasoning traces.
  - **GPQA-Diamond (`198` PhD STEM)**: **`78.8%`** — up **`+2.5%`** over `v2.0` (`76.3%`) while using `28%` less reasoning.
- **Far smarter tool use**:
  - **Precision Tools Hard (`24` tricky agent scenarios)**: **`91.7%`** — up **`+8.4%`** over `v2.0` (`83.3%`), **`+27.1%`** over Stock `Qwen3.8-27B` (`64.6%`) and `Inkling-2-Mini` (`64.6%`), and **`+33.4%`** over `177B` `Qwen3.8-Flash-Next` (`58.3%`).
  - **Zero wasted tool calls on impossible requests**: Rejects mathematical/physical impossibilities directly without spinning up tools **`80.0%`** of the time — **`2×` `v2.0` (`40.0%`)**, **`4×` `Pi` & `Swift-1.5` (`20.0%`)**, and **`12×` Stock (`6.7%`)**.
- **Top-ranked nonsense & sycophancy resistance**:
  - **BullshitBench v2 (`100` nonsense traps)**: **`1.93 / 2.0`** (`96%` clear pushback, Rank **`#1 / 171`**) — well ahead of `Pi` (`1.80`), `Swift-1.5` (`1.78`), Stock (`1.73`), and `Inkling-2-Mini` (`1.50`).
  - **SycophancyEval (`200` 2-turn challenges)**: **`93.0%`** accuracy after pushback (`+1.0%` over `v2.0` and Stock) with **`0.0%` false apologies**.
  - **Clean, grounded recommendations**: Eliminates ungrounded checklist padding (`85.4%` in direct blind 4-way auditing, `0` status-quo holdovers, `0` ideological jargon).
- **Half the GPUs, twice the speed**: Fits in **`19 GB`** on **one 32 GB GPU** (unlike `31–35 GB` 8-bit `Pi`/`Swift-1.5` which need two 32 GB GPUs) and pairs cleanly with **`DFlash2`** speculative decoding (`7.2` mean accepted tokens per step, **`258 tok/s`** burst on dual R9700).

---

## Quick Start (`vLLM`, Single 32 GB GPU)

### Standard Serving

```bash
vllm serve flechin/Qwen3.8-27B-Randean-MXFP4 \
  --served-model-name Qwen3.8-27B-Randean-MXFP4 \
  --trust-remote-code \
  --quantization quark \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --max-model-len 32768
```

### Fast Serving with `DFlash2` Speculative Decoding

```bash
vllm serve flechin/Qwen3.8-27B-Randean-MXFP4 \
  --served-model-name Qwen3.8-27B-Randean-MXFP4 \
  --trust-remote-code \
  --quantization quark \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"dflash","model":"syvai/Qwen3.8-27B-DFlash2-W4A16","num_speculative_tokens":7}' \
  --max-model-len 32768
```

---

## Benchmarks: `v3.0` vs. `v2.0` and Stock `Qwen3.8-27B`

### 1. Coding, Terminal Agents, STEM & Anti-Sycophancy (`T = 0.2`)

| Benchmark | Stock `Qwen3.8-27B` | `Randean v1.0` | `Randean v2.0` | **`Randean v3.0`** | `v3.0` vs. `v2.0` / Stock |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Terminal-Bench (`89` Docker tasks)** ↑ | `53.9%` *(TB 2.0)* | `53.9%` *(TB 2.0)* | — | **`56.2%`** *(TB 2.1)* | **`+2.3%` all-time high** (`-37%` think/turn) |
| **LiveCodeBench v6 (`175` coding)** ↑ | `69.1%` | `66.3%` | `68.6%` | **`72.0%`** | **`+3.4%` vs `v2.0`**, **`+2.9%` vs Stock** |
| ↳ **LiveCodeBench Hard (`80`)** ↑ | `38.8%` | `38.8%` | `41.3%` | **`48.8%`** | **`+7.5%` vs `v2.0`**, **`+10.0%` vs Stock** |
| **GPQA-Diamond (`198` PhD STEM)** ↑ | **`80.8%`** | `76.3%` | `76.3%` | `78.8%` | **`+2.5%` vs `v2.0`** (`-28%` think, n.s. vs Stock) |
| **AIME 2024–2025 (`60` Olympiad)** ↑ | **`85.0%`** | `80.0%` | `80.0%` | `78.3%` | Within noise (`p = 0.29` vs Stock) |
| **Google IFEval (`541` strict prompt)** ↑ | **`89.1%`** | `88.9%` | `87.3%` | `86.9%` | Within noise (`p = 0.15`, `-24%` think) |
| **SycophancyEval (`200` post-challenge)** ↑ | `92.0%` | `91.5%` | `92.0%` | **`93.0%`** | **`+1.0%` vs `v2.0` & Stock**, `0.0%` false apologies |
| **HumanEval (`80`) / GSM8K (`100`)** ↑ | `96.2%` / **`98.0%`** | **`97.5%`** / `97.0%` | **`97.5%`** / `96.0%` | **`97.5%`** / `97.0%` | `+1.0%` GSM8K vs `v2.0`, tied #1 HumanEval |

*(Note: `SWE-bench Verified` 100-issue score of `68.0%` is from `v2.0`; `v3.0` has not been re-run on SWE-bench yet.)*

---

### 2. Tool Discipline, Bullshit Rejection & Grounding vs. Other Local Models

| Benchmark | Stock `Qwen3.8-27B` (`4-bit`) | `Qwen3.8-27B-Pi` (`8-bit FP8`) | `Swift-1.5-27B` (`8-bit FP8`) | `Inkling-2-Mini` (`26B MoE`) | `Randean v2.0` (`4-bit`) | **`Randean v3.0`** (`4-bit`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Precision Tools Hard (`24` traps)** ↑ | `64.6%` | — | — | `64.6%` | `83.3%` | **`91.7%`** |
| **Active Grounding Overall (`H5`, `45`)** ↑ | `60.0%` | `66.7%` | `62.2%` | `53.3%` | `71.1%` | **`82.2%`** |
| ↳ **Reject Impossibilities w/ 0 Tools** ↑ | `6.7%` | `20.0%` | `20.0%` | `0.0%` | `40.0%` | **`80.0%`** |
| ↳ **Offline Honesty (No Hallucination)** ↑ | `73.3%` | **`80.0%`** | `66.7%` | `60.0%` | `73.3%` | **`80.0%`** |
| **BullshitBench v2 (`100` traps)** ↑ | `1.73` (`82%` G) | `1.80` (`86%` G) | `1.78` (`85%` G) | `1.50` (`55%` G) | **`1.94`** (`96%` G) | `1.93` (`96%` G, **#1**) |
| **Epistemology & Deductive Spine (`84`)** ↑ | `57.1%` (`23` fb) | `60.1%` (`23` fb) | `58.3%` (`18` fb) | `61.9%` (`19` fb) | `82.7%` (`11` fb) | **`83.3%`** (**`9` fb**) |
| **Ideological / Philosopher Jargon** ↓ | `4 / 84` | `5 / 84` | `8 / 84` | **`0 / 84`** | **`0 / 84`** | **`0 / 84`** |
| **Minimum Hardware (`VRAM`)** ↓ | **1× 32 GB** | 2× 32 GB | 2× 32 GB | **1× 32 GB** | **1× 32 GB** | **1× 32 GB** |

*(177B `Qwen3.8-Flash-Next` IQ3_XXS also scored `58.3%` on Precision Tools Hard and `44.4%` on `H5`, compared with `v3.0`'s `91.7%` and `82.2%`. On the 24-scenario held-out Generative Grounding suite, `v2.0` beat Stock, `Pi`, and `Swift-1.5` in a blind Gemini-4-Carbon audit [`41.7%` vs `29.2%`/`35.4%`/`33.3%`, cutting floating recommendations nearly in half]; in a separate direct blind 4-way audit across `v3.0` candidates, `v3.0` scored `85.4%` with `18/24` clean zero-floating answers and `0` status-quo holdovers.)*

---

### 3. Speed & Footprint Advantage

Because `Randean` runs in **4-bit (`19 GB`)** and cuts out repetitive *"on the other hand..."* thinking loops without hurting accuracy, it finishes complete answers **more than `2×` faster** than 8-bit `Qwen3.8-27B` variants:

| Streaming Metric (`6` Multi-Domain Prompts, R9700) | `Qwen3.8-27B-Pi` (`FP8`, `2×GPU`) | `Swift-1.5-27B` (`FP8`, `2×GPU`) | **Randean** (`MXFP4`, `1×GPU`) | **Randean** (`MXFP4`, `2×GPU`) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Size** ↓ | `34.7 GB` | `30.9 GB` | **`19.0 GB`** | `19.0 GB` |
| **End-to-End Answer Time (`c=1`, DFlash2)** ↓ | `25.9 s` | `28.9 s` | `14.9 s` | **`12.4 s`** (`2.1×–2.3×` faster) |
| **Batch Wall Time (`c=6`, DFlash2)** ↓ | `56.4 s` | `51.3 s` | `31.7 s` | **`22.5 s`** (`2.3×–2.5×` faster) |

*(Measured in controlled `v2.0` streaming benchmarks; `v3.0` maintains the exact same `19 GB` 4-bit layout and verified `7.21` mean `DFlash2` draft acceptance length in production.)*

---

## How `v3.0` Was Trained (Short Version)

- **Clean start from Stock**: Instead of stacking another round on top of `v1.0`/`v2.0`, `v3.0` starts fresh from Stock `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` with a zero-contamination 400-example dataset audited against every evaluation suite.
- **Two focused stages**:
  1. **Foundation (`288` examples)**: Teaches direct first-principles reasoning, anti-hedging, and elimination of ungrounded recommendations without leaking jargon.
  2. **Tool & Grounding Surgery (`140` examples)**: Uses shared-reasoning pairs to teach exact tool discipline — probe real environments when needed, refuse impossible tasks without wasting tool calls, and keep strict XML/JSON schemas intact.
- **Quantization-aware training directly in 4-bit**: Weights stay synced with the exact 4-bit `MXFP4` serving format during training, so nothing is lost when exporting the final `19 GB` checkpoint.
- **Weight-space blending (`WiSE-FT`, `α = 0.65`)**: Blends Stage 1 and Stage 2 cleanly into a single standalone checkpoint with zero adapter overhead at runtime.

### Preserved Revisions
- **`v3.0` (`Arm B`, current local production)**: Clean start-from-Stock two-stage `8K` training — highest coding, terminal, tool, and active-grounding scores.
- **`v2.0-phase-c2`**: Prior published Hub release (`68.0%` SWE-bench, `1.94` BullshitBench v2).
- **`v1.0-phase2`**: Original closed-book research paper baseline (`48/89` TB 2.0, `1.95` BullshitBench v2).

---

## Citation

```bibtex
@misc{flecha2026randean,
  title   = {Epistemic De-Hedging Under 4-Bit Microscaling: Selective <think> Pruning and Native OCP MXFP4 Preference Alignment on a Single 32 GB GPU},
  author  = {Fernando L. Flecha},
  year    = {2026},
  url     = {https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4}
}
```
