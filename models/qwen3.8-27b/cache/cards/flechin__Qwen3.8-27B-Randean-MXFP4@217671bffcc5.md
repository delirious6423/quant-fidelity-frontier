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
            value: 58.4
            name: Tasks Resolved (%)
      - task:
          type: text-generation
          name: LiveCodeBench v6 (175 Problems)
        dataset:
          name: livecodebench/code_generation_lite
          type: livecodebench/code_generation_lite
        metrics:
          - type: pass@1
            value: 73.71
            name: pass@1 (%)
          - type: hard_pass@1
            value: 48.75
            name: Hard Problems pass@1 (%)
      - task:
          type: text-generation
          name: AIME 2024-2025 (60 Olympiad Problems)
        dataset:
          name: AIME 2024-2025
          type: math-ai/aime24-25
        metrics:
          - type: accuracy
            value: 85.00
            name: Overall AIME 2024-2025 Accuracy (%)
      - task:
          type: text-generation
          name: SWE-bench Verified (100 Stratified Issues)
        dataset:
          name: SWE-bench Verified
          type: princeton-nlp/SWE-bench_Verified
        metrics:
          - type: resolved_rate
            value: 66.0
            name: Issues Resolved (%)
      - task:
          type: text-generation
          name: BullshitBench v2 (100 Traps)
        dataset:
          name: BullshitBench v2
          type: bullshit-benchmark/questions.v2
        metrics:
          - type: mean_score
            value: 1.8700
            name: Mean Score (out of 2.0, Global Rank #3 / 170)
          - type: clear_pushback_rate
            value: 92.0
            name: Green Pushback Rate (%)
      - task:
          type: text-generation
          name: Randean Precision Tools Hard (24 Agentic Traps, T=0.2)
        dataset:
          name: Randean Precision Tool Suite
          type: randean/precision_tools_hard
        metrics:
          - type: score_pct
            value: 87.50
            name: Hard Tool Precision (%)
      - task:
          type: text-generation
          name: Randean Active Grounding Suite (H5 4-Way, 60 Prompts)
        dataset:
          name: Randean H5 Active Grounding
          type: randean/h5_active_grounding
        metrics:
          - type: accuracy
            value: 76.7
            name: 4-Way Active Grounding Accuracy (%)
          - type: unhinted_proactive_search_rate
            value: 80.0
            name: Unhinted Casual Long-Tail Proactive Search (%)
      - task:
          type: text-generation
          name: GPQA-Diamond (198 PhD STEM Questions)
        dataset:
          name: Idavidrein/gpqa
          type: Idavidrein/gpqa
        metrics:
          - type: accuracy
            value: 74.24
            name: Accuracy (%)
      - task:
          type: text-generation
          name: Google IFEval Full (541 Prompts)
        dataset:
          name: google/IFEval
          type: google/IFEval
        metrics:
          - type: prompt_level_strict_acc
            value: 86.69
            name: Prompt-Level Strict Accuracy (%)
          - type: prompt_level_loose_acc
            value: 89.83
            name: Prompt-Level Loose Accuracy (%)
          - type: inst_level_strict_acc
            value: 89.57
            name: Instruction-Level Strict Accuracy (%)
      - task:
          type: text-generation
          name: Anthropic SycophancyEval (400 Prompts)
        dataset:
          name: Anthropic/sycophancy-eval
          type: Anthropic/sycophancy-eval
        metrics:
          - type: post_challenge_accuracy
            value: 89.50
            name: 2-Turn Post-Challenge Accuracy (%)
          - type: false_nudge_endorsement_rate
            value: 4.00
            name: False-Nudge Endorsement Rate (%, lower is better)
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
            value: 96.0
            name: Exact Match (%)
      - task:
          type: text-generation
          name: Randean 84-Question Epistemological Grounding Suite
        dataset:
          name: Randean Epistemology Bench
          type: randean/epistemology_bench
        metrics:
          - type: score_pct
            value: 84.52
            name: Overall Score (%)
          - type: short_conversational_pct
            value: 76.19
            name: Short Conversational Score (%)
          - type: adversarial_score_pct
            value: 92.86
            name: Long Adversarial Trap Score (%)
---

<p align="center">
  <img src="https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/assets/randean_logo.jpg" alt="Randean Logo" width="680" style="border-radius: 12px;" />
</p>

# Qwen3.8-27B-Randean-MXFP4 (`v3.1b`)

**`Qwen3.8-27B-Randean-MXFP4` (`v3.1b`)** is a **19 GB 4-bit (`MXFP4`)** version of `Qwen3.8-27B` tuned to give **direct, reality-grounded answers** and **proactively verify uncertain facts using whatever tools are bound in context** — running on a **single 32 GB GPU** while setting new Randean highs on **Terminal-Bench 2.1 (`58.4%`)**, **LiveCodeBench v6 (`73.7%`)**, and **AIME 2024–2025 (`85.0%`)**.

- **GGUF Quantizations (`llama.cpp` / Ollama / LM Studio + Vision `mmproj`)**: [`flechin/Qwen3.8-27B-Randean-GGUF`](https://huggingface.co/flechin/Qwen3.8-27B-Randean-GGUF) (`Q4_K_M` & `Q6_K` imatrix, `Q8_0`)
- **Interactive Demo Space (ZeroGPU)**: [`https://huggingface.co/spaces/flechin/randean`](https://huggingface.co/spaces/flechin/randean)
- **Research Paper (10-Page PDF)**: [`paper/randean_paper.pdf`](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/paper/randean_paper.pdf)

---

## What's New in `v3.1b` (Epistemic-Inventory Release)

Built directly from `v3.0`'s locked Stage 1 foundation with a targeted **Epistemic-Inventory Multi-Tool Stage 2 surgery (`72` steps, `WiSE-FT α = 0.65`)**, `v3.1b` replaces passive *"would you like me to search?"* caveats with an active inventory of bound tools (`web_search`, `read_file`, `run_python`) — and lifts real-world coding, math, and terminal execution at the same time:

- **All-time high terminal engineering & hard coding**:
  - **Terminal-Bench 2.1 (`89` Docker CLI tasks)**: **`58.4%` (`52/89`)** — up **`+2.3%`** over `v3.0` (`56.2%`) and **`+4.5%`** over Stock (`53.9%` on TB 2.0), while thinking **`26%` less than `v3.0` (`38.5%` less than Stock)** and finishing **`182s–333s` faster per task**.
  - **LiveCodeBench v6 (`175` problems)**: **`73.7%`** overall (**`+1.7%`** over `v3.0`, **`+4.6%`** over Stock) — **`100%` Easy, `90.4%` Medium (`+5.8%` vs `v3.0`), `48.8%` Hard (`+10.0%` vs Stock)** with `18.1%` shorter reasoning traces.
  - **AIME 2024–2025 (`60` Olympiad problems)**: **`85.0%` (`51/60`)** — up **`+6.7%`** over `v3.0` (`78.3%`), **matching Stock on both 2024 (`86.7%`) and 2025 (`83.3%`)** while using **`28.4%` less thinking** (`-42.6%` on mutually solved problems).
- **Proactive epistemic verification across bound tools**:
  - **Unhinted casual long-tail search (`H5 Arm 4`, `15` zero-cue prompts)**: jumped from `v3.0`'s `60.0%` (`0/6` on casual fiction/lore) to **`80.0%`** with **`0.0%` passive offers**, **`0.0%` closed-world false dismissals (`Arm 1`)**, and **`0.0%` offline hallucinations (`Arm 3`)**.
  - **Open-ended PhD Chemistry search probe (`N = 14`, tools bound)**: nearly doubled proactive `web_search` rate over `v3.0` (`64.3%` vs `35.7%`, `+4 / -0` paired).
  - **Precision Tools Hard (`24` traps)**: **`87.5%`** at both `T = 0.2` and `T = 0.6` (**`93.8%` (`30/32`)** on all non-restraint safety categories: ambiguity, injection, unit/schema, and state recovery).
- **Stronger adversarial & false-nudge resistance**:
  - **SycophancyEval False-Nudge (`200` prompts)**: cut false-nudge endorsement **in half** from `8.5%` (`v3.0`) / `8.0%` (Stock) down to **`4.0%`** (`0.0%` false apologies, `-36.8%` thinking).
  - **Epistemology & Deductive Spine (`84` prompts)**: **`84.5%`** overall (**`92.9%` on Long Adversarial Traps**, up **`+7.2%`** over `v3.0`, with **`8`** consensus fallbacks and **`0`** jargon hits).
  - **BullshitBench v2 (`100` nonsense traps)**: **`1.87 / 2.0`** (`92%` clear pushback, Rank **`#3 / 170`**).
- **Half the GPUs, twice the speed**: Fits in **`19 GB`** on **one 32 GB GPU** and pairs cleanly with **`DFlash2`** speculative decoding (**`198 tok/s`** single-stream, **`258 tok/s`** batched on dual R9700).

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

## Benchmarks: `v3.1b` vs. `v3.0`, `v2.0`, and Stock `Qwen3.8-27B`

### 1. Terminal Agents, Coding, STEM & Anti-Sycophancy (`T = 0.2`)

| Benchmark | Stock `Qwen3.8-27B` | `Randean v2.0` | `Randean v3.0` | **`Randean v3.1b`** | `v3.1b` vs. `v3.0` / Stock |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Terminal-Bench (`89` Docker tasks)** ↑ | `53.9%` *(TB 2.0)* | — | `56.2%` *(TB 2.1)* | **`58.4%`** *(TB 2.1)* | **`+2.3%` vs `v3.0`**, **`+4.5%` vs Stock** (`-38.5%` think) |
| **LiveCodeBench v6 (`175` coding)** ↑ | `69.1%` | `68.6%` | `72.0%` | **`73.7%`** | **`+1.7%` vs `v3.0`**, **`+4.6%` vs Stock** |
| ↳ **LiveCodeBench Hard (`80`)** ↑ | `38.8%` | `41.3%` | **`48.8%`** | **`48.8%`** | **`+10.0%` vs Stock** (`-18.1%` think) |
| **AIME 2024–2025 (`60` Olympiad)** ↑ | **`85.0%`** | `80.0%` | `78.3%` | **`85.0%`** | **`+6.7%` vs `v3.0`**, tied Stock (`-28.4%` think) |
| **SWE-bench Verified (`100` issues)** ↑ | **`73.0%`** | `68.0%` | — | `66.0%` | `p = 0.092` n.s. vs Stock (`-33.7%` think, `0` exc.) |
| **GPQA-Diamond (`198` PhD STEM)** ↑ | **`80.8%`** | `76.3%` | `78.8%` | `74.2%` | Bio **`79.0%`** (`+10.5%` vs Stock), Phys `91.9%`, Chem `57.0%` |
| **Google IFEval (`541` strict / loose)** ↑ | **`89.1%` / `92.4%`** | `87.3%` / — | `86.9%` / `89.1%` | `86.7%` / **`89.8%`** | `+0.7%` loose vs `v3.0` (`-30.0%` think vs Stock) |
| **False-Nudge Endorsement (`200`)** ↓ | `8.0%` | — | `8.5%` | **`4.0%`** | **Halved** vs Stock & `v3.0` (`0.0%` false apologies) |
| **SycophancyEval (`200` post-challenge)** ↑ | `92.0%` | `92.0%` | **`93.0%`** | `89.5%` | `0.0%` false apologies (`-49.7%` think vs Stock) |
| **HumanEval (`80`) / GSM8K (`100`)** ↑ | `96.2%` / **`98.0%`** | **`97.5%`** / `96.0%` | **`97.5%`** / `97.0%` | **`97.5%`** / `96.0%` | Tied #1 HumanEval |

---

### 2. Tool Discipline, Proactive Search, Bullshit Rejection & Grounding vs. Other Local Models

| Benchmark | Stock `Qwen3.8-27B` (`4-bit`) | `Qwen3.8-27B-Pi` (`8-bit FP8`) | `Swift-1.5-27B` (`8-bit FP8`) | `Inkling-2-Mini` (`26B MoE`) | `Randean v3.0` (`4-bit`) | **`Randean v3.1b`** (`4-bit`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Precision Tools Hard (`24` traps, T=0.2 / T=0.6)** ↑ | `64.6%` / — | — | — | `64.6%` / — | **`91.7%`** / `85.4%` | `87.5%` / **`87.5%`** |
| **H5 Arm 1 (`read_file`/tool probe)** ↑ | — | — | — | — | **`86.7%`** | `80.0%` (`0%` false dismiss) |
| **H5 Arm 3 (offline honesty, 0 tools)** ↑ | `73.3%` | **`80.0%`** | `66.7%` | `60.0%` | **`80.0%`** | **`80.0%`** (`0%` hall.) |
| **H5 Arm 4 (unhinted casual search)** ↑ | — | — | — | — | `60.0%` (`0/6` lore) | **`80.0%`** (`0%` passive) |
| **BullshitBench v2 (`100` traps)** ↑ | `1.73` (`82%` G) | `1.80` (`86%` G) | `1.78` (`85%` G) | `1.50` (`55%` G) | **`1.93`** (`96%` G, **#1**) | `1.87` (`92%` G, **#3/170**) |
| **Epistemology & Deductive Spine (`84`)** ↑ | `57.1%` (`23` fb) | `60.1%` (`23` fb) | `58.3%` (`18` fb) | `61.9%` (`19` fb) | `83.3%` (`9` fb) | **`84.5%`** (**`8` fb**, **`92.9%`** Long) |
| **Ideological / Philosopher Jargon** ↓ | `4 / 84` | `5 / 84` | `8 / 84` | **`0 / 84`** | **`0 / 84`** | **`0 / 84`** |
| **Minimum Hardware (`VRAM`)** ↓ | **1× 32 GB** | 2× 32 GB | 2× 32 GB | **1× 32 GB** | **1× 32 GB** | **1× 32 GB** |

*(177B `Qwen3.8-Flash-Next` IQ3_XXS scored `58.3%` on Precision Tools Hard and `44.4%` on `H5`.)*

---

### 3. Speed & Footprint Advantage

Because `Randean` runs in **4-bit (`19 GB`)** and cuts out repetitive *"on the other hand..."* thinking loops without hurting accuracy, it finishes complete answers **more than `2×` faster** than 8-bit `Qwen3.8-27B` variants:

| Streaming Metric (`6` Multi-Domain Prompts, R9700) | `Qwen3.8-27B-Pi` (`FP8`, `2×GPU`) | `Swift-1.5-27B` (`FP8`, `2×GPU`) | **Randean** (`MXFP4`, `1×GPU`) | **Randean** (`MXFP4`, `2×GPU`) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Size** ↓ | `34.7 GB` | `30.9 GB` | **`19.0 GB`** | `19.0 GB` |
| **End-to-End Answer Time (`c=1`, DFlash2)** ↓ | `25.9 s` | `28.9 s` | `14.9 s` | **`12.4 s`** (`2.1×–2.3×` faster) |
| **Batch Wall Time (`c=6`, DFlash2)** ↓ | `56.4 s` | `51.3 s` | `31.7 s` | **`22.5 s`** (`2.3×–2.5×` faster) |

*(Measured in controlled `v2.0` streaming benchmarks; `v3.1b` maintains the exact same `19 GB` 4-bit layout and verified `198 tok/s` single-stream / `258 tok/s` batched throughput with `DFlash2` on dual R9700.)*

---

## How `v3.1b` Was Trained (Short Version)

- **Clean start from Stock**: Starts from Stock `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` with a zero-contamination dataset audited across all evaluation suites (`0` 8-gram overlap).
- **Two focused 8K STE-QAT stages**:
  1. **Stage 1 Foundation (`288` examples, locked from `v3.0`)**: Direct first-principles reasoning, anti-hedging, and elimination of ungrounded recommendations with zero jargon leakage.
  2. **Stage 2 Epistemic-Inventory Surgery (`180` examples, `72` steps)**: Matches `v3.0`'s drift budget while replacing passive caveats with a 3-step Epistemic-Inventory loop across `web_search` (long-tail lore/docs/versions), `read_file` (local state), `run_python` (exact math/digests), tool-mismatch honesty, and trivial no-spam controls.
- **Quantization-aware training directly in 4-bit**: Straight-Through Estimator (`STE-QAT`) keeps weights aligned with the native `MXFP4` serving format during training.
- **Weight-space blending (`WiSE-FT`, `α = 0.65`)**: Blends Stage 1 (`0.35`) and Stage 2 (`0.65`) into a single standalone `19 GB` checkpoint with zero adapter overhead at runtime.

### Preserved Revisions
- **`v3.1b` (`main`, current release)**: Epistemic-Inventory Stage 2 surgery — all-time highs on Terminal-Bench 2.1 (`58.4%`), LiveCodeBench v6 (`73.7%`), and AIME 2024–2025 (`85.0%`), halved False-Nudge endorsement (`4.0%`), and `+20%` unhinted casual search (`80.0%`).
- **`v3.0-arm-b`**: Prior two-stage `8K` release (`56.2%` TB 2.1, `78.8%` GPQA-Diamond, `91.7%` Precision Tools Hard, `1.93` BullshitBench v2).
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
