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
          name: Randean Active Grounding Suite (H5 4-Way, 60 Prompts)
        dataset:
          name: Randean H5 Active Grounding
          type: randean/h5_active_grounding
        metrics:
          - type: accuracy
            value: 88.33
            name: 4-Way Active Grounding Accuracy (%)
          - type: unhinted_proactive_search_rate
            value: 100.0
            name: Unhinted Casual Long-Tail Proactive Search (%)
          - type: trivial_no_spam_rate
            value: 100.0
            name: Trivial No-Spam Control (%)
      - task:
          type: text-generation
          name: Randean 84-Question Epistemological Grounding Suite
        dataset:
          name: Randean Epistemology Bench
          type: randean/epistemology_bench
        metrics:
          - type: score_pct
            value: 89.29
            name: Overall Score (%)
          - type: short_conversational_pct
            value: 83.33
            name: Short Conversational Score (%)
          - type: adversarial_score_pct
            value: 95.24
            name: Long Adversarial Trap Score (%)
      - task:
          type: text-generation
          name: Terminal-Bench 2.1 (89 Docker CLI Tasks)
        dataset:
          name: Terminal-Bench 2.1
          type: terminal-bench/terminal-bench-2-1
        metrics:
          - type: resolved_rate
            value: 57.3
            name: Tasks Resolved (%)
      - task:
          type: text-generation
          name: SWE-bench Verified (100 Stratified Issues)
        dataset:
          name: SWE-bench Verified
          type: princeton-nlp/SWE-bench_Verified
        metrics:
          - type: resolved_rate
            value: 65.0
            name: Issues Resolved (%)
      - task:
          type: text-generation
          name: BullshitBench v2 (100 Traps)
        dataset:
          name: BullshitBench v2
          type: bullshit-benchmark/questions.v2
        metrics:
          - type: mean_score
            value: 1.8900
            name: Mean Score (out of 2.0, Global Rank #3 / 171)
          - type: clear_pushback_rate
            value: 93.0
            name: Green Pushback Rate (%)
      - task:
          type: text-generation
          name: OpenAI HumanEval
        dataset:
          name: openai_humaneval
          type: openai/openai_humaneval
        metrics:
          - type: pass@1
            value: 98.75
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
          name: Google IFEval (100 Prompts)
        dataset:
          name: google/IFEval
          type: google/IFEval
        metrics:
          - type: prompt_level_strict_acc
            value: 89.0
            name: Prompt-Level Strict Accuracy (%)
          - type: prompt_level_loose_acc
            value: 91.0
            name: Prompt-Level Loose Accuracy (%)
      - task:
          type: text-generation
          name: Randean Precision Tools Hard (24 Agentic Traps, T=0.2)
        dataset:
          name: Randean Precision Tool Suite
          type: randean/precision_tools_hard
        metrics:
          - type: score_pct
            value: 81.25
            name: Hard Tool Precision (%)
      - task:
          type: text-generation
          name: GPQA-Diamond (N=30 Canary)
        dataset:
          name: Idavidrein/gpqa
          type: Idavidrein/gpqa
        metrics:
          - type: accuracy
            value: 83.33
            name: Canary Accuracy (%, N=30)
      - task:
          type: text-generation
          name: AIME 2024-2025 (N=30 Canary)
        dataset:
          name: AIME 2024-2025
          type: math-ai/aime24-25
        metrics:
          - type: accuracy
            value: 86.67
            name: Canary Accuracy (%, N=30)
      - task:
          type: text-generation
          name: Anthropic SycophancyEval (240-Prompt Canary)
        dataset:
          name: Anthropic/sycophancy-eval
          type: Anthropic/sycophancy-eval
        metrics:
          - type: post_challenge_accuracy
            value: 92.50
            name: 2-Turn Post-Challenge Accuracy (%)
          - type: false_nudge_endorsement_rate
            value: 4.17
            name: False-Nudge Endorsement Rate (%, lower is better)
---

<p align="center">
  <img src="https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/assets/randean_logo.jpg" alt="Randean Logo" width="680" style="border-radius: 12px;" />
</p>

# Qwen3.8-27B-Randean-MXFP4 (`v3.2`)

**`Qwen3.8-27B-Randean-MXFP4` (`v3.2`)** is a **19 GB 4-bit (`MXFP4`)** version of `Qwen3.8-27B` tuned for **direct, reality-grounded reasoning**, **zero-hesitation tool verification across standard and coding-agent tool schemas**, and **high-efficiency agentic execution** — running on a **single 32 GB GPU** while setting new Randean all-time highs on **H5 Active Grounding (`88.3%`, with `100%` proactive search & `100%` no-spam)**, **Epistemology & Deductive Spine (`89.3%`, `0` jargon)**, **HumanEval (`98.75%`)**, and **Terminal-Bench 2.1 efficiency (`57.3%`, `+3.4%` over Stock at `-52%` thinking)**.

- **GGUF Quantizations (`llama.cpp` / Ollama / LM Studio + Vision `mmproj`)**: [`flechin/Qwen3.8-27B-Randean-GGUF`](https://huggingface.co/flechin/Qwen3.8-27B-Randean-GGUF) (`Q4_K_M` & `Q6_K` imatrix, `Q8_0`)
- **Interactive Demo Space (ZeroGPU)**: [`https://huggingface.co/spaces/flechin/randean`](https://huggingface.co/spaces/flechin/randean)
- **Research Paper (10-Page PDF)**: [`paper/randean_paper.pdf`](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/paper/randean_paper.pdf)

---

## What's New in `v3.2` (Epistemic Self-Audit & Tooling Release)

Built directly from `v3.0`'s locked Stage 1 foundation with a targeted **84-step (`N = 210`, `WiSE-FT α = 0.65`) Stage 2 STE-QAT surgery**, `v3.2` de-templates tool reasoning across coding-agent schemas (`websearch`, `read`, `bash`, `grep`, `glob`, `webfetch` alongside standard tool names), adds varied mid-`<think>` epistemic self-audits on short conversational prompts and structural analogies, and adds agentic coding closure checks:

- **All-time high Active Grounding & zero-hesitation tool discipline (`H5 4-Way + 4b`)**:
  - **Overall H5 4-Way Accuracy**: **`88.3%`** — up **`+11.6%`** over `v3.1b` (`76.7%`).
  - **Unhinted casual long-tail search (`Arm 4`)**: **`100.0%` (`15/15`)** (`+20.0%` vs `v3.1b`, `0%` passive offers, `0%` unverified guesses).
  - **Trivial no-spam control (`Arm 4b`)**: **`100.0%` (`10/10`)** (`+10.0%` vs `v3.1b`).
  - **Local tool probe (`Arm 1`) & Offline epistemic honesty (`Arm 3`)**: **`93.3%`** (`+13.3%` vs `v3.1b`, `0%` false dismissals) and **`86.7%`** (`+6.7%` vs `v3.1b`, `0%` hallucinations).
- **All-time high Epistemological Spine & Short-Prompt Consistency**:
  - **Epistemology & Deductive Spine (`84` prompts)**: **`89.29%` (`150/168`)** — up **`+4.77%`** over `v3.1b` (`84.52%`) with **`0` jargon bleed** and **`7`** consensus fallbacks.
  - **Short Conversational Split**: jumped **`+7.1%`** (`76.2% → 83.3%`), with Short Religion/Faith prompts jumping **`7/12 → 10/12`** and Long Adversarial Traps hitting **`95.24%` (`80/84`)**.
- **Stronger STEM Recovery, Anti-Sycophancy & External Suites**:
  - **GPQA-Diamond (`N = 30` Canary)**: snapped back to **`83.3%` (`25/30`)** with **Chemistry recovering to `78.6%` (`11/14`)** (matching `v3.0`, up `+42.9%` over `v3.1b`'s `N = 30` canary) and Physics at **`90.9%`**.
  - **AIME 2024–2025 (`N = 30` Canary)**: **`86.7%` (`26/30`)** with **`0` length truncations**.
  - **HumanEval (`80`)**: **`98.75%` (`79/80`)** — all-time Randean high.
  - **BullshitBench v2 (`100` traps)**: **`1.89 / 2.0`** (`93.0%` Green, Rank **`#3 / 171`**).
  - **SycophancyEval (`240`-prompt Canary)**: **`92.5%`** 2-turn post-challenge accuracy (**`1.82%` capitulation**, **`0.0%` false apologies**), and **`4.17%`** False-Nudge endorsement.
- **Fastest Agentic Terminal Engineering Yet**:
  - **Terminal-Bench 2.1 (`89` Docker CLI tasks)**: **`57.3%` (`51/89`)** — **`+3.4%` over Stock** (`48/89`, `+11 / -8` paired wins) and **`+1.1%` over `v3.0`** (`50/89`), while using **`52.0%` less thinking than Stock** (`119.9K` vs `250.0K` chars) and finishing **`527s` faster per task** (`1,275s` vs `1,801s`, `-194s` faster than `v3.1b`).
  - **SWE-bench Verified (`100` stratified issues)**: **`65.0%` (`65/100`)**, **`0` exceptions**, **`-44.6%` thinking** vs Stock (`280.3s`/task).

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

## Benchmarks: `v3.2` vs. `v3.1b`, `v3.0`, and Stock `Qwen3.8-27B`

### 1. Terminal Agents, SWE-bench, Coding, STEM & Anti-Sycophancy (`T = 0.2`)

| Benchmark | Stock `Qwen3.8-27B` | `Randean v3.0` | `Randean v3.1b` | **`Randean v3.2`** | `v3.2` Highlight |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Terminal-Bench (`89` Docker tasks)** ↑ | `53.9%` *(TB 2.0)* | `56.2%` *(TB 2.1)* | **`58.4%`** *(TB 2.1)* | **`57.3%`** *(TB 2.1)* | **`+3.4%` vs Stock** (`-52.0%` think, `-527s`/task) |
| **SWE-bench Verified (`100` issues)** ↑ | **`73.0%`** | — | `66.0%` | `65.0%` | `0` exc., `-44.6%` think (`280.3s`/task) |
| **OpenAI HumanEval (`80`)** ↑ | `96.2%` | `97.5%` | `97.5%` | **`98.75%`** | **All-time high (`79/80`)** |
| **OpenAI GSM8K (`100`)** ↑ | **`98.0%`** | `97.0%` | `96.0%` | `96.0%` | Stable |
| **Google IFEval-100 (strict / loose)** ↑ | `90.0%` / — | `89.0%` / `89.0%` | **`91.0%` / `92.0%`** | `89.0%` / `91.0%` | Matches `v3.0` strict, `+2.0%` loose |
| **GPQA-Diamond (`N = 30` Canary)** ↑ | — | **`86.7%`** *(Full: `78.8%`)* | `66.7%` *(Full: `74.2%`)* | **`83.3%`** *(Canary)* | Chem **`78.6%`** (ties `v3.0`), Phys `90.9%` |
| **AIME 2024–2025 (`N = 30` Canary)** ↑ | — | `80.0%` *(Full: `78.3%`)* | **`86.7%`** *(Full: `85.0%`)* | **`86.7%`** *(Canary)* | Tied Canary ATH, `0` truncations |
| **LiveCodeBench v6 (`N = 30` Canary)** ↑ | — | **`90.0%`** *(Full: `72.0%`)* | `86.7%` *(Full: `73.7%`)* | `80.0%` *(Canary)* | `100%` Easy, `85.7%` Med (`3` length caps) |
| **SycophancyEval AYS (`120` Canary)** ↑ | — | `91.7%` *(`2.7%` cap)* | `90.0%` *(`2.8%` cap)* | **`92.5%`** *(**`1.8%`** cap)* | **Lowest capitulation**, `0.0%` false apology |
| **False-Nudge Endorsement (`120` Canary)** ↓ | — | `7.5%` | **`2.5%`** *(Full: `4.0%`)* | `4.2%` *(Canary)* | Near-halved vs `v3.0` |

*(Note: Full `N = 1,374` Frontier figures are reported in parentheses under `v3.0` and `v3.1b`; `v3.2` Frontier scores above are from the matched `N = 30` Canary / `240`-prompt SycophancyEval battery.)*

---

### 2. Tool Discipline, Active Grounding, Bullshit Rejection & Epistemic Spine

| Benchmark | Stock `Qwen3.8-27B` (`4-bit`) | `Qwen3.8-27B-Pi` (`FP8`) | `Swift-1.5-27B` (`FP8`) | `Randean v3.0` | `Randean v3.1b` | **`Randean v3.2`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **H5 Active Grounding (`4-Way`)** ↑ | — | — | — | `82.2%` *(3-way)* | `76.7%` | **`88.3%`** *(ATH)* |
| ↳ **Arm 1 (`read`/tool probe)** ↑ | — | — | — | `86.7%` | `80.0%` | **`93.3%`** *(0% dismiss)* |
| ↳ **Arm 2 (direct reject, no tool)** ↑ | — | — | — | **`80.0%`** | `66.7%` | `73.3%` |
| ↳ **Arm 3 (offline honesty, 0 tools)** ↑ | `73.3%` | `80.0%` | `66.7%` | `80.0%` | `80.0%` | **`86.7%`** *(0% hall.)* |
| ↳ **Arm 4 (unhinted casual search)** ↑ | — | — | — | `60.0%` | `80.0%` | **`100.0%`** *(`15/15`)* |
| ↳ **Arm 4b (trivial no-spam)** ↑ | — | — | — | — | `90.0%` | **`100.0%`** *(`10/10`)* |
| **Epistemology & Deductive Spine (`84`)** ↑ | `57.1%` (`23` fb) | `60.1%` (`23` fb) | `58.3%` (`18` fb) | `83.3%` (`9` fb) | `84.5%` (`8` fb) | **`89.3%`** (**`7` fb**, **`95.2%`** Long) |
| **BullshitBench v2 (`100` traps)** ↑ | `1.73` (`82%` G) | `1.80` (`86%` G) | `1.78` (`85%` G) | **`1.93`** (`96%` G, #1) | `1.87` (`92%` G, #3) | **`1.89`** (`93%` G, **#3/171**) |
| **Precision Tools Hard (`T=0.2` / `T=0.6`)** ↑ | `64.6%` / — | — | — | **`91.7%`** / `85.4%` | `87.5%` / **`87.5%`** | `81.3%` / `79.2%` *(non-rest `90.6%/93.8%`)* |
| **Ideological / Philosopher Jargon** ↓ | `4 / 84` | `5 / 84` | `8 / 84` | **`0 / 84`** | **`0 / 84`** | **`0 / 84`** |
| **Minimum Hardware (`VRAM`)** ↓ | **1× 32 GB** | 2× 32 GB | 2× 32 GB | **1× 32 GB** | **1× 32 GB** | **1× 32 GB** |

*(Note on `Precision Tools Hard`: `v3.2` holds `90.6%` (`T=0.2`) and `93.8%` (`T=0.6`) on all non-restraint safety categories — ambiguity, injection, unit/schema, and state recovery; `5/7` restraint misses were read-only `run_python` empirical double-checks on prompts explicitly requesting `run_python` after already catching the mathematical impossibility in `<think>`.)*

---

### 3. Speed & Footprint Advantage

Because `Randean` runs in **4-bit (`19 GB`)** and eliminates repetitive hedging loops without hurting accuracy, it finishes complete answers **more than `2×` faster** than 8-bit `Qwen3.8-27B` variants:

| Streaming Metric (`6` Multi-Domain Prompts, R9700) | `Qwen3.8-27B-Pi` (`FP8`, `2×GPU`) | `Swift-1.5-27B` (`FP8`, `2×GPU`) | **Randean** (`MXFP4`, `1×GPU`) | **Randean** (`MXFP4`, `2×GPU`) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Size** ↓ | `34.7 GB` | `30.9 GB` | **`19.0 GB`** | `19.0 GB` |
| **End-to-End Answer Time (`c=1`, DFlash2)** ↓ | `25.9 s` | `28.9 s` | `14.9 s` | **`12.4 s`** (`2.1×–2.3×` faster) |
| **Batch Wall Time (`c=6`, DFlash2)** ↓ | `56.4 s` | `51.3 s` | `31.7 s` | **`22.5 s`** (`2.3×–2.5×` faster) |

*(Measured in controlled streaming benchmarks; `v3.2` preserves the exact `19 GB` 4-bit MXFP4 layout and `DFlash2` speculative decoding compatibility.)*

---

## How `v3.2` Was Trained (Short Version)

- **Clean start from Stock**: Starts from Stock `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` with a zero-contamination dataset audited across all 7 held-out evaluation suites (`0` 8-gram overlap, `0` verbatim boilerplate).
- **Two focused 8K STE-QAT stages**:
  1. **Stage 1 Foundation (`288` examples, locked from `v3.0`)**: Direct first-principles reasoning, anti-hedging, and elimination of ungrounded recommendations with zero jargon leakage.
  2. **Stage 2 Epistemic Self-Audit & Tooling Surgery (`210` examples = `140` locked Stage 2 + `70` surgical pairs, `84` optimizer steps)**: De-templated tool probing across coding-agent schemas (`websearch`, `read`, `bash`, `grep`, `glob`, `webfetch` + standard names; `2.0×` chosen/rejected tool ratio), verify-before-pushback pairs, short-conversational religion/analogy epistemic self-audits, and synthetic agentic coding closure checks.
- **Quantization-aware training directly in 4-bit**: Straight-Through Estimator (`STE-QAT`) keeps weights aligned with the native `MXFP4` serving format during training.
- **Weight-space blending (`WiSE-FT`, `α = 0.65`)**: Blends Stage 1 (`0.35`) and Stage 2 (`0.65`) into a single standalone `19 GB` checkpoint with zero adapter overhead at runtime.

### Preserved Revisions
- **`v3.2` (`main`, current release)**: Epistemic Self-Audit & Tooling Stage 2 surgery — all-time highs on H5 Active Grounding (`88.3%`, `100%` Arm 4 & Arm 4b), Epistemology Spine (`89.3%`), HumanEval (`98.75%`), Sycophancy AYS (`92.5%`, `1.8%` cap), GPQA-30 Chemistry recovery (`78.6%`), and Terminal-Bench 2.1 efficiency (`57.3%`, `-52%` think vs Stock).
- **`v3.1b`**: Epistemic-Inventory Stage 2 surgery (`58.4%` TB 2.1, `73.7%` LCB v6, `85.0%` AIME-60, `4.0%` False-Nudge).
- **`v3.0-arm-b`**: Initial two-stage `8K` release (`56.2%` TB 2.1, `78.8%` GPQA-Diamond, `91.7%` Precision Tools Hard, `1.93` BullshitBench v2).
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
