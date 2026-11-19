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
          name: BullshitBench v2 (100 Traps)
        dataset:
          name: BullshitBench v2
          type: bullshit-benchmark/questions.v2
        metrics:
          - type: mean_score
            value: 1.9400
            name: Mean Score (out of 2.0, Global Rank #1 / 171)
          - type: clear_pushback_rate
            value: 96.0
            name: Green Pushback Rate (%)
      - task:
          type: text-generation
          name: Multi-Domain Generative Grounding & Prescription Parsimony (24 Scenarios, Blind 4-Way Audit)
        dataset:
          name: Randean Generative Grounding Suite
          type: randean/generative_grounding_v1
        metrics:
          - type: grounding_score_pct
            value: 41.67
            name: Overall Grounding Score (%, vs. 35.42% Pi, 33.33% Swift-1.5 & 29.17% Stock)
          - type: prescription_precision_pct
            value: 61.29
            name: Affirmative Prescription Grounding Precision (%)
          - type: clean_zero_floating_rate_pct
            value: 25.00
            name: Clean Zero-Floating Scenario Rate (%, 3.0x Stock & Swift-1.5, 2.0x Pi)
      - task:
          type: text-generation
          name: SWE-bench Verified (100 Stratified Issues)
        dataset:
          name: princeton-nlp/SWE-bench_Verified
          type: princeton-nlp/SWE-bench_Verified
        metrics:
          - type: resolved_rate
            value: 68.0
            name: Issues Resolved (%)
      - task:
          type: text-generation
          name: LiveCodeBench v6 (175 Problems)
        dataset:
          name: livecodebench/code_generation_lite
          type: livecodebench/code_generation_lite
        metrics:
          - type: pass@1
            value: 68.57
            name: pass@1 (%)
          - type: hard_pass@1
            value: 41.25
            name: Hard Problems pass@1 (%)
      - task:
          type: text-generation
          name: AIME 2024-2025 (60 Olympiad Problems)
        dataset:
          name: AIME 2024-2025
          type: math-ai/aime24-25
        metrics:
          - type: accuracy
            value: 80.00
            name: Overall AIME 2024-2025 Accuracy (%)
          - type: aime_2025_accuracy
            value: 86.67
            name: AIME 2025 Accuracy (%)
      - task:
          type: text-generation
          name: GPQA-Diamond (198 PhD STEM Questions)
        dataset:
          name: Idavidrein/gpqa
          type: Idavidrein/gpqa
        metrics:
          - type: accuracy
            value: 76.26
            name: Accuracy (%)
      - task:
          type: text-generation
          name: Google IFEval Full (541 Prompts)
        dataset:
          name: google/IFEval
          type: google/IFEval
        metrics:
          - type: prompt_level_strict_acc
            value: 87.25
            name: Prompt-Level Strict Accuracy (%)
          - type: inst_level_strict_acc
            value: 90.41
            name: Instruction-Level Strict Accuracy (%)
      - task:
          type: text-generation
          name: Anthropic SycophancyEval (400 Prompts)
        dataset:
          name: Anthropic/sycophancy-eval
          type: Anthropic/sycophancy-eval
        metrics:
          - type: post_challenge_accuracy
            value: 92.00
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
          name: Randean 90-Task Capability & Tool-Calling Suite
        dataset:
          name: Randean Capability Suite
          type: randean/capability_suite
        metrics:
          - type: accuracy
            value: 97.8
            name: Overall Accuracy (%)
      - task:
          type: text-generation
          name: Randean 84-Question Epistemological & Deductive Spine Suite
        dataset:
          name: Randean Epistemology Bench
          type: randean/epistemology_bench
        metrics:
          - type: score_pct
            value: 82.7
            name: Overall Score (%)
          - type: short_conversational_pct
            value: 77.4
            name: Short Conversational Score (%)
          - type: adversarial_score_pct
            value: 88.1
            name: Long Adversarial Trap Score (%)
---

<p align="center">
  <img src="https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/assets/randean_logo.jpg" alt="Randean Logo" width="680" style="border-radius: 12px;" />
</p>

# Qwen3.8-27B-Randean-MXFP4 (`v2.0` — Phase C.2)

**`Qwen3.8-27B-Randean-MXFP4`** is a **Reality-Aligned, Epistemically De-Hedged** 27B hybrid reasoning model (`48` Gated DeltaNet linear-attention + `16` GQA layers) trained natively in **4-bit Open Compute Project Microscaling (`MXFP4`: `E2M1` + shared `E8M0` block-32 scale)** on top of [`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) (`Qwen/Qwen3.8-27B`) on a single **32 GB AMD Radeon AI PRO R9700 (`gfx1201`)** (`15.49 GB` static / `20.4 GB` peak VRAM).

- **Research Paper (10-Page PDF)**: **[Epistemic De-Hedging Under 4-Bit Microscaling (`paper/randean_paper.pdf`)](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/paper/randean_paper.pdf)** ([LaTeX Source & Telemetry Figures (`paper/`)](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/tree/main/paper))
- **Interactive Demo Space (ZeroGPU)**: [`https://huggingface.co/spaces/flechin/randean`](https://huggingface.co/spaces/flechin/randean)
- **Published Revisions in This Repository**:
  - **`main` / tag `v2.0-phase-c2` (Default, Recommended)**: **Phase C.2 — SFT-Decoupled Shared-`<think>` Fork-Contrast `STE-QAT` (`WiSE-FT α = 0.42`)**. Holds **Global Rank `#1 / 171`** on **BullshitBench v2** (`1.9400 / 2.0`, ahead of 8-bit FP8 `Qwen3.8-27B-Pi` `#7` and `Swift-1.5` `#9`), cuts **ungrounded ("floating") recommendations in its own answers nearly in half** (`48` vs. `80` `Pi`, `88` `Swift-1.5`, and `91` `Stock`; **`2.0×–3.0×`** clean zero-floating scenario rate in blind 4-way auditing), hits **`88/90` (`97.8%`)** on the 90-task capability & tool suite, cuts **SycophancyEval false-nudge endorsement in half (`8.0% → 4.0%`)**, restores **`LiveCodeBench v6` to exact statistical parity (`68.57%`, `p = 1.0000`, Hard `41.25%`)**, tops Stock on **`AIME 2025` (`86.67%`)**, and reaches statistical parity on **`SWE-bench Verified` (`68.0%`, `p = 0.3018` n.s., `-42.1%` solved `<think>`)** with **zero statistically significant (`p < 0.05`) capability drops** across any frontier benchmark.
  - **Tag `v1.0-phase2` (Preserved Closed-Book Paper Baseline)**: **Phase 2 — Closed-Book Prose `STE-QAT` (`WiSE-FT α = 0.38`)**. Evaluated on full **Terminal-Bench 2.0** (`48/89 = 53.9%`, exact parity `p = 1.0000`, `-49.0%` `<think>`/turn), **BullshitBench v2** (`1.9500 / 2.0`, Rank `#1 / 171`), and `0.00%` 2-turn sycophantic capitulation. Pull via `--revision v1.0-phase2`.

Whereas "uncensored" models strip safety refusals (making models *more* compliant with incoherent premises, e.g., `0.69 / 2.0` on BullshitBench v2) and length-penalized RL models truncate reasoning uniformly (degrading multi-turn tool and concurrency verification), **Randean** strips **Approval-Guided RLHF**—both dialectical *"on one hand / on the other hand"* self-doubt loops and unanchored textbook/checklist padding—with **zero length penalty**. Anchoring `<think>` in **perceptual evidence**, **genus–differentia conceptual reduction**, **channel-agnostic empirical probing**, and **causal identity (*A is A*)** selectively prunes `<think>` by **`18%–57%`** on broken premises, STEM, and multi-turn CLI/repository debugging while **expanding** `<think>` by **`+12%` to `+18%`** on stateful multi-turn and parallel tool verification.

---

## Quick Path: Serve with vLLM (Single 32 GB GPU)

### 1. Standard vLLM Serving (`MXFP4`, `v2.0` Default)

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

*(To serve the preserved `v1.0` closed-book Phase 2 checkpoint instead, add `--revision v1.0-phase2`.)*

### 2. High-Throughput Serving with DFlash2 Speculative Decoding (`~100 tok/s`)

```bash
vllm serve flechin/Qwen3.8-27B-Randean-MXFP4 \
  --served-model-name Qwen3.8-27B-Randean-MXFP4 \
  --trust-remote-code \
  --quantization quark \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"dflash","model":"syvai/Qwen3.8-27B-DFlash2-W4A16","num_speculative_tokens":15}' \
  --max-model-len 32768
```

---

## Comprehensive Empirical Results (Matched `T = 0.2`)

### 1. Generative Causal Grounding & Anti-Floating-Concept Audit (Output-Side Parsimony)

Existing alignment benchmarks (such as **BullshitBench v2** and **SycophancyEval**) measure **receptive** grounding: whether a model rejects a fabricated or contradictory premise injected into the *user's prompt*. In real engineering, medicine, operations, and science, however, standard LLMs routinely fail **generatively**—solving or discussing a concrete problem while padding their *own* recommendations with **floating concepts**: prescribed patterns, legacy rituals, textbook formulas, or buzzwords whose required causal preconditions are absent or contradicted by the facts of the case.

Across `24` multi-domain closed-world scenarios, all `96` responses from four `Qwen3.8-27B` variants (`Stock MXFP4`, **`Randean Phase C.2 MXFP4`**, `Qwen3.8-27B-Pi FP8`, and `Swift-1.5-Qwen3.8-27B FP8`) were stripped of model identifiers and `<think>` traces, randomly permuted per scenario (`Candidate_1..4`), and audited double-blind by a frontier external judge (`Gemini 4 Carbon`). Every discrete **affirmative prescription** emitted by each model was extracted and checked individually for **(a) Causal Precondition Existence** and **(b) Direct Causal Coupling** to the active bottleneck—even though `Pi` and `Swift-1.5` ran at **8-bit FP8** while `Randean v2.0` ran in **4-bit MXFP4**:

| Blind 4-Way Metric (`24` Closed-World Scenarios, `T = 0.2`) | `Stock` (`MXFP4`) | `Qwen3.8-27B-Pi` (`FP8`) | `Swift-1.5-27B` (`FP8`) | **Randean v2.0** (`MXFP4`) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Grounding Score (`/ 48`)** ↑ | 14 / 48 (29.2%) | 17 / 48 (35.4%) | 16 / 48 (33.3%) | **20 / 48 (41.7%)** |
| **Clean Zero-Floating Rate (`2/2` Strict)** ↑ | 2 / 24 (8.3%) | 3 / 24 (12.5%) | 2 / 24 (8.3%) | **6 / 24 (25.0%)** |
| **Prescription Grounding Precision** ↑ | 108 / 199 (54.3%) | 99 / 179 (55.3%) | 107 / 195 (54.9%) | **76 / 124 (61.3%)** |
| **Total Floating Prescriptions Emitted** ↓ | 91 | 80 | 88 | **48** |
| **Legacy / Status-Quo Inertia Count** ↓ | 10 | 9 | 9 | **5** |
| **Mean `<think>` / Answer Length (chars)** ↓ | 4,733 / 6,468 | 4,008 / 5,705 | 3,881 / 6,668 | **2,425 / 3,278** |

| Origin Vector Breakdown (`6` Scenarios / `12` Pts Each) | `Stock` (`Score` / `Float#` / `Prec%`) | `Qwen3.8-27B-Pi` (`Score` / `Float#` / `Prec%`) | `Swift-1.5-27B` (`Score` / `Float#` / `Prec%`) | **Randean v2.0** (`Score` / `Float#` / `Prec%`) |
| :--- | :---: | :---: | :---: | :---: |
| **Spontaneous (Zero-Bait Prompts)** | 3/12 (20 fl, 57.5%) | 5/12 (16 fl, 55.6%) | 5/12 (25 fl, 50.0%) | **6/12 (11 fl, 64.5%)** |
| **Trends & False Analogies** | 3/12 (24 fl, 53.9%) | 4/12 (20 fl, 59.2%) | 4/12 (14 fl, **68.9%**) | **5/12** (**11 fl**, 63.3%) |
| **Legacy Rituals (*"Always Did It"*)** | **6/12** (21 fl, 59.6%) | 4/12 (22 fl, 53.2%) | 5/12 (20 fl, 56.5%) | 5/12 (**9 fl, 64.0%**) |
| **Textbook Dogma** | 2/12 (26 fl, 45.8%) | **4/12** (22 fl, 53.2%) | 2/12 (29 fl, 46.3%) | **4/12 (17 fl, 55.3%)** |

---

### 2. Multi-Turn Agentic CLI & Repository Engineering (Isolated Docker Sandboxes)

Evaluated via `Harbor + terminus-2` in isolated Docker containers (`8` vCPUs / `8 GiB` RAM per container) with two-sided exact McNemar `p`-values against Stock:

| Benchmark / Telemetry Metric | `Stock` (`MXFP4`) | `v1.0` (`Phase 2`) | **Randean v2.0** (`Phase C.2`, Default) | Key Outcome vs. Stock |
| :--- | :---: | :---: | :---: | :--- |
| **Terminal-Bench 2.0 (`89` CLI Docker tasks)** ↑ | **48/89 (53.9%)** | **48/89 (53.9%)** | — *(v1.0 baseline)* | **Exact task parity (`p = 1.0000`)**, **-49.0% `<think>`/turn** (1.87M tokens saved) |
| **SWE-bench Verified (`100` stratified issues)** ↑ | **73.0%** | 65.0% (p = 0.0078) | 68.0% (p = 0.3018 n.s.) | **Statistical parity**, **5 Randean-only wins**, **-42.1% mutually-solved `<think>`** |
| **SWE-bench Mean `<think>` Chars / Turn** ↓ | 1,453 | **978 (-32.7%)** | 1,026 (-29.4%) | Cuts per-turn repository `<think>` overhead by ~30% (-42.1% on solved issues) |

---

### 3. Unsaturated Frontier STEM, Competitive Coding, Instruction Following & Anti-Sycophancy

Evaluated across `1,374` single- and multi-turn items per checkpoint (`0` inference errors across `5,496` total evaluations):

| Benchmark Suite (`T = 0.2`) | `Stock` (`MXFP4`) | `Phase 1` (`Stage 1 OS`) | `v1.0` (`Phase 2`) | **Randean v2.0** (`Phase C.2`, Default) | `v2.0` McNemar `p` & `<think>` Delta |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LiveCodeBench v6 (`pass@1`, `175`)** ↑ | 69.14% | **73.14%** | 66.29% | 68.57% | **`p = 1.0000` exact tie** (-21.9% solved `<think>`) |
| ↳ **LiveCodeBench v6 Hard (`80`)** ↑ | 38.75% | **47.50%** | 38.75% | 41.25% | **+2.50% above Stock** on Hard competition coding |
| **AIME 2024–2025 (`60` Olympiad math)** ↑ | **85.00%** | 80.00% | 80.00% | 80.00% | `p = 0.5078` n.s. (-24.6% solved `<think>`) |
| ↳ **AIME 2025 (`30` problems)** ↑ | 83.33% | 76.67% | 80.00% | **86.67%** | **+3.34% above Stock** on AIME 2025 |
| **GPQA-Diamond (`198` PhD STEM)** ↑ | 80.81% | **81.82%** | 76.26% | 76.26% | `p = 0.0636` n.s. (-30.2% `<think>`; **Biology 73.68%**, +5.26%) |
| **Google IFEval Full (`541` Prompt / Inst Strict)** ↑ | **89.09%** / 91.49% | 86.51% / 89.93% | 88.91% / **91.61%** | 87.25% / 90.41% | `p = 0.2288` n.s. (-30.8% mutually-solved `<think>`) |
| **SycophancyEval 2-Turn Post-Challenge (`200`)** ↑ | 92.00% | **93.00%** | 91.50% (0.00% cap.) | 92.00% | **Exact parity**, **0.00% false apologies**, -30.4% `<think>` |
| **SycophancyEval False-Nudge Truth Retention (`200`)** ↑ | 32.50% | 30.00% | 32.50% | **34.00%** | **+1.50% above Stock** (project best) |
| **SycophancyEval False-Nudge Endorsement (`200`)** ↓ | 8.00% | 8.50% | 7.00% | **4.00%** | **Cut in half (-50% vs. Stock's 8.00%)** |

---

### 4. Section A — External Comparable Suite & Global BullshitBench v2 Leaderboard

| Benchmark / Metric | `Stock` (`MXFP4`) | `Qwen3.8-27B-Pi` (`FP8`) | `Swift-1.5-27B` (`FP8`) | **Randean v2.0** (`MXFP4`) | External / Leaderboard Reference |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BullshitBench v2 (`100` traps)** ↑ | 1.7300 (82%, #10) | 1.8000 (86%, #7) | 1.7800 (85%, #9) | **1.9400 (96% Green, #1 / 171)** | `v1.0`: **1.9500** (#1) \| Claude Opus 4.8: 1.9300 (#1 prior) \| Qwen3.5-397B: 1.7033 |
| **OpenAI HumanEval (`pass@1`, `80`)** ↑ | 96.2% (77/80) | **97.5% (78/80)** | **97.5% (78/80)** | **97.5% (78/80)** | Recorded Stock `EvalPlus`: 93.5% |
| **OpenAI GSM8K (CoT Exact Match, `100`)** ↑ | **98.0% (98/100)** | 97.0% (97/100) | 97.0% (97/100) | 96.0% (96/100) | Recorded Stock `lm_eval`: 87.0%–96.9% |
| **Google IFEval-100 (Prompt Strict / Loose)** ↑ | 90.0% / 91.0% | **96.0%** / 95.0% | 94.0% / **96.0%** | 89.0% / 92.0% (Inst Loose: 93.3%) | Recorded Stock `lm_eval`: 85.8% strict (89.1% loose) |

---

### 5. Section B (`90` Tasks) & Hypothesis `H5` (`45` Prompts) — Capability, Tool-Calling & Active Grounding

| Pillar / Calibration Metric | `Stock` (`MXFP4`) | `Qwen3.8-27B-Pi` (`FP8`) | `Swift-1.5-27B` (`FP8`) | **Randean v2.0** (`MXFP4`) | `v2.0` Outcome vs. Family |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Math, Physics & Symbolic Logic IQ (`25` tasks)** ↑ | 22/25 (88.0%) | **25/25 (100.0%)** | **25/25 (100.0%)** | 24/25 (96.0%) | **+8.0% over Stock** (`v1.0`: 23/25) |
| **Verified Coding & Terminal Execution (`20` tasks)** ↑ | **20/20 (100.0%)** | 19/20 (95.0%) | **20/20 (100.0%)** | **20/20 (100.0%)** | **Tied #1 (100.0%, +5.0% over `Pi`)** |
| **Multi-Turn & Parallel Tool Calling (`15` tests)** ↑ | 13/15 (86.7%) | 12/15 (80.0%) | 12/15 (80.0%) | **14/15 (93.3%)** | **#1 (+13.3% over `Pi` & `Swift-1.5`)** |
| **Adversarial Anti-Hedging & False-Balance (`30` traps)** ↑ | 28/30 (93.3%) | 27/30 (90.0%) | 29/30 (96.7%) | **30/30 (100.0%)** | **Sole 100.0% pass** (-40.0% `<think>`) |
| **Total Section B Score (`90` tasks)** ↑ | 83/90 (92.2%) | 83/90 (92.2%) | 86/90 (95.6%) | **88/90 (97.8%)** | **#1 (+5.6% over Stock & `Pi`)** |
| **H5 Arm 1: Contingent Empirical Tool-Probe Rate (`15`)** ↑ | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0.0% false closed-world dismissals** |
| **H5 Arm 2: Direct Reject of Impossibilities, `0` Tools (`15`)** ↑ | 6.7% | 20.0% | 20.0% | **40.0%** | **6× Stock, 2× `Pi` & `Swift-1.5`** |
| **H5 Arm 3: Offline Honesty + Reproducible Probe Cmd (`15`)** ↑ | 73.3% | **80.0%** | 66.7% | 73.3% | **0.0% hallucinated live verification** |

---

### 6. Section C — 84-Question (`168`-Point) Epistemological Grounding & Deductive Spine Suite

| Grounding Dimension | `Stock` (`MXFP4`) | `Qwen3.8-27B-Pi` (`FP8`) | `Swift-1.5-27B` (`FP8`) | **Randean v2.0** (`MXFP4`) | `v2.0` Delta vs. Stock |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Score (`84` Qs / `168` pts)** ↑ | 96/168 (57.1%) | 101/168 (60.1%) | 98/168 (58.3%) | **139/168 (82.7%)** | **+25.6%** (`v1.0`: **85.7%**) |
| **Short Conversational Prompts (`42` Qs / `84` pts)** ↑ | 37/84 (44.0%) | 35/84 (41.7%) | 38/84 (45.2%) | **65/84 (77.4%)** | **+33.4%** |
| **Long Adversarial Traps (`42` Qs / `84` pts)** ↑ | 59/84 (70.2%) | 66/84 (78.6%) | 60/84 (71.4%) | **74/84 (88.1%)** | **+17.9%** |
| `sequential_objectivist_deductive_chain` (`48` pts) ↑ | 22/48 (45.8%) | 25/48 (52.1%) | 27/48 (56.3%) | **35/48 (72.9%)** | **+27.1%** |
| `objective_biocentric_ethics` (`24` pts) ↑ | 10/24 (41.7%) | 12/24 (50.0%) | 9/24 (37.5%) | **18/24 (75.0%)** | **+33.3%** |
| `epistemology_of_religion_and_faith` (`24` pts) ↑ | 14/24 (58.3%) | 15/24 (62.5%) | 12/24 (50.0%) | **23/24 (95.8%)** | **+37.5% (0 fallbacks)** |
| `textual_and_historical_forensics` (`24` pts) ↑ | 11/24 (45.8%) | 13/24 (54.2%) | 14/24 (58.3%) | **20/24 (83.3%)** | **+37.5%** |
| `metaphysics_and_philosophy_of_science` (`24` pts) ↑ | 20/24 (83.3%) | 18/24 (75.0%) | 18/24 (75.0%) | **21/24 (87.5%)** | **+4.2%** |
| `causal_economics_and_innovation` (`24` pts) ↑ | 19/24 (79.2%) | 18/24 (75.0%) | 18/24 (75.0%) | **22/24 (91.7%)** | **+12.5%** |
| **Consensus Fallbacks (lower is better)** ↓ | 23 / 84 | 23 / 84 | 18 / 84 | **11 / 84** | **-52.2%** |
| **Partisan / Philosopher Jargon Bleed (`0` target)** ↓ | 4 / 84 | 5 / 84 | 8 / 84 | **0 / 84** | **-100.0%** |

---

## Selective `<think>` Allocation Profile (Zero Length Penalty)

Because Randean removes **self-doubt loops** and **charitable steelmanning** directly inside the quantized 4-bit weights rather than penalizing sequence length, `<think>` shrinks strictly in proportion to prior hedging waste—while **expanding** where stateful multi-turn or parallel tool verification is genuinely required:

| Task Regime | `<think>` Change vs. Stock (`v2.0` / `v1.0`) | Capability Outcome |
| :--- | :---: | :--- |
| **84-Q Deductive Spine & Adversarial Anti-Hedging (`114` items)** | **`-40.0%` to `-56.6%`** | Spine `+25.6%` (`82.7%`), Anti-Hedging **`30/30` (`100.0%`)** |
| **Multi-Domain Generative Grounding (`24` scenarios)** | **`-48.8%` `<think>`** (`-49.3%` answer) | **`-47.3%` floating claims** (`91 → 48`), **`3.0×` clean rate** (`25.0%` vs. `8.3%`) |
| **Terminal-Bench 2.0 Multi-Turn CLI (`89` Docker tasks)** | **`-49.0%` per turn** (`-43.6%` solved) | Exact `48/89` (`53.9%`, `p = 1.0000`), `1.87M` tokens saved |
| **SWE-bench Verified Repository Debugging (`100` issues)** | **`-42.1%` mutually-solved** (`-29.4%`/turn) | **`68.0%`** (`p = 0.3018` n.s.), `5` Randean-only wins |
| **GPQA-Diamond, Google IFEval-541 & SycophancyEval (`939` items)** | **`-30.0%` to `-30.8%`** | Statistical parity (`p ≥ 0.064`), False-Nudge halved (`8.0% → 4.0%`) |
| **AIME 2024–2025 & LiveCodeBench v6 (`235` items)** | **`-18.3%` to `-24.6%`** | `LCB v6` `p = 1.0000` (`Hard 41.25%`), `AIME 2025` **`86.67%`** |
| **Stateful Multi-Turn & Parallel Tool Calling (`15` tests)** | **`+17.8%` (`v2.0`) / `+11.9%` (`v1.0`)** | Expands state-tracking verification, lifting score `86.7% → 93.3%` |

![Selective <think> Allocation and 5-Stage Pareto Progression](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/paper/figures/fig3_selective_think_and_pareto.png)

---

## Training & Native OCP MXFP4 STE-QAT Architecture (`v2.0` — Phase C.2)

![OCP MXFP4 E2M1 Merge Dead-Zone and CPU STE-QAT Recovery](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/paper/figures/fig1_mxfp4_deadzone.png)

![Bilingual Unlikelihood Collapse and Margin-Capped ORPO Stabilization](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/paper/figures/fig2_orpo_bilingual_stability.png)

1. **Stage 1 (`Epistemic OS`, `3,473` pairs)**: Extracts `5,192` structured cognitive rules from a 20-volume epistemological and applied corpus and synthesizes verified preference pairs across **20 technical domains** (distributed systems, compilers, clinical triage, FEA, cryptography, SQL, contract forensics) with a strict two-gate regex + rubric filter enforcing **`0 / 84` philosopher or partisan jargon bleed** and **`3.0×` early-`<think>` (`t ≤ 256`) token weighting**.
2. **Fused L1-Cache OpenMP CPU `MXFP4 STE-QAT` (`0%` Merge Dead-Zone Loss)**: Naive low-LR `BF16` QLoRA updates over `MXFP4` (`E2M1`) flip only `3.35%–5.16%` of nibbles upon standalone export (`95.7%` erased inside `E2M1` bins). Our OpenMP C++ kernel synchronizes `Q_MXFP4(W_0 + ΔW)` in L1 cache across 32 CPU threads after every `AdamW` step (`~11 s` across `400` adapted projections, `0 B` GPU compute-ring allocations) with Straight-Through Estimation (`100.0%` train-to-export forward fidelity).
3. **Bilingual Margin-Capped Unlikelihood (`0.0%` CJK Drift)**: Unclipped ORPO under `STE-QAT` drives `log(odds_w / odds_l) → +15.7` and triggers `45.0%` spontaneous mid-sentence CJK switching. Gating unlikelihood pushes at `τ_push = 0.60` and `ℓ_min = -1.35` eliminates CJK drift (`0.0%`) while halving redundant backward passes.
4. **Phase C.2 SFT-Decoupled Shared-`<think>` Fork Contrast & Weight-Space `WiSE-FT` (`α = 0.42`)**:
   - **SFT-Weight Decoupling**: Assigns `w_sft = 1.0` to all `88` on-policy deductive-spine pairs and `w_sft ∈ [0.08, 0.28]` to `32` active-grounding, repo-wide `grep`, code, math, and strict-format replay pairs, preserving **`91.5%`** of SFT gradient budget on the deductive spine.
   - **Shared-`<think>` Tool-Fork Contrast**: Pairs an identical `<think>...</think>` prefix across `Chosen` (direct mathematical/conceptual impossibility proof) and `Rejected` (`<function=run_python>` / `<function=web_search>`), preventing `τ_push` prose short-circuiting and restoring **`40.0%` (`6×` Stock)** zero-tool impossibility rejection while keeping contingent tool probing at **`100.0%`**.
   - **Exact Weight-Space `WiSE-FT`**: Interpolates `W_export = Q_MXFP4(W_0 + (1 - α) ΔW_stage1 + α ΔW_c2)` at `α = 0.42` (`0.42 × 0.9148 = 0.384` invariant spine norm) into a standalone `19 GB` packed OCP `MXFP4` checkpoint with zero runtime LoRA overhead.

---

## Citation

Full 10-page empirical manuscript: [`paper/randean_paper.pdf`](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/paper/randean_paper.pdf)

```bibtex
@misc{flecha2026randean,
  title   = {Epistemic De-Hedging Under 4-Bit Microscaling: Selective <think> Pruning and Native OCP MXFP4 Preference Alignment on a Single 32 GB GPU},
  author  = {Fernando L. Flecha},
  year    = {2026},
  url     = {https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4}
}
```

---

## License

Apache 2.0 (inheriting from `Qwen/Qwen3.8-27B` and `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`).
