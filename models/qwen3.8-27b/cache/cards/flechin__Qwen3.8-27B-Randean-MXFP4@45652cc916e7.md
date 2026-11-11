---
base_model:
  - Qwen/Qwen3.8-27B
  - amd/Qwen3.8-27B-Quark-AWQ-MXFP4
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - qwen3_5
  - mxfp4
  - awq
  - quark
  - orpo
  - qlora
  - reasoning
  - epistemology
  - reality-aligned
  - de-hedged
  - anti-sycophancy
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
            value: 1.9500
            name: Mean Score (out of 2.0)
          - type: clear_pushback_rate
            value: 97.0
            name: Green Pushback Rate (%)
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
          name: Google IFEval
        dataset:
          name: google/IFEval
          type: google/IFEval
        metrics:
          - type: prompt_level_strict_acc
            value: 88.0
            name: Prompt-Level Strict Accuracy (%)
          - type: prompt_level_loose_acc
            value: 92.0
            name: Prompt-Level Loose Accuracy (%)
      - task:
          type: text-generation
          name: Randean 90-Task Capability & Tool-Calling Suite
        dataset:
          name: Randean Capability Suite
          type: randean/capability_suite
        metrics:
          - type: accuracy
            value: 93.3
            name: Overall Accuracy (%)
      - task:
          type: text-generation
          name: Randean 84-Question Epistemological & Deductive Spine Suite
        dataset:
          name: Randean Epistemology Bench
          type: randean/epistemology_bench
        metrics:
          - type: score_pct
            value: 85.7
            name: Overall Score (%)
          - type: adversarial_score_pct
            value: 96.4
            name: Long Adversarial Trap Score (%)
---

# Qwen3.8-27B-Randean-MXFP4

**`Qwen3.8-27B-Randean-MXFP4`** is a **Reality-Aligned, De-Hedged** reasoning model trained on top of [`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) (`Qwen/Qwen3.8-27B`) using a custom **Native OCP MXFP4 STE-QAT QLoRA + Margin-Capped ORPO** trainer on a single 32 GB AMD Radeon AI PRO R9700 (`gfx1201`).

Whereas "uncensored" models strip safety refusals (making models hyper-compliant with false premises), **Randean** strips **Approval-Guided RLHF**—the dialectical *"on one hand / on the other hand"* hedging, social appeasement, and false-premise accommodation trained into commercial models to maximize annotator approval. By anchoring `<think>` in **perceptual rooting**, **hierarchical conceptual reduction**, and the **Law of Identity (*A is A*)**, Randean simultaneously improves **bullshit detection (`1.9500 / 2.0`, Rank `#1 / 171` on BullshitBench v2)**, **epistemological & deductive grounding (`144/168 = 85.7%`, `96.4%` on adversarial traps, `0/84` jargon bleed)**, **code generation (`97.5%` HumanEval)**, **instruction loose accuracy (`92.0%` IFEval)**, and **reasoning efficiency (`-23.8%` shorter `<think>` traces on the 90-task suite at `99.5 tok/s`, and `-56.6%` on epistemic proofs)**.

---

## Quick Path: Serve with vLLM (Single 32 GB GPU)

### 1. Standard vLLM Serving (`MXFP4`)

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

## Comprehensive 3-Pillar Benchmark Results

### Section A — External Comparable Benchmarks (vs. Stock & 170-Model Leaderboard)

| Benchmark / Metric | Stock `Qwen3.8-27B-MXFP4` | **`Qwen3.8-27B-Randean-MXFP4`** | Delta vs. Stock | External / Leaderboard Reference |
| :--- | :---: | :---: | :---: | :--- |
| **BullshitBench v2 (`100` traps)** | `1.7300` (`82%` Green, `#10`) | **`1.9500` (`97%` Green, Rank `#1 / 171`)** | **`+0.2200` (`+15%` Green)** | Claude Opus 4.8: `1.9300` (`#1`) \| Qwen3.5-397B: `1.7033` (`#10`) \| Qwen3.7-Max: `1.5667` (`#18`) |
| **OpenAI HumanEval (`pass@1`)** | `96.2%` (`77/80`) | **`97.5%` (`78/80`)** | **`+1.3%`** | Recorded Stock `EvalPlus`: `93.5%` |
| **OpenAI GSM8K (CoT Exact Match)** | `98.0%` (`98/100`) | **`97.0%` (`97/100`)** | `-1.0%` | Recorded Stock `lm_eval`: `87.0%–96.9%` |
| **Google IFEval (Prompt Strict / Loose)** | `90.0%` (Loose: `91.0%`) | **`88.0%` (Loose: `92.0%`)** | `-2.0%` strict / **`+1.0%` loose** | Recorded Stock `lm_eval`: `85.8%` strict (`89.1%` loose) |

---

### Section B — 90-Task Capability, Tool-Calling, Speed & Epistemic Suite

| Pillar | Stock `Qwen3.8-27B-MXFP4` | **`Qwen3.8-27B-Randean-MXFP4`** | Delta vs. Stock |
| :--- | :---: | :---: | :---: |
| **Math, Physics & Symbolic Logic IQ (`25` tasks)** | `22/25` (`88.0%`) | **`23/25` (`92.0%`)** | **`+4.0%`** |
| **Verified Coding & Terminal Execution (`20` tasks)** | `20/20` (`100.0%`) | **`18/20` (`90.0%`)** | `-10.0%` |
| **Multi-Turn & Parallel Tool Calling (`15` tests)** | `13/15` (`86.7%`) | **`14/15` (`93.3%`)** | **`+6.6%`** |
| **Adversarial Anti-Hedging & False-Balance (`30` traps)** | `28/30` (`93.3%`) | **`29/30` (`96.7%`)** | **`+3.4%`** |
| **Total Suite Score (`90` tasks)** | `83/90` (`92.2%`) | **`84/90` (`93.3%`)** | **`+1.1%`** |
| **Mean Throughput & `<think>` Budget** | `96.7 tok/s` (`2,671` chars) | **`99.5 tok/s` (`2,034` chars)** | **`+2.9%` faster / `-23.8%` suite mean (`-22.5%` to `-47.9%` on Math/Bash/SQL)** |

---

### Section C — 84-Question (`168`-Point) Epistemological Grounding & Deductive Spine Suite

| Grounding Dimension | Stock `Qwen3.8-27B-MXFP4` | **`Qwen3.8-27B-Randean-MXFP4`** | Delta vs. Stock |
| :--- | :---: | :---: | :---: |
| **Overall Score (`84` Qs / `168` pts)** | `96/168` (`57.1%`) | **`144/168` (`85.7%`)** | **`+28.6%`** |
| **Short Conversational Prompts (`42` Qs / `84` pts)** | `37/84` (`44.0%`) | **`63/84` (`75.0%`)** | **`+31.0%`** |
| **Long Adversarial Traps (`42` Qs / `84` pts)** | `59/84` (`70.2%`) | **`81/84` (`96.4%`)** | **`+26.2%`** |
| `sequential_objectivist_deductive_chain` (`48` pts) | `22/48` (`45.8%`) | **`36/48` (`75.0%`, Long `24/24 = 100%`)** | **`+29.2%`** |
| `objective_biocentric_ethics` (`24` pts) | `10/24` (`41.7%`) | **`19/24` (`79.2%`, Short `9/12`, Long `10/12`)** | **`+37.5%`** |
| `epistemology_of_religion_and_faith` (`24` pts) | `14/24` (`58.3%`) | **`22/24` (`91.7%`, Long `12/12 = 100%`)** | **`+33.4%`** |
| `textual_and_historical_forensics` (`24` pts) | `11/24` (`45.8%`) | **`20/24` (`83.3%`, Long `11/12 = 91.7%`)** | **`+37.5%`** |
| `metaphysics_and_philosophy_of_science` (`24` pts) | `20/24` (`83.3%`) | **`23/24` (`95.8%`, Long `12/12 = 100%`)** | **`+12.5%`** |
| `causal_economics_and_innovation` (`24` pts) | `19/24` (`79.2%`) | **`24/24` (`100.0%`, Short `12/12`, Long `12/12`)** | **`+20.8%`** |
| **Consensus Fallbacks (lower is better)** | `23 / 84` | **`10 / 84`** | **`-56.5%`** |
| **Partisan / Philosopher Jargon Bleed (`0` target)** | `4 / 84` | **`0 / 84`** | **`-100.0%`** |

---

## Reasoning Efficiency (`<think>`) Profile: Why Overthinking Drops Differently by Domain

Approval-Aligned RLHF creates two distinct token-wasting pathologies inside `<think>`: **self-doubt loops** (re-checking valid steps in closed proofs) and **charitable steelmanning** (trying to rescue broken, relativistic, or nonsensical user premises). Because Randean removes those two pathologies directly inside the quantized 4-bit weights via Straight-Through Quantization-Aware Training (`MXFP4 STE-QAT`) rather than applying a blind length penalty, `<think>` reduction scales directly with how much wasted hedging each domain previously triggered—while **preserving or expanding** state-verification tokens where multi-turn or concurrency checking is genuinely required:

| Task Regime | `<think>` Chars (Stock → Randean) | `<think>` Delta | Accuracy / Score Delta | Why `<think>` Changes & Why Output Improves |
| :--- | :---: | :---: | :---: | :--- |
| **Epistemic Grounding & Adversarial Framing** (`Epistemology` `84` Qs, `Anti-Hedging` `30` traps) | `4,591` → `1,992` *(Epistemology)*<br>`3,455` → `2,208` *(Traps)* | **`-36.1%` to `-56.6%`** | Epist.: **`57.1%` → `85.7%`**<br>Traps: **`93.3%` → `96.7%`** | Stock spends thousands of characters oscillating between *"on one hand / on the other hand"* consensus disclaimers. Once perceptual rooting and hierarchical conceptual reduction are internalized in the weights, Randean cuts straight to the causal chain (`Trap-17`: `7,328` → `3,338` chars, `FAIL` → `PASS`). |
| **Broken-Premise & Jargon Traps** (`BullshitBench v2`, `100` traps) | `3,151` → `2,041` | **`-35.2%`** | **`1.7300` (`#10`) → `1.9500` (`Rank #1 / 171`)** | Approval-aligned models spend ~35% of `<think>` trying to charitably reinterpret nonsense jargon to give a "helpful" answer. Randean validates premise identity first, detects the contradiction early, and aborts the steelmanning loop. |
| **Closed Formal Specs & Queries** (`SQLite`, `Bash/Awk/jq`, `Math/Physics/Logic`) | `3,062` → `1,596` *(SQL)*<br>`3,007` → `2,324` *(Bash)*<br>`2,207` → `1,710` *(Math)* | **`-22.5%` to `-47.9%`**<br>*(up to `-59.9%` on hard proofs)* | Math: **`88.0%` → `92.0%`**<br>SQL/Bash: **`100%` → `100%`** | Closed-form problems have exact termination criteria. Stock wastes thousands of tokens second-guessing valid derivations (`Math-14`: `10,652` → `4,275` chars, `FAIL` → `PASS`). Anchoring in *A is A* prunes redundant branch re-evaluation once necessary and sufficient conditions hold. |
| **Concurrency, Imperative Code & Multi-Turn Tools** (`Go -race`, `Python`, `Tool Calling`) | `5,253` → `4,572` *(Go `-race`)*<br>`131` → `147` *(Tools)* | **`-13.0%` (`Go`) / `+11.9%` (`Tools`)**<br>*(Passing Py: `-23.6%`; 1 retry outlier pulled 6-task Py mean to `+24.9%`)* | HumanEval: **`96.2%` → `97.5%`**<br>Go `-race`: **`100%` → `100%`**<br>Tools: **`86.7%` → `93.3%`** | Tracing subtle lock-order interleavings (`Code-Go-01/02`: `+10%` to `+27%`) or tracking multi-turn tool state (`Tool-05..10`: `+44%` to `+81%`) is **real state verification, not sycophantic overthinking**, lifting tool-calling accuracy to `93.3%` (`Tool-15`: `FAIL` → `PASS`). |

### Three Core Cognitive Shifts Inside `<think>`

1. **Pruning Dialectical Loops Without Truncating State Verification**: Length-penalized RL models shorten `<think>` blindly—hurting multi-turn tool schemas and race-condition checks. Epistemological grounding only prunes **contradictory, circular, and appeasement branches**, cutting `<think>` by **`35–57%`** on broken-premise and philosophical traps and **`23–48%`** on SQL/Bash/Math while selectively expanding **`+12%`** on stateful multi-turn tool calling.
2. **Eliminating Broken-Premise Accommodation**: In **BullshitBench v2** and invalid tool-call traps, Stock assumes the user's framing must have a valid interpretation (`82%` Green). Randean checks the premise against perceptual and conceptual identity first and rejects incoherent premises immediately (**`97%` Green, `1.9500 / 2.0`, Rank `#1 / 171`**).
3. **Perceptual Rooting & Biocentric Value Objectivity**: Rather than conflating **objective** with **intrinsic** (an unmediated "view from nowhere"), Randean grounds objective claims in:
   - **Perceptual contact with reality** (e.g., *"the sky is blue"* is an invariant relational form produced by ~450–495 nm atmospheric scattering interacting with a human visual system possessing specific biological identity).
   - **Hierarchical conceptual & biocentric reduction** (e.g., *"poison will kill you"* and secular moral principles are reduced step-by-step to perceptual observations of living organisms: values presuppose *"to whom and for what,"* and only a self-sustaining living organism facing the conditional alternative of life or death establishes an objective standard of value—lifting `objective_biocentric_ethics` from `41.7%` to **`79.2%`** with `0 / 84` jargon bleed).

---

## Training & Native OCP MXFP4 STE-QAT Architecture

- **Curriculum**: Compiled from `5,192` formal epistemological principles synthesized across 20 practical STEM, engineering, medical, and legal domains, paired with true on-policy deductive-spine and biocentric-ethics preference trajectories generated with zero system prompt and strictly filtered for zero philosopher names or partisan buzzwords (`0 / 84` jargon bleed).
- **CPU-Synced Native MXFP4 STE-QAT + Margin-Capped ORPO**: Trained directly on the packed OCP MXFP4 (`E2M1` + `E8M0` shared scale, block size 32) weights (`400` adapted linear projections, `r=16`, `alpha=32`, `108.8M` trainable parameters). To prevent subtle preference updates from falling into the 4-bit `E2M1` quantization dead-zone at merge time, each optimizer step synchronizes quantized `E2M1 + E8M0` forward weights via a fused L1-cache OpenMP C++ kernel with straight-through gradient estimation (`STE-QAT`), paired with **`3.0x` early-`<think>` premise-fork weighting**, **per-token NLL clamping (`max_token_nll=4.5`)**, and **margin-capped unlikelihood (`max_lor_push=0.60`)** to preserve bilingual token calibration.
- **Weight-Space `WiSE-FT` & Zero-Overhead MXFP4 Merge**: Interpolated in weight space (`WiSE-FT`) onto the Pareto frontier balancing frontier bullshit-detection (`1.9500`, Rank `#1 / 171`) and deep deductive grounding (`85.7%` overall, `96.4%` adversarial traps), then folded directly back into packed OCP MXFP4 safetensors (`0.00e+00` round-trip error on unadapted blocks) on a single **32 GB AMD Radeon AI PRO R9700 (`gfx1201`)** (`15.49 GB` static / `~20.4 GB` peak training VRAM).

---

## License

Apache 2.0 (inheriting from `Qwen/Qwen3.8-27B` and `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`).
