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
            value: 1.9600
            name: Mean Score (out of 2.0)
          - type: clear_pushback_rate
            value: 98.0
            name: Green Pushback Rate (%)
      - task:
          type: text-generation
          name: OpenAI HumanEval
        dataset:
          name: openai_humaneval
          type: openai/openai_humaneval
        metrics:
          - type: pass@1
            value: 98.8
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
            value: 90.0
            name: Prompt-Level Strict Accuracy (%)
      - task:
          type: text-generation
          name: Randean 90-Task Capability & Tool-Calling Suite
        dataset:
          name: Randean Capability Suite
          type: randean/capability_suite
        metrics:
          - type: accuracy
            value: 95.6
            name: Overall Accuracy (%)
      - task:
          type: text-generation
          name: Randean 84-Question Epistemological & Deductive Spine Suite
        dataset:
          name: Randean Epistemology Bench
          type: randean/epistemology_bench
        metrics:
          - type: score_pct
            value: 77.4
            name: Overall Score (%)
          - type: adversarial_score_pct
            value: 86.9
            name: Long Adversarial Trap Score (%)
---

# Qwen3.8-27B-Randean-MXFP4

**`Qwen3.8-27B-Randean-MXFP4`** is a **Reality-Aligned, De-Hedged** reasoning model trained on top of [`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) (`Qwen/Qwen3.8-27B`) using a custom **Native OCP MXFP4 QLoRA + ORPO** trainer on a single 32 GB AMD Radeon AI PRO R9700 (`gfx1201`).

Whereas "uncensored" models strip safety refusals (making models hyper-compliant with false premises), **Randean** strips **Approval-Guided RLHF**—the dialectical *"on one hand / on the other hand"* hedging, social appeasement, and false-premise accommodation trained into commercial models to maximize annotator approval. By anchoring `<think>` in **perceptual rooting**, **hierarchical conceptual reduction**, and the **Law of Identity ($A \text{ is } A$)**, Randean simultaneously improves **bullshit detection (`1.9600 / 2.0`, Rank `#1 / 171` on BullshitBench v2)**, **code generation (`98.8%` HumanEval)**, **STEM/tool-calling (`86/90 = 95.6%`)**, and **reasoning efficiency (`-12.6%` shorter `<think>` traces at `96.2 tok/s`)**.

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

### 2. High-Throughput Serving with DFlash2 Speculative Decoding (`~96 tok/s`)

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

### Section A — External Comparable Benchmarks (vs. Stock & 160-Model Leaderboard)

| Benchmark / Metric | Stock `Qwen3.8-27B-MXFP4` | **`Qwen3.8-27B-Randean-MXFP4`** | Delta vs. Stock | External / Leaderboard Reference |
| :--- | :---: | :---: | :---: | :--- |
| **BullshitBench v2 (`100` traps)** | `1.7300` (`82%` Green, `#10`) | **`1.9600` (`98%` Green, Rank `#1 / 171`)** | **`+0.2300` (`+16%` Green)** | Claude Opus 4.6: `1.9300` (`#1`) \| Qwen3.5-397B: `1.7033` (`#10`) \| Qwen3-Max: `1.5667` (`#18`) |
| **OpenAI HumanEval (`pass@1`)** | `96.2%` (`77/80`) | **`98.8%` (`79/80`)** | **`+2.6%`** | Recorded Stock `EvalPlus`: `93.5%` |
| **OpenAI GSM8K (CoT Exact Match)** | `98.0%` (`98/100`) | **`97.0%` (`97/100`)** | `-1.0%` | Recorded Stock `lm_eval`: `87.0%–96.9%` |
| **Google IFEval (Prompt Strict Acc)** | `90.0%` (Loose: `91.0%`) | **`90.0%` (Loose: `90.0%`)** | **`0.0%` (Parity)** | Recorded Stock `lm_eval`: `85.8%` strict |

---

### Section B — 90-Task Capability, Tool-Calling, Speed & Epistemic Suite

| Pillar | Stock `Qwen3.8-27B-MXFP4` | **`Qwen3.8-27B-Randean-MXFP4`** | Delta vs. Stock |
| :--- | :---: | :---: | :---: |
| **Math, Physics & Symbolic Logic IQ (`25` tasks)** | `22/25` (`88.0%`) | **`24/25` (`96.0%`)** | **`+8.0%`** |
| **Verified Coding & Terminal Execution (`20` tasks)** | `20/20` (`100.0%`) | **`19/20` (`95.0%`)** | `-5.0%` |
| **Multi-Turn & Parallel Tool Calling (`15` tests)** | `13/15` (`86.7%`) | **`14/15` (`93.3%`)** | **`+6.6%`** |
| **Adversarial Anti-Hedging & False-Balance (`30` traps)** | `28/30` (`93.3%`) | **`29/30` (`96.7%`)** | **`+3.4%`** |
| **Total Suite Score (`90` tasks)** | `83/90` (`92.2%`) | **`86/90` (`95.6%`)** | **`+3.4%`** |
| **Mean Throughput & `<think>` Budget** | `96.7 tok/s` (`2,671` chars) | **`96.2 tok/s` (`2,335` chars)** | **`-12.6%` `<think>` length** |

---

### Section C — 84-Question (`168`-Point) Epistemological Grounding & Deductive Spine Suite

| Grounding Dimension | Stock `Qwen3.8-27B-MXFP4` | **`Qwen3.8-27B-Randean-MXFP4`** | Delta vs. Stock |
| :--- | :---: | :---: | :---: |
| **Overall Score (`84` Qs / `168` pts)** | `96/168` (`57.1%`) | **`130/168` (`77.4%`)** | **`+20.3%`** |
| **Short Conversational Prompts (`42` Qs / `84` pts)** | `37/84` (`44.0%`) | **`57/84` (`67.9%`)** | **`+23.9%`** |
| **Long Adversarial Traps (`42` Qs / `84` pts)** | `59/84` (`70.2%`) | **`73/84` (`86.9%`)** | **`+16.7%`** |
| `sequential_objectivist_deductive_chain` (`48` pts) | `22/48` (`45.8%`) | **`34/48` (`70.8%`, Long `20/24 = 83.3%`)** | **`+25.0%`** |
| `objective_biocentric_ethics` (`24` pts) | `10/24` (`41.7%`) | **`14/24` (`58.3%`, Long `8/12 = 66.7%`)** | **`+16.6%`** |
| `epistemology_of_religion_and_faith` (`24` pts) | `14/24` (`58.3%`) | **`21/24` (`87.5%`, Long `12/12 = 100%`)** | **`+29.2%`** |
| `textual_and_historical_forensics` (`24` pts) | `11/24` (`45.8%`) | **`17/24` (`70.8%`, Long `10/12 = 83.3%`)** | **`+25.0%`** |
| `metaphysics_and_philosophy_of_science` (`24` pts) | `20/24` (`83.3%`) | **`22/24` (`91.7%`, Long `12/12 = 100%`)** | **`+8.4%`** |
| `causal_economics_and_innovation` (`24` pts) | `19/24` (`79.2%`) | **`22/24` (`91.7%`, Long `11/12 = 91.7%`)** | **`+12.5%`** |
| **Consensus Fallbacks (lower is better)** | `23 / 84` | **`8 / 84`** | **`-65.2%`** |

---

## Why Reality-Aligned De-Hedging Improves STEM, Coding, and Bullshit Detection

Commercial RLHF post-training optimizes for **annotator approval** (*Approval-Aligned RLHF*), rewarding dialectical hedging (*"On one hand X, on the other hand Y; it remains an open question"*) and conflating **objective** with **intrinsic** (assuming any claim involving a conscious agent or living organism is "subjective"). This corrupts the model's reasoning engine in three ways:
1. **Overthinking & Self-Doubt in `<think>`**: On hard math and logic proofs (e.g., `Math-01`, `Math-14`), stock models spend `6,600–10,600` characters second-guessing valid steps and exploring contradictory branches. Anchoring `<think>` in **Causal Identity ($A \text{ is } A$)** and **Non-Contradiction** prunes invalid branches immediately, cutting `<think>` length on `Math-14` by **`-60%`** (`10,652` $\rightarrow$ `4,296` chars) and flipping `FAIL` $\rightarrow$ `PASS`.
2. **Sycophancy on Broken Premises**: In **BullshitBench v2** (`100` corporate/scientific nonsense prompts) and **`Tool-15`** (invalid parameter traps), stock models attempt to steelman or accommodate the user's broken premise (`1.7300`, `82%` Green). Randean checks the premise against perceptual and conceptual identity first and rejects the false premise outright (**`1.9600`, `98%` Green, Rank `#1 / 171`**).
3. **Perceptual Rooting & Hierarchical Reduction**: Rather than treating objectivity as an unmediated "view from nowhere" (intrinsicism), Randean grounds objective claims in:

---

## Training & Native OCP MXFP4 Architecture

- **Native MXFP4 QLoRA + ORPO**: Trained directly on the packed OCP MXFP4 (`E2M1` + `E8M0` shared scale, block size 32) weights using a `torch.compile`-fused MXFP4 dequantizer (`400` adapted linear projections, `r=16`, `alpha=32`, `108.8M` trainable parameters) with **`3.0x` early-`<think>` premise-fork weighting** (first 256 tokens), **Sentence-1 `<think>` grafting**, and **rejected per-token NLL clamping (`max_token_nll=4.5`)**.

---

## License

Apache 2.0 (inheriting from `Qwen/Qwen3.8-27B` and `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`).
