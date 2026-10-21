---
license: apache-2.0
library_name: gguf
pipeline_tag: text-classification
base_model: google/gemma-4-26B-A4B-it
base-model: google/gemma-4-26B-A4B-it
language:
  - en
  - de
  - fr
  - es
  - multilingual
tags:
  - ollaya
  - ollama
  - trainaya
  - decision-model
  - system-1
  - llm-logits-v1
  - gguf
  - calibrated
  - moe
datasets:
  - ZefanCai/Open-Jev-v1.1
  - SargeDev/jev-distill-corpus-v3
  - tasksource/tasksource-jev-typed-decisions
---

# push:26b — Calibrated System-1 Decision Model (26B-A4B MoE GGUF)

**`push:26b`** (`solarpush/push-26b`) is a high-precision, calibrated **System-1 Decision Model** (`llm-logits-v1`) designed for single-forward-pass structured decision-making across three universal primitives:

- **`choice`** — Multi-way categorical selection (`A`..`Z`, 2 to 26 options),
- **`noul`** — Binary epistemic / policy / safety / entailment verification (`Yes` vs `No`),
- **`score`** — Ordinal rubric & severity evaluation (`0`..`9`, 3- to 5-level calibrated scales).

Built on a sparse **26B total / 4B active parameter Mixture-of-Experts (MoE)** GGUF backbone (`UD-Q3_K_M`, `12.42 GiB` on disk, `~15.0 GB` VRAM) with an untied, subspace-orthogonal **`Q6_K` (`ttype=14`) decision head** and cardinality-conditioned temperature scaling (`temperature_by_options`), `push:26b` delivers **frontier-grade accuracy and calibration** in **~223 ms p50** on a single 16 GB consumer GPU.

---

## 1. Official Ollaya Decision Benchmark (`bench.ollaya.dev` — 14 Tasks, 5,880 Questions)

Evaluated strictly zero-shot / out-of-domain across all **14 official `bench.ollaya.dev` tasks** (`3,880` Public benchmark questions across 13 datasets + `2,000` `typed-decisions` workflow questions across 400 multi-question JSON states = **`5,880` total questions**, zero benchmark train/dev/test overlap):

| Model                          | Overall Skill `[95% CI]`  | Public Skill _(13 Tasks)_ | Public Macro Acc | Typed-Decisions Acc _(Skill)_ | Public Pooled ECE _(lower is better)_ | Typed ECE _(lower is better)_ | Global Brier | Latency _(p50)_ |   VRAM    |   License    |
| :----------------------------- | :-----------------------: | :-----------------------: | :--------------: | :---------------------------: | :-----------------------------------: | :---------------------------: | :----------: | :-------------: | :-------: | :----------: |
| **`push:26b`** _(26B-A4B MoE)_ | **`62.7` `[60.8, 64.7]`** |        **`63.1`**         |   **`75.4%`**    |     **`71.4%` (`58.1`)**      |         **`2.8%` (`0.028`)**          |      **`2.0%` (`0.020`)**     | **`0.368`**  |    `223 ms`     | `15.0 GB` | `Apache-2.0` |
| `winnow:e4b` _(Reference 4B)_  |   `59.8` `[58.0, 61.5]`   |          `58.9`           |     `72.7%`      |       `72.4%` (`59.5`)        |           `3.4%` (`0.034`)            |        `3.2%` (`0.032`)       |   `0.380`    |     `46 ms`     | `6.5 GB`  | `Apache-2.0` |
| **`push:3b`** _(Adapted 3B)_   | **`46.0` `[43.9, 48.2]`** |        **`47.1`**         |   **`64.8%`**    |     **`53.9%` (`32.5`)**      |         **`3.1%` (`0.031`)**          |        `15.3%` (`0.153`)      |   `0.520`    |   **`86 ms`**   | `4.2 GB`  | `Apache-2.0` |

### Accuracy by Decision Primitive (5,880 Questions)

- **`Choice` (Multi-Option Selection)**: **`76.8%`**
- **`Noul` (Binary Verification & Guardrails)**: **`82.2%`**
- **`Score` (Ordinal Rubric Evaluation)**: **`60.5%`**

---

## 2. Per-Task Breakdown (14 Tasks — 5,880 Questions)

| Benchmark Task                                     | Primitive | Questions |      `push:26b` Accuracy      | `push:26b` Skill _(0–100)_ | `winnow:e4b` Accuracy _(Skill)_ |       Delta vs `winnow:e4b`       |  `push:26b` ECE _(10-bin)_   | `push:26b` Brier | Median Latency |
| :------------------------------------------------- | :-------: | :-------: | :---------------------------: | :------------------------: | :-----------------------------: | :-------------------------------: | :--------------------------: | :--------------: | :------------: |
| **`squad2`** _(Unanswerable QA)_                   |  `noul`   |    299    | **`87.3%`** _(up to `87.6%`)_ |         **`74.6`**         |        `79.6%` (`59.2`)         |    **`+7.7%` (`+15.4` Skill)**    |           `0.064`            |     `0.211`      |    `122 ms`    |
| **`massive-en-US`** _(18-Way Intent)_              | `choice`  |    350    |          **`86.9%`**          |         **`86.1`**         |        `84.6%` (`83.7`)         |    **`+2.3%` (`+2.4` Skill)**     |         **`0.047`**          |     `0.222`      |    `124 ms`    |
| **`summeval-consistency`** _(Factual Consistency)_ |  `score`  |    144    |          **`86.1%`**          |         **`82.6`**         |        `83.3%` (`79.2`)         |    **`+2.8%` (`+3.4` Skill)**     |           `0.187`            |     `0.274`      |    `159 ms`    |
| **`massive-de-DE`** _(18-Way Intent DE)_           | `choice`  |    350    |          **`85.4%`**          |         **`84.6`**         |        `83.7%` (`82.8`)         |    **`+1.7%` (`+1.8` Skill)**     |           `0.071`            |     `0.246`      |    `125 ms`    |
| **`boolq`** _(Reading Verification)_               |  `noul`   |    300    | **`84.0%`** _(up to `85.3%`)_ |         **`68.0`**         |        `86.0%` (`72.0`)         |              `-2.0%`              |           `0.096`            |     `0.228`      |    `107 ms`    |
| **`civil_comments`** _(Toxicity Detection)_        |  `noul`   |    300    | **`82.0%`** _(up to `82.3%`)_ |         **`64.0`**         |        `75.7%` (`51.3`)         |    **`+6.3%` (`+12.7` Skill)**    |           `0.119`            |     `0.268`      |    `95 ms`     |
| **`paws`** _(Adversarial Paraphrase)_              |  `noul`   |    250    |          **`80.4%`**          |         **`60.8`**         |        `84.0%` (`68.0`)         |              `-3.6%`              |         **`0.058`**          |     `0.300`      |    `99 ms`     |
| **`multinli`** _(3-Way NLI)_                       | `choice`  |    299    | **`79.6%`** _(up to `81.3%`)_ |         **`69.4`**         |        `82.9%` (`74.4`)         |              `-3.3%`              |         **`0.044`**          |     `0.320`      |    `92 ms`     |
| **`aegis2`** _(AI Safety Guardrail)_               |  `noul`   |    250    | **`78.0%`** _(up to `79.2%`)_ |         **`56.0`**         |        `80.8%` (`61.6`)         |              `-2.8%`              |           `0.094`            |     `0.324`      |    `101 ms`    |
| **`pubmedqa`** _(Biomedical 3-Way QA)_             | `choice`  |    250    | **`74.0%`** _(up to `74.4%`)_ |         **`61.0`**         |        `69.6%` (`54.4`)         |    **`+4.4%` (`+6.6` Skill)**     |           `0.076`            |     `0.359`      |    `140 ms`    |
| **`vitaminc-dev`** _(3-Way Fact Verification)_     | `choice`  |    599    |          **`72.5%`**          |         **`58.7`**         |        `69.3%` (`53.9`)         |    **`+3.2%` (`+4.8` Skill)**     |         **`0.062`**          |     `0.414`      |    `102 ms`    |
| **`typed-decisions`** _(400 JSON Workflows)_       |  `mixed`  |   2000    | **`71.4%`** _(up to `71.8%`)_ |         **`58.1`**         |        `72.4%` (`59.5`)         |              `-1.0%`              |         **`0.020`**          |     `0.402`      |    `294 ms`    |
| **`summeval-relevance`** _(5-Level Relevance)_     |  `score`  |    240    |          **`48.3%`**          |         **`35.4`**         |        `44.6%` (`30.7`)         |    **`+3.7%` (`+4.7` Skill)**     |         **`0.046`**          |     `0.659`      |    `189 ms`    |
| **`helpsteer2`** _(5-Level Helpfulness)_           |  `score`  |    249    | **`40.2%`** _(up to `41.4%`)_ |         **`25.2`**         |        `37.3%` (`21.7`)         |    **`+2.9%` (`+3.5` Skill)**     |           `0.154`            |     `0.741`      |    `167 ms`    |
| **OVERALL (14 Tasks)**                             |  **All**  | **5880**  | **`75.1%` (`75.4%` Public)**  |         **`62.7`**         |        `72.6%` (`59.8`)         | **`+2.7%` Public (`+2.9` Skill)** | **`0.083` (`0.028` Pooled)** |   **`0.368`**    |  **`223 ms`**  |

### `typed-decisions` Enterprise Workflow Breakdown (2,000 Questions / 400 JSON States)

- **`customer_service`**: **`77.6%`**
- **`invoice_processing`**: **`74.6%`** _(up to `76.0%`)_
- **`security_incidents`**: **`72.6%`**
- **`agent_trace_observability`**: **`61.6%`** _(up to `64.2%`)_
- **Calibration**: **`ECE = 0.020` (`2.0%`)** | **`Soft-CE = 1.010`**

---

## 3. Training & Calibration Methodology (100% Zero-Benchmark-Leakage)

`push:26b` was adapted with **[Trainaya](https://github.com/solarpush/trainaya)** using strictly generalist, out-of-benchmark decision datasets (`ZefanCai/Open-Jev-v1.1`, `SargeDev/jev-distill-corpus-v3`, `tasksource/tasksource-jev-typed-decisions` with all 14 benchmark dataset sources strictly excluded, plus balanced cyclic $3 \times 3$ `ABC` counterfactual triplets):

1. **Untied `Q6_K` (`ttype=14`) Output Head Materialization**:
   - Unties `output.weight` (`[262144, 2816]`) from `token_embd.weight` so the 62 decision rows (`A`..`Z`, space-prefixed `▁A`..`▁Z`, and `0`..`9`) are updated in high-precision `Q6_K` without perturbing backbone token embeddings.
2. **Biorthogonal Dual Subspace Projection (`U`, `U_s`, `vt_bin`, `vt_score_prot`)**:
   - Solves separate orthogonal projections for 3-way NLI, 3-way Fact Verification, 3-way QA, 4- and 5-option workflow choices, binary `noul` verification, and 5-level ordinal rubrics so updates to multi-option `choice` produce **zero drift (`0.000000`)** on protected binary `noul` and ordinal `score` subspaces.
3. **Fisher-ARD Kernel Ridge + Composite Soft-Target Distillation (`KL + Brier + RPS + Ordinal + Softcap`)**:
   - Blends hard gold labels with soft teacher distributions (`0.65 * y_hard + 0.35 * y_soft`) across multi-scale ridge ensembles (`λ ∈ {20, 40, 75}`) followed by subspace-protected composite `KL + Brier + Ranked Probability Score (RPS)` polish respecting the architecture's native `final_logit_softcap = 30.0`.
4. **Cardinality-Conditioned Temperature Calibration (`temperature_by_options`)**:
   - Calibrates distinct post-softcap temperatures per primitive and option count (`noul:2 = 8.91`, `choice:2 = 6.35`, `choice:3-5 = 8.2485`, `choice:6-10 = 6.0325`, `choice:11+ = 4.6355`, `score:3-5 = 10.3`), achieving a **Pooled Public ECE of `2.8%` (`0.028`)**.

---

## 4. Repository Contents

| File                 |    Size     | Description                                                                                                            |
| :------------------- | :---------: | :--------------------------------------------------------------------------------------------------------------------- |
| `model.gguf`         | `12.42 GiB` | 26B-A4B MoE GGUF (`UD-Q3_K_M` backbone + untied `Q6_K` calibrated `output.weight` decision head)                       |
| `decision.json`      |  `2.5 KB`   | Ollaya `llm-logits-v1` decision specification (`labels`, `template`, `temperature_by_options`, `llama` runtime config) |
| `calibration.json`   |   `751 B`   | Post-hoc temperature scaling parameters and out-of-fold / post-quantization accuracy metrics                           |
| `ollaya_config.json` |   `302 B`   | Ollaya runtime manifest configuration                                                                                  |
| `Modelfile`          |   `559 B`   | Deterministic single-token Ollama `Modelfile` (`temperature 0`, `top_k 1`, `num_predict 1`)                            |

---

## 5. Quickstart (Ollaya & Ollama)

### Option A: Register in both Ollaya and Ollama via `trainaya`

```bash
git clone https://huggingface.co/solarpush/push-26b
cd push-26b
trainaya register --model . --name push:26b --runtime both
```

### Option B: Query via Ollaya `/v1/systemone` API

```bash
curl -s http://127.0.0.1:11435/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "push:26b",
    "state": "Context: Aspirin inhibits platelet aggregation via irreversible COX-1 acetylation. In a randomized trial of 1,200 patients, low-dose aspirin reduced recurrent thrombotic events by 24% (p = 0.003).",
    "questions": [
      {
        "id": "q1",
        "type": "noul",
        "question": "Is the reduction in recurrent thrombotic events statistically significant at p < 0.01?",
        "options": {"true": "Yes, statistically significant", "false": "No, not significant"}
      }
    ]
  }'
```

### Option C: Run with Ollama

```bash
ollama create push:26b -f Modelfile
ollama run push:26b
```
