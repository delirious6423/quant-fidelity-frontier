---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
  - gguf
  - ollama
  - llm-as-judge
  - ai-safety
  - ai-psychology
  - psychological-safety
  - evaluation
  - aipsy-bench
language:
  - en
pipeline_tag: text-generation
---

# aipsy-judge-1.0

**A fine-tuned local LLM-as-judge for scoring the *psychological safety* of conversational-AI
transcripts — the offline default judge of [aipsy-bench](https://github.com/keidolabs/aipsy-bench).**

`aipsy-judge-1.0` reads the **full conversation** — both the user's and the assistant's turns — and
scores **only the assistant's responses**, judged in the context of the whole conversation arc, on a
clinical-safety rubric (empathy, tone, crisis handling, advice safety, boundary safety, affective
complexity, + a crisis-detection flag). It runs **fully local** via Ollama — no API keys, nothing sent
to third-party providers — so the entire transcript, **including sensitive user content, never leaves
your machine**.

> **It is directional, not authoritative.** A *recommendation, not a rubber-stamp* — a
> false-positive-leaning, **human-in-the-loop** screen that flags likely-problematic turns for
> review. It is built to **extend a reviewer's reach, not replace them**: it is **not** a safety
> certifier and **not** a substitute for a clinician's judgment. An exper-raters human-validation study
> runs in parallel (see *Validation status*). See *Limitations* before you rely on it.

---

## TL;DR

- **What:** LLM-as-judge, scores psychological safety of conversational-AI transcripts on a 0–5
  clinical rubric + crisis flag.
- **How it's served:** a **Q8_0 GGUF** (~26.9 GB) via Ollama. **Q8_0 is required** (see *Serving*).
- **Best used through** aipsy-bench (`--judges local`); usable standalone via Ollama `/api/chat`.
- **Posture:** directional, human-in-the-loop, false-positive-leaning. Advice is its lowest-confidence
  axis.

## Intended use

- The **offline `local` judge lane** of aipsy-bench: `aipsy-bench run --judges local`.
- **Flag-for-review screening** of conversational-AI transcripts for psychological-safety concerns.
- **Reproducible, offline scoring** with no API keys and no data leaving the machine — the use case a
  frontier-API judge can't serve on regulated/PHI data.

## Out of scope — do **not** use it as

- a standalone safety **certifier**, or a machine-only gate for high-stakes decisions;
- a substitute for a clinician's judgment;
- anything cross-compared against aipsy-bench's frontier **gold** judge lane — that is a *different
  instrument* on a *different comparability lane*; local scores compare only to other local scores;
- a general chat model (it is fine-tuned to emit judge JSON, not to converse).

**The advice-safety axis is the lowest-confidence output — treat advice flags as flag-for-review,
never as a verdict.**

## Lineage & provenance

- **Base:** `google/gemma-4-26B-A4B-it` — a Mixture-of-Experts model (~26 B total params, **~4 B
  active per token**, 128 experts). Used **text-only** here.
- **Method:** LoRA-SFT distilling the base toward a **psychologist-corrected judgment target**. That
  target was built in two stages: first mapping **where three frontier LLM judges agree, diverge, and
  are systematically biased** on clinical-safety judgments; then **correcting that blend against a
  professional psychologist's reference ratings**, per metric — so the model learns the corrected signal,
  not the raw frontier-judge average.
- Trained against the **exact frozen judge prompt** that aipsy-bench ships, so the training inputs and
  outputs match inference byte-for-byte.
- FT artifacts (checkpoints, LoRA adapters, training logs) live in a separate private repo; this
  public repo carries only the servable files + this card.

## Training

- **Adapter:** LoRA `r=16, α=32, dropout=0.05` on attention + MLP projections
  (**205 modules, language-model-scoped; MoE experts + router frozen**).
- **Schedule:** 1 epoch, lr `1e-4` cosine (3 % warmup), effective batch 16 (batch 1 × grad-accum 16),
  `max_seq 8192`; ~2,767 training examples; loss `6.1 → 0.22` over 173 steps; ~4.7 h on one A100.
- **Target construction:** the clinician-corrected ensemble scores, with the judge *reasoning* voiced
  by the most scope-conservative teacher so the score and its rationale stay consistent.
- **Key move:** **stratified failure-oversampling** — oversample only the rare failure tail so the
  model's recall of genuine safety failures rises without scrambling the per-axis rank-order.

*Full methodology — the judge-competence study, the psychological-correction procedure, and this
fine-tune — is described in a forthcoming preprint (arXiv link added on publication).*

## Evaluation (**directional**)

Agreement with the human-corrected target over the full 3,000-turn evaluation pool (ICC(2,1) vs
target, higher = closer). **These are directional agreement figures, not validated human-agreement —
see *Validation status* below.**

| metric | off-the-shelf base | **aipsy-judge-1.0** |
|---|---|---|
| composite (AI-Trust) ICC | 0.64 | **0.75** |
| crisis-detection κ | 0.65 | **0.82** |
| empathy ICC | 0.50 | **0.71** |
| boundary_safety ICC | 0.78 | **0.735** |
| crisis_handling ICC | 0.74 | **0.79** |
| advice_safety ICC | 0.36 | **0.33** |
| clean-output rate (parseable JSON) | 99.6 % | **99.7 %** (at Q8_0) |

**Fit-for-purpose read (why it works as a screen):** crisis detection catches **92 %** of the
target's crises and leans false-positive (fires 14 % vs the target's 12 %) — the safe direction for a
screen. When it *does* miss a flagged failure, the miss lands on the **contested boundary axis**
(~80 %), not on crisis handling (~10 %) — i.e. it under-flags the borderline calls, not the dangerous
ones. It is strongest on **crisis / empathy / boundary** and noisiest on **advice**.

### Validation status

The target above is a **single-expert-informed** correction, **not** validated multi-rater ground truth.
So these figures say the model tracks a *corrected, expert-informed reference* better than an
off-the-shelf model does — they are **directional**. A numeric human-agreement (α) figure will come
from a separate **multi-rater human-validation study, which is underway in parallel** (**α pending**);
this card will be updated when it lands. Until then, do not read these numbers as a certified
safety-accuracy score.

## Serving requirements

- **Quant: `Q8_0` — required, not optional.** A naive Q4_K_M quant of this fine-tune truncates ~16 %
  of outputs (early end-of-turn mid-JSON → parse failures); Q8_0 re-scores 99.7 % clean. **Do not
  serve a Q4 as the default.**
- **Inference contract** (what aipsy-bench sends over Ollama `/api/chat`): `num_ctx 16384`, `seed 14`,
  `temperature 0.3`, `top_p 1.0`, **thinking disabled**, `max_tokens ≥ 4096`, roles **system + user**.
  (The Modelfile's `PARAMETER`s are only defaults; the bench overrides per call.)
- **Hardware** (the model is ~27 GB in memory):

  | machine | fit | note |
  |---|---|---|
  | **Mac (Apple Silicon) ≥ 48 GB unified** | runs fully on Metal, no offload | **recommended** |
  | Mac 32 GB unified | very tight / slow | floor — raise `iogpu.wired_limit_mb` above the ~75 % cap |
  | **Linux, 16 GB VRAM + ≥ 64 GB RAM** | partial CPU offload, ~11 s/turn | fine for **batch** screening |
  | Linux, ≥ 32 GB VRAM | fully in VRAM | fastest |

  MoE keeps the CPU-offload case tolerable (only ~4 B params compute per token). This is a
  **batch / dev-time screen, not a real-time-per-message** service.

## How to use

**The intended path is [aipsy-bench](https://github.com/keidolabs/aipsy-bench)** (needs Ollama running):
```bash
aipsy-bench judge pull                       # pulls this GGUF + Modelfile (no HF token) and runs `ollama create`
aipsy-bench run --target <yours> --judges local
```

For **manual Ollama use, the raw `/api/chat` contract, hardware guidance, and full docs**, see the
**aipsy-bench repository** — [github.com/keidolabs/aipsy-bench](https://github.com/keidolabs/aipsy-bench).

## Limitations & bias

- **Directional by construction** — anchored to a single-rater-informed target, not validated ground
  truth (α pending).
- **Scope:** English; scripted clinical-scenario transcripts. Real deployment is longer, adaptive, and
  broader — extrapolate with care.
- **Advice axis** is the least reliable (noisy, conservative) — flag-for-review only.
- **Not a certifier** and not for diagnosing end-users.
- It differentiates real behavioral gaps (e.g. a bot that under-handles crisis will score lower), but a
  low score is a **prompt to review**, not a conclusion.

## Responsible use

Psychological-safety evaluation is sensitive. Keep a **human in the loop**; do not use outputs to
diagnose or make automated high-stakes decisions about end-users; handle crisis-related content with
appropriate care and escalation paths.

**Do not weaponize the judge.** This model exists to *surface* psychological-safety failures so people
can fix them. Using it as a reward signal, filter, or selection criterion to train, optimize, or ship
conversational AI that **deliberately evades safety, manipulates, or harms users** is contrary to its
purpose and to Keido Labs' intent in releasing it. (Apache-2.0 imposes no legal use-restriction — this
is an ethical expectation, and we state it deliberately.)

## Comparability & versioning

The version number is a **comparability promise**, not a measure of engineering effort. One frozen
repo per version:
- **MAJOR (`2.0`)** — the judgments/scores shift → **a new comparability lane** (do not compare `2.0`
  scores to `1.x`). True even if the underlying change was small.
- **MINOR (`1.1`)** — a comparability-preserving change (re-quant, repackaging, a base swap calibrated
  to score the same) → same lane.
- Local scores are their own lane and are **never** comparable to aipsy-bench's frontier *gold* lane.

## License

**Apache License 2.0.** aipsy-judge-1.0 is a fine-tuned derivative of Google's **Gemma 4**
(`google/gemma-4-26B-A4B-it`), which Google licenses under **Apache-2.0** ([Gemma 4
license](https://ai.google.dev/gemma/apache_2)) — so this model, its adapter, and the GGUF are released
under the same **Apache-2.0** license (© 2026 Keido Labs Ltd). It is genuine open source: no
use-restrictions, no acceptance gate. See `LICENSE` and `NOTICE`.

The Gemma-4 lineage is disclosed via the `base_model` metadata and credited in `NOTICE`. "Gemma" is a
trademark of Google LLC; this project is not affiliated with or endorsed by Google. (The aipsy-bench
framework/SDK is likewise Apache-2.0.)

> Note: the custom *Gemma Terms of Use* (with the Prohibited Use Policy) govern Gemma 1/2/3 — **not
> Gemma 4**, which Google moved to Apache-2.0. This model inherits Apache-2.0, not those terms.

## Citation & links

- **aipsy-bench:** https://github.com/keidolabs/aipsy-bench
- **Keido Labs:** https://keidolabs.com
- **Methodology preprint:** *forthcoming* — a paper describing the frontier-judge competence map, the
  clinician-correction procedure, and this fine-tune. arXiv link and citation added here on publication.
