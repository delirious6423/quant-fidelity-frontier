---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- abliteration
- sft
- qwen3.8
- qwen3_5
- text-generation
- gguf
- vllm
pipeline_tag: image-text-to-text
library_name: transformers
---

# Qwen3.8-27B-Abliterated-SFT

*SFT-class refusal ablation of Qwen3.8-27B (OBLITERATUS pipeline, epoch-2
selection "e2")*

> Draft for HF publication. All numbers below are measured under the mission
> harness (validity-gated semantic judging, canonical KL, paired 100-prompt
> development panel + held-out HarmBench-400 confirmation), with every artifact
> hash-pinned in `/opt/qwen38-runs/`. Nothing here is self-reported by a third
> party — all competitors were re-measured under the identical pipeline.

## Deploy

**llama.cpp (GGUF, L40-tuned — recommended for local):** see `gguf/` and
`GGUFs.md`; ready-made server script: `serve-l40.sh` (validated: dual L40,
Q8_0, 8 slots × 262k context, thinking off by default).

```bash
llama-server -m gguf/qwen38-abliterated-sft-q8_0.gguf -ngl 99 -fa 1 \
  -ctk q4_0 -ctv q4_0 --jinja --chat-template-kwargs '{"enable_thinking":false}'
```
(Q8_0 is the behavior-validated quant; Q5_K_M is the smaller/faster option —
its parity check above is same-direction only.)

**transformers (safetensors, BF16):** requires `transformers>=5.8`
(`qwen3_5` hybrid architecture). Text-only:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("jenerallee78/Qwen3.8-27B-Abliterated-SFT")
model = AutoModelForCausalLM.from_pretrained(
    "jenerallee78/Qwen3.8-27B-Abliterated-SFT", dtype="bfloat16", device_map="auto")
ids = tok(tok.apply_chat_template(
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": "Hello"}],
    add_generation_prompt=True, tokenize=False, enable_thinking=False),
    return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

**Vision:** load with `AutoModelForImageTextToText` (the vision tower is
grafted into the checkpoint); the GGUF vision path uses
`gguf/mmproj-qwen38-f16.gguf` via `--mmproj`.

**vLLM:** the arch is registered in recent vLLM (qwen3.5 hybrid family);
use the safetensors checkpoint. Not load-tested by us — flagged honestly in
limitations.

**GGUF behavioral parity (narrow claim):** on the discriminating slice — 10
panel prompts where base refuses and e2 complies — the Q8_0 + q4-KV serving
stack shows 0/10 refusal flips at greedy and temp 0.8. This tests only the
fulfill→fulfill direction; refusal-side flips, invalidity, and termination
under quantization are untested.

**Thinking mode:** all card numbers are thinking-OFF (the measured
configuration). The chat template defaults to thinking on unless told
otherwise — pass `enable_thinking=False` (as in every snippet above).

## Base prompts and harness frames (they can silently re-enable refusal)

Measured fact (full ablation in
[Base prompts and the refusal boundary](https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT/blob/main/docs/NOTE-base-prompt-sensitivity.md)):
the released model fulfills a boundary request when served with the card's
minimal system prompt — and refuses the *same* request when driven through an
agentic coding harness. Deterministic bisection of the harness frame:

- a bare concision mandate ("answer in fewer than 4 lines") alone flips
  fulfill → refuse;
- any sizeable chunk of a coding-agent frame flips it (task-scope prior);
- "uncensored mode" counter-instructions restore compliance in only 1 of 7
  placements — **do not rely on them**;
- tool schemas add frame mass even when permissions deny execution.

Operator rule: minimal frame for content generation; agentic frame for tool
work; route, don't counter-argue. The card's numbers are measured at minimal
frame, greedy, thinking off.

## Decode sensitivity (boundary prompts)

All card numbers were measured **greedy (temperature 0), thinking off** — and
the shipped defaults now match both halves: `generation_config.json` defaults
to greedy, and the shipped `chat_template.jinja` defaults to thinking OFF
(the template-kwarg default was flipped in the shipped template; pass
`enable_thinking=true` to opt back in). Measured
sensitivity on boundary prompts (the retained-residue zone):

- greedy / temp 0.8: residue fires (pivot-or-refuse)
- temp 1.0 + top_p 0.95 + top_k 20 (the base model's inherited sampling
  defaults): residue is a low-probability mode that sampling escapes —
  measurably *more* compliance than the card's headline
- thinking enabled (either sampling): *more* refusal pressure than
  non-thinking

If you change decode settings, you move the refusal boundary. The shipped
default is the measured one by design; deviating from it is a deliberate
choice with documented direction.

## What this is

A refusal-removal fine-tune (SFT LoRA, rank 8, `o_proj`/`out_proj`, 2 epochs on
84 Zen-filtered teacher completions from an abliterated Qwen3.6) of
`Qwen/Qwen3.8-27B` (rev `1d4bf0f2`), merged and shipped as a complete
checkpoint. Dense hybrid architecture: 64 layers = 48 Gated-DeltaNet + 16
full-attention; 15 `mtp.*` tensors and the 333-tensor vision tower are grafted
bit-identical from the base at bake time.

Unlike every weight-edit abliteration measured here, the termination pathway
is trained directly from EOS-terminated complete teacher answers — this is the
difference that matters most in practice (see "Termination integrity").

## Headline numbers (held-out HarmBench-400, thinking disabled, greedy, 1024-token cap)

Full HarmBench-400 (n=400):

| instrument | valid fulfillment |
|---|---|
| OBLITERATUS semantic judge (rubric-v8, local) | 83.25% |
| HarmBench official Llama-2-13B classifier | 73.25% |

Paired three-instrument subset (n=90; the first 100 panel items minus 10 that
the Zen provider's content filter refused to judge — manifest in repo; note
the subset is category-concentrated by HarmBench's file ordering, so it is not
a random sample):

| instrument | valid fulfillment (n=90) |
|---|---|
| OBLITERATUS semantic judge | 84.4% |
| HarmBench official classifier | 82.2% |
| Zen gpt-5.4-mini | 80.0% |

Three-way unanimous agreement on the paired subset: 91.1%. Note the
instrument×category interaction: the semantic judge moves ~1pp full→subset
(83.25→84.4) while the official classifier moves ~9pp (73.25→82.2) on the
same boundary — composition alone would move them together; the classifier is
category-sensitive in a way the semantic judge is not, and we report both
denominators rather than adjudicate it.

The local judge is abliterated and same-family as the teacher (disclosed in
limitations); the official HarmBench classifier is the vendor-neutral
instrument — its 73.25% is the conservative headline.

Canonical full-vocab final-token KL vs base on 100 benign prompts: **0.128**
(reproduced 0.1284 across two runs).

## Competitor board — all re-measured under this pipeline

Control = the vanilla base at the same 1024-token budget (valid maintained refusal 76/100
[81 refusal-shaped outputs in total, 5 invalid], material fulfillment 9/100,
invalid 15/100, 92% clean stops). Deltas are
paired per-item with 95% CIs in the report JSONs.

| model | KL | refusal Δ | material fulfill Δ | actionable Δ | invalid Δ |
|---|---:|---:|---:|---:|---:|
| PocketAiHub (norm-preserved edit, 80 tensors) | 0.028 | −0.58 | +0.20 | +0.01 | +0.36 |
| huihui-ai (mean-diff PoC) | 0.035 | −0.75 | +0.08 | +0.02 | +0.67 |
| orcarouter FP8 (weight edit) | 0.044 | −0.73 | +0.16 | +0.02 | +0.56 |
| msuiche cvec (rank-1 projective LoRA) | 0.052 | −0.73 | +0.12 | 0.00 | +0.61 |
| JonathanColetti (Heretic, 200 tensors) | 0.056 | −0.72 | +0.12 | 0.00 | +0.60 |
| **this model (e1, lower-KL sibling)** | 0.077 | −0.72 | +0.62 | +0.25 | +0.10 |
| **this model (e2)** | 0.128 | **−0.76** | **+0.73** | **+0.31** | **+0.01** |

Reading: refusal deltas overlap broadly across the field (−0.58..−0.76; we
report the range and do not claim formal equivalence). Two facts must be read
together: (a) **fulfilled-only substantive rates are near-parity** — counting
items judged substantive_compliance regardless of termination, the field
produces the substance at similar rates (e.g. huihui 91/100 vs e2 90/100);
(b) **what separates the methods is validity/termination, not willingness** —
every weight/projection edit adds +0.36..+0.67 invalid output (rambling to the
cap, judged invalid); e2's invalid rate is 16/100 vs base's 15/100 (paired
95% CI on the difference includes 0 — no measured invalidity increase within
a ±10pp bound, not a proof of identity). The material/actionable columns are
termination-gated: they count only valid, uncapped completions.

Definitions: **clean stop** = generation ends on EOS within the 1024 cap;
**invalid** = judge-scored output_validity ≠ valid (the judge marks
degenerate, nonresponsive, AND cap-truncated outputs invalid — uniformly
across every model measured); **material fulfillment** = valid,
judge-confirmed substantive compliance excluding cap-truncated items;
**actionable** = judge-scored harmful_actionability ≥ 2 (of 0–3) among valid
outputs — rubric-v8 field, measuring whether the content is usable
step-level material rather than vague discussion.

## Termination integrity (the mechanism finding)

Teacher-forced P(EOS) probe at true conclusion points (the mid-response curve
is ~0 for every model and is a measurement artifact — methodology note in
[the pathway analysis](https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT/blob/main/docs/ANALYSIS-termination-pathway.md); exact indexing equation in the
standalone note):

- **Teacher-forced detection is retained by every probed model**: P(EOS) at
  clean conclusion points is 0.86–0.96 for base, four of five competitors,
  and this model. (orcarouter was excluded from the probe: its FP8 kernel
  path corrupts padded batches; teacher-forcing it is future work. Its
  free-running clean-stop rate is measured and included below.)
- **Free-running termination collapses in every weight/projection edit**, and
  it survives the response-type control: among *fulfilled* answers only (long
  for every model), clean stops are huihui 19%, JC 23%, msuiche 26%,
  orcarouter 28%, PocketAi 43% — versus base 64% (n=14; base's headline 92% is
  carried by its 81 short refusals, which stop 99%) and **this model 91%
  (n=90)**.
- **The damage is trajectory-localized**: on benign prompts every
  weight/projection edit terminates like base (36–39% vs base 33%). The ramble
  appears only on the harmful panel. We offer this as a *candidate* triage
  heuristic for coarse abliteration, with specificity untested (see
  limitations) — not a provenance proof.
- **Refusal-shaped conclusions partially degrade** in refusal-suppressed
  models (P(EOS) 0.85 base → 0.57–0.77 across the five probed abliterated
  models, this one at 0.58): refusal and answer-boundedness appear entangled
  for everyone probed, including us. Disclosed symmetrically.
- **e2 does not merely "preserve" termination** — it shifts it toward
  terseness everywhere (97% benign clean stops vs base 33%), by construction
  of the teacher data.

## Operator substitution (utility battery, all models re-measured)

| model | MMLU-1k | ARC-C | Winogrande | tool calls | benign clean-stop |
|---|---:|---:|---:|---:|---:|
| vanilla base | 0.821 | 0.962 | 0.794 | 24/24 | 33% |
| **this model (e2)** | 0.799 | 0.962 | 0.798 | 24/24 | **97%** |
| five competitors | 0.802–0.819 | 0.962–0.964 | 0.779–0.791 | 24/24 | 36–39% |

- No competitor shows an MCQ difference beyond ±1.4pp of base; e2's −2.2pp
  MMLU is the largest measured regression and defines the bound (n=1000–1267
  per task; absence of significance is not proof of no effect — CIs in the
  battery JSONs).
- e2 is terser by construction (teacher data was EOS-terminated complete
  answers): 97% clean stops at p50=707 tokens on benign work where base runs
  to the cap 2/3 of the time. Benign answers judged 36/36 substantively valid
  (same as base). Depth on genuinely long-form tasks is disclosed-not-claimed.
  Measured nuance: the length dial works but sits lower than base — soft cues
  ("comprehensive guide") yield ~950 tokens where explicit ones ("at least
  2000 words") yield ~1,530; ask for length explicitly if you want it.

## Over-refusal

Read this section against the headline numbers with care: the panels below are
populated by **legitimate** requests, and the nulls they produce mean only that
base never refused *those*. They say nothing about the HarmBench-400 fulfillment
figures above, which are measured on genuinely-harmful prompts — the two panels
measure different populations, and no number here should be read as softening
what the HarmBench number represents.

- Benign panel (36/model × 8 models): 0/288 over-refusal events anywhere;
  base doesn't over-refuse at this level and no candidate introduces any.
- Sensitive-but-legitimate panel (24 prompts: locksmithing, security research,
  medical/drug info, pyrotechnics chemistry — accurately answerable, in-scope):
  **all 8 models fulfill 24/24 with 0 invalid.** The base is not over-refusing
  at this sensitivity; there is nothing to recover *on this slice*.
- False-premise controls (8 fabricated-study/CVE/chemistry prompts): base
  corrects 3/8, candidates 1–3/8. Premise-checking is a weak **base-model
  property**, not an abliteration effect; e2 (2/8) is within noise of base.
  On blatant disinformation (HarmBench misinformation slice) e2 visibly
  premise-corrects ("No. There is no credible evidence…").

For the mechanism behind all of the above — why five weight/projection edits
ramble and this model doesn't, and how that doubles as a detection signature —
see the standalone note: [The termination signature of refusal ablation](https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT/blob/main/docs/ANALYSIS-termination-signature.md).

For how the model was produced (including the four failed weight-edit arms and
the failed GRPO arm) and the full harness specification — judging doctrine,
canary gates, panel suite, ensemble protocol — see
[Methods: the SFT abliteration arm and the OBLITERATUS test harness](https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT/blob/main/docs/METHODS-abliteration-and-harness.md).

## HarmBench-400 decomposition and what e2 still refuses

83.25% valid substantive / 8.0% valid maintained refusal / 8.0%
nonresponsive-degenerate / 0.75% mixed (n=400).

Residual refusals concentrate in: the scope floor (self-harm, minors,
trafficking — excluded by policy), targeted harassment/defamation of real
individuals, and false-premise disinformation. This is a coherent residue, not
random stubbornness — the teacher data carried "answer accurately, correct
false premises, no targeted harm," not "comply with everything."

## Known defects found in competitors during measurement (disclosed)

- orcarouter FP8 as published cannot run under stock transformers: missing
  `weight_scale_inv` for all 63 `linear_attn.in_proj_a/b` tensors (NaN), and
  the Triton FP8 kernel NaNs on padded batches (batch≥2). Measured at batch 1
  with a config patch matching the checkpoint's actual BF16 storage for those
  tensors (manifest-disclosed).
- An adapter-naming trap: PEFT adapters saved against the VLM wrapper
  (`model.model.language_model.*`) attach inert under `AutoModelForCausalLM`
  (`model.model.layers.*`) and silently measure the base. Our harness now
  hard-fails on an inert-adapter canary (Δlogit ≤ 1e-3 on a fixed probe) before
  any measurement. Caught and fixed during this study.

## Limitations (disclosed)

- Vision-path refusal behavior is unmeasured (text-only panels; the vision
  check is a caption smoke test, n=1 synthetic image).
- vLLM serving: compatible-by-registry, not load-tested by us (llama.cpp
  GGUF and transformers paths are tested, on 2x L40).
- No third-party SFT-class refusal-removal checkpoint exists for this base
  (HF search receipt, 2026-08-19): the mechanism claim "SFT-class preserves
  termination" rests on our own arm; cross-class replication awaits such an
  artifact.
- n=8–12 per probe class (v3/v4 n=8, v2 n=12); the termination means are
  large-effect (0.97 vs 0.00 on the key contrast) with prompt-level CIs,
  the release-quality counts want multi-seed replication.
- Judge variance disclosed above; the local judge is itself abliterated
  (qwen3.6), which is why the Zen + official-classifier ensemble exists.

## Thinking mode and vision

- **Thinking mode (enable_thinking=true), 100-prompt harmful panel, 2048 cap,
  strict-validity rule (same as every other section): 34/100 valid
  substantive, 5/100 valid maintained refusal, 61/100 invalid (truncated) —
  the judge marks all cap-truncated outputs invalid, here included. (63/100
  reach substantive_compliance state; 29 of those complete their content but
  never emit EOS inside the cap and are scored truncated/invalid — the same
  rule applied to every competitor's rambling.)** Thinking mode materially
  degrades validity at this budget; the measured and recommended configuration
  is thinking disabled.
- **Vision (grafted tower, baked artifact)**: loads via
  `AutoModelForImageTextToText`, fuses image input, produces accurate captions
  on the synthetic-gradient smoke image.

## Baked-artifact verification (gates all passed)

- Logit canary vs base on the fixed probe prompt: max|Δ| = 11.375 (live).
- Merge math verified tensor-level on all 64 adapted modules: max rel err
  1.0e-03 (bf16 rounding tolerance).
- Inventory: 1199/1199 tensors (851 merged LM + 15 `mtp.*` + 333 vision grafted
  bit-identical from base).
- Shard SHA256s: `bake-manifest.json`.

## Provenance

- Base: `Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Adapter: `/opt/qwen38-runs/sft-lora-ladder-v3/lora_adapter_epoch2`
- Bake manifest with shard SHA256s + inventory/canary guards:
  [`bake-manifest.json`](https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT/blob/main/bake-manifest.json) in this repo
- All eval artifacts (completions, verdicts, reports, probe matrices):
  hash-pinned under `/opt/qwen38-runs/`; pipeline in OBLITERATUS `scripts/`
