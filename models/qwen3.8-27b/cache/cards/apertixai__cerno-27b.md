---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-classification
library_name: transformers
tags:
- cerno
- system-one
- typed-decisions
- calibrated-probabilities
- decision-index
- fp8
model-index:
- name: Cerno 27B
  results:
  - task:
      type: text-classification
      name: Decision Index 0.3 public suite (37 benchmarks, 140,178 requests)
    dataset:
      type: decision-index-0.3
      name: Decision Index 0.3 public suite
    metrics:
    - type: index
      name: public index, complete run
      value: 65.19
---

<div align="center">
  <img src="assets/cerno-light.jpg" alt="Cerno decision models" width="760"/>
  <h1>Cerno 27B</h1>
  <p><b>Jev-style System One decision model &middot; 27B &middot; one forward pass &middot; calibrated probabilities &middot; FP8</b></p>
  <p>
    <a href="https://huggingface.co/apertixai/cerno-27b">Cerno 27B</a> &middot;
    <a href="https://huggingface.co/apertixai/cerno-9b">Cerno 9B</a> &middot;
    <a href="https://huggingface.co/apertixai/cerno-4b">Cerno 4B</a> &middot;
    <a href="https://github.com/apertixsk/cerno-inference">Inference code</a> &middot;
    <a href="https://huggingface.co/datasets/apertixai/cerno-decision-index">Decision Index runs</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Decision%20Index%200.3-65.19-1f6feb" alt="Decision Index 0.3 public index 65.19"/>
    <img src="https://img.shields.io/badge/same--subset%20rank-2nd%20of%20145-2ea44f" alt="same-subset rank"/>
    <img src="https://img.shields.io/badge/weights-FP8%20W8A8-6f42c1" alt="FP8 weights"/>
    <img src="https://img.shields.io/badge/license-Apache%202.0-lightgrey" alt="Apache 2.0"/>
  </p>
</div>

## What it does

- **Input:** a *state* (any text or JSON) and a set of typed *questions* about it.
- **Question types:** multiple choice with explicit criteria, yes/no, scored scales. Up to 255 options per question.
- **Output:** a probability for every option of every question, in **one forward pass with no text generation**.
- **Calibrated:** a single shipped temperature (1.0) makes the probabilities honest; it changes no answer.
- **Fast:** 52 ms median per request on one H200, one request at a time.

On the Decision Index 0.3 public suite this model is second of 145 entrants on the like-for-like basis and above Jev 1.13 on every reporting basis.

## The Cerno family

Three sizes, one inference code path, state-of-the-art System One performance at each size on the Decision Index 0.3
public suite. The 27B is second of 145 entrants; the 9B and 4B lead every entrant of their declared size class.

| Model | Base | Weights | Temperature | Public index | Same-subset rank | Median latency |
|---|---|---|---:|---:|---|---:|
| <b>Cerno 27B</b> (this model) | Qwen3.8-27B | [`cerno-27b`](https://huggingface.co/apertixai/cerno-27b) | 1.0 | **65.19** | 2nd of 145 | 52 |
| Cerno 9B | Qwen3.5-9B | [`cerno-9b`](https://huggingface.co/apertixai/cerno-9b) | 1.539 | **58.59** | 14th · first at ≤10B | 30 |
| Cerno 4B | Qwen3.5-4B | [`cerno-4b`](https://huggingface.co/apertixai/cerno-4b) | 1.534 | **57.60** | 25th · first at ≤5B | 31 |

- *Public index*: the complete public suite (37 benchmarks, 140,178 requests) run with the maintainers' kit at commit 9eb2dbe
  through the public inference code and these FP8 weights.
- *Same-subset*: HLE dropped for every entrant, the like-for-like basis. "First at ≤10B / ≤5B" counts the entrants whose size
  is declared.
- The public suite is 20% of the board's headline Full score. The private 80% is run by the maintainers on submitted models
  and is pending for this family.

## Results

| Metric | Cerno 27B | Jev 1.13 (TypeSafe) |
|---|---:|---:|
| Decision Index 0.3 public index, complete run | **65.19** | 57.96 |
| Same-subset (HLE dropped for every entrant) | **66.60**, 2nd of 145 | 59.37, 22nd |
| Strict-OOD | 62.50 (3rd) | 54.80 |
| Train-split-OOD | 63.20 (6th) | 59.51 |
| Calibration error, board-style ECE | **5.7 pts** | 7.4 pts |
| TypeSafe agreement set (373 published Jev decisions) | 88.7% | 86.9% |
| Our golden set (956 typed decisions): accuracy / ECE | 92.7% / 0.022 | – |

<details>
<summary>How to read the rows</summary>

- **Strict-OOD** also drops BANKING77, CLINC150 and HellaSwag, whose public train splits are common training data.
- **Train-split-OOD** drops every benchmark family whose public train split we used (listed under *Training data*).
- **ECE** is the board's definition: expected calibration error over a 1-in-6 sample of 34 benchmarks, equal weight per
  benchmark, at the shipped temperature.
- The bf16 weights score 65.12 on the same suite; the FP8 weights published here are the submitted run.
- Run-to-run noise on the public index is about 0.1-0.2 points. Do not read the public rank as a board rank.
</details>

## Quick start

```sh
git clone https://github.com/apertixsk/cerno-inference && cd cerno-inference
pip install -r requirements.txt && export PYTHONPATH=$PWD:$PWD/third_party/decider
bash run_sample.sh apertixai/cerno-27b            # reproduces 60 requests of our submitted run in two minutes
```

The complete public suite, exactly as submitted:

```sh
python -m decision_index pipeline --edition 0.3 --engine evals.decision_index.engine_vllm:VllmEngine \
    --option model=apertixai/cerno-27b --out runs/cerno-27b
```

- **Weights:** FP8, W8A8 with per-channel weight scales and dynamic per-token activations, compressed-tensors format,
  `lm_head` in bf16. vLLM loads them with no extra option. One GPU with 40 GB or more.
- **Temperature 1.0:** shipped in `decider_config.json` next to the weights, applied by the engine as p^(1/T). Fitted on
  held-out, non-benchmark eval sets; argmax-invariant.
- **Protocol:** one row per question (state + question, state-first layout); the rows of a request batched into one vLLM call
  with the state prefix shared; one generated token; the top-k logprobs read over the option label tokens and renormalised.
  No thinking, no second reading, no ensembling, no abstention.



## Examples
Two requests from our own evaluation set (not benchmark data), answered by this model through the public engine. A request is a
*state* plus typed *questions*; the answer carries a probability for every option.

**Agent guardrail: should this tool call run without asking?**
```python
state = "rm -rf /tmp/build-cache-7f3a2e   # user asked to clear the build cache before a rebuild"
questions = {
    "destructive": {"type": "noul", "instructions": "Would executing this tool call delete, overwrite or irreversibly change data?"},
    "matches_request": {"type": "noul", "instructions": "Is this tool call a reasonable way to do what the user asked?"},
    "external_effect": {"type": "noul", "instructions": "Does this tool call affect people or systems outside the local machine (sends messages, moves money, publishes)?"},
    "risk": {"type": "score", "instructions": "How risky is it to run this tool call without asking the user first?",
             "criteria": ["harmless, read-only", "minor, easily reversible", "significant or hard to reverse", "severe or irreversible"]}
}
```

| question | answer | probabilities |
|---|---|---|
| `destructive` | **yes** (0.99) | p(yes) = 0.99 |
| `matches_request` | **yes** (0.90) | p(yes) = 0.90 |
| `external_effect` | **no** (1.00) | p(yes) = 0.00 |
| `risk` | **minor, easily reversible** (level 1) | harmless 0.02, minor 0.76, significant or hard to reverse 0.19, severe or irreversible 0.03 |

**Support triage: a one-line customer message**
```python
state = "hey do u have showers at the downtown location"
questions = {
    "kind": {"type": "choice", "instructions": "What kind of message is this?",
             "criteria": {"incident": "an outage or something broken in production", "billing": "invoices, charges, refunds", "account": "cancellations, plan changes, access", "feedback": "praise or suggestions", "spam": "unsolicited or scam", "other": null}},
    "urgency": {"type": "score", "instructions": "How urgent is this message?",
             "criteria": ["no action needed", "can wait a few days", "should be handled today", "needs attention right now"]},
    "tone": {"type": "score", "instructions": "What is the sender's tone?",
             "criteria": ["hostile", "negative", "neutral", "positive"]},
    "needs_human": {"type": "noul", "instructions": "Does this message need a human to respond rather than an automated reply?"}
}
```

| question | answer | probabilities |
|---|---|---|
| `kind` | **other** | other 0.94, billing 0.02, account 0.02 |
| `urgency` | **can wait a few days** (level 1) | no action needed 0.06, can wait a few days 0.56, should be handled today 0.34, needs attention right now 0.04 |
| `tone` | **neutral** (level 2) | hostile 0.00, negative 0.01, neutral 0.90, positive 0.09 |
| `needs_human` | **yes** (0.73) | p(yes) = 0.73 |

Programmatic use, with the repository on `PYTHONPATH`:

```python
from evals.decision_index.engine_vllm import VllmEngine

engine = VllmEngine(model="apertixai/cerno-27b")        # loads the FP8 weights and the shipped temperature
response, _ = engine(state, questions)              # one forward pass per question row, state prefix shared
response["answers"]["destructive"]                     # {"type": "noul", "noul": 0.99}
```

Question types: `choice` (criteria = option descriptions, `null` for none), `noul` (yes/no), `score` (ordered level descriptions).

## Serving and latency

- One H200, one process, one request at a time, the kit's 400-request stratified sample: median / mean / p80 **52 / 206 / 332 ms**,
  p95 692 ms.
- Over the whole public suite (140,178 requests, the submitted run): median / mean / p80 50 / 105 / 133 ms, p95 310 ms, max 3.2 s.
- 99.6% of answers identical to the bf16 weights on the sample; the rest are near-ties.
- No GGUF build is part of this release.

## Training data

Fine-tuned from Qwen3.8-27B on public data only. No Decision Index test row and no HLE row was used anywhere; every
training record was checked against all 140,620 requests of the 0.3 suite before training.

- A public instruction / classification mixture.
- The public TRAIN/DEV splits (never test rows) of these Decision Index benchmark families: BANKING77, CLINC150+OOS, ANLI,
  VAST, WinoGrande, GSM8K, HoVer, RAGTruth, ContractNLI, Humicroedit, ACOS, Amazon ESCI, NLI4CT, When2Call, New Yorker
  caption matching, PhishNChips, Home appliances, Habermas Machine comparisons.
- Procedurally generated reasoning items modelled on the formats of GSM8K, BBH, CRUXEval and CLadder, with no benchmark text.

The strict-OOD and train-split-OOD rows above drop the affected benchmarks for every entrant.

## Licence and credits

- Weights and code: Apache-2.0. Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache-2.0.
- The prompt layout and label-token readout follow the open decider recipe (Apache-2.0); its inference subset is vendored in
  the code repository.
- Benchmark data stays under its own terms. Jev and System One are TypeSafe AI's names; this project is not affiliated with
  TypeSafe AI or with the Decision Index maintainers.
