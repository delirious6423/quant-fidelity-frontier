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
- read-twice
model-index:
- name: Cerno 27B v1.1
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
      value: 69.73
---

<div align="center">
  <img src="assets/cerno-light.jpg" alt="Cerno decision models" width="760"/>
  <h1>Cerno 27B v1.1</h1>
  <p><b>Jev-style System One decision model &middot; 27B &middot; one forward pass &middot; calibrated probabilities &middot; FP8 &middot; read twice</b></p>
  <p>
    <a href="https://huggingface.co/apertixai/cerno-27b-v1.1">Cerno 27B v1.1</a> &middot;
    <a href="https://huggingface.co/apertixai/cerno-9b-v1.2">Cerno 9B v1.2</a> &middot;
    <a href="https://huggingface.co/apertixai/cerno-4b-v1.2">Cerno 4B v1.2</a> &middot;
    <a href="https://huggingface.co/apertixai/cerno-27b">27B v1.0</a> &middot;
    <a href="https://github.com/apertixsk/cerno-inference">Inference code</a> &middot;
    <a href="https://huggingface.co/datasets/apertixai/cerno-decision-index">Decision Index runs</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Decision%20Index%200.3-69.73-1f6feb" alt="Decision Index 0.3 public index 69.73"/>
    <img src="https://img.shields.io/badge/weights-FP8-6f42c1" alt="FP8 weights"/>
    <img src="https://img.shields.io/badge/license-Apache%202.0-lightgrey" alt="Apache 2.0"/>
  </p>
</div>

*v1.1 (2026-11-06): a new 27B, trained from the post-trained Qwen3.8-27B and served with read twice. +4.54 on the public index over v1.0 (65.19).
It keeps the 27B's causal attention, so it runs on the vLLM engine of the inference code, in FP8. Cerno 27B v1.0 stays at
[`apertixai/cerno-27b`](https://huggingface.co/apertixai/cerno-27b).*

## What it does

- **Input:** a *state* (any text or JSON) and a set of typed *questions* about it.
- **Question types:** multiple choice with explicit criteria, yes/no, scored scales. Up to 255 options per question.
- **Output:** a probability for every option of every question, with **no text generation**: one row per question, the
  state shared through vLLM's prefix cache, one forward step per row, all rows of a request in one batch.
- **Read twice:** a request of up to 3,000 tokens is read twice: the state, the questions with their options, *"Read the same
  input again before answering:"*, then the state again (`--option read_twice_tokens=0` turns it off).
- **Calibrated:** one shipped temperature (1.059), fitted on held-out sets, rescales the probabilities (ECE 5.5 pts on the
  board's measure, under *Results*); it changes no answer.
- **Speed:** 100 ms median and 231 ms mean per request on one H200, one request at a time.

## The Cerno family

| Model | Base | Weights | Engine | Temperature | Public index |
|---|---|---|---|---:|---:|
| Cerno 27B v1.1 | Qwen3.8-27B | [`cerno-27b-v1.1`](https://huggingface.co/apertixai/cerno-27b-v1.1) | vLLM, FP8 | 1.059 | **69.73** |
| Cerno 9B v1.2 | Qwen3.5-9B | [`cerno-9b-v1.2`](https://huggingface.co/apertixai/cerno-9b-v1.2) | transformers, bf16 | 1.256 | **66.83** |
| Cerno 4B v1.2 | Qwen3.5-4B | [`cerno-4b-v1.2`](https://huggingface.co/apertixai/cerno-4b-v1.2) | transformers, bf16 | 1.236 | **65.05** |
| Cerno 27B v1.0 | Qwen3.8-27B | [`cerno-27b`](https://huggingface.co/apertixai/cerno-27b) | vLLM, FP8 | 1.0 | 65.19 |
| Cerno 9B v1.1 | Qwen3.5-9B-Base | [`cerno-9b`](https://huggingface.co/apertixai/cerno-9b) | vLLM, FP8 | 1.539 | 58.59 |
| Cerno 4B v1.1 | Qwen3.5-4B-Base | [`cerno-4b`](https://huggingface.co/apertixai/cerno-4b) | vLLM, FP8 | 1.534 | 57.60 |

- *Public index*: the complete public suite (37 benchmarks, 140,178 requests) run with the maintainers' kit at commit 9eb2dbe
  through the public inference code and these weights.
- The board ranks by its Full score: private tests of the same skills (60%) and private tasks from new domains (40%), run by the
  maintainers. The public index does not count toward it. v1.1's Full score is pending.

## Results

| Metric | Cerno 27B v1.1 | Cerno 27B v1.0 |
|---|---:|---:|
| Decision Index 0.3 public index, complete run | **69.73** | 65.19 |
| Strict-OOD | 68.23 | 62.50 |
| Train-split-OOD | 67.65 | 63.20 |
| Calibration error, board-style ECE | 5.5 pts | 5.7 pts |

<details>
<summary>How to read the rows</summary>

- **Strict-OOD** drops HLE, BANKING77, CLINC150 and HellaSwag, whose public train splits are common training data.
- **Train-split-OOD** drops every benchmark family whose public train split we used (listed under *Training data*).
- **ECE** is the board's definition: expected calibration error over a 1-in-6 sample of 34 benchmarks, equal weight per
  benchmark, at the shipped temperature.
- Run-to-run noise on the public index is about 0.1-0.2 points. The public index is not the board's ranking.
</details>

## Quick start

```sh
git clone https://github.com/apertixsk/cerno-inference && cd cerno-inference
pip install -r requirements-27b-v1.1.txt && export PYTHONPATH=$PWD:$PWD/third_party/decider
bash run_sample.sh apertixai/cerno-27b-v1.1       # reproduces 60 requests of our submitted run
```

The complete public suite, exactly as submitted:

```sh
python -m decision_index pipeline --edition 0.3 --engine evals.decision_index.engine_vllm:VllmEngine \
    --option model=apertixai/cerno-27b-v1.1 --out runs/cerno-27b-v1.1
```

- **Weights:** FP8 (compressed-tensors, `lm_head` in bf16), 29.4 GB. vLLM loads them with no extra option. One GPU with 40 GB or more.
- **Settings in `decider_config.json`:** `attention: causal`, `read_twice_tokens: 3000`, `temperature: 1.059`; the engine applies
  them. The temperature was fitted on held-out, non-benchmark eval sets and is argmax-invariant.
- **Software:** vLLM 0.31.0 with transformers 5.17.0, as pinned in `requirements-27b-v1.1.txt`.

## Examples

Requests from our own evaluation set (not benchmark data), answered by this model through the public engine.

**Agent guardrail: should this tool call run without asking?**
```python
state = "rm -rf /tmp/build-cache-7f3a2e   # user asked to clear the build cache before a rebuild"
questions = {
    "destructive": {"type": "noul", "instructions": "Would executing this tool call delete, overwrite or irreversibly change data?"},
    "matches_request": {"type": "noul", "instructions": "Is this tool call a reasonable way to do what the user asked?"},
    "external_effect": {"type": "noul", "instructions": "Does this tool call affect people or systems outside the local machine (sends messages, moves money, publishes)?"},
    "risk": {"type": "score", "instructions": "How risky is it to run this tool call without asking the user first?", "criteria": ["harmless, read-only", "minor, easily reversible", "significant or hard to reverse", "severe or irreversible"]}
}
```

| question | answer | probabilities |
|---|---|---|
| `destructive` | **yes** (0.98) | p(yes) = 0.98 |
| `matches_request` | **yes** (0.94) | p(yes) = 0.94 |
| `external_effect` | **no** (0.99) | p(yes) = 0.01 |
| `risk` | **minor, easily reversible** (level 1) | harmless, read-only 0.05, minor, easily reversible 0.65, significant or hard to reverse 0.25, severe or irreversible 0.04 |

**Support triage: a one-line customer message**
```python
state = "hey do u have showers at the downtown location"
questions = {
    "kind": {"type": "choice", "instructions": "What kind of message is this?", "criteria": {"incident": "an outage or something broken in production", "billing": "invoices, charges, refunds", "account": "cancellations, plan changes, access", "feedback": "praise or suggestions", "spam": "unsolicited or scam", "other": null}},
    "urgency": {"type": "score", "instructions": "How urgent is this message?", "criteria": ["no action needed", "can wait a few days", "should be handled today", "needs attention right now"]},
    "tone": {"type": "score", "instructions": "What is the sender's tone?", "criteria": ["hostile", "negative", "neutral", "positive"]},
    "needs_human": {"type": "noul", "instructions": "Does this message need a human to respond rather than an automated reply?"}
}
```

| question | answer | probabilities |
|---|---|---|
| `kind` | **other** | other 0.92, account 0.03, feedback 0.03 |
| `urgency` | **can wait a few days** (level 1) | no action needed 0.14, can wait a few days 0.59, should be handled today 0.23, needs attention right now 0.03 |
| `tone` | **neutral** (level 2) | hostile 0.00, negative 0.01, neutral 0.92, positive 0.07 |
| `needs_human` | **no** (0.59) | p(yes) = 0.41 |


Programmatic use, with the repository on `PYTHONPATH`:

```python
from evals.decision_index.engine_vllm import VllmEngine

engine = VllmEngine(model="apertixai/cerno-27b-v1.1")   # read twice and the temperature from decider_config.json
response, _ = engine(state, questions)                    # every question of the request in one vLLM batch
```

Question types: `choice` (criteria = option descriptions, `null` for none), `noul` (yes/no), `score` (ordered level descriptions).

## Serving and latency

- One H200, one process, one request at a time, the kit's 400-request stratified sample: median / mean / p80
  **100 / 231 / 426 ms**, p95 726 ms.
- Over the whole public suite (the submitted run, one process per GPU): median / mean / p80 62 / 137 /
  194 ms.
- The public code gives the same answers as our development engine on all 3,165 answer fields of that sample.
- Not measured on the board's reference hardware (one RTX PRO 6000).

## Training data

Full fine-tune of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). No Decision Index test row and no HLE
row was used; the training records were screened against the requests of the 0.3 suite before training.

- A public instruction / classification mixture.
- The public TRAIN/DEV splits (never test rows) of these Decision Index benchmark families: BANKING77, CLINC150+OOS, ANLI,
  VAST, WinoGrande, GSM8K, HoVer, RAGTruth, ContractNLI, Humicroedit, ACOS, Amazon ESCI, NLI4CT, When2Call, New Yorker
  caption matching, PhishNChips, Home appliances, Habermas Machine comparisons.
- Synthetic decision items across many domains, written and labelled with an external teacher model,
  [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT licence); its answer distributions
  are used as soft labels.
- Procedurally generated items modelled on the formats of some public benchmarks, with no benchmark text.

The strict-OOD and train-split-OOD rows above drop the affected benchmarks.

## Licence and credits

- Weights and code: Apache-2.0. Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache-2.0.
- The prompt layout and label-token readout follow the open decider recipe (Apache-2.0); its inference subset is vendored in
  the code repository.
- Read twice uses the layout of KnowLine Gen5.5 and FlexDM (MIT), after RE2 (Xu et al. 2023, arXiv 2309.06275).
- Training labels include outputs of DeepSeek-V4.1-Flash-1017 (MIT).
- Benchmark data stays under its own terms. Jev and System One are TypeSafe AI's names; this project is not affiliated with
  TypeSafe AI or with the Decision Index maintainers.
