---
license: apache-2.0
library_name: vllm
pipeline_tag: text-generation
language:
- en
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- medical
- clinical
- nvfp4
- modelopt
- speculative-decoding
- qwen3.8
- ertas
- medertas
---

![MedErtas v0.1](medertas-banner.png)

**[Use MedErtas on the Ertas AI gateway →](https://www.ertas.ai/medertas)**

# MedErtas v0.1 27B LS

MedErtas v0.1 27B LS is an open-weight language model for medical AI products, built for fast, low-cost serving. It is strongest on clinical calculation, with or without tools, and on long records with short answers, and it serves on vLLM and NVIDIA Blackwell GPUs with Perigee, its medical drafter.

> **22 GB**, from 55.6 GB at BF16 | **98.2%** of BF16's mean score across 12 medical sets | **207 tokens/s** single-user decode speed on clinical calculation, against 26 at BF16

[Whitepaper](https://www.ertas.ai/research/medertas-v01-specialisation-inference-optimisation) | [Model page](https://www.ertas.ai/medertas) | [Collections](https://huggingface.co/collections/ErtasAI) | [Try it on the Ertas API](https://app.ertas.ai)

**LS (Lightweight Study)** is the MedErtas build profile for compressed models. LS weights are quantised with calibration on medical data, and every LS build is scored against its full-precision base on medical tasks before release. The [whitepaper](https://www.ertas.ai/research/medertas-v01-specialisation-inference-optimisation) explains how it works.

## Which MedErtas model to pick

| If your product mostly does | Pick |
|---|---|
| Clinical calculation: doses, scores, risk calculators | **This model**: MedCalc-Bench 77.3 |
| Exam-style answers, ICD-10 coding, checking answers for hallucinations | **[MedErtas-v0.1-31B-LS](https://huggingface.co/ErtasAI/MedErtas-v0.1-31B-LS)**: MedQA 93.0 against 86.7 here, MedConceptsQA hard 69.1 against 60.9, MedHallu hard 64.1 against 59.9 |
| Facts from long patient records | Either: LongHealth 88.5 to 90.2 across both |
| Your own product's tasks, on your own data | A MedErtas model trained for your use case: [talk to us](https://www.calendly.com/hello-ertas/enterprise) |

## Build at a glance

Built on Qwen3.8-27B from the Qwen team. We quantised its weights with calibration on medical data and left the architecture unchanged.

| | |
|---|---|
| Base | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), architecture unchanged |
| Profile | LS v0.1: domain-aware quantisation, calibrated on medical data |
| Format | NVFP4 (W4A4) with attention in FP8. This repository holds the NVFP4 build |
| Size | 22 GB |
| Drafter | [MedErtas-v0.1-27B-LS-Perigee](https://huggingface.co/ErtasAI/MedErtas-v0.1-27B-LS-Perigee), 4.0 GB: Red Hat's DSpark drafter for Qwen3.8-27B, trained further on medical answers |
| Hardware | NVIDIA Blackwell GPUs, which NVFP4 W4A4 needs. Measured on one RTX PRO 6000 (96 GB); smaller Blackwell cards are unmeasured |
| Runtime | vLLM 0.32.0 |
| Context | Measured at 65,536 tokens. The base model's longer context is inherited and unmeasured with these weights |
| Licence | Apache 2.0, from Qwen3.8 |

## Precision layout

| Part | Format |
|---|---|
| MLP projections and LM head | NVFP4 |
| Attention projections, full and linear | FP8 |
| Gated DeltaNet gating projections | BF16 |
| Embeddings and vision tower | BF16 |

This is the same layout as NVIDIA's NVFP4 build of Qwen3.8-27B. The difference is the medical calibration.

## Results

Scored on the [Medmarks](https://medmarks.ai) harness with thinking off, on vLLM 0.32.0. Each build is paired against BF16 on the same items. Differences are in accuracy points with 95% intervals, computed per item before rounding.

| Set (items) | What it tests | BF16 | This model | Δ (95% interval) | Red Hat NVFP4 | NVIDIA NVFP4 |
|---|---|---|---|---|---|---|
| MedCalc-Bench 1.2 (1,100) | Clinical calculation | 80.9 | 77.3 | −3.6 (−5.9 to −1.4) | 75.4 | 76.6 |
| MedCalc-Bench 1.2, with tools (1,100) | Clinical calculation with a calculator tool | 79.5 | 76.3 | −3.2 (−5.6 to −0.7) | 75.6 | 76.1 |
| MedQA (1,273) | Exam questions | 88.2 | 86.7 | −1.5 (−3.3 to +0.3) | 85.7 | 88.1 |
| MetaMedQA (1,373) | Recognising unanswerable questions | 65.6 | 65.4 | −0.2 (−1.9 to +1.5) | 65.0 | 63.7 |
| MedHallu easy (548) | Hallucination detection | 82.7 | 81.6 | −1.1 (−4.0 to +1.8) | 79.7 | 82.3 |
| MedHallu medium (636) | Hallucination detection | 76.4 | 75.5 | −0.9 (−4.1 to +2.2) | 74.2 | 77.4 |
| MedHallu hard (816) | Hallucination detection | 59.2 | 59.9 | +0.7 (−2.4 to +3.9) | 57.5 | 58.5 |
| MedConceptsQA easy (2,000) | ICD-10 coding | 97.5 | 97.5 | −0.1 (−0.8 to +0.6) | 96.6 | 96.9 |
| MedConceptsQA medium (2,000) | ICD-10 coding | 67.2 | 65.4 | −1.9 (−3.9 to +0.2) | 67.4 | 63.1 |
| MedConceptsQA hard (2,000) | ICD-10 coding | 64.5 | 60.9 | −3.6 (−5.7 to −1.5) | 61.8 | 60.5 |
| LongHealth task 1 (400) | Facts from long patient records | 89.2 | 88.5 | −0.8 (−2.6 to +1.1) | 88.5 | 88.2 |
| LongHealth task 2 (800) | Facts from long patient records | 89.2 | 89.4 | +0.1 (−0.8 to +1.1) | 89.8 | 89.4 |
| **Mean, 12 sets** | | **78.4** | **77.0** | | 76.4 | 76.7 |

The mean is above both the Red Hat and NVIDIA NVFP4 builds. Clinical calculation is where this build still trails BF16, by 3.6 points on MedCalc-Bench, and it is the focus of the next version. The harness samples at temperature 1.0, and each build ran once.

## Speed

Single-user decode speed on clinical calculation, in output tokens per second, measured on one RTX PRO 6000 with vLLM 0.32.0 in October 2026.

| Configuration | Decode speed (tokens/s) |
|---|---|
| Qwen3.8-27B, BF16 | 26 |
| This model | 60 |
| This model with Red Hat's DSpark drafter | 188 |
| This model with Perigee, the MedErtas medical drafter | **207** |

With speculative decoding, the main model checks every drafted token and keeps only those it would produce itself, so answers follow this model's own distribution. The drafter changes how fast they arrive.

**Choosing a drafter.** Perigee is the fastest single-user option on clinical calculation and exam answers. On long patient records, Red Hat's drafter is faster. With many users at once, the model's built-in multi-token prediction gives the most total throughput on calculation.

## Serving

Text only, with Perigee. These are the settings the speed and quality results above were measured with.

```shell
vllm serve ErtasAI/MedErtas-v0.1-27B-LS \
  --language-model-only \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"dspark","model":"ErtasAI/MedErtas-v0.1-27B-LS-Perigee","num_speculative_tokens":7}'
```

For the built-in multi-token prediction instead, use `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`. Without speculative decoding, leave the flag out.

Sampling, as on the Qwen3.8 card and in our runs. Thinking off: `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0`, `presence_penalty=1.5`, with `chat_template_kwargs: {"enable_thinking": false}`. Thinking on: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0`. The results above are thinking off.

## What makes it medical

The main weights are Qwen3.8-27B's, compressed with calibration on medical data, so the compression costs as little as possible on medical tasks. Perigee, the drafter, is trained on medical answers, and the release is scored on medical evaluation sets. The main weights are not post-trained on medical data; a post-trained MedErtas build would carry the QS profile.

## Train a stronger model on your data

MedErtas v0.1 is a strong starting base for medical AI products. Trained on the data your product actually handles, such as your notes, coding conventions, patient conversations and clinical workflows, it can become a much stronger model for your use case, scored against your own evaluations before it goes live. We build these for medical AI companies and healthcare enterprises, and run them on infrastructure we manage, in your own cloud or on-premise.

[Talk to us](https://www.calendly.com/hello-ertas/enterprise) about a MedErtas model trained on your data.

## Intended use and limitations

MedErtas is a research model for teams building medical AI products, such as clinical documentation, patient-facing assistants and decision-support tools, as a starting point for their own validation.

- It is not a medical device, is not cleared for clinical use, and must not be used for diagnosis or treatment decisions without qualified clinical oversight.
- It has been evaluated on benchmarks only. Validate it on your own data and workflow before any use with patients.
- It inherits the behaviour and limitations of Qwen3.8-27B. Like its base, it is weaker at ICD-10 code lookup than at clinical reasoning, so pair it with a coding database for coding tasks.
- Image and video input are unmeasured with these weights.
- English only in our evaluation.

The hosted MedErtas API is in beta, with a production offering coming soon.

## Licence and credits

Apache 2.0, the licence of Qwen3.8. Built on Qwen3.8-27B by the Qwen team and quantised with NVIDIA Model Optimizer. The medical drafter starts from Red Hat's DSpark drafter for Qwen3.8-27B. Comparison builds from Red Hat and NVIDIA. Medical evaluation sets through Medmarks (MedARC).

## Citation

```bibtex
@techreport{ertas2026medertas,
  title       = {Specialisation as Inference Optimisation: Domain-Aware Quantisation and Drafting for Open Medical {LLMs}},
  author      = {Yang, Edward Xi},
  institution = {Ertas AI},
  year        = {2026},
  month       = oct,
  note        = {MedErtas v0.1},
  url         = {https://www.ertas.ai/research/medertas-v01-specialisation-inference-optimisation}
}
```
