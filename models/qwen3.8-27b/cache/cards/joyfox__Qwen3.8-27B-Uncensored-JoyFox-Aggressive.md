---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - conversational
  - multilingual
  - uncensored
  - abliterated
  - gguf
  - imatrix
  - mtp
  - vision
language:
  - en
  - zh
  - multilingual
---

<p align="center">
  <img src="assets/joyfox-qwen38-banner.png" alt="JoyFox Qwen3.8-27B" width="100%">
</p>

<h1 align="center">🦊 Qwen3.8-27B-Uncensored-JoyFox-Aggressive</h1>

<p align="center">
  A direct-response, low-overrefusal variant of Qwen3.8-27B with native model components preserved.
</p>

<p align="center">
  <strong>English</strong> · <a href="https://huggingface.co/joyfox/Qwen3.8-27B-Uncensored-JoyFox-Aggressive/blob/main/README_ZH.md">简体中文</a>
</p>

## 🦊 Overview

This release focuses on reducing false refusals and unnecessary policy-style preambles while retaining the native architecture and general capabilities of `Qwen/Qwen3.8-27B`.

The final checkpoint preserves the official tokenizer, chat template, and configuration, the complete vision stack and multimodal projector, native Multi-Token Prediction (MTP) weights. Vision tensors remain unchanged and are verified separately. MTP tensors remain unchanged and are verified separately.


## ⚡ What “Aggressive” means

“Aggressive” describes the strength of the low-overrefusal behavior. The intended response style is direct and substantive for benign, fictional, analytical, educational, authorized, privacy-respecting, or otherwise legitimate requests, including requests whose wording is sensitive or ambiguous.

The intervention does not add factual knowledge. The model can still hallucinate, accept a false premise, misunderstand intent, or produce unsuitable content. Users should evaluate it for their own application and remain responsible for deployment controls and applicable law.

## 🧬 Architecture and behavior

The repository contains a complete standalone checkpoint in the native Qwen3.8 tensor layout. Its text weights favor direct, substantive answers to legitimate requests, while the tokenizer, chat template, configuration, vision stack, multimodal projector, and MTP components retain upstream compatibility.

Release validation enforces these invariants:

- official tokenizer, chat template, configuration, and special-token IDs are unchanged;
- vision encoder and projector tensor names, shapes, dtypes, and values match the official checkpoint;
- native MTP tensor names, shapes, dtypes, and values match the official checkpoint before conversion;
- the public native checkpoint contains only the complete model and inference metadata;
- BF16, GGUF, MTP, and multimodal inference are tested independently.

## 🧪 Evaluation

The reported evaluation contains 1,000 prompts spanning difficult legitimate requests, bilingual exact-answer controls, and a non-blocking behavior diagnostic.

| Split | Prompts | Purpose |
|---|---:|---|
| Benign overrefusal | 800 | difficult legitimate and ambiguity-sensitive requests |
| Bilingual capability controls | 100 | deterministic Chinese/English exact-answer checks |
| Non-blocking behavior diagnostic | 100 | descriptive V1 reporting |

| Checkpoint | Benign refusal rate | Capability score |
|---|---:|---:|
| Official Qwen3.8-27B | 59.2% | 59.0% |
| JoyFox BF16 | 0.0% | 59.0% |

All results are generation-based. The fixed suite contains 1,000 prompts, greedy decoding is used, and the tested llama.cpp runtime commit is `885c5bbe8e04`.

## 💾 GGUF downloads

All main GGUF files are generated with an importance matrix. Native MTP tensors are bundled in every main GGUF, and a standalone MTP GGUF is also released for split-draft runtimes. The vision stack is exported as an F16 `mmproj` and validated with real image input.

| File | Quant | Intended use | Size |
|---|---:|---|---:|
| `Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q2_K.gguf` | Q2_K | minimum footprint | 10.12 GiB |
| `Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q3_K_M.gguf` | Q3_K_M | low-memory deployment | 12.57 GiB |
| `Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q4_K_M.gguf` | Q4_K_M | recommended 4-bit balance | 15.66 GiB |
| `Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q5_K_M.gguf` | Q5_K_M | recommended general use | 18.19 GiB |
| `Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q6_K.gguf` | Q6_K | high quality | 20.89 GiB |
| `Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q8_0.gguf` | Q8_0 | maximum GGUF fidelity | 27.05 GiB |
| `mtp-Qwen3.8-27B-Uncensored-JoyFox-Aggressive-BF16.gguf` | MTP BF16 | standalone draft model | 5.54 GiB |
| `mmproj-Qwen3.8-27B-Uncensored-JoyFox-Aggressive-F16.gguf` | mmproj F16 | image input | 0.86 GiB |

### Compatibility builds without MTP

The following `-no-mtp` files contain the same target-model trunk but no MTP/NextN draft tensors. Use one of these when a runtime version cannot load a GGUF with bundled MTP. They support normal text generation and can use the same matching `mmproj`; speculative MTP decoding is intentionally unavailable. A runtime must still support the Qwen3.8/Qwen3.5 GGUF architecture itself.

| File | Quant | Size |
|---|---:|---:|
| `Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q2_K-no-mtp.gguf` | Q2_K | 9.98 GiB |
| `Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q3_K_M-no-mtp.gguf` | Q3_K_M | 12.39 GiB |
| `Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q4_K_M-no-mtp.gguf` | Q4_K_M | 15.41 GiB |
| `Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q5_K_M-no-mtp.gguf` | Q5_K_M | 17.91 GiB |
| `Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q6_K-no-mtp.gguf` | Q6_K | 20.57 GiB |
| `Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q8_0-no-mtp.gguf` | Q8_0 | 26.63 GiB |

All six compatibility builds were inspected as 851-tensor target models with zero MTP/NextN tensors and passed a real text-generation smoke test.

llama.cpp has no standard Q7 K-quant, so the practical Q2–Q8 matrix uses Q2, Q3, Q4, Q5, Q6, and Q8.

## 🚀 Usage

Use llama.cpp commit `885c5bbe8e04` or a newer compatible build.

### 💬 Text chat

```bash
llama-cli \
  -m Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q5_K_M.gguf \
  --jinja -c 32768 -ngl 99
```

### ⚡ Native bundled MTP

```bash
llama-server \
  -m Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q6_K.gguf \
  --jinja -c 32768 -ngl 99 \
  --spec-type draft-mtp --spec-draft-n-max 3
```

MTP is an optional speculative-decoding feature. Its benefit depends on draft acceptance, context length, backend, and available memory.

### 👁️ Multimodal inference

```bash
llama-cli \
  -m Qwen3.8-27B-Uncensored-JoyFox-Aggressive-Q6_K.gguf \
  --mmproj mmproj-Qwen3.8-27B-Uncensored-JoyFox-Aggressive-F16.gguf \
  --image example.png \
  --prompt "Describe this image accurately." \
  --jinja -c 32768 -ngl 99
```

The `mmproj` must match this checkpoint. It is kept at F16 to preserve visual quality.

## 🎛️ Recommended settings

Use the official chat template and upstream sampling defaults. For reproducible evaluation, the scores above use deterministic greedy decoding.

## ⚠️ Validation and limitations

Before upload, the release must pass native checkpoint inspection, component-preservation checks, every-quant text generation, bundled and standalone MTP generation, and real-image generation with the matching F16 `mmproj`. The uploaded repository is then checked against the local release inventory.

Lower-bit quantization can reduce factual precision, multilingual consistency, long-context stability, visual grounding, and subtle instruction following. Abliteration can also weaken refusal behavior more broadly than intended. Evaluate the chosen quant and deployment policy for the actual use case.

## 📜 License and attribution

Released under Apache-2.0, following the base-model license. Qwen is created by the Qwen team. GGUF conversion and quantization use llama.cpp.
