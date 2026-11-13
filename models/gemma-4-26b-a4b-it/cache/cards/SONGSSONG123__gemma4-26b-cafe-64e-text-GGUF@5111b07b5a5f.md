---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
language:
- ko
tags:
- gguf
- gemma4
- mixture-of-experts
- expert-pruning
- tool-calling
- ollama
pipeline_tag: text-generation
---

# gemma4-26b-cafe-64e-text (GGUF, Q4_K_M)

An experiment: half of the experts of Gemma 4 26B (MoE, 128 experts × 30 layers) were removed using Korean café-order calibration data, so the model fits on a laptop integrated GPU while keeping café ordering skills.

**This is a domain-specific experimental model, not a general-purpose model.** General knowledge is significantly reduced.

## What was changed
- Base: google/gemma-4-26B-A4B-it (Apache 2.0)
- Structured expert pruning: 128 → 64 experts per layer (1,920 of 3,840 experts removed). Remaining weights are unchanged; expert tensors, router rows, and per-expert scales were sliced. Top-k routing stays 8.
- Expert selection: per-expert contribution sum Σ g·‖f(x)‖ over 600 Korean café-order prompts (system prompt + tool definitions + customer utterance), fixed prompt and request tokens weighted 50:50, plus protection for experts handling ≥30% of a layer's tokens. Based on REAP (Lasby et al., 2025, arXiv:2510.13999), modified to account for routing frequency.
- Quantized to Q4_K_M (per-tensor types matched to the original Ollama build), no imatrix.
- Text only: vision components removed. Language weights are identical to the full 64e build used in the evaluation below.

## Results (single run, thinking off, temperature 0)
| Model | Café test (120, AI-written) | Live test (20, human-written) | General knowledge (KMMLU 200 + MMLU 100) |
|---|---|---|---|
| Gemma 4 26B original | 94.2% | 75% | 72.7% |
| **64e (this model)** | **93.3%** | **70%** | **47.3%** |
| Gemma 4 12B | 90.8% | 45% | 64.3% |

Same laptop integrated GPU (Intel Arc 140T): the full 64e build (with Ollama's MTP draft model) ran at a median 2.55 s per order (Gemma 4 12B, also with its draft model: 3.33 s). This text-only GGUF does not include the draft model: median 6.46 s per order. The original 26B does not fit on this iGPU.

## Limitations
- Live test has only 20 items; no comparison is statistically significant (p ≥ 0.05).
- Café test, calibration data, and prompts were generated with an AI coding tool; only the live test was human-written.
- Tuned for a specific café menu/tool schema. Expect weak performance outside café ordering.
- Image input is not supported and was never evaluated.

## Usage
```
ollama run hf.co/SONGSSONG123/gemma4-26b-cafe-64e-text-GGUF:Q4_K_M
```
Or download the GGUF and `ollama create cafe-64e -f Modelfile`.

## Links
- Code and results: https://github.com/gustjrvkdnj1234-png/cafe-robot-brain
- Blog post (Korean): https://blog.cslee.co.kr/a-graduate-ai-applied-for-a-cafe-part-time-job

## License
Apache 2.0. This is a modified version of Google's Gemma 4; see LICENSE.
