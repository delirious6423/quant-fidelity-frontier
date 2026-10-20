---
license: gemma
base_model: google/gemma-4-26B-A4B-it
tags:
  - gguf
  - gemma4
  - style
  - style-classification
  - prose-evaluator
  - fine-tuned
  - llama.cpp
  - lmstudio
quantized_by: alexokita
pipeline_tag: text-generation
language:
  - en
---

# Gemma 4 Style (step 3000) — GGUF

Merged **Gemma 4 26B-A4B** style-evaluator LoRA at **checkpoint step 3000**, converted to GGUF for llama.cpp / LM Studio.

This is a **prose style judge**, not a fiction writer. Full intended use, training details, data sources, and the **14 Style-in-Fiction rubric metrics** are documented on the companion LoRA card:

→ [`alexokita/gemma4-style-lora-step3000`](https://huggingface.co/alexokita/gemma4-style-lora-step3000)

## Files

| File | Quant | Size (approx) | Notes |
|------|-------|---------------|-------|
| `gemma4-style-step3000-q5_k_m.gguf` | Q5_K_M | ~18 GB | Higher fidelity |
| `gemma4-style-step3000-q4_k_m.gguf` | Q4_K_M | ~16 GB | Recommended for 24 GB GPUs |

F16 intermediate (~48 GB) was built locally for conversion and is not uploaded here.

## Intended use (short)

- Emit JSON `style_profile`s and short judgments over the 14 SiF-derived metrics
- Score drafts against steering cards (voice / style / tone)
- Power evaluator bake-offs and later writer ↔ evaluator loops

**Not for:** open-ended novel writing.

## Recommended sampling (LM Studio / llama.cpp)

From the [Gemma 4 QAT model card](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF): temperature `1.0`, top_p `0.95`, top_k `64`, **thinking off** for JSON / classification.

## How it was trained (summary)

| Item | Value |
|---|---|
| Base | Gemma 4 26B-A4B instruct QAT (`unsloth/gemma-4-26B-A4B-it-qat-q4_0-unquantized`) |
| Method | LoRA SFT (r=16, α=16) via Unsloth + TRL on DGX Spark |
| Steps | **3000** (best eval loss ≈ 2.378) |
| Data | ~430k / 48k style instruction pairs from Phase 2–3 labeled fiction |
| Conversion | merge LoRA → HF F16 → `convert_hf_to_gguf.py` → `llama-quantize` Q5_K_M / Q4_K_M |

## The 14 Style-in-Fiction metrics

Operationalized from Leech & Short, *Style in Fiction* (2nd ed.) in `style_rubric.json`:

1. `lexical_complexity` — LLM Pass 1  
2. `lexical_density` — computable (spaCy)  
3. `register` — LLM Pass 1  
4. `sentence_complexity` — LLM Pass 1  
5. `sentence_length_mean` — computable  
6. `subordination_ratio` — computable  
7. `figurative_density` — LLM Pass 1  
8. `cohesion` — LLM Pass 1  
9. `dialogue_ratio` — computable  
10. `pov` — LLM Pass 1  
11. `mind_style` — LLM Pass 2  
12. `narrative_distance` — LLM Pass 2  
13. `free_indirect_discourse` — LLM Pass 2  
14. `tone` — LLM Pass 2  

Labels: Phase 1 distilled the rubric from the book; Phase 2 classified prose with computables + multi-pass LLM + RAG; Phase 3 built SFT pairs. See the LoRA card for the full table and value enums.

## License

Use of the base Gemma 4 weights is subject to the [Gemma license](https://ai.google.dev/gemma/terms).
