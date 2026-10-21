---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: image-text-to-text
language: [zh, en]
base_model:
  - Qwen/Qwen3.8-27B
  - prism-ml/Ternary-Bonsai-27B-gguf
tags: [gguf, vision-language, reasoning, bounded-thinking, qwen3-vl, mmproj, ternary, q2_0, lm-studio, experimental]
inference: false
---

# Vireqo-27VT-Plus-260818

`Vireqo-27VT-Plus-260818` is the independent ternary Vision + bounded-Thinking product. It reuses the unchanged Plus language payload and shared Q8 projector, with the validated `Think512-Concise` configuration.

## Required files

| File | SHA-256 |
|---|---|
| `Vireqo-27VT-Plus-260818.gguf` | `a32a8ec286a11c6534bf29d1ee20bd4c02064032b51ae8310bb1216e2de17e03` |
| `mmproj-Vireqo-27VT-Plus-260818-Q8_0.gguf` | `eb561d41a7bbeb0fcf04883c8af11078ef6cae0a66862a0b68443cfca495269d` |

No additional physical main/projector copy was created.

## Thinking preset

Reasoning Budget `512`, maximum response `768`, temperature `0`, repeat penalty `1.08`, context `2048`, parallel `1`.

```text
思考预算已到。复核已有结论后，只输出最终答案；禁止重复或展示思考过程。
```

## Validation

Text Thinking answers were correct for capital, multiplication, and chicken/rabbit probes. Visual Thinking correctly read and reasoned over the chart values, highest year, and difference; reasoning/final separation was clean. See [`thinking-validation.md`](thinking-validation.md) and [`thinking-vision-validation.md`](thinking-vision-validation.md).

## Limitations

- Unrestricted Thinking is not accepted.
- This preset is concise, not a general benchmark claim.
- Natural-image benchmarks and video are untested.
- All underlying Plus and VLM limitations remain.

See [`LM-STUDIO-使用指南.md`](LM-STUDIO-使用指南.md), [`TECHNICAL_README.md`](TECHNICAL_README.md), and [`bundle-provenance.json`](bundle-provenance.json).
