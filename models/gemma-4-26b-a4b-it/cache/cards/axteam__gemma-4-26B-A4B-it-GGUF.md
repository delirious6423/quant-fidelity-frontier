---
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
tags:
  - gemma
  - gemma4
  - gguf
  - llama.cpp
  - quantized
---
# gemma-4-26B-A4B-it · GGUF

GGUF builds of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) for **llama.cpp / Ollama / LM Studio**. Converted by **axteam** with llama.cpp (converted from the BF16 original; the chat template is embedded in the GGUF).

| File | Quant | Size | Notes |
|---|---|---|---|
| [`gemma-4-26B-A4B-it-GGUF-BF16.gguf`](gemma-4-26B-A4B-it-GGUF-BF16.gguf) | BF16 | 47.0 GB | unquantized 16-bit (reference; for re-quantizing or max quality) |
| [`gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf`](gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf) | Q4_K_M | 15.6 GB | 4-bit, best size/quality balance — the most common choice |
| [`gemma-4-26B-A4B-it-GGUF-Q8_0.gguf`](gemma-4-26B-A4B-it-GGUF-Q8_0.gguf) | Q8_0 | 25.0 GB | 8-bit, near-lossless (larger) |

> **Q4_K_M** is the recommended default (best balance). **Q8_0** for near-lossless quality. **BF16** is the unquantized reference.
> Architecture: Gemma 4 · Mixture-of-Experts (multimodal; the GGUF text model runs the language side).

## Quantization method
These GGUF files are **post-training quantization (PTQ)** made with llama.cpp — the BF16 weights are quantized **block-wise** (llama.cpp *k-quants*: per-block scales with mixed bit-width allocation, a refined round-to-nearest), with **no retraining**. This is a different toolchain from the AWQ / NVFP4 vLLM builds, and it is **not** QAT (quantization-aware training, which fine-tunes the weights). `Q8_0` is near-lossless, `Q4_K_M` is the best size/quality balance, and `BF16` is the unquantized reference for re-quantizing.

## Run with llama.cpp

```bash
# download one file (e.g. Q4_K_M) then:
llama-server -m gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf -c 8192
# or one-shot generation:
llama-cli   -m gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf -p "대한민국의 수도는?"
```

### Reduce KV-cache memory (optional, run-time)
KV-cache type is a **run-time** choice in llama.cpp (not baked into the file). Halve it with:
```bash
llama-server -m gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf --cache-type-k q8_0 --cache-type-v q8_0 -fa on
```

## Ollama
```bash
ollama run hf.co/axteam/gemma-4-26B-A4B-it-GGUF:Q4_K_M
```

> Recommended sampling (per Google): `temperature=1.0`, `top_p=0.95`, `top_k=64`.

## License
Quantized derivative of Google's Gemma 4 — governed by the
[Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license). Weights, tokenizer,
and chat template belong to Google DeepMind.

---

## 한국어 요약
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) 의 **GGUF** 빌드 (llama.cpp·Ollama·LM Studio용). 변환: **axteam** (BF16 원본에서 llama.cpp로 변환, chat_template은 GGUF에 내장됨).

- **파일**: 위 표 — `Q4_K_M`(권장·균형), `Q8_0`(거의 무손실), `BF16`(비양자화 기준).
- **양자화 방식**: llama.cpp **k-quant**(블록 단위 스케일 + 혼합 비트 할당, 정교한 반올림)로 만든 **학습 후 양자화(PTQ)** — 재학습 없음. AWQ/NVFP4 vLLM 빌드와는 다른 툴체인이며 **QAT 아님**(QAT는 재학습으로 가중치를 적응시키는 방식).
- **실행**: `llama-server -m gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf -c 8192`
- **KV 캐시 절감(런타임)**: `--cache-type-k q8_0 --cache-type-v q8_0 -fa on` (파일에 안 구움 — 실행 때 선택)
- **Ollama**: `ollama run hf.co/axteam/gemma-4-26B-A4B-it-GGUF:Q4_K_M`
- 라이선스: 구글 Gemma 4 파생물([라이선스](https://ai.google.dev/gemma/docs/gemma_4_license)).
