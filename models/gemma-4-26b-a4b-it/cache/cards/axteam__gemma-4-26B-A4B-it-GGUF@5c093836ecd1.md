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
| [`gemma-4-26B-A4B-it-GGUF-BF16.gguf`](gemma-4-26B-A4B-it-GGUF-BF16.gguf) | BF16 | 50.5 GB | unquantized 16-bit (reference; for re-quantizing or max quality) |
| [`gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf`](gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf) | Q4_K_M | 16.8 GB | 4-bit, the usual size/quality trade-off — the most common choice |
| [`gemma-4-26B-A4B-it-GGUF-Q8_0.gguf`](gemma-4-26B-A4B-it-GGUF-Q8_0.gguf) | Q8_0 | 26.9 GB | 8-bit, closest to BF16 (larger) |

> **Q4_K_M** is the usual default (size/quality trade-off). **Q8_0** stays closest to BF16. **BF16** is the unquantized reference. GGUF quality was not measured for this model.
> Architecture: Gemma 4 · Mixture-of-Experts (multimodal; the GGUF text model runs the language side).

## Quantization method
These GGUF files are **post-training quantization (PTQ)** made with llama.cpp — the BF16 weights are quantized **block-wise** (llama.cpp *k-quants*: per-block scales with mixed bit-width allocation, a refined round-to-nearest), with **no retraining**. This is a different toolchain from the AWQ / NVFP4 vLLM builds, and it is **not** QAT (quantization-aware training, which fine-tunes the weights). `Q8_0` stays closest to BF16, `Q4_K_M` is the usual size/quality trade-off (not measured for this model), and `BF16` is the unquantized reference for re-quantizing.

## Requirements

| | |
|---|---|
| **llama.cpp** | build **≥ b8637** — architecture `gemma4` needs ≥ b8637, and the commands on this card use `-fa on` (≥ b6325). Load-checked here with b12005 (commit e0f142d) and b10724 on CPU — the loader accepts the model, tokenizer and the `-fa on` / q8_0 KV-cache flags; no generation was run in that check. |
| **Memory** | the chosen GGUF file (sizes in the table) plus KV cache for the context (`-c`); `--cache-type-k/v q8_0 -fa on` halves the KV cache. |
| **Input** | **text only** — this repo has no `mmproj` (vision projector) file. |
| **Ollama / LM Studio** | not tested here. |

## Run with llama.cpp

```bash
# download one file (e.g. Q4_K_M) then:
llama-server -m gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf -c 8192
# or a single answer and exit (-st):
llama-cli   -m gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf -p "대한민국의 수도는?" -st
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

**[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)의 GGUF 양자화 빌드 (llama.cpp·Ollama·LM Studio용)** · 제작 axteam

### 개요
| 항목 | 내용 |
|---|---|
| 원본 모델 | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| 형식 | GGUF (llama.cpp) |
| 변환 도구 | llama.cpp `convert_hf_to_gguf.py` → `llama-quantize` |
| 모델 구조 | Gemma 4 · MoE (원본은 멀티모달, GGUF는 언어 모델 부분만 실행) |
| 파일 수 | 3개 |

### 양자화 상세
- **양자화 절차**: PTQ(학습 후 양자화, 재학습 없음), llama.cpp 블록 단위 양자화(k-quant `Q*_K`, `Q8_0`)
- **chat template**: GGUF 파일에 내장
- **품질**: 미측정 — 아래 파일별 설명은 llama.cpp 양자화 단계의 일반적 경향

**제공 파일**

| 파일 | 크기 | 설명 |
|---|---|---|
| `gemma-4-26B-A4B-it-GGUF-BF16.gguf` | 50.5 GB | 양자화하지 않은 16비트 — 기준 파일 (재양자화용·최고 품질) |
| `gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf` | 16.8 GB | 4비트 — 크기·품질 균형, 가장 많이 쓰는 선택 (**권장**) |
| `gemma-4-26B-A4B-it-GGUF-Q8_0.gguf` | 26.9 GB | 8비트 — BF16에 가장 가까움, 용량 큼 |

### 요구사항
- **llama.cpp**: b8637 이상 (`gemma4` 구조 지원 b8637부터, 예시 명령의 `-fa on` 옵션 b6325부터)
- **로딩 확인**: b12005(커밋 e0f142d)와 b10724에서 CPU로 모델·토크나이저 로딩과 `-fa on`·q8_0 KV 캐시 옵션 인식만 확인 (생성은 실행 미확인)
- **메모리**: 선택한 GGUF 파일 크기 + 컨텍스트(`-c`)만큼의 KV 캐시 (KV 캐시 줄이는 방법은 아래 사용법 참고)
- **입력**: 텍스트 전용 — 비전 인코더·투영층 파일(`mmproj`)이 없어 이미지 입력 불가
- **Ollama·LM Studio**: 제작 측에서 실행 미확인

### 사용법
- **파일 받기**: `hf download axteam/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf --local-dir .` (아래 `-m` 명령에 필요, 다른 양자화는 파일명의 `Q4_K_M`만 변경)
- **서버 실행**: `llama-server -m gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf -c 8192`
- **한 번 질문 (답변 후 종료)**: `llama-cli -m gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf -p "대한민국의 수도는?" -st`
- **Ollama**: `ollama run hf.co/axteam/gemma-4-26B-A4B-it-GGUF:Q4_K_M`
- **KV 캐시 절감(선택)**: `--cache-type-k q8_0 --cache-type-v q8_0 -fa on` — KV 캐시 약 절반, GGUF 파일과 상관없이 실행 시 지정
- **권장 샘플링**: `temperature=1.0`, `top_p=0.95`, `top_k=64` (Google 권장값)

### 라이선스
- **라이선스**: Apache-2.0 ([Gemma 4 라이선스 안내](https://ai.google.dev/gemma/docs/gemma_4_license)) — 원본 모델의 라이선스를 그대로 따름
- **권리**: 원본 가중치·토크나이저·chat template은 Google DeepMind 소유, 이 리포는 양자화 파생물
