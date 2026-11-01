---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
tags:
  - qwen
  - qwen3_5
  - gguf
  - llama.cpp
  - quantized
---

# Qwen3.8-27B · GGUF (llama.cpp)

GGUF quantizations of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) for **llama.cpp / Ollama / LM Studio**. Produced by **axteam**.

## Files

| File | Size | Notes |
|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | 16.8 GB | 4-bit k-quant (medium) — **recommended** size/quality balance. |
| `Qwen3.8-27B-Q8_0.gguf` | 29.0 GB | 8-bit — closest to the original. |

> Text generation via llama.cpp. Converted with llama.cpp `convert_hf_to_gguf.py`; the original `chat_template` is embedded. (The vision encoder is not included in these GGUF files.)

## Quantization method
These GGUF files are **post-training quantization (PTQ)** made with llama.cpp (`convert_hf_to_gguf.py` → `llama-quantize`): the BF16 weights are quantized **block-wise** (k-quants: per-block scales with mixed bit allocation), with **no retraining**. The original `chat_template` is embedded in each file.

## Requirements

| | |
|---|---|
| **llama.cpp** | build **≥ b9180** — architecture `qwen35` needs ≥ b9180, and the commands on this card use `-fa on` (≥ b6325). Load-checked here with b12005 (commit e0f142d) and b10724 on CPU — the loader accepts the model, tokenizer and the `-fa on` / q8_0 KV-cache flags; no generation was run in that check. |
| **Memory** | the chosen GGUF file (sizes in the table) plus KV cache for the context (`-c`); `--cache-type-k/v q8_0 -fa on` halves the KV cache. |
| **Input** | **text only** — this repo has no `mmproj` (vision projector) file. |
| **Ollama / LM Studio** | not tested here. |

## Run with llama.cpp

```bash
# pull one quant straight from the Hub and serve an OpenAI-compatible API on :8080
llama-server -hf axteam/Qwen3.8-27B-GGUF:Q4_K_M -c 8192 --jinja

# or download a single file and chat in the terminal
hf download axteam/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf --local-dir .
llama-cli -m Qwen3.8-27B-Q4_K_M.gguf -c 8192 --jinja
```

Query the server:

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "대한민국의 수도는 어디인가요?"}],
  "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 512
}'
```

### Reduce KV-cache memory (optional, run-time)
KV-cache type is chosen at run time in llama.cpp (it is not baked into the file):
```bash
llama-server -hf axteam/Qwen3.8-27B-GGUF:Q4_K_M -c 8192 --jinja --cache-type-k q8_0 --cache-type-v q8_0 -fa on
```

## Ollama
```bash
ollama run hf.co/axteam/Qwen3.8-27B-GGUF:Q4_K_M
```

## LM Studio
Search for `axteam/Qwen3.8-27B-GGUF` in the model browser and pick a quant (e.g. `Q4_K_M`).

> Recommended sampling (from the original `generation_config.json`): `temperature=1.0`, `top_p=0.95`, `top_k=20`.

## License
Quantized derivative of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). Original weights & tokenizer belong to the Qwen team.

---
## 한국어 요약

**[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)의 GGUF 양자화 빌드 (llama.cpp·Ollama·LM Studio용)** · 제작 axteam

### 개요
| 항목 | 내용 |
|---|---|
| 원본 모델 | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| 형식 | GGUF (llama.cpp) |
| 변환 도구 | llama.cpp `convert_hf_to_gguf.py` → `llama-quantize` |
| 포함 | 텍스트 백본(언어 모델)만 — 비전 인코더 미포함, 이 GGUF로는 OCR 불가 |
| 파일 수 | 2개 |

### 양자화 상세
- **양자화 절차**: PTQ(학습 후 양자화, 재학습 없음), llama.cpp 블록 단위 양자화(k-quant `Q*_K`, `Q8_0`)
- **chat template**: GGUF 파일에 내장
- **품질**: 미측정 — 아래 파일별 설명은 llama.cpp 양자화 단계의 일반적 경향

**제공 파일**

| 파일 | 크기 | 설명 |
|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | 16.8 GB | 4비트 k-quant — 크기·품질 균형 (**권장**) |
| `Qwen3.8-27B-Q8_0.gguf` | 29.0 GB | 8비트 — 원본에 가장 가까움 |

### 요구사항
- **llama.cpp**: b9180 이상 (`qwen35` 구조 지원 b9180부터, 예시 명령의 `-fa on` 옵션 b6325부터)
- **로딩 확인**: b12005(커밋 e0f142d)와 b10724에서 CPU로 모델·토크나이저 로딩과 `-fa on`·q8_0 KV 캐시 옵션 인식만 확인 (생성은 실행 미확인)
- **메모리**: 선택한 GGUF 파일 크기 + 컨텍스트(`-c`)만큼의 KV 캐시 (KV 캐시 줄이는 방법은 아래 사용법 참고)
- **입력**: 텍스트 전용 — 비전 인코더·투영층 파일(`mmproj`)이 없어 이미지 입력 불가
- **Ollama·LM Studio**: 제작 측에서 실행 미확인

### 사용법
- **파일 받기**: `hf download axteam/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf --local-dir .` (아래 `-m` 명령에 필요, 다른 양자화는 파일명의 `Q4_K_M`만 변경)
- **서버 실행**: `llama-server -hf axteam/Qwen3.8-27B-GGUF:Q4_K_M -c 8192 --jinja` — OpenAI 호환 API
- **터미널 대화**: `llama-cli -m Qwen3.8-27B-Q4_K_M.gguf -c 8192 --jinja`
- **Ollama**: `ollama run hf.co/axteam/Qwen3.8-27B-GGUF:Q4_K_M`
- **LM Studio**: 모델 검색에서 `axteam/Qwen3.8-27B-GGUF` 입력 후 양자화 단계 선택
- **KV 캐시 절감(선택)**: `--cache-type-k q8_0 --cache-type-v q8_0 -fa on` — KV 캐시 약 절반, GGUF 파일과 상관없이 실행 시 지정
- **권장 샘플링**: `temperature=1.0`, `top_p=0.95`, `top_k=20` (원본 `generation_config.json` 기준)

### 라이선스
- **라이선스**: Apache-2.0 — 원본 모델의 라이선스를 그대로 따름
- **권리**: 원본 가중치·토크나이저는 Qwen 팀(Alibaba) 소유, 이 리포는 양자화 파생물
