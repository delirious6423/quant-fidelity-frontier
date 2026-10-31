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
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) 를 llama.cpp 용 **GGUF(Q8_0·Q4_K_M)** 로 변환. llama.cpp/Ollama/LM Studio 에서 사용. **Q8_0**=8비트 최고품질, **Q4_K_M**=4비트(Q8_0 의 약 60% 크기). 원본 chat_template 내장(비전 인코더는 GGUF에 미포함, 텍스트 전용). 라이선스 Apache-2.0(원본 Qwen).
- **파일** — `Q4_K_M`(4비트·권장), `Q8_0`(8비트·원본에 가장 가까움).
- **요구사항** — llama.cpp b9180 이상(구조 `qwen35` 는 b9180+, 카드 명령의 `-fa on` 은 b6325+). b12005·b10724 에서 CPU 로드 점검(생성은 안 해 봄). 텍스트 전용(mmproj 없음). Ollama·LM Studio 는 미확인.
- **실행** — `llama-server -hf axteam/Qwen3.8-27B-GGUF:Q4_K_M -c 8192 --jinja` (OpenAI 호환 서버 :8080) 또는 `llama-cli -m Qwen3.8-27B-Q4_K_M.gguf --jinja`.
- **Ollama** — `ollama run hf.co/axteam/Qwen3.8-27B-GGUF:Q4_K_M`. LM Studio 에서는 `axteam/Qwen3.8-27B-GGUF` 검색.
- **KV 캐시 절감(런타임)** — `--cache-type-k q8_0 --cache-type-v q8_0 -fa on`.
- 권장 샘플링 `temperature=1.0`, `top_p=0.95`, `top_k=20`.
- 텍스트 전용(비전 인코더 미포함).
