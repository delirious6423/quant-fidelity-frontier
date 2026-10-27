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

| File | Size |
|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | 16.8 GB |
| `Qwen3.8-27B-Q8_0.gguf` | 29.0 GB |

- **Q8_0** — 8-bit, highest GGUF quality, closest to the original.
- **Q4_K_M** — 4-bit (k-quant medium), ~half the size, small quality trade-off.

> Text generation via llama.cpp. Converted with llama.cpp `convert_hf_to_gguf.py`; the original `chat_template` is embedded. (The vision encoder is not included in these GGUF files.)

## Usage

```bash
# llama.cpp (pulls one quant by tag)
llama-server -hf axteam/Qwen3.8-27B-GGUF:Q4_K_M -c 8192
# or with a downloaded file
llama-cli -m Qwen3.8-27B-Q8_0.gguf -p "안녕하세요"
```
```bash
# Ollama
ollama run hf.co/axteam/Qwen3.8-27B-GGUF:Q4_K_M
```

## License
Quantized derivative of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). Original weights & tokenizer belong to the Qwen team.

---
## 한국어 요약
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) 를 llama.cpp 용 **GGUF(Q8_0·Q4_K_M)** 로 변환. llama.cpp/Ollama/LM Studio 에서 사용. **Q8_0**=8비트 최고품질, **Q4_K_M**=4비트 절반 용량. 원본 chat_template 내장(비전 인코더는 GGUF에 미포함, 텍스트 전용). 라이선스 Apache-2.0(원본 Qwen).
