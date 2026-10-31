---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
tags:
  - qwen
  - qwen3_5
  - compressed-tensors
  - vllm
  - quantized
  - conversational
  - multimodal
  - image-text-to-text
  - awq
  - int4
---
# Qwen3.8-27B · AWQ INT4 (W4A16)

4-bit quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), produced for efficient **vLLM serving**. Quantized by **axteam**.

| | |
|---|---|
| **Base model** | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Method** | AWQ INT4 (W4A16) |
| **Format** | compressed-tensors `pack-quantized` |
| **Weights / activations** | W4A16 — INT4 asymmetric weights, group size 32 (BF16 scale + INT4 zero-point), BF16 activations |
| **Architecture** | Hybrid (linear-attention + full-attention) · Dense · multimodal (vision) · MTP head |
| **Size on disk** | ~20.9 GB |
| **Calibration** | 256 samples (Korean-heavy mixed corpus) |
| **License** | [Apache-2.0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) |

## Quantization details

- **Scheme** — W4A16 — INT4 asymmetric weights, group size 32 (BF16 scale + INT4 zero-point), BF16 activations, stored as compressed-tensors `pack-quantized`.
- **Kept in BF16** — the vision tower & projector (`model.visual.*`), the multi-token-prediction head (`mtp.*`), all RMSNorms, token embeddings, the LM head, the SSM non-linear parts (`linear_attn.conv1d/A_log/dt_bias`). Left unquantized to preserve quality (about 12% of all parameters in this build).
- **Tokenizer / `chat_template` / `generation_config`** are identical to the original. `config.json` is the original plus a `quantization_config` block.

This is **post-training quantization (PTQ)** — no retraining. AWQ uses a small calibration set (no gradient) to protect salient weight channels before rounding.

## Evaluation (measured)

Fidelity to the BF16 original on an internal mixed English/Korean set (100 prompts, BF16 greedy answers scored under each build — same harness for every row).

| Metric | This model | BF16 original |
|---|---|---|
| Perplexity (lower = closer) | 1.3896 | 1.3586 (**+2.3%**) |
| Top-1 next-token agreement | 94.6% | 99.4% |

> These measure fidelity to the base model, not absolute task scores.

## Usage (vLLM)

Start an OpenAI-compatible server:

```bash
vllm serve axteam/Qwen3.8-27B-AWQ-4bit --max-model-len 32768
```

Query it:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "axteam/Qwen3.8-27B-AWQ-4bit",
  "messages": [{"role": "user", "content": "대한민국의 수도는 어디인가요?"}],
  "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 512
}'
```

Offline inference in Python:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="axteam/Qwen3.8-27B-AWQ-4bit", max_model_len=32768)
params = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=1024)
out = llm.chat([{"role": "user", "content": "Explain model quantization in two sentences."}], params)
print(out[0].outputs[0].text)

# image input (URL or data: URI)
messages = [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://example.com/your-image.png"}},
    {"type": "text", "text": "Describe this image."},
]}]
out = llm.chat(messages, params)
print(out[0].outputs[0].text)
```

> Recommended sampling (from the original `generation_config.json`): `temperature=1.0`, `top_p=0.95`, `top_k=20`.
> Thinking mode is on by default in the chat template; to turn it off pass `chat_template_kwargs={"enable_thinking": False}` (Python `llm.chat(..., chat_template_kwargs=...)`, or `"chat_template_kwargs": {"enable_thinking": false}` in the request body).
> Text-only serving: add `--limit-mm-per-prompt '{"image": 0, "video": 0}'` to skip loading the vision encoder's memory budget.

## Reducing KV-cache memory (optional, serve-time)

The KV cache is kept in BF16 by default. KV-cache precision is a **runtime** choice, independent of the
weight quantization above; for long contexts or large batches FP8 roughly halves KV-cache memory:

```bash
vllm serve axteam/Qwen3.8-27B-AWQ-4bit --kv-cache-dtype fp8
```

## License

Quantized derivative of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). All original weights, tokenizer and chat template belong to the Qwen team.

---

## 한국어 요약

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) 를 **AWQ INT4 (W4A16)** 로 양자화한 서빙용(4-bit) 모델입니다. 작성: **axteam**.

- **방식** — INT4 비대칭, 32개 묶음(BF16 스케일+INT4 영점), 활성화 BF16(W4A16). AWQ로 배율 이동 후 양자화. 형식 compressed-tensors `pack-quantized`.
- **BF16 유지** — 비전 타워·MTP 헤드·모든 norm·임베딩/LM 헤드·SSM 비선형부(conv1d/A_log/dt_bias).
- **PTQ** — 재학습 없음.
- **품질(실측)** — BF16 원본 대비 perplexity 1.3896(원본 1.3586, **+2.3%**), top-1 일치도 94.6%(원본 99.4%). 내부 영/한 100문항, 같은 하네스.
- **라이선스** — 원본 Apache-2.0 파생물. 가중치·토크나이저·chat_template 권리는 Qwen 팀.
- **실행** — `vllm serve axteam/Qwen3.8-27B-AWQ-4bit --max-model-len 32768` 로 OpenAI 호환 서버(포트 8000), 또는 Python `LLM(model="axteam/Qwen3.8-27B-AWQ-4bit").chat(...)` (위 예시). 권장 샘플링 `temperature=1.0`, `top_p=0.95`, `top_k=20`. thinking 끄기: `chat_template_kwargs={"enable_thinking": False}`. 이미지 입력은 `image_url` 메시지로.
- **KV 캐시 절감(런타임)** — `--kv-cache-dtype fp8`.
