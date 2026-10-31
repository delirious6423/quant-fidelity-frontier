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
  - fp8
---
# Qwen3.8-27B · FP8 (dynamic)

8-bit quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), produced for efficient **vLLM serving**. Quantized by **axteam**.

| | |
|---|---|
| **Base model** | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Method** | FP8 (dynamic) |
| **Format** | compressed-tensors `float-quantized` (scheme `FP8_DYNAMIC`) |
| **Weights / activations** | W8A8 — FP8 E4M3 weights (per-output-channel scale), dynamic per-token FP8 activations |
| **Architecture** | Hybrid (linear-attention + full-attention) · Dense · multimodal (vision) · MTP head |
| **Size on disk** | ~31.2 GB |
| **Calibration** | None (data-free static weight scales) |
| **License** | [Apache-2.0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) |

## Quantization details

- **Scheme** — W8A8 — FP8 E4M3 weights (per-output-channel scale), dynamic per-token FP8 activations, stored as compressed-tensors `float-quantized` (scheme `FP8_DYNAMIC`).
- **Kept in BF16** — the vision tower & projector (`model.visual.*`), the multi-token-prediction head (`mtp.*`), all RMSNorms, token embeddings, the LM head, the SSM non-linear parts (`linear_attn.conv1d/A_log/dt_bias`). Left unquantized to preserve quality (about 12% of all parameters in this build).
- **Tokenizer / `chat_template` / `generation_config`** are identical to the original. `config.json` is the original plus a `quantization_config` block.

This is **post-training quantization (PTQ)** — no retraining. FP8 dynamic is calibration-free.

## Evaluation (measured)

Fidelity to the BF16 original on an internal mixed English/Korean set (100 prompts, BF16 greedy answers scored under each build — same harness for every row).

| Metric | This model | BF16 original |
|---|---|---|
| Perplexity (lower = closer) | 1.3651 | 1.3586 (**+0.5%**) |
| Top-1 next-token agreement | 97.3% | 99.4% |

> These measure fidelity to the base model, not absolute task scores.

## Usage (vLLM)

Start an OpenAI-compatible server:

```bash
vllm serve axteam/Qwen3.8-27B-FP8-dynamic --max-model-len 32768
```

Query it:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "axteam/Qwen3.8-27B-FP8-dynamic",
  "messages": [{"role": "user", "content": "대한민국의 수도는 어디인가요?"}],
  "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 512
}'
```

Offline inference in Python:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="axteam/Qwen3.8-27B-FP8-dynamic", max_model_len=32768)
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
vllm serve axteam/Qwen3.8-27B-FP8-dynamic --kv-cache-dtype fp8
```

## License

Quantized derivative of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). All original weights, tokenizer and chat template belong to the Qwen team.

---

## 한국어 요약

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) 를 **FP8 (dynamic)** 로 양자화한 서빙용(8-bit) 모델입니다. 작성: **axteam**.

- **방식** — FP8 E4M3. 가중치는 출력 채널마다 스케일 하나(정적·보정 불필요), 활성화는 추론 시 토큰마다 동적 FP8. 형식 compressed-tensors `float-quantized` (scheme `FP8_DYNAMIC`).
- **BF16 유지** — 비전 타워·MTP 헤드·모든 norm·임베딩/LM 헤드·SSM 비선형부(conv1d/A_log/dt_bias).
- **PTQ** — 재학습 없음.
- **품질(실측)** — BF16 원본 대비 perplexity 1.3651(원본 1.3586, **+0.5%**), top-1 일치도 97.3%(원본 99.4%). 내부 영/한 100문항, 같은 하네스.
- **라이선스** — 원본 Apache-2.0 파생물. 가중치·토크나이저·chat_template 권리는 Qwen 팀.
- **실행** — `vllm serve axteam/Qwen3.8-27B-FP8-dynamic --max-model-len 32768` 로 OpenAI 호환 서버(포트 8000), 또는 Python `LLM(model="axteam/Qwen3.8-27B-FP8-dynamic").chat(...)` (위 예시). 권장 샘플링 `temperature=1.0`, `top_p=0.95`, `top_k=20`. thinking 끄기: `chat_template_kwargs={"enable_thinking": False}`. 이미지 입력은 `image_url` 메시지로.
- **KV 캐시 절감(런타임)** — `--kv-cache-dtype fp8`.
