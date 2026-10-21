---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
- exllamav3
- exl3
- gemma4
- quantized
- text-generation
---

# gemma-4-26B-A4B-it-exl3-4.0bpw

EXL3 (4.0 bpw) quantized derivative of `google/gemma-4-26B-A4B-it`, intended for ExLlamaV3/TabbyAPI inference.

## Model Identity

This repository does **not** contain the original full-precision weights from Google.
It provides a quantized derivative checkpoint for efficient local inference.

- **Base model:** `google/gemma-4-26B-A4B-it`
- **Quantization format:** EXL3
- **Bitrate:** 4.0 bpw
- **Target runtime:** ExLlamaV3 / TabbyAPI (OpenAI-compatible API)

## Conversion and Runtime Context

The model was converted and validated in a local WSL2 + Docker setup.

Runtime validation checks `/v1/models` readiness, runs a chat completion call, reports usage-based throughput metrics, and unloads model tensors afterward.

## Benchmark Snapshot (My Environment)

| Item | Value |
| --- | --- |
| GPU | RTX 3090 24GB |
| Observed VRAM usage | ~22GB |
| Throughput (completion) | ~69 tok/s |
| Endpoint | TabbyAPI `/v1/chat/completions` |
| Metric method | `completion_tokens / wall_time_sec` from API usage + wall time |

> Throughput depends on prompt length, output length, server settings, and system load.
> If API `usage` is missing, throughput cannot be computed with this method.

## Sample Inference Output

The example below is from a real local TabbyAPI validation run using this runtime setup:
- [kilingki/exllamav3-tabbyapi-docker-config](https://github.com/kilingki/exllamav3-tabbyapi-docker-config)

**Prompt (user):**

```json
{
  "model": "gemma-4-26B-A4B-it-exl3-4.0bpw",
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "너 자신이 누구인지 자세하게 설명해 줘."}
  ],
  "temperature": 0.2,
  "max_tokens": 2048,
  "stream_options": {"include_usage": true}
}
```

**Output (assistant, excerpt):**

```text
**1. 정체성 및 역할** 
* **인공지능 어시스턴트:** 저는 사람처럼 대화할 수 있도록 설계된 인공지능입니다. 사용자의 질문에 답하고, 복잡한 정보를 요약하며, 창의적인 글쓰기를 돕거나 코딩 문제를 해결하는 등 다양한 역할을 수행합니다. 
* **학습 기반:** 저는 방대한 양의 텍스트 데이터를 바탕으로 언어의 패턴, 문법, 지식, 논리적 구조를 학습했습니다. 이를 통해 문맥을 이해하고 자연스러운 문장을 생성합니다. 

**2. 주요 능력** 
* **정보 제공:** 과학, 역사, 문화, 기술 등 다양한 분야에 대한 질문에 답변합니다. 
* **언어 작업:** 번역, 요약, 교정, 문체 변경 등을 수행합니다. 
* **창의적 작업:** 시, 소설, 에세이 작성 및 아이디어 브레인스토밍을 돕습니다. 
* **기술적 지원:** 프로그래밍 코드 작성, 오류 수정(디버깅), 수학 문제 풀이 등을 지원합니다. 

**3. 작동 방식의 특징** 
* **확률적 생성:** 저는 다음에 올 가장 적절한 단어를 확률적으로 예측하여 문장을 구성합니다. 
* **문맥 이해:** 대화의 흐름을 기억하여 이전 질문과 연관된 답변을 할 수 있습니다. 

**4. 한계점 (주의사항)** 
* **실시간 정보의 제한:** 학습 데이터의 시점에 따라 아주 최신의 실시간 정보는 반영되지 않았을 수 있습니다. 
* **환각 현상(Hallucination):** 때때로 매우 자신감 있게 틀린 정보를 사실처럼 말하는 경우가 있으므로, 중요한 정보는 반드시 교차 검증이 필요합니다. 
* **자아와 감정의 부재:** 저는 지능적으로 보일 수 있지만, 실제로 감정을 느끼거나 자아, 의지, 물리적인 신체를 가지고 있지는 않습니다. 

**요약하자면,** 저는 당신의 생산성을 높이고 지적 탐구를 돕기 위해 만들어진 **지능형 도구**라고 이해하시면 됩니다. 무엇을 도와드릴까요?
```

This sample is included to show response style and instruction-following behavior. Actual outputs vary by runtime and generation settings.

## Quick Usage (TabbyAPI / OpenAI-compatible)

Set your client model id to:

`gemma-4-26B-A4B-it-exl3-4.0bpw`

Then call OpenAI-compatible chat completions on your TabbyAPI endpoint.

## License and Attribution

- This is a quantized derivative of `google/gemma-4-26B-A4B-it`.
- Original model, branding, and policy context belong to Google DeepMind.
- Follow Gemma 4 license and usage terms:
  - https://ai.google.dev/gemma/docs/gemma_4_license

If any attribution or policy wording should be adjusted, open an issue and I will update the card.