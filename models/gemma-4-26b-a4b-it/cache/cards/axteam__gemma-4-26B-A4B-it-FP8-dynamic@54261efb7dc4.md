---
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: transformers
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
tags:
  - gemma
  - gemma4
  - compressed-tensors
  - vllm
  - quantized
  - fp8
  - conversational
---
# gemma-4-26B-A4B-it · FP8 (dynamic)

8-bit quantization of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (commit `4d7ae49`), produced for efficient **vLLM serving**. Quantized by **axteam**.

| | |
|---|---|
| **Base model** | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (commit `4d7ae49`) |
| **Method** | FP8 (dynamic) |
| **Format** | compressed-tensors `FP8_DYNAMIC` |
| **Weights / activations** | W8A8 — FP8 E4M3 weights (per-output-channel scale), dynamic per-token FP8 activations |
| **Architecture** | Gemma 4 · Mixture-of-Experts (8 active / 128 total + 1 shared) |
| **Size on disk** | ~25.3 GB |
| **Calibration** | None (data-free static weight scales) |
| **License** | [Gemma 4 / Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license) |

## Quantization details

- **Scheme** — W8A8 — FP8 E4M3 weights (per-output-channel scale), dynamic per-token FP8 activations, stored as compressed-tensors `FP8_DYNAMIC`.
- **Kept in BF16** — vision tower, vision projection, token embeddings (tied with lm_head), MoE router, and all norms, plus the shared MLP. These are left unquantized to preserve quality (a small fraction of total parameters).
- **Tokenizer / `chat_template` / `processor_config` / `generation_config`** are byte-identical to the original, including Google's July-2026 corrected `chat_template`. `config.json` is the original plus a `quantization_config` block.

## Quantization method — PTQ (AWQ / RTN) vs QAT

This model is a **post-training quantization (PTQ)**: the released BF16 weights are quantized *after* training, with **no retraining**, so it stays faithful to the original model's behaviour.

- **RTN** (round-to-nearest) — weights are simply rounded to the quantization grid with per-group scales; no calibration data. Fastest, but outliers aren't compensated, so 4-bit fidelity is the lowest. Used only as an internal baseline here, **not published**.
- **AWQ** (activation-aware weight quantization) — a small calibration set finds the weight channels that multiply the largest activations; those salient channels are protected by per-channel scaling (the scale is folded into adjacent layers, e.g. LayerNorm → q/k/v) before rounding. Still PTQ (no gradient, seconds–minutes), and far better than RTN at 4-bit.
- **QAT** (quantization-aware training) — the model is *fine-tuned* with quantization simulated in the forward pass, so the weights adapt to quantization error via gradient descent. Best low-bit fidelity, but needs the full training stack, data and GPU hours. Google ships separate Gemma QAT checkpoints; **this repo is not one of them**.

**Rough low-bit fidelity:** QAT ≥ AWQ > RTN.  **Cost:** QAT ≫ AWQ > RTN.
_This build: **FP8 dynamic** — a calibration-free PTQ (static per-channel weight scales + runtime per-token activation scales)._

## Evaluation (measured)

Measured against the BF16 original on an internal mixed English/Korean set. Quantization variant: FP8 dynamic (W8A8).

| Metric | This model | BF16 original |
|---|---|---|
| Perplexity (lower is better) | 1.1531 | 1.1284 (**+2.2%**) |
| Top-1 next-token agreement | 96.3% | 98.7% |

> These numbers measure fidelity to the base model, not absolute task scores.

## Usage (vLLM)

```bash
vllm serve axteam/gemma-4-26B-A4B-it-FP8-dynamic
```

```python
from vllm import LLM, SamplingParams

llm = LLM(model="axteam/gemma-4-26B-A4B-it-FP8-dynamic")
params = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=1024)
out = llm.chat(
    [{"role": "user", "content": "Write a short joke about saving RAM."}],
    params,
)
print(out[0].outputs[0].text)
```

> Recommended sampling (per Google): `temperature=1.0`, `top_p=0.95`, `top_k=64`.
> Thinking mode: add the `<|think|>` token at the start of the system prompt to enable step-by-step reasoning.

## Reducing KV-cache memory (optional, serve-time)

The KV cache is **unquantized (BF16)** here — the standard for quantized checkpoints
(RedHatAI, NVIDIA, etc. ship the same way). KV-cache precision is a **runtime** choice,
independent of the weight quantization above. For long context or large batches you can
roughly halve KV-cache memory with FP8:

```bash
vllm serve axteam/gemma-4-26B-A4B-it-FP8-dynamic --kv-cache-dtype fp8
```

> `--kv-cache-dtype` accepts `auto` (default, BF16), `fp8` / `fp8_e4m3`, `fp8_e5m2`.
> For a GGUF build the llama.cpp equivalent is `--cache-type-k q8_0 --cache-type-v q8_0`.

## License

This is a quantized derivative of Google's Gemma 4. Use is governed by the
[Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license). All original
weights, tokenizer, and chat template belong to Google DeepMind.

---

## 한국어 요약

[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) 를 **FP8 (dynamic)** 로 양자화한 서빙용(8-bit) 모델입니다. 작성: **axteam**.

- **방식** — FP8 E4M3. 가중치는 출력 채널마다 스케일 하나(정적·보정 불필요), 활성화는 추론 시 토큰마다 동적 FP8. 형식은 compressed-tensors `FP8_DYNAMIC`.
- **BF16 유지** — 비전 타워·비전 투영·토큰 임베딩(lm_head와 묶임)·MoE 라우터·노름·공유 MLP (전체의 극히 일부).
- **토크나이저·chat_template·processor·generation_config** 는 원본과 바이트까지 동일 (구글이 2026-07에 고친 chat_template 그대로). `config.json` 은 원본 + `quantization_config`.
- **양자화 방식(PTQ vs QAT)** — 학습 후(PTQ) 방식이라 **재학습 없이** 원본 가중치를 저비트로 바꿉니다. **RTN**=보정 없이 반올림만(가장 빠르나 이상치 보상이 안 돼 4비트 품질 최저, 내부 기준용·비공개) · **AWQ**=소량 보정으로 큰 활성화와 곱해지는 중요 채널을 채널 스케일로 보호한 뒤 반올림(경사 없음, 4비트서 RTN보다 훨씬 좋음) · **QAT**=양자화를 흉내 내며 **재학습**해 가중치가 오차에 적응(저비트 최상이나 학습 스택·데이터·GPU 필요, 구글 Gemma QAT 체크포인트는 별도 — **본 레포는 QAT 아님**). 저비트 품질 ≈ QAT ≥ AWQ > RTN. **본 빌드: FP8 동적 — 보정 불필요 PTQ(가중치는 정적 채널 스케일, 활성화는 런타임 토큰별).**
- **품질(실측)** — BF16 원본 대비 perplexity 1.1531(원본 1.1284, **+2.2%**), top-1 일치도 96.3%(원본 98.7%). 내부 영/한 평가셋에서 측정. 변형: FP8 dynamic (W8A8).
- **사용** — `vllm serve axteam/gemma-4-26B-A4B-it-FP8-dynamic`. 권장 샘플링 `temperature=1.0, top_p=0.95, top_k=64`.
- **KV 캐시 절감(선택, 서빙 시)** — 기본은 BF16 KV(양자화 체크포인트 표준, RedHat·NVIDIA 동일). 가중치 양자화와 **무관한 런타임 선택**. 긴 문맥·대배치에선 `vllm serve … --kv-cache-dtype fp8` 로 KV 메모리 절반. (GGUF라면 llama.cpp `--cache-type-k q8_0 --cache-type-v q8_0`)
- **라이선스** — 구글 Gemma 4 파생물. [Gemma 4 라이선스](https://ai.google.dev/gemma/docs/gemma_4_license) 적용. 원본 가중치·토크나이저·chat_template 권리는 Google DeepMind.
