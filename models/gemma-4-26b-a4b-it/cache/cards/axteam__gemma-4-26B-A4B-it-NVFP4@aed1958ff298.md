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
  - nvfp4
  - fp4
  - conversational
---
# gemma-4-26B-A4B-it · NVFP4 (W4A4, FP4)

4-bit quantization of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (commit `4d7ae49`), produced for efficient **vLLM serving**. Quantized by **axteam**.

| | |
|---|---|
| **Base model** | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (commit `4d7ae49`) |
| **Method** | NVFP4 (W4A4, FP4) |
| **Format** | compressed-tensors `nvfp4-pack-quantized` |
| **Weights / activations** | W4A4 — FP4 (E2M1) weights group 16 (FP8 E4M3 scale + per-tensor FP32 global scale), FP4 activations |
| **Architecture** | Gemma 4 · Mixture-of-Experts (8 active / 128 total + 1 shared) |
| **Size on disk** | ~17.2 GB |
| **Calibration** | 256 samples (Korean-heavy; -it includes 32 self-generated thinking-mode answers) |
| **License** | [Gemma 4 / Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license) |

## Quantization details

- **Scheme** — W4A4 — FP4 (E2M1) weights group 16 (FP8 E4M3 scale + per-tensor FP32 global scale), FP4 activations, stored as compressed-tensors `nvfp4-pack-quantized`.
- **Kept in BF16** — vision tower, vision projection, token embeddings (tied with lm_head), MoE router, and all norms, plus the shared MLP. These stay unquantized to preserve quality — about 7% of all parameters in this build.
- **Tokenizer / `chat_template` / `processor_config` / `generation_config`** are byte-identical to the original, including Google's July-2026 corrected `chat_template`. `config.json` is the original plus a `quantization_config` block.

## Quantization method — PTQ (AWQ / RTN) vs QAT

This model is a **post-training quantization (PTQ)**: the released BF16 weights are quantized *after* training, with **no retraining**. How close each build stays to BF16 is reported in the measured sections where available.

- **RTN** (round-to-nearest) — weights are simply rounded to the quantization grid with per-group scales; no calibration data. Fastest, but outliers aren't compensated, so 4-bit fidelity is the lowest. Used only as an internal baseline here, **not published**.
- **AWQ** (activation-aware weight quantization) — a small calibration set finds the weight channels that multiply the largest activations; those salient channels are protected by per-channel scaling (the scale is folded into adjacent layers, e.g. LayerNorm → q/k/v) before rounding. Still PTQ (no gradient, seconds–minutes), and far better than RTN at 4-bit.
- **QAT** (quantization-aware training) — the model is *fine-tuned* with quantization simulated in the forward pass, so the weights adapt to quantization error via gradient descent. Best low-bit fidelity, but needs the full training stack, data and GPU hours. Google ships separate Gemma QAT checkpoints; **this repo is not one of them**.

**Rough low-bit fidelity:** QAT ≥ AWQ > RTN.  **Cost:** QAT ≫ AWQ > RTN.
_This build: **AWQ** — a calibration-based PTQ (256-sample calibration, no gradient)._

## Evaluation (measured)

Measured against the BF16 original on an internal mixed English/Korean set. Quantization variant: AWQ attention + per-expert (full).

| Metric | This model | BF16 original |
|---|---|---|
| Perplexity (lower is better) | 1.2802 | 1.1284 (**+13.5%**) |
| Top-1 next-token agreement | 91.0% | 98.7% |

> These numbers measure fidelity to the base model, not absolute task scores.

## Usage (vLLM)

```bash
vllm serve axteam/gemma-4-26B-A4B-it-NVFP4
```

```python
from vllm import LLM, SamplingParams

llm = LLM(model="axteam/gemma-4-26B-A4B-it-NVFP4")
params = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=1024)
out = llm.chat(
    [{"role": "user", "content": "Write a short joke about saving RAM."}],
    params,
)
print(out[0].outputs[0].text)
```

> Recommended sampling (per Google): `temperature=1.0`, `top_p=0.95`, `top_k=64`.
> Full speed needs Blackwell FP4 tensor cores (e.g. B200, RTX PRO 6000); older GPUs run it through slower fallback kernels.

## Reducing KV-cache memory (optional, serve-time)

The KV cache is **unquantized (BF16)** here — the standard for quantized checkpoints
(RedHatAI, NVIDIA, etc. ship the same way). KV-cache precision is a **runtime** choice,
independent of the weight quantization above. For long context or large batches you can
roughly halve KV-cache memory with FP8:

```bash
vllm serve axteam/gemma-4-26B-A4B-it-NVFP4 --kv-cache-dtype fp8
```

> `--kv-cache-dtype` accepts `auto` (default, BF16), `fp8` / `fp8_e4m3`, `fp8_e5m2`.
> For a GGUF build the llama.cpp equivalent is `--cache-type-k q8_0 --cache-type-v q8_0`.

## License

This is a quantized derivative of Google's Gemma 4. Use is governed by the
[Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license). All original
weights, tokenizer, and chat template belong to Google DeepMind.

---

## 한국어 요약

[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) 를 **NVFP4 (W4A4, FP4)** 로 양자화한 서빙용(4-bit) 모델입니다. 작성: **axteam**.

- **방식** — 가중치·활성화 모두 FP4(W4A4). Blackwell(B200 등) FP4 텐서코어에서 제 속도. 활성화도 FP4라 품질 손실이 가장 클 수 있음 — 품질 우선이면 8비트(FP8·AWQ-8bit) 권장. 형식은 compressed-tensors `nvfp4-pack-quantized`.
- **BF16 유지** — 비전 타워·비전 투영·토큰 임베딩(lm_head와 묶임)·MoE 라우터·노름·공유 MLP (이 빌드 전체 파라미터의 약 7%).
- **토크나이저·chat_template·processor·generation_config** 는 원본과 바이트까지 동일 (구글이 2026-07에 고친 chat_template 그대로). `config.json` 은 원본 + `quantization_config`.
- **양자화 방식(PTQ vs QAT)** — 학습 후(PTQ) 방식이라 **재학습 없이** 원본 가중치를 저비트로 바꿉니다. **RTN**=보정 없이 반올림만(가장 빠르나 이상치 보상이 안 돼 4비트 품질 최저, 내부 기준용·비공개) · **AWQ**=소량 보정으로 큰 활성화와 곱해지는 중요 채널을 채널 스케일로 보호한 뒤 반올림(경사 없음, 4비트서 RTN보다 훨씬 좋음) · **QAT**=양자화를 흉내 내며 **재학습**해 가중치가 오차에 적응(저비트 최상이나 학습 스택·데이터·GPU 필요, 구글 Gemma QAT 체크포인트는 별도 — **본 레포는 QAT 아님**). 저비트 품질 ≈ QAT ≥ AWQ > RTN. **본 빌드: AWQ — 보정 기반 PTQ(256문항 보정, 경사 없음).**
- **품질(실측)** — BF16 원본 대비 perplexity 1.2802(원본 1.1284, **+13.5%**), top-1 일치도 91.0%(원본 98.7%). 내부 영/한 평가셋에서 측정. 변형: AWQ attention + per-expert (full).
- **사용** — `vllm serve axteam/gemma-4-26B-A4B-it-NVFP4`. 권장 샘플링 `temperature=1.0, top_p=0.95, top_k=64`.
- **KV 캐시 절감(선택, 서빙 시)** — 기본은 BF16 KV(양자화 체크포인트 표준, RedHat·NVIDIA 동일). 가중치 양자화와 **무관한 런타임 선택**. 긴 문맥·대배치에선 `vllm serve … --kv-cache-dtype fp8` 로 KV 메모리 절반. (GGUF라면 llama.cpp `--cache-type-k q8_0 --cache-type-v q8_0`)
- **라이선스** — 구글 Gemma 4 파생물. [Gemma 4 라이선스](https://ai.google.dev/gemma/docs/gemma_4_license) 적용. 원본 가중치·토크나이저·chat_template 권리는 Google DeepMind.
