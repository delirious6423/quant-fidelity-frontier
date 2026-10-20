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
  - w4a16
  - conversational
---
# gemma-4-26B-A4B-it · NVFP4A16 (W4A16, FP4 weights)

4-bit quantization of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (commit `4d7ae49`), produced for efficient **vLLM serving**. Quantized by **axteam**.

| | |
|---|---|
| **Base model** | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (commit `4d7ae49`) |
| **Method** | NVFP4A16 (W4A16, FP4 weights) |
| **Format** | compressed-tensors `nvfp4-pack-quantized` |
| **Weights / activations** | W4A16 — FP4 (E2M1) weights group 16 (FP8 E4M3 scale + per-tensor FP32 global scale), BF16 activations |
| **Architecture** | Gemma 4 · Mixture-of-Experts (8 active / 128 total + 1 shared) |
| **Size on disk** | ~16.0 GB |
| **Calibration** | 256 samples (Korean-heavy; -it includes thinking-mode answers) |
| **License** | [Gemma 4 / Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license) |

## Quantization details

- **Scheme** — W4A16 — FP4 (E2M1) weights group 16 (FP8 E4M3 scale + per-tensor FP32 global scale), BF16 activations, stored as compressed-tensors `nvfp4-pack-quantized`.
- **Kept in BF16** — vision tower, vision projection, token embeddings (tied with lm_head), MoE router, and all norms, plus the shared MLP. These are left unquantized to preserve quality (a small fraction of total parameters).
- **Tokenizer / `chat_template` / `processor_config` / `generation_config`** are byte-identical to the original, including Google's July-2026 corrected `chat_template`. `config.json` is the original plus a `quantization_config` block.

## Quantization method — PTQ (AWQ / RTN) vs QAT

This model is a **post-training quantization (PTQ)**: the released BF16 weights are quantized *after* training, with **no retraining**, so it stays faithful to the original model's behaviour.

- **RTN** (round-to-nearest) — weights are simply rounded to the quantization grid with per-group scales; no calibration data. Fastest, but outliers aren't compensated, so 4-bit fidelity is the lowest. Used only as an internal baseline here, **not published**.
- **AWQ** (activation-aware weight quantization) — a small calibration set finds the weight channels that multiply the largest activations; those salient channels are protected by per-channel scaling (the scale is folded into adjacent layers, e.g. LayerNorm → q/k/v) before rounding. Still PTQ (no gradient, seconds–minutes), and far better than RTN at 4-bit.
- **QAT** (quantization-aware training) — the model is *fine-tuned* with quantization simulated in the forward pass, so the weights adapt to quantization error via gradient descent. Best low-bit fidelity, but needs the full training stack, data and GPU hours. Google ships separate Gemma QAT checkpoints; **this repo is not one of them**.

**Rough low-bit fidelity:** QAT ≥ AWQ > RTN.  **Cost:** QAT ≫ AWQ > RTN.
_This build: **AWQ** — a calibration-based PTQ (256-sample calibration, no gradient)._

## Evaluation

Quantization fidelity was checked against the BF16 original using perplexity and top-1 next-token agreement on an internal mixed English/Korean set. In our runs the 8-bit variants (AWQ INT8 / FP8) stay closest to the original, while the 4-bit variants (AWQ INT4 / NVFP4) trade a little quality for roughly half the footprint.

## Usage (vLLM)

```bash
vllm serve axteam/gemma-4-26B-A4B-it-NVFP4A16
```

```python
from vllm import LLM, SamplingParams

llm = LLM(model="axteam/gemma-4-26B-A4B-it-NVFP4A16")
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
vllm serve axteam/gemma-4-26B-A4B-it-NVFP4A16 --kv-cache-dtype fp8
```

> `--kv-cache-dtype` accepts `auto` (default, BF16), `fp8` / `fp8_e4m3`, `fp8_e5m2`.
> For a GGUF build the llama.cpp equivalent is `--cache-type-k q8_0 --cache-type-v q8_0`.

## License

This is a quantized derivative of Google's Gemma 4. Use is governed by the
[Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license). All original
weights, tokenizer, and chat template belong to Google DeepMind.

---

## 한국어 요약

[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) 를 **NVFP4A16 (W4A16, FP4 weights)** 로 양자화한 서빙용(4-bit) 모델입니다. 작성: **axteam**.

- **방식** — FP4(E2M1) 가중치 16묶음 + 활성화는 BF16(W4A16). W4A4 NVFP4보다 품질이 크게 좋음(활성화 손실 없음), Blackwell 활성화 속도만 포기. 4비트 중 품질 최상급. 형식은 compressed-tensors `nvfp4-pack-quantized`.
- **BF16 유지** — 비전 타워·비전 투영·토큰 임베딩(lm_head와 묶임)·MoE 라우터·노름·공유 MLP (전체의 극히 일부).
- **토크나이저·chat_template·processor·generation_config** 는 원본과 바이트까지 동일 (구글이 2026-07에 고친 chat_template 그대로). `config.json` 은 원본 + `quantization_config`.
- **양자화 방식(PTQ vs QAT)** — 학습 후(PTQ) 방식이라 **재학습 없이** 원본 가중치를 저비트로 바꿉니다. **RTN**=보정 없이 반올림만(가장 빠르나 이상치 보상이 안 돼 4비트 품질 최저, 내부 기준용·비공개) · **AWQ**=소량 보정으로 큰 활성화와 곱해지는 중요 채널을 채널 스케일로 보호한 뒤 반올림(경사 없음, 4비트서 RTN보다 훨씬 좋음) · **QAT**=양자화를 흉내 내며 **재학습**해 가중치가 오차에 적응(저비트 최상이나 학습 스택·데이터·GPU 필요, 구글 Gemma QAT 체크포인트는 별도 — **본 레포는 QAT 아님**). 저비트 품질 ≈ QAT ≥ AWQ > RTN. **본 빌드: AWQ — 보정 기반 PTQ(256문항 보정, 경사 없음).**
- **품질** — BF16 원본 대비 perplexity·top-1 일치도로 검증(영/한 혼합셋). 8비트(AWQ INT8·FP8)가 원본에 가장 가깝고, 4비트(AWQ INT4·NVFP4)는 용량을 절반으로 줄이는 대신 품질을 조금 양보.
- **사용** — `vllm serve axteam/gemma-4-26B-A4B-it-NVFP4A16`. 권장 샘플링 `temperature=1.0, top_p=0.95, top_k=64`.
- **KV 캐시 절감(선택, 서빙 시)** — 기본은 BF16 KV(양자화 체크포인트 표준, RedHat·NVIDIA 동일). 가중치 양자화와 **무관한 런타임 선택**. 긴 문맥·대배치에선 `vllm serve … --kv-cache-dtype fp8` 로 KV 메모리 절반. (GGUF라면 llama.cpp `--cache-type-k q8_0 --cache-type-v q8_0`)
- **라이선스** — 구글 Gemma 4 파생물. [Gemma 4 라이선스](https://ai.google.dev/gemma/docs/gemma_4_license) 적용. 원본 가중치·토크나이저·chat_template 권리는 Google DeepMind.
