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
| **Format** | compressed-tensors `float-quantized` (scheme `FP8_DYNAMIC`) |
| **Weights / activations** | W8A8 — FP8 E4M3 weights (per-output-channel scale), dynamic per-token FP8 activations |
| **Architecture** | Gemma 4 · Mixture-of-Experts (8 active / 128 total + 1 shared) |
| **Size on disk** | ~27.2 GB |
| **Calibration** | None (data-free static weight scales) |
| **License** | [Gemma 4 / Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license) |

## Quantization details

- **Scheme** — W8A8 — FP8 E4M3 weights (per-output-channel scale), dynamic per-token FP8 activations, stored as compressed-tensors `float-quantized` (scheme `FP8_DYNAMIC`).
- **Kept in BF16** — vision tower, vision projection, token embeddings (tied with lm_head), MoE router, and all norms. These stay unquantized to preserve quality — about 5% of all parameters in this build.
- **Tokenizer / `chat_template` / `processor_config` / `generation_config`** are byte-identical to the original, including Google's July-2026 corrected `chat_template`. `config.json` is the original plus a `quantization_config` block.

## Quantization method — PTQ (AWQ / RTN) vs QAT

This model is a **post-training quantization (PTQ)**: the released BF16 weights are quantized *after* training, with **no retraining**. How close each build stays to BF16 is reported in the measured sections where available.

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

## Requirements

| | |
|---|---|
| **Runtime** | vLLM — the architecture is in vLLM ≥ 0.19.0; this build was run here only for the evaluation above, not yet with vLLM 0.32.0 (other builds of this model were, with vLLM 0.32.0). |
| **vLLM install** | vLLM 0.32 itself needs Python 3.11–3.14, torch 2.13 (CUDA 13.0 build, so an NVIDIA driver with CUDA 13 support), transformers ≥5.16.1,<5.20 and compressed-tensors 0.17.0 (both pinned by vLLM; 0.17.0 reads this checkpoint, which was saved with 0.19.0 — do not upgrade it separately). |
| **GPU** | NVIDIA Turing (sm75, e.g. T4) or newer. FP8 math runs on Ada / Hopper / Blackwell (sm89+); on Turing and Ampere vLLM switches to weight-only FP8 automatically. V100 (sm70) is not supported. (Kernel choice per GPU from vLLM 0.32's selection code; this build was run here only for the evaluation above.) |
| **Memory** | Weights: **27.2 GB**. Add KV cache (grows with context length × concurrent requests) and a few GB for activations / CUDA graphs; use `--tensor-parallel-size N` when the weights do not fit one GPU. |
| **Transformers (without vLLM)** | Not loadable by Transformers (checked up to 5.22.1: no conversion rule for per-expert checkpoints, the experts stay uninitialised) — use vLLM. |
| **trust_remote_code** | not needed |

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
> Thinking mode is off by default; turn it on with `chat_template_kwargs={"enable_thinking": True}` (Python `llm.chat(..., chat_template_kwargs=...)` or the request body), and add `--reasoning-parser gemma4` to `vllm serve` to return the reasoning separately.

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

**[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)의 FP8 (W8A8) 양자화 빌드** · 제작 axteam

### 개요
| 항목 | 내용 |
|---|---|
| 원본 모델 | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (커밋 `4d7ae49`) |
| 양자화 방식 | FP8 동적 양자화 (W8A8) |
| 저장 형식 | compressed-tensors `float-quantized` (scheme `FP8_DYNAMIC`) |
| 모델 구조 | Gemma 4 · MoE (전문가 128개 중 8개 활성 + 항상 쓰는 공유 MLP) |
| 용량 | 약 27.2 GB |
| 보정 데이터 | 없음 (가중치 스케일은 가중치만으로 계산, 활성화 스케일은 실행 시 계산) |

### 양자화 상세
- **가중치**: FP8(E4M3), 출력 채널별 스케일
- **활성화**: 실행 시 토큰별 동적 FP8
- **양자화 절차**: PTQ(학습 후 양자화, 재학습 없음), 가중치 스케일은 미리 계산하고 활성화 스케일은 실행 시 토큰마다 계산
- **BF16 유지**: 비전 타워, 비전 투영층, 토큰 임베딩(LM 헤드와 공유), MoE 라우터, 모든 정규화층 — 전체 파라미터의 약 5%
- **원본 파일**: 토크나이저·chat template(Google의 2026년 7월 수정본)·processor·generation config를 바이트 단위로 그대로 사용, `config.json`은 원본에 `quantization_config` 블록만 추가

### 품질 (실측)
- **원본 대비 충실도**: PPL 1.1531 (원본 1.1284, +2.2%, 낮을수록 원본에 가까움), 다음 토큰 top-1 일치율 96.3% (원본 자체 기준값 98.7%) — 내부 영어·한국어 혼합 평가셋, 원본(BF16)과 비교 측정

### 요구사항
- **실행 엔진**: vLLM 0.19.0 이상(이 구조를 지원하는 첫 버전), 이 빌드는 위 품질 측정 때 실행됨, vLLM 0.32.0에서는 아직 실행 미확인 (같은 모델의 다른 빌드는 vLLM 0.32.0에서 실행 확인)
- **vLLM 0.32 설치 조건**: Python 3.11~3.14, torch 2.13(CUDA 13.0 빌드, CUDA 13 지원 드라이버 필요), transformers 5.16.1 이상 5.20 미만, compressed-tensors 0.17.0 — transformers·compressed-tensors는 vLLM이 버전을 고정하므로 compressed-tensors를 따로 업그레이드하지 말 것 (이 체크포인트는 0.19.0으로 저장됐지만 0.17.0에서도 로드됨)
- **GPU**: NVIDIA Turing(sm75, 예: T4) 이상, V100(sm70) 미지원 — FP8 연산은 Ada·Hopper·Blackwell(sm89 이상)에서 수행, Turing·Ampere는 vLLM이 가중치 전용 FP8로 자동 전환 (GPU별 지원 여부는 vLLM 0.32 커널 선택 코드 기준, 이 빌드는 품질 측정 때만 실행됨)
- **메모리**: 가중치 27.2 GB + KV 캐시(컨텍스트 길이 × 동시 요청 수에 비례) + 활성화·CUDA graph용 수 GB — GPU 한 장에 다 들어가지 않으면 `--tensor-parallel-size N`으로 분산
- **Transformers로 직접 사용**: 불가 — 5.22.1까지 로드되지 않음 확인 (전문가별 체크포인트 변환 규칙이 없어 전문가가 초기화되지 않음), vLLM 사용
- **trust_remote_code**: 불필요

### 사용법
- **서버 실행**: `vllm serve axteam/gemma-4-26B-A4B-it-FP8-dynamic`
- **Python**: `LLM(model="axteam/gemma-4-26B-A4B-it-FP8-dynamic").chat(...)` — 전체 예시는 위 영어 본문 'Usage (vLLM)' 절 참고
- **권장 샘플링**: `temperature=1.0`, `top_p=0.95`, `top_k=64` (Google 권장값)
- **thinking 모드**: 기본 꺼짐 — 켜려면 `chat_template_kwargs={"enable_thinking": True}` 전달(Python은 `llm.chat()` 인자, HTTP는 요청 본문의 `"chat_template_kwargs": {"enable_thinking": true}`), 추론 과정을 답변과 분리해 받으려면 `vllm serve`에 `--reasoning-parser gemma4` 추가
- **KV 캐시 절감(선택)**: `vllm serve`에 `--kv-cache-dtype fp8` 추가 — 기본 BF16인 KV 캐시를 FP8로 저장해 KV 캐시 메모리를 약 절반으로 줄임 (긴 컨텍스트·동시 요청이 많을 때 유리, 가중치 양자화와 별개인 실행 옵션)

### 라이선스
- **라이선스**: Apache-2.0 ([Gemma 4 라이선스 안내](https://ai.google.dev/gemma/docs/gemma_4_license)) — 원본 모델의 라이선스를 그대로 따름
- **권리**: 원본 가중치·토크나이저·chat template은 Google DeepMind 소유, 이 리포는 양자화 파생물
