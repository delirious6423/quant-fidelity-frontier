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
  - awq
  - int4
  - conversational
---
# gemma-4-26B-A4B-it · AWQ INT4 (W4A16)

4-bit quantization of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (commit `4d7ae49`), produced for efficient **vLLM serving**. Quantized by **axteam**.

| | |
|---|---|
| **Base model** | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (commit `4d7ae49`) |
| **Method** | AWQ INT4 (W4A16) |
| **Format** | compressed-tensors `pack-quantized` |
| **Weights / activations** | W4A16 — INT4 asymmetric weights, group size 32 (BF16 scale + INT4 zero-point), BF16 activations |
| **Architecture** | Gemma 4 · Mixture-of-Experts (8 active / 128 total + 1 shared) |
| **Size on disk** | ~17.6 GB |
| **Calibration** | 256 samples (Korean-heavy; -it includes 32 self-generated thinking-mode answers) |
| **License** | [Gemma 4 / Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license) |

## Quantization details

- **Scheme** — W4A16 — INT4 asymmetric weights, group size 32 (BF16 scale + INT4 zero-point), BF16 activations, stored as compressed-tensors `pack-quantized`.
- **Kept in BF16** — vision tower, vision projection, token embeddings (tied with lm_head), MoE router, and all norms, plus the shared MLP. These stay unquantized to preserve quality — about 7% of all parameters in this build.
- **Tokenizer / `chat_template` / `processor_config` / `generation_config`** are byte-identical to the original, including Google's July-2026 corrected `chat_template`. `config.json` is the original plus a `quantization_config` block.

## Quantization method — PTQ (AWQ / RTN) vs QAT

This model is a **post-training quantization (PTQ)**: the released BF16 weights are quantized *after* training, with **no retraining**. How close each build stays to BF16 is reported in the measured sections where available.

- **RTN** (round-to-nearest) — weights are simply rounded to the quantization grid with per-group scales; no calibration data. Fastest, but outliers aren't compensated, so 4-bit fidelity is the lowest. Used only as an internal baseline here, **not published**.
- **AWQ** (activation-aware weight quantization) — a small calibration set finds the weight channels that multiply the largest activations; those salient channels are protected by per-channel scaling (the scale is folded into adjacent layers, e.g. LayerNorm → q/k/v) before rounding. Still PTQ (no gradient, seconds–minutes), and far better than RTN at 4-bit.
- **QAT** (quantization-aware training) — the model is *fine-tuned* with quantization simulated in the forward pass, so the weights adapt to quantization error via gradient descent. Best low-bit fidelity, but needs the full training stack, data and GPU hours. Google ships separate Gemma QAT checkpoints; **this repo is not one of them**.

**Rough low-bit fidelity:** QAT ≥ AWQ > RTN.  **Cost:** QAT ≫ AWQ > RTN.
_This build: **AWQ** — a calibration-based PTQ (256-sample calibration, no gradient)._

## Benchmarks (measured)

Every column was measured with the same harness and settings — [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on vLLM, greedy, completion-style prompts without a chat template (0- or 5-shot, as listed in each row). Percentages are recovery vs. the BF16 original.

| Benchmark | BF16 (original) | FP8-dynamic | AWQ-8bit | NVFP4A16 | NVFP4 | **AWQ-4bit (this repo)** |
|---|---|---|---|---|---|---|
| MMLU (5-shot) | 40.23 | 35.77 (88.9%) | 38.72 (96.2%) | 30.44 (75.7%) | 31.39 (78.0%) | **33.21 (82.5%)** |
| GSM8K (5-shot, strict) | 66.19 | 67.70 (102.3%) | 67.10 (101.4%) | 50.42 (76.2%) | 60.88 (92.0%) | **55.12 (83.3%)** |
| ARC-Challenge (0-shot) | 45.39 | 45.82 (100.9%) | 44.88 (98.9%) | 44.62 (98.3%) | 44.28 (97.6%) | **44.62 (98.3%)** |
| Winogrande (5-shot) | 58.48 | 60.22 (103.0%) | 59.91 (102.4%) | 58.64 (100.3%) | 57.77 (98.8%) | **58.48 (100.0%)** |
| KMMLU (5-shot, direct) | 36.52 | 34.81 (95.3%) | 35.84 (98.1%) | 33.42 (91.5%) | 34.37 (94.1%) | **33.91 (92.9%)** |
| HAE-RAE (0-shot) | 76.17 | 75.62 (99.3%) | 75.99 (99.8%) | 73.60 (96.6%) | 72.87 (95.7%) | **70.76 (92.9%)** |
| **Average** | **53.83** | **53.32 (99.1%)** | **53.74 (99.8%)** | **48.53 (90.1%)** | **50.26 (93.4%)** | **49.35 (91.7%)** |

> ⚠️ **Note:** average recovery is 91.7% — noticeable degradation at this precision; prefer an 8-bit build if quality matters most.

## Evaluation (measured)

Measured against the BF16 original on an internal mixed English/Korean set. Quantization variant: AWQ attention + per-expert (full).

| Metric | This model | BF16 original |
|---|---|---|
| Perplexity (lower is better) | 1.2161 | 1.1284 (**+7.8%**) |
| Top-1 next-token agreement | 93.0% | 98.7% |

> These numbers measure fidelity to the base model, not absolute task scores.

### Comparison with other public quantizations

Same eval set and harness for every row — closer to the BF16 original is better.

| Model | Perplexity | Top-1 |
|---|---|---|
| BF16 original (upper bound) | 1.1284 | 98.7% |
| **This model (`axteam`)** | **1.2161** | **93.0%** |
| [cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit](https://huggingface.co/cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit) | 1.2308 | 92.5% |

On our harness this model is on par with or ahead of the public same-format quantizations above. It also ships Google's corrected July-2026 `chat_template`; some earlier community builds were based on the older template.

## Requirements

| | |
|---|---|
| **Runtime** | vLLM — the architecture is in vLLM ≥ 0.19.0; **tested here with vLLM 0.32.0** (text: the benchmarks above; image: an image-description check on 2026-11-02). |
| **vLLM install** | vLLM 0.32 itself needs Python 3.11–3.14, torch 2.13 (CUDA 13.0 build, so an NVIDIA driver with CUDA 13 support), transformers ≥5.16.1,<5.20 and compressed-tensors 0.17.0 (both pinned by vLLM; 0.17.0 reads this checkpoint, which was saved with 0.19.0 — do not upgrade it separately). |
| **GPU** | NVIDIA Turing (sm75) or newer — weight-only INT, no special tensor cores needed. V100 (sm70) is not supported. (Kernel choice per GPU from vLLM 0.32's selection code; only B200 was run here.) |
| **Memory** | Weights: **17.6 GB**. Add KV cache (grows with context length × concurrent requests) and a few GB for activations / CUDA graphs; use `--tensor-parallel-size N` when the weights do not fit one GPU. |
| **Transformers (without vLLM)** | Not loadable by Transformers (checked up to 5.22.1: no conversion rule for per-expert checkpoints, the experts stay uninitialised) — use vLLM. |
| **trust_remote_code** | not needed |

## Usage (vLLM)

```bash
vllm serve axteam/gemma-4-26B-A4B-it-AWQ-4bit
```

```python
from vllm import LLM, SamplingParams

llm = LLM(model="axteam/gemma-4-26B-A4B-it-AWQ-4bit")
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
vllm serve axteam/gemma-4-26B-A4B-it-AWQ-4bit --kv-cache-dtype fp8
```

> `--kv-cache-dtype` accepts `auto` (default, BF16), `fp8` / `fp8_e4m3`, `fp8_e5m2`.
> For a GGUF build the llama.cpp equivalent is `--cache-type-k q8_0 --cache-type-v q8_0`.

## License

This is a quantized derivative of Google's Gemma 4. Use is governed by the
[Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license). All original
weights, tokenizer, and chat template belong to Google DeepMind.

---

## 한국어 요약

**[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)의 AWQ INT4 (W4A16) 양자화 빌드** · 제작 axteam

### 개요
| 항목 | 내용 |
|---|---|
| 원본 모델 | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (커밋 `4d7ae49`) |
| 양자화 방식 | AWQ INT4 (W4A16) |
| 저장 형식 | compressed-tensors `pack-quantized` |
| 모델 구조 | Gemma 4 · MoE (전문가 128개 중 8개 활성 + 항상 쓰는 공유 MLP) |
| 용량 | 약 17.6 GB |
| 보정 데이터 | 샘플 256개 (한국어 위주, 원본 모델이 스스로 생성한 thinking 모드 답변 32개 포함) |

### 양자화 상세
- **가중치**: INT4 비대칭, 그룹 크기 32 (BF16 스케일 + INT4 영점)
- **활성화**: BF16 (양자화하지 않음)
- **양자화 절차**: PTQ(학습 후 양자화, 재학습 없음), AWQ 방식 — 보정 데이터로 활성화가 큰 가중치 채널을 찾아 채널별 스케일로 보호한 뒤 INT4로 반올림
- **AWQ 적용 범위**: 어텐션 입력·전문가별 MLP 전체
- **BF16 유지**: 비전 타워, 비전 투영층, 토큰 임베딩(LM 헤드와 공유), MoE 라우터, 모든 정규화층, 공유 MLP — 전체 파라미터의 약 7%
- **원본 파일**: 토크나이저·chat template(Google의 2026년 7월 수정본)·processor·generation config를 바이트 단위로 그대로 사용, `config.json`은 원본에 `quantization_config` 블록만 추가

### 품질 (실측)
- **원본 대비 충실도**: PPL 1.2161 (원본 1.1284, +7.8%, 낮을수록 원본에 가까움), 다음 토큰 top-1 일치율 93.0% (원본 자체 기준값 98.7%) — 내부 영어·한국어 혼합 평가셋, 원본(BF16)과 비교 측정
- **다른 공개 빌드와 비교**: 같은 형식의 [cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit](https://huggingface.co/cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit) PPL 1.2308 · top-1 92.5%, 이 빌드 PPL 1.2161 · top-1 93.0% (같은 평가셋·같은 평가 코드) — 이 빌드가 동등 이상, Google의 2026년 7월 수정 chat template 포함 (일부 이전 커뮤니티 빌드는 수정 전 템플릿 기반)
- **벤치마크 평균**: 49.35 (원본 53.83, 91.7%) — MMLU·GSM8K·ARC-Challenge·Winogrande·KMMLU·HAE-RAE 6종 평균, lm-evaluation-harness + vLLM, chat template 없는 완성형 프롬프트(과제별 0·5-shot, 표 참고)·greedy, 모든 빌드를 같은 설정으로 측정
- **한국어 벤치마크**: KMMLU 33.91 (원본 36.52, 92.9%) · HAE-RAE 70.76 (원본 76.17, 92.9%)
- **주의**: ⚠️ 이 정밀도에서 품질 저하가 뚜렷함 (벤치마크 평균 원본 대비 91.7%) — 품질이 중요하면 8비트(FP8·AWQ-8bit) 빌드 권장

### 요구사항
- **실행 엔진**: vLLM 0.19.0 이상(이 구조를 지원하는 첫 버전), 이 빌드는 vLLM 0.32.0에서 실행 확인 (텍스트 — 위 벤치마크, 이미지 — 2026-11-02 그림 설명 점검)
- **vLLM 0.32 설치 조건**: Python 3.11~3.14, torch 2.13(CUDA 13.0 빌드, CUDA 13 지원 드라이버 필요), transformers 5.16.1 이상 5.20 미만, compressed-tensors 0.17.0 — transformers·compressed-tensors는 vLLM이 버전을 고정하므로 compressed-tensors를 따로 업그레이드하지 말 것 (이 체크포인트는 0.19.0으로 저장됐지만 0.17.0에서도 로드됨)
- **GPU**: NVIDIA Turing(sm75) 이상, V100(sm70) 미지원 — 가중치 전용 정수 형식이라 FP8·FP4 전용 텐서코어 불필요 (GPU별 지원 여부는 vLLM 0.32 커널 선택 코드 기준, 실제 실행은 B200에서만 확인)
- **메모리**: 가중치 17.6 GB + KV 캐시(컨텍스트 길이 × 동시 요청 수에 비례) + 활성화·CUDA graph용 수 GB — GPU 한 장에 다 들어가지 않으면 `--tensor-parallel-size N`으로 분산
- **Transformers로 직접 사용**: 불가 — 5.22.1까지 로드되지 않음 확인 (전문가별 체크포인트 변환 규칙이 없어 전문가가 초기화되지 않음), vLLM 사용
- **trust_remote_code**: 불필요

### 사용법
- **서버 실행**: `vllm serve axteam/gemma-4-26B-A4B-it-AWQ-4bit`
- **Python**: `LLM(model="axteam/gemma-4-26B-A4B-it-AWQ-4bit").chat(...)` — 전체 예시는 위 영어 본문 'Usage (vLLM)' 절 참고
- **권장 샘플링**: `temperature=1.0`, `top_p=0.95`, `top_k=64` (Google 권장값)
- **thinking 모드**: 기본 꺼짐 — 켜려면 `chat_template_kwargs={"enable_thinking": True}` 전달(Python은 `llm.chat()` 인자, HTTP는 요청 본문의 `"chat_template_kwargs": {"enable_thinking": true}`), 추론 과정을 답변과 분리해 받으려면 `vllm serve`에 `--reasoning-parser gemma4` 추가
- **KV 캐시 절감(선택)**: `vllm serve`에 `--kv-cache-dtype fp8` 추가 — 기본 BF16인 KV 캐시를 FP8로 저장해 KV 캐시 메모리를 약 절반으로 줄임 (긴 컨텍스트·동시 요청이 많을 때 유리, 가중치 양자화와 별개인 실행 옵션)

### 라이선스
- **라이선스**: Apache-2.0 ([Gemma 4 라이선스 안내](https://ai.google.dev/gemma/docs/gemma_4_license)) — 원본 모델의 라이선스를 그대로 따름
- **권리**: 원본 가중치·토크나이저·chat template은 Google DeepMind 소유, 이 리포는 양자화 파생물
