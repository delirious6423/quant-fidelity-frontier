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
  - nvfp4
  - fp4
---
# Qwen3.8-27B · NVFP4 (W4A4, FP4)

4-bit quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), produced for efficient **vLLM serving**. Quantized by **axteam**.

> **Revision 2026-11-01 (rebuilt):** Channels whose original norm gain is zero are now left out of AWQ's scale normalization, so the smoothed norms keep their BF16 precision (checked: the median rounding error of every smoothed norm is below 1%). The first upload of this repo lost norm precision in layer 7 because of a zero-gain channel (median per-channel gain rounding error about 10%). If you downloaded this repo before 2026-11-01 11:00:07 UTC (revision `89c28ec`), download it again. The benchmark and fidelity figures on this card were measured on the rebuilt files.
<!-- axteam-rebuild: 1 -->

| | |
|---|---|
| **Base model** | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Method** | NVFP4 (W4A4, FP4) |
| **Format** | compressed-tensors `nvfp4-pack-quantized` |
| **Weights / activations** | W4A4 — FP4 (E2M1) weights group 16 (FP8 E4M3 scale + per-tensor FP32 global scale), FP4 activations |
| **Architecture** | Hybrid (linear-attention + full-attention) · Dense · multimodal (vision) · MTP head |
| **Size on disk** | ~28.6 GB |
| **Calibration** | 256 samples (Korean-heavy mixed corpus) |
| **License** | [Apache-2.0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) |

## Quantization details

- **Scheme** — W4A4 — FP4 (E2M1) weights group 16 (FP8 E4M3 scale + per-tensor FP32 global scale), FP4 activations, stored as compressed-tensors `nvfp4-pack-quantized`.
- **Kept in BF16** — the vision tower & projector (`model.visual.*`), the multi-token-prediction head (`mtp.*`), all RMSNorms, token embeddings, the LM head, **the entire linear-attention (GatedDeltaNet/SSM) path** — FP4 is too coarse for the SSM weights, so the full `linear_attn.*` stays BF16. Left unquantized to preserve quality (about 32% of all parameters in this build).
- **Tokenizer / `chat_template` / `generation_config`** are identical to the original. `config.json` is the original plus a `quantization_config` block.

This is **post-training quantization (PTQ)** — no retraining. AWQ uses a small calibration set (no gradient) to protect salient weight channels before rounding.

## Benchmarks (measured)

Every column was measured with the same harness and settings — [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on vLLM, greedy, few-shot completion (no chat template). Percentages are recovery vs. the BF16 original.

| Benchmark | BF16 (original) | FP8-dynamic | NVFP4A16 | **NVFP4 (this repo)** |
|---|---|---|---|---|
| MMLU (5-shot) | 62.42 | 62.15 (99.6%) | 62.55 (100.2%) | **63.47 (101.7%)** |
| GSM8K (5-shot, strict) | 70.43 | 76.88 (109.1%) | 49.81 (70.7%) | **47.54 (67.5%)** |
| ARC-Challenge (0-shot) | 58.62 | 58.36 (99.6%) | 58.70 (100.1%) | **57.76 (98.5%)** |
| Winogrande (5-shot) | 79.08 | 79.56 (100.6%) | 79.48 (100.5%) | **80.03 (101.2%)** |
| KMMLU (5-shot, direct) | 65.72 | 65.71 (100.0%) | 65.37 (99.5%) | **64.82 (98.6%)** |
| HAE-RAE (0-shot) | 76.35 | 76.54 (100.2%) | 75.89 (99.4%) | **75.80 (99.3%)** |
| **Average** | **68.77** | **69.87 (101.6%)** | **65.30 (95.0%)** | **64.90 (94.4%)** |

> ⚠️ **Note:** average recovery is 94.4% — noticeable degradation at this precision; prefer an 8-bit build if quality matters most.

## Evaluation (measured)

Fidelity to the BF16 original on an internal mixed English/Korean set (100 prompts: the BF16 original's greedy answers, scored under each build with the same prompts and scoring code). This build was measured on the rebuilt files with vLLM 0.32 on 2026-11-01; the BF16 reference column was measured on 2026-10-27.

| Metric | This model | BF16 original |
|---|---|---|
| Perplexity (lower = closer) | 1.3954 | 1.3586 (**+2.7%**) |
| Top-1 next-token agreement | 93.7% | 99.4% |

> These measure fidelity to the base model, not absolute task scores.

## Requirements

| | |
|---|---|
| **Runtime** | vLLM — the architecture is in vLLM ≥ 0.17.0; **tested here with vLLM 0.32.0**. |
| **vLLM install** | vLLM 0.32 itself needs Python 3.11–3.14, torch 2.13 (CUDA 13.0 build, so an NVIDIA driver with CUDA 13 support), transformers ≥5.16.1,<5.20 and compressed-tensors 0.17.0 (both pinned by vLLM; 0.17.0 reads this checkpoint, which was saved with 0.19.0 — do not upgrade it separately). |
| **GPU** | Full FP4 (W4A4) speed on Blackwell — B200/GB200 (sm100), B300 (sm103), RTX 50xx / RTX PRO 6000 (sm120). Turing–Hopper (sm75–90) fall back to weight-only FP4 (slower, vLLM warns). V100 (sm70) is not supported. (Kernel choice per GPU from vLLM 0.32's selection code; only B200 was run here.) |
| **Memory** | Weights: **28.6 GB**. Add KV cache (grows with context length × concurrent requests) and a few GB for activations / CUDA graphs; use `--tensor-parallel-size N` when the weights do not fit one GPU. |
| **Transformers (without vLLM)** | `transformers>=5.2.0` + `compressed-tensors>=0.15.0` (checked on CPU with Qwen3.5-0.8B FP8 / AWQ-4bit / NVFP4A16 / NVFP4). |
| **trust_remote_code** | not needed |

## Usage (vLLM)

Start an OpenAI-compatible server:

```bash
vllm serve axteam/Qwen3.8-27B-NVFP4 --max-model-len 32768
```

Query it:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "axteam/Qwen3.8-27B-NVFP4",
  "messages": [{"role": "user", "content": "대한민국의 수도는 어디인가요?"}],
  "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 512
}'
```

Offline inference in Python:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="axteam/Qwen3.8-27B-NVFP4", max_model_len=32768)
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
vllm serve axteam/Qwen3.8-27B-NVFP4 --kv-cache-dtype fp8
```

## License

Quantized derivative of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). All original weights, tokenizer and chat template belong to the Qwen team.

---

## 한국어 요약

**[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)의 NVFP4 (W4A4) 양자화 빌드** · 제작 axteam

### 개요
| 항목 | 내용 |
|---|---|
| 원본 모델 | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| 양자화 방식 | NVFP4 (W4A4, 가중치·활성화 FP4) |
| 저장 형식 | compressed-tensors `nvfp4-pack-quantized` |
| 모델 구조 | 하이브리드 어텐션(선형 어텐션 + 풀 어텐션) · Dense · 멀티모달(비전) · MTP 헤드 |
| 용량 | 약 28.6 GB |
| 보정 데이터 | 샘플 256개 (한국어 위주 혼합 코퍼스) |

### 양자화 상세
- **가중치**: FP4(E2M1), 그룹 크기 16 (그룹별 FP8(E4M3) 스케일 + 텐서별 FP32 전역 스케일)
- **활성화**: 실행 시 FP4 (Blackwell FP4 텐서코어에서 가속)
- **양자화 절차**: PTQ(학습 후 양자화, 재학습 없음), AWQ 방식 — 보정 데이터로 활성화가 큰 가중치 채널을 찾아 채널별 스케일로 보호한 뒤 FP4로 반올림
- **BF16 유지**: 비전 타워·투영층(`model.visual.*`), MTP 헤드(`mtp.*`), 모든 RMSNorm, 토큰 임베딩, LM 헤드, 선형 어텐션(GatedDeltaNet/SSM) 경로 전체(`linear_attn.*`, FP4로는 정밀도 부족) — 전체 파라미터의 약 32%
- **원본 파일**: 토크나이저·chat template·generation config는 원본과 동일, `config.json`은 원본에 `quantization_config` 블록만 추가

### 품질 (실측)
- **원본 대비 충실도**: PPL 1.3954 (원본 1.3586, +2.7%, 낮을수록 원본에 가까움), 다음 토큰 top-1 일치율 93.7% (원본 자체 기준값 99.4%) — 내부 영어·한국어 혼합 프롬프트 100개, 원본(BF16)의 greedy 답변을 기준 텍스트로 측정
- **벤치마크 평균**: 64.90 (원본 68.77, 94.4%) — MMLU·GSM8K·ARC-Challenge·Winogrande·KMMLU·HAE-RAE 6종 평균, lm-evaluation-harness + vLLM, chat template 없는 few-shot·greedy, 모든 빌드를 같은 설정으로 측정
- **한국어 벤치마크**: KMMLU 64.82 (원본 65.72, 98.6%) · HAE-RAE 75.80 (원본 76.35, 99.3%)
- **과제별 편차**: GSM8K 47.54 (원본 70.43, 67.5%) — 평균에 가려지는 차이, 과제별 값은 위 영어 본문 Benchmarks 표
- **주의**: ⚠️ 이 정밀도에서 품질 저하가 뚜렷함 (벤치마크 평균 원본 대비 94.4%) — 품질이 중요하면 8비트(FP8·AWQ-8bit) 빌드 권장

### 요구사항
- **실행 엔진**: vLLM 0.17.0 이상(이 구조를 지원하는 첫 버전), 이 빌드는 vLLM 0.32.0에서 실행 확인
- **vLLM 0.32 설치 조건**: Python 3.11~3.14, torch 2.13(CUDA 13.0 빌드, CUDA 13 지원 드라이버 필요), transformers 5.16.1 이상 5.20 미만, compressed-tensors 0.17.0 — transformers·compressed-tensors는 vLLM이 버전을 고정하므로 compressed-tensors를 따로 업그레이드하지 말 것 (이 체크포인트는 0.19.0으로 저장됐지만 0.17.0에서도 로드됨)
- **GPU**: Blackwell(B200·GB200 sm100, B300 sm103, RTX 50xx·RTX PRO 6000 sm120)에서 FP4(W4A4) 연산 가속, V100(sm70) 미지원 — Turing~Hopper(sm75~90)는 가중치 전용 FP4로 자동 전환되어 속도 저하, vLLM 경고 출력 (GPU별 지원 여부는 vLLM 0.32 커널 선택 코드 기준, 실제 실행은 B200에서만 확인)
- **메모리**: 가중치 28.6 GB + KV 캐시(컨텍스트 길이 × 동시 요청 수에 비례) + 활성화·CUDA graph용 수 GB — GPU 한 장에 다 들어가지 않으면 `--tensor-parallel-size N`으로 분산
- **Transformers로 직접 사용**: transformers 5.2.0 이상 + compressed-tensors 0.15.0 이상 (Qwen3.5-0.8B의 FP8·AWQ-4bit·NVFP4A16·NVFP4 빌드로 CPU에서 확인)
- **trust_remote_code**: 불필요

### 사용법
- **서버 실행**: `vllm serve axteam/Qwen3.8-27B-NVFP4 --max-model-len 32768` — OpenAI 호환 API
- **Python**: `LLM(model="axteam/Qwen3.8-27B-NVFP4", max_model_len=32768).chat(...)` — 전체 예시는 위 영어 본문 'Usage (vLLM)' 절 참고
- **권장 샘플링**: `temperature=1.0`, `top_p=0.95`, `top_k=20` (원본 `generation_config.json` 기준)
- **thinking 모드**: 기본 켜짐 — 끄려면 `chat_template_kwargs={"enable_thinking": False}` 전달(Python은 `llm.chat()` 인자, HTTP는 요청 본문의 `"chat_template_kwargs": {"enable_thinking": false}`)
- **이미지 입력**: 메시지에 `image_url` 항목 사용
- **텍스트만 사용할 때**: `--limit-mm-per-prompt '{"image": 0, "video": 0}'` — 비전 타워용 메모리 예약을 생략해 메모리 절약
- **KV 캐시 절감(선택)**: `vllm serve`에 `--kv-cache-dtype fp8` 추가 — 기본 BF16인 KV 캐시를 FP8로 저장해 KV 캐시 메모리를 약 절반으로 줄임 (긴 컨텍스트·동시 요청이 많을 때 유리, 가중치 양자화와 별개인 실행 옵션)

### 참고
- **수정 이력 (2026-11-01, 재양자화)**: 원본 norm 이득이 0인 채널을 AWQ 스케일 정규화에서 빼 스무딩된 norm의 BF16 정밀도 유지 (스무딩된 모든 norm의 반올림 오차 중앙값 1% 미만 확인) — 이전 업로드의 결함: 첫 번째 업로드는 7번 층의 이득 0 채널 탓에 norm 정밀도 손실(채널별 이득 반올림 오차 중앙값 약 10%) — 2026-11-01 11:00:07 UTC(리비전 `89c28ec`) 이전에 받았다면 다시 받을 것, 품질 항목의 벤치마크·충실도는 재양자화 빌드로 잰 값

### 라이선스
- **라이선스**: Apache-2.0 — 원본 모델의 라이선스를 그대로 따름
- **권리**: 원본 가중치·토크나이저·chat template은 Qwen 팀(Alibaba) 소유, 이 리포는 양자화 파생물
