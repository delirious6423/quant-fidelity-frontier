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
| **Size on disk** | ~19.5 GB |
| **Calibration** | 256 samples (Korean-heavy mixed corpus) |
| **License** | [Apache-2.0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) |

## Quantization details

- **Scheme** — W4A16 — INT4 asymmetric weights, group size 32 (BF16 scale + INT4 zero-point), BF16 activations, stored as compressed-tensors `pack-quantized`.
- **Kept in BF16** — the vision tower & projector (`model.visual.*`), the multi-token-prediction head (`mtp.*`), all RMSNorms, token embeddings, the LM head, the SSM non-linear parts (`linear_attn.conv1d/A_log/dt_bias`). Left unquantized to preserve quality (a small fraction of parameters).
- **Tokenizer / `chat_template` / `generation_config`** are identical to the original. `config.json` is the original plus a `quantization_config` block.

This is **post-training quantization (PTQ)** — no retraining, so behaviour stays faithful to the original. AWQ uses a small calibration set (no gradient) to protect salient weight channels before rounding.

## Evaluation (measured)

Fidelity to the BF16 original on an internal mixed English/Korean set (100 prompts, BF16 greedy answers scored under each build — same harness for every row).

| Metric | This model | BF16 original |
|---|---|---|
| Perplexity (lower = closer) | 1.3896 | 1.3586 (**+2.3%**) |
| Top-1 next-token agreement | 94.6% | 99.4% |

> These measure fidelity to the base model, not absolute task scores.

## Usage (vLLM)

```bash
vllm serve axteam/Qwen3.8-27B-AWQ-4bit
```
> Recommended sampling (Qwen): `temperature=0.7, top_p=0.8, top_k=20`.

## License

Quantized derivative of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). All original weights, tokenizer and chat template belong to the Qwen team.

---

## 한국어 요약

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) 를 **AWQ INT4 (W4A16)** 로 양자화한 서빙용(4-bit) 모델입니다. 작성: **axteam**.

- **방식** — INT4 비대칭, 32개 묶음(BF16 스케일+INT4 영점), 활성화 BF16(W4A16). AWQ로 배율 이동 후 양자화. 형식 compressed-tensors `pack-quantized`.
- **BF16 유지** — 비전 타워·MTP 헤드·모든 norm·임베딩/LM 헤드·SSM 비선형부(conv1d/A_log/dt_bias).
- **PTQ** — 재학습 없음, 원본 동작 유지.
- **품질(실측)** — BF16 원본 대비 perplexity 1.3896(원본 1.3586, **+2.3%**), top-1 일치도 94.6%(원본 99.4%). 내부 영/한 100문항, 같은 하네스.
- **사용** — `vllm serve axteam/Qwen3.8-27B-AWQ-4bit`, 권장 샘플링 `temperature=0.7, top_p=0.8, top_k=20`.
- **라이선스** — 원본 Apache-2.0 파생물. 가중치·토크나이저·chat_template 권리는 Qwen 팀.
