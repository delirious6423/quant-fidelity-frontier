---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- exl3
- exllamav3
- deepseek
- deepseek_v41
- moe
- quantized
datasets:
- nvidia/Nemotron-SFT-Multilingual-v2
language:
- ko
---

# Deepseek-V4.1-Flash-1017-EXL3-2.9bpw-ko

EXL3 checkpoint with Korean-focused calibration datasets.

Compatible with [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks), [bartholomus](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) TensorFold fork and recipe.

## Notice

* This repo does **NOT** contains engrams. (Shard 47-48)
* This checkpoint uses chat_template.jinja from [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)

## Calibration datasets

| Category | Source | Share |
|---|---|---:|
| Fiction | Private | 40% |
| General prose | Private | 10% |
| Code | [NVIDIA](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v2) `code_ko` |  35% |
| Math | [NVIDIA](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v2) `math_ko` | 10% | 
| STEM | [NVIDIA](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v2) `stem_ko` | 5% |

* All datasets are in Korean. (excluding reasoning traces)
* Approx. ~511,000 tokens.

----

한국어 교정 데이터셋을 사용한 EXL3 체크포인트입니다.

[MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks), [bartholomus](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) TensorFold 포크, 레시피와 호환됩니다.

## 공지 

* Engram은 **미포함**입니다. (47-48번 Shard)
* 이 체크포인트는 [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)의 chat_template.jinja를 사용하고 있습니다.

## 교정 데이터셋

| 주제 | 출처 | 비율 |
|---|---|---:|
| 소설 | 개인 | 40% |
| 일반 산문 | 개인 | 10% |
| 코드 | [NVIDIA](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v2) `code_ko` |  35% |
| 수학 | [NVIDIA](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v2) `math_ko` | 10% | 
| 과학 | [NVIDIA](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v2) `stem_ko` | 5% |

* 모든 데이터셋은 한국어입니다. (추론 제외)
* 대략 ~511,000 토큰입니다.