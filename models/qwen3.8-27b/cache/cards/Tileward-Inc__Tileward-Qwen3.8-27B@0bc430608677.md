---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
model-index:
  - name: Tileward-Qwen3.8-27B
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: MMLU (5-shot)
          type: cais/mmlu
        metrics:
          - type: acc
            value: 82.4
            name: accuracy
            verified: false
---

# Tileward-Qwen3.8-27B

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), compressed by
[Tileward](https://tileward.com) to 64% smaller (2.8x), and level with NVIDIA's NVFP4 build of the
same model on MMLU from a file 8.5% smaller. Takes text and image input.

![Weights on disk and MMLU for Qwen3.8-27B, NVIDIA's NVFP4 build and Tileward-Qwen3.8-27B](results.png)

| | Size | MMLU (5-shot) |
|---|---|---|
| Qwen3.8-27B (original) | 55.6 GB | – |
| Qwen3.8-27B-NVFP4 (NVIDIA's build) | 21.9 GB | 82.1 |
| **Tileward-Qwen3.8-27B** | **20.1 GB** | **82.4** |

Against NVIDIA's NVFP4 build, each pair in one session on one card. NVIDIA's build stays closer
to the original on next-word agreement:

| | NVIDIA's NVFP4 build | Tileward-Qwen3.8-27B |
|---|---|---|
| MMLU-Redux (5,700 questions, thinking on) | 89.4% | 89.4% |
| Facts made up when writing out a document's facts (lower is better) | 4.7% | 3.8% |
| Tool calls right, of 240 | 240 | 240 |
| Next-word agreement with the original | 88.9% | 87.8% |
| Loaded on the card | 20.6 GiB | 18.9 GiB |

## Research

- [We made Qwen3.8-27B 8.5% smaller than NVIDIA's NVFP4 build, and beyond](https://tileward.com/research/a-smaller-qwen3-8-27b/)
- [How each figure was measured, on tileward.com/evals](https://tileward.com/evals/#compressed-model-27b)
- [All of Tileward's research](https://tileward.com/research/)

## Try it

- **[demo.tileward.com](https://demo.tileward.com)**: chat with this model in your browser. No
  sign-up.
- **[app.tileward.com](https://app.tileward.com/?join=explore)**: create an account and an API
  key. Every account starts with $25 of credit and needs no card. The API is OpenAI-compatible
  at `https://api.tileward.com/v1`, and this model's id is `Tileward-Qwen3.8-27b`.

## Availability

Weights are not published here. The model runs on Tileward's hosted API, in your own cloud
account, or air-gapped.

Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team,
Apache-2.0.
