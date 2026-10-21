---
license: apache-2.0
language:
- ko
- en
base_model: google/gemma-4-26B-A4B-it
library_name: transformers
pipeline_tag: text-generation
tags:
- sglang
- fp8
- decision-making
- multiple-choice
---

# korjev-26b4a-v1

A Korean finite-choice decision model based on [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it), with 3.8B active parameters (rounded to `4a` in the name). It scores 2–26 choices through the original LM head and a masked softmax over the corresponding A–Z tokens, without generating a reasoning trace.

This repository contains **merged FP8 W8A8 weights**: the trained LoRA is already merged into the backbone. The LM head, embeddings, routers and normalization layers retain higher precision. Training and evaluation use text inputs.

## Results

![Model comparison](model_overview_highlight.png)

| Model | Precision / service | C* | Decisions/s ↑ | HTTP p50 ms ↓ | Korean accuracy ↑ | English accuracy ↑ | Order KO ↑ | Order EN ↑ | Order macro ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 31B | FP8 | 64 | 45.2 | 1343.3 | 94.40% | 57.64% | 97.68% | 55.00% | 76.34% |
| Qwen3.8 27B | FP8 | 64 | 52.9 | 1162.4 | 89.02% | 60.40% | 95.29% | 61.38% | 78.34% |
| Gemma 4 26B-A4B † | FP8 | 128 | 195.3 | 633.5 | 90.37% | 45.36% | 95.63% | 41.75% | 68.69% |
| Gemma 4 12B | FP8 | 64 | 114.4 | 533.4 | 89.58% | 39.69% | 92.72% | 36.46% | 64.59% |
| Cygnet · Gemma 12B | BF16 | 64 | 49.9 | 1244.7 | 89.18% | 54.84% | 95.47% | 58.73% | 77.10% |
| Mica v0.1 4B | BF16 | 64 | 152.0 | 432.7 | 81.47% | 52.96% | 89.83% | 61.25% | 75.54% |
| K-Decision 4B | BF16 | 256 | 102.7 | 2451.4 | 85.41% | 48.94% | 93.44% | 53.12% | 73.28% |
| Qwen3.5 4B | BF16 | 64 | 185.6 | 332.5 | 78.77% | 46.24% | 77.49% | 44.75% | 61.12% |
| **korjev-26b4a-v1** † | FP8 | 128 | 195.3 | 633.5 | 92.16% | 65.66% | 96.73% | 71.89% | 84.31% |
| Decision 2.0 Vega 27B | BF16 / FP32 head · official max_speed | 1 | 11.1 | 71.3 | 92.38% | 64.59% | 96.84% | 73.78% | 85.31% |
| JEV 1.13 · hosted API | Hosted API | 64 | 88.4 | 543.7 | 88.93% | 82.50% | 96.39% | 89.28% | 92.83% |

**Shared serving mean (†):** The 26B backbone and korjev share the mean of their measured FP8 throughput and HTTP-p50 summaries at C*=128. Accuracy and order metrics remain individual.

Local models use one RTX PRO 6000 Blackwell Server 96GB GPU, with 256 warmup requests per timed pass. JEV is hosted through OpenRouter; its hardware is undisclosed. C* maximizes observed throughput across concurrency 1/8/32/64/128/256/512 on the same 2,048-item workload. HTTP p50 is measured at that C* and includes server/API queueing and transport after submission; it excludes local caller-slot wait.

Accuracy uses 25,418 Korean items and 12,032 English MMLU-Pro items (zero-shot direct decisions, without reasoning). Order agreement compares original and exact-reverse choices after semantic alignment; Order macro gives the two languages equal weight. Agreement can include consistently wrong answers.

**Radar coverage:** All eleven measured models, including Vega and hosted JEV, appear in both the table and radar. Throughput is normalized across these eleven models. HTTP p50 uses a reversed linear millisecond axis: 3,000 ms at the center, 0 ms at the rim, with 500 ms ticks. Farther outward means lower latency.

[Measurement details](evaluation.md) · [Complete metrics](comparison.json).

## Training

Training used two stages of LoRA fine-tuning (`r=64`):

1. Initial training on **4,200 Korean decision items**.
2. Continued training with **3,511 additional items** and the **4,200-item replay pool**, with every batch split equally between additional and replay items.

The combined corpus contains **7,711 unique training items**; no English training items were added. Cross-entropy trains the answer choice. An exact-reverse JS loss aligns the choice distributions between a shuffled choice order and its reverse to improve order stability. In the second stage, a reference KL loss on eligible replay items helps retain the first-stage model's predictions. Checkpoints were selected using private validation data, before public benchmark evaluation.

## Serve the merged FP8 model with SGLang

The following uses the tested SGLang 0.5.21 container and one GPU. The model's quantization configuration is loaded automatically. `--dtype bfloat16` controls the unquantized tensors; the quantized matrix weights remain FP8.

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  -p 127.0.0.1:30000:30000 \
  -v "$PWD/hf-cache:/root/.cache/huggingface" \
  lmsysorg/sglang@sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75 \
  python3 -m sglang.launch_server \
    --model-path DaehanKim/korjev-26b4a-v1 \
    --host 0.0.0.0 --port 30000 --tp 1 \
    --dtype bfloat16 --context-length 32768 \
    --mem-fraction-static 0.88 --max-running-requests 512 \
    --chunked-prefill-size 2048 --swa-full-tokens-ratio 0.25 \
    --attention-backend triton \
    --cuda-graph-backend-decode disabled \
    --cuda-graph-backend-prefill disabled
```

After `/health` returns success, use [`score_example.py`](score_example.py) to obtain candidate probabilities through `/v1/score`:

```bash
pip install 'transformers==5.12.1' requests
python score_example.py --warmup 256
```

The example applies the non-thinking chat template once, scores only the candidate letter tokens, and returns their probabilities and the selected choice. Edit its task, input and choices for your application. Warmup requests should be excluded from your timing measurements. The reported throughput was measured with concurrent requests; this small example sends one request at a time.

## Credits and artifacts

Thanks to Google DeepMind for Gemma, the original benchmark dataset authors, and the authors of the comparison models. The backbone is released under Apache 2.0; see [LICENSE](LICENSE). [Weight provenance and quantization](release_provenance.json) · [Machine-readable comparison](comparison.json).
