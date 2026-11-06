---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
license: apache-2.0
language:
  - en
tags:
  - int4
  - w4a16
  - gptq
  - compressed-tensors
  - auto-round
  - vllm
  - quantized
  - qwen3
---

# Qwen3.8-27B — W4A16 AutoRound

Int4 weight-only quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [Intel AutoRound](https://github.com/intel/auto-round) and packed as
compressed-tensors.

> **Mirror maintained by [Arkane Cloud](https://arkanecloud.com).** These are the weights we
> serve in production on our Lyon datacenter. The quantization recipe and the original model
> card are the work of
> [dbirks/Qwen3.8-27B-W4A16-AutoRound](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound)
> — all credit there. What we add below is a production serving configuration, an independent
> benchmark on our own hardware, and a hosted endpoint if you would rather not deploy anything.

---

## Try it without deploying anything

The model runs in production on our French datacenter. You can call it directly:

**→ [console.arkanecloud.com](https://console.arkanecloud.com)**

OpenAI-compatible API:

```bash
curl https://console.arkanecloud.com/api/v2/chat/completions \
  -H "Authorization: Bearer $ARKANE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38-27b",
    "messages": [{"role": "user", "content": "Bonjour"}],
    "temperature": 0
  }'
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://console.arkanecloud.com/api/v2",
    api_key="<your key>",
)

resp = client.chat.completions.create(
    model="qwen38-27b",
    messages=[{"role": "user", "content": "Bonjour"}],
    temperature=0,
)
print(resp.choices[0].message.content)
```

Two profiles are served: a 1 000 000-token window on a tensor-parallel node, and a
150 000-token window on a single GPU for latency-sensitive traffic.

---

## Quantization recipe

Credit: this section describes the upstream recipe.

- Int4 weight-only (W4A16), group size 128, symmetric. Activations stay in BF16.
- Kept in BF16: `in_proj_a`, `in_proj_b`, the vision tower `visual.*`, the `mtp` head
  and `lm_head`.
- Calibration on `NeelNanda/pile-10k`, 128 samples, seqlen 2048, 200 iterations.
- Format: compressed-tensors (`pack-quantized`), int4 Marlin kernels, Ampere or newer.

Weights occupy **14.26 GiB of VRAM** at load time, against 52 GB for the BF16 checkpoint.

---

## Serving it yourself

We run it with vLLM, Marlin int4 kernels and an FP8 KV cache. Both configurations below
are the ones we measure in the benchmark section.

**Single GPU — 24 GB, 150k context**

```bash
vllm serve Arkane-cloud/qwen38-27b-w4a16-autoround \
  --max-model-len 150000 \
  --kv-cache-dtype fp8 \
  --trust-remote-code
```

**Tensor-parallel 4 — 96 GB, 1M context**

```bash
vllm serve Arkane-cloud/qwen38-27b-w4a16-autoround \
  --tensor-parallel-size 4 \
  --max-model-len 1000000 \
  --kv-cache-dtype fp8 \
  --trust-remote-code
```

The FP8 KV cache is what makes the 1M window fit. On 4× RTX A5000 (96 GB total), the
quantized weights leave **2 117 834 tokens** of KV cache, against **515 576** for the BF16
checkpoint on the same cards. On a single 24 GB A5000, you get 217 268 tokens.

---

## Benchmark on our hardware

Protocol: OpenAI MRCR, 8-needle, 32 strictly paired prompts run on every pod, two
independent runs, temperature 0, thinking disabled, autoscaling frozen, pod-direct access
without a gateway. Seeds, scripts and raw JSON are versioned.

### Quality

| Configuration | MRCR | Perfect scores |
|---|---|---|
| BF16 vanilla — 4× RTX A5000, TP=4 *(control)* | 51.7 % | 15 / 32 |
| W4A16 AutoRound — 4× RTX A5000, TP=4 | 50.7 % | 15 / 32 |
| W4A16 AutoRound — 1× RTX A5000 | 50.1 % | 15 / 32 |

1.6 point between the original model on four cards and the quantized one on a single
card — and the exact same count of perfect retrievals across all three.

### Throughput — 128 in / 1024 out

| Configuration | Single user | ITL | C=32 aggregate | Per GPU |
|---|---|---|---|---|
| 1× RTX A5000 | 39.9 tok/s | 24.8 ms | 516 tok/s | 516 tok/s |
| 4× RTX A5000, TP=4 | 80.9 tok/s | 12.2 ms | 850 tok/s | 212 tok/s |
| 4× RTX A5000, TP=4, BF16 | 38.8 tok/s | 25.6 ms | 399 tok/s | 100 tok/s |
| 1× RTX PRO 6000 (Blackwell) | 74.3 tok/s | 13.4 ms | 1 572 tok/s | 1 572 tok/s |

Tensor parallelism buys latency, not volume: four cards halve the inter-token latency but
produce half as many tokens per card.

### Long context

The 1M window is confirmed live. Retrieval is clean up to 290 021 real tokens (single
needle, cold, no prefix cache, three seeds per size). The official 8-needle MRCR was scored
up to 946 185 input tokens and averages 33.9 % at that tier, against 50.7 % below 131k —
retrieval degrades past roughly 250k and we publish the figure. A request of 997 953 input
tokens is refused for exceeding the 1 000 000 window with its 2 048 output tokens, which is
the cleanest proof the window is real.

---

## A note on Blackwell

The upstream card suggests the NVFP4 variant for Blackwell GPUs. On our RTX PRO 6000 we
measured the opposite for decoding — same pod, same sweep, same day:

| | W4A16 AutoRound | NVFP4 (`nvidia/Qwen3.8-27B-NVFP4`) |
|---|---|---|
| Weights in VRAM | 14.26 GiB | 18.80 GiB |
| Single user | 74.3 tok/s · ITL 13.4 ms | 60.7 tok/s · ITL 16.4 ms |
| C=32 aggregate | 1 572 tok/s | 1 085 tok/s |
| KV cache pool | 2.19 M tokens | 2.00 M tokens |

NVFP4 does take the lead on long prefill: 39.4 against 30.3 tok/s at 32k input, single
request — the FP4 tensor cores work better on dense attention matrices. Pick by your
workload, not by a ranking.

---

## Citation

AutoRound:

```bibtex
@article{cheng2023optimize,
  title={Optimize weight rounding via signed gradient descent for the quantization of LLMs},
  author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao and Liu, Yi},
  journal={arXiv preprint arXiv:2309.05516},
  year={2023}
}
```
