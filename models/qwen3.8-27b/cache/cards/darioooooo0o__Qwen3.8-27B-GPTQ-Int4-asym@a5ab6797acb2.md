---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - gptq
  - int4
  - qwen3.8
  - intel-arc
  - xpu
  - vllm
---

# Qwen3.8-27B GPTQ int4 (asymmetric)

An int4 quant of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) that I made to run on a single Intel Arc
Pro B70. It's asymmetric int4, group 128, GPTQ with an MSE clip search, stored in **standard GPTQ v1 format**
(`sym: false`). The vision tower, the MTP head and the lm_head stay BF16. Compared with a regular symmetric GPTQ int4
of the same size, it has about half the KL divergence from BF16.

Made by [@imdariotoo](https://x.com/imdariotoo). I post my B70 and local-model releases on X, so follow there for the next ones.

## Quality vs BF16

KL divergence over the top-20 logits on held-out 2048-token windows (lower is better):

| | code | technical prose | chat (private samples) | all | top-1 agree |
|---|---|---|---|---|---|
| [SergiioB's GPTQ int4](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) (sym g128, GPTQModel) | 0.087 | 0.103 | 0.050 | 0.078 | 91.3% |
| **this quant** | **0.037** | **0.057** | **0.025** | **0.039** | **93.8%** |
| this quant + fp8 lm_head (as served in the cookbook) | 0.038 | 0.058 | 0.025 | 0.039 | 93.5% |

In-head arithmetic, greedy, thinking off, same server:
- the B70 cookbook's 6-question canary: this quant 4/6, SergiioB's 6/6 (6 near-tie questions, so mostly noise)
- 100 seeded questions of the same kind: this quant 48/100, SergiioB's 49/100

The `sum(i*i for i in range(4))` check (SergiioB reports every symmetric int4 export of this model answers 30 instead of 14; BF16, EXL3, OpenVINO int4 and Q4_K_M answer 14). Measured 2026-10-15 on one B70, served as in the cookbook:
- thinking off, greedy: 5/6. Of two phrasings of that question, one answers 30, the other 14. The other four (9*7-3, 17*23, 2**10, and a control whose right answer is 30) are correct.
- thinking on (temperature 0.6, top_p 0.95, 2 samples each): 12/12, both phrasings answer 14.
So with thinking on it answers correctly; with thinking off it can still slip on this one question depending on wording. The BF16 base was not re-run on these exact phrasings.

## Running it

Built for and tested on **Intel Arc Pro B70 + vLLM XPU v0.31.0** (the v0.32.0 image runs the same code path; tested there
with the Uncensored variant) with the launcher and B70 vLLM changes in
**[riceharvest/b70-qwen38-cookbook](https://github.com/riceharvest/b70-qwen38-cookbook)**. That setup gives 262K context
with vision, MTP k=4, and fp8 KV on one card.

```bash
hf download darioooooo0o/Qwen3.8-27B-GPTQ-Int4-asym --local-dir qwen38-b70
MODEL_DIR=$PWD/qwen38-b70 ./serve.sh     # from the cookbook repo
```

On a B70 at 275 W, one request at a time:

| | |
|---|---|
| Decode, short prompts | 66-119 tok/s greedy depending on content (35 ms per MTP verify step) |
| Decode at 65K / 100K context | ~72 / 65-67 tok/s greedy, 73-77 / 66-69 sampled (43 / 46 ms per step) |
| Prefill | 2042 tok/s at 5.7K, 1422 at 74K, 824 at 237K |
| Context | 262,144 tokens with vision; 237K-token needle retrieved |

**Prebuilt image:** `ghcr.io/riceharvest/b70-vllm:v0.32.0-b70.7` (= `:latest`, vLLM v0.32.0) has the B70 vLLM changes and kernels
baked in: `docker run -d --device /dev/dri -v /dev/dri:/dev/dri:ro --shm-size 8g -p 8000:8000 -v ~/models:/models -e
MODEL=darioooooo0o/Qwen3.8-27B-GPTQ-Int4-asym ghcr.io/riceharvest/b70-vllm:latest`.

**Many concurrent requests:** add `--mamba-ssm-cache-dtype float16` (after the image name, or `B70_GDN_STATE=fp16 ./serve.sh`).
Qwen3.8 stores its linear-attention (GDN) state in fp32 by default, and with many requests that state is a large part of what
the GPU reads per step. Measured on Qwen3.6-35B-A3B, which has the same GDN layers: 64 concurrent users went from 49 to 42-45
ms per token, with decode quality at the fp32 noise floor and gsm8k 98/100 vs 94/100
([details](https://riceharvest.github.io/awesome-b70/results/vllm-64-users.html)). Not measured on this model yet.

The image (v0.31.0-b70.3) with this quant: 102 / 77 tok/s code / prose for one request, 0/32 corrupted outputs with
8 concurrent thinking-length requests.

**Other runtimes:** the format is standard asymmetric GPTQ v1, so GPTQModel / transformers / vLLM on CUDA should load
it. I only tested on Intel XPU. Stock vLLM XPU (v0.31, v0.32) does **not** load asym GPTQ correctly without the cookbook's
asym GPTQ change (upstream PR [vllm#61731](https://github.com/vllm-project/vllm/pull/61731)).

## How it was made

- Quantizer: `b70quant.py` from the cookbook. GPTQ (damp 0.01, act-order off), asym uint4 with per-group
  scale + zero, and a clip search over 11 ratios minimizing |err|^2.4. One layer at a time on the B70, ~1 h.
- Calibration: 128 x 2048 tokens: 40 reasoning/agentic samples from my own datasets (not released), 44 code windows
  (Python source), 44 technical prose windows (Linux docs). Disjoint from the eval windows.
- Not quantized: lm_head, MTP layers, vision tower, the GDN `in_proj_a`/`in_proj_b`.
- Export: GPTQ v1 (qzeros store z-1). The two groups with z=0 were set to z=1 (two weights changed). It was checked
  against the raw quant using GPTQModel's dequant semantics.

Credits: Qwen team (base model), [SergiioB](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) (B70 vLLM cookbook this builds on).

Questions or issues: reply to [@imdariotoo](https://x.com/imdariotoo) on X or open a discussion on this repo.
