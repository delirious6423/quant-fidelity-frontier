---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
language:
- en
- zh
tags:
- autotrust
- glm5_next
- moe
- nvfp4
- modelopt
- vllm
- dgx-spark
- gb10
- blackwell
- neural-architecture-search
- mtp
- speculative-decoding
---

# GLM5.3-Flash-E256-DGX-Spark

[![AutoTrust/GuruSearch recommendation & search live demo: news.guru.so](https://huggingface.co/autotrust/JEV-27B/resolve/main/assets/autotrust_gurusearch_demo.png)](https://news.guru.so/ui)

## 🔴 Live: AutoTrust/GuruSearch recommendation & search demo → [news.guru.so](https://news.guru.so/ui)

> [!IMPORTANT]
> **Try AutoTrust/GuruSearch live (new, 11 October 2026).** A recommendation and search demo in which every ranking is a
> calibrated System 1 decision, made in about half a second:
> * **Recommendation:** the latest headlines from 6 news feeds, ranked by importance.
> * **Search:** results from Google News, Bing News and Yahoo News re-ranked by relevance, side by side with the
>   search engines' own order, plus a short answer with citations.

**autotrust/GLM5.3-Flash-E256-DGX-Spark is a compact build of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) for Blackwell systems, from DGX Spark clusters to a single B200. It's an unofficial derivative.**

It keeps 256 of the 288 routed experts in each layer by **Neural Architecture Search (NAS)**, uses NVFP4 for the experts and still activates 18 B parameters per token. It's the largest architecture in the family, with the best Chinese, code and vision scores. The KV cache uses FP8.

The repo includes a **ready-to-use NVFP4 MTP draft** in [`mtp-nvfp4/`](mtp-nvfp4/) (6.95 GB) for speculative decoding. On a single B200 it gives **1.84× single-stream decode speed** (1.94× with 3 draft tokens), and output quality is unchanged.

The model keeps the full 154,880-token vocabulary and has the vision tower intact.

| | GLM-5.3-Flash (FP8) | GLM-5.3-Flash NVFP4 (288 experts) | [E224](https://huggingface.co/autotrust/GLM5.3-Flash-E224-DGX-Spark) | **This model (E256)** |
|---|---|---|---|---|
| Disk / weight memory | 306 GB | ~191 GB | 151.5 GB = 141 GiB | **170.5 GB = 159 GiB** |
| Routed experts / layer | 288 | 288 | 224 | **256** |
| Active params / token | 18 B | 18 B | 18 B | 18 B (top-8 of 256) |
| KV cache | BF16 | FP8 | BF16 (FP8 optional) | **FP8 (declared in config)** |
| 1× B200 / GB200 (≥180 GB) | ❌ | ❌ | ✅ | ✅ (tight, see Deployment) |
| 2× DGX Spark (2 × 128 GB) | ❌ | ❌ | ✅ | ⚠️ ~80 GiB weights per node, small KV pool (not validated) |
| Vision (image / video) | ✅ | ✅ | ✅ | ✅ |
| MTP speculative decoding | ✅ | ✅ | BF16 `mtp/` + NVFP4 `mtp-nvfp4/` | **NVFP4 draft (`mtp-nvfp4/`), included** |

## Benchmarks

Measured on one B200 with vLLM. Sampling follows the base model's recipe (`temperature=1.0, top_p=0.95`); HumanEval uses greedy decoding. `reasoning_effort` is the GLM-5.3-Flash chat-template thinking budget (`low` / `high` / `max`). The token budget is 65,536 (16,384 for MMMU).

**Scoring is strict:** a response that runs out of tokens before giving a final answer counts as wrong. All numbers are single runs unless noted. MoE decoding in vLLM isn't bit-deterministic, so treat ±2–3 points as noise. E224 numbers come from the same harness on the same machine.

| Capability | Benchmark | Setting | **This model (E256)** | E224 |
|---|---|---|---|---|
| Code | HumanEval (164) | greedy | **97.6 %** (160/164) | 95.7 % |
| Science reasoning | GPQA-Diamond (198) | **effort=low** (quick self-test) | 77.3 % (153/198) · 77.8 % tolerant extraction | 78.3 % |
| Chinese knowledge | C-Eval val (1,606, 52 subjects) | effort=low | **89.4 %** (1,436/1,606) | 84.0 % |
| Math | AIME 2025 (30 × 4 samples, pass@1) | effort=high | 74.2 % (89/120) · 28/30 solved in ≥1 sample | 75.0 % |
| Vision | MMMU val (900) | effort=low | **76.1 %** (685/900) | 73.6 % |
| Tool use | BFCL v4 Non-Live AST | template default | 87.7 % | 88.3 % |
| | BFCL v4 Live AST (weighted) | template default | **80.5 %** | 80.3 % |
| | BFCL v4 Multi-Turn Base (200) | two runs | 73.0 % / 75.0 % | 80.0 % |

> ⚡ **About the GPQA-Diamond number:** it is a **quick self-test at `reasoning_effort=low`** with a 65,536-token budget, run to compare builds against each other, not the model's best score. At `low` the model thinks briefly and some answers are cut off before they finish. At the base model's recommended `reasoning_effort=max` with a 163,840-token budget, this family scores **90.9 %** on GPQA-Diamond (measured on E224).

### Tool use: BFCL v4 (function calling, AST match)

| Category | Accuracy |
|---|---|
| simple (Python) | 95.8 % |
| simple (Java) | 59.0 % |
| simple (JavaScript) | 74.0 % |
| multiple | 95.0 % |
| parallel | 93.5 % |
| parallel-multiple | 86.0 % |
| irrelevance detection | 69.2 % |
| live simple | 88.4 % |
| live multiple | 78.9 % |
| live parallel | 75.0 % |
| live parallel-multiple | 66.7 % |
| live irrelevance | 71.6 % |
| live relevance | 81.3 % |
| **Multi-Turn Base** (200) | 73.0 % / 75.0 % |

BFCL was run with `bfcl-eval` v4 against the local OpenAI-compatible endpoint (`--tool-call-parser glm47`, template-default effort, `--num-threads 32`). **For multi-turn agent workloads, prefer [E224](https://huggingface.co/autotrust/GLM5.3-Flash-E224-DGX-Spark)**: it scores about 5 points higher on Multi-Turn Base.

### Throughput and MTP speculative decoding (single B200)

`reasoning_effort=low`, 1,024 output tokens, short prompts, 8 requests per level, served with `--max-model-len 8192 --max-num-seqs 8 --gpu-memory-utilization 0.985`:

| Draft | `num_speculative_tokens` | Mean acceptance length | Draft acceptance | Single-stream decode tok/s | Speed-up (1 stream) | Aggregate tok/s @ 8 |
|---|---|---|---|---|---|---|
| off | — | — | — | 134 | 1.00× | 472 |
| **`mtp-nvfp4/`** | **2** (recommended) | **2.48** | **74.2 %** | **247** | **1.84×** | **709** |
| `mtp-nvfp4/` | 3 | 2.85 | 61.7 % | 261 | 1.94× | 665 |

On reasoning-heavy HumanEval traffic with 3 draft tokens, acceptance reached 81 % (3.43 tokens per step; per-position 0.93 / 0.82 / 0.69).

Speculative decoding is lossless: the target model verifies every drafted token, so the draft can only change speed, not output quality. The draft costs memory, though. On one B200 it adds ~4 GiB of weights and shrinks the KV pool, so use MTP for interactive, low-concurrency serving and turn it off for high-concurrency batch serving.

### The `mtp-nvfp4/` draft

* It is the **original GLM-5.3-Flash MTP layer** (one nextn layer, all 288 routed experts). Only the storage precision of its MoE weights changed.
* It is exported with NVIDIA ModelOpt (`NVFP4QTensor.quantize`), the same format as the [E224 repo's `mtp-nvfp4/`](https://huggingface.co/autotrust/GLM5.3-Flash-E224-DGX-Spark/blob/main/MTP-NVFP4.md): the 288 routed experts **and the shared expert** are NVFP4 (e2m1 weights packed two per byte, E4M3 block scales over 16-element groups). Attention, the DSA indexer, the router, `eh_proj`, norms, embeddings and `shared_head` stay BF16.
* One file, 6.95 GB, vs 17.4 GB for the BF16 draft.
* The same draft works with E192, E224 and E256: the MTP layer doesn't depend on how many experts the main model keeps.
* To regenerate it from the BF16 MTP layer (the `mtp/` folder of [E224](https://huggingface.co/autotrust/GLM5.3-Flash-E224-DGX-Spark/tree/main/mtp)), run [`scripts/export_mtp_nvfp4.py`](scripts/export_mtp_nvfp4.py) (needs `nvidia-modelopt`):

```bash
# expects <dir>/mtp (BF16 draft), writes <dir>/mtp-nvfp4
python3 scripts/export_mtp_nvfp4.py <dir>
```

> **Fixed 2026-10-10.** The first upload of `mtp-nvfp4/` (two shards) kept the shared expert in BF16, but its quantization manifest only excluded it under a module name that vLLM 0.31 doesn't use. On vLLM 0.31 the draft then failed to load with a shape assertion. It is now replaced by the ModelOpt export above. If you downloaded the old two-shard version, re-download `mtp-nvfp4/`.

## Deployment

### Requirements

* vLLM with GLM-5.3-Flash support. Validated on the `ZJY0516/vllm@glm-release` branch ([vllm-project/vllm#53906](https://github.com/vllm-project/vllm/pull/53906)) at commit `7e2d791` plus `8f8cc41` ("Make GLM-5.3 kpool metadata graph-safe"). Without that fix, full CUDA graphs can crash under concurrency.
* **For MTP:** apply [`scripts/vllm_glm5next_standalone_mtp_draft.patch`](scripts/vllm_glm5next_standalone_mtp_draft.patch) to `vllm/models/glm5next/nvidia/mtp.py`. The draft is a standalone directory with its own config (288 experts, `num_nextn_predict_layers=1`), and the patch makes vLLM build the MTP layer from the draft's config instead of the target's (256 experts, no MTP layer).
* `transformers >= 5.16.1`
* Set `VLLM_USE_DEEP_GEMM=0`.

### Single B200 / GB200 (validated)

Without MTP, at 66 K context:

```bash
export VLLM_USE_DEEP_GEMM=0
vllm serve autotrust/GLM5.3-Flash-E256-DGX-Spark \
  --served-model-name glm53-flash-e256 \
  --max-model-len 66560 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.97 \
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
```

This loads 159.1 GiB of weights and leaves 6.5 GiB of FP8 KV cache: 532 K tokens, or 8 concurrent requests at 66 K.

With the NVFP4 MTP draft (interactive use):

```bash
export VLLM_USE_DEEP_GEMM=0
vllm serve autotrust/GLM5.3-Flash-E256-DGX-Spark \
  --served-model-name glm53-flash-e256 \
  --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.985 \
  --speculative-config '{"method": "mtp", "model": "<local-path-to-this-repo>/mtp-nvfp4", "num_speculative_tokens": 2}' \
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
```

Draft and target together load 163.2 GiB, which leaves a 183 K-token KV pool at 32 K context. `reasoning_effort=max` (needs ~172 K context) isn't practical on one 180 GB card. For that, use two GPUs, a B300, or E224.

### 2× DGX Spark (not validated)

The [E224 two-Spark runbook](https://huggingface.co/autotrust/GLM5.3-Flash-E224-DGX-Spark/blob/main/DEPLOY-2X-DGX-SPARK.md) (network setup, NCCL/CX7, swap, launch scripts) applies unchanged. Replace the model path and served name. The memory budget is the difference: at TP=2, this model puts about **80 GiB of weights on each node** (E224: 71 GiB), so the KV pool is much smaller than with E224. Start with `--max-model-len 65536`, keep `--max-num-seqs` low, and size `--gpu-memory-utilization` to what your node leaves free. The NVFP4 draft adds ~3.5 GiB per node.

> ⚠️ We haven't run this model on DGX Spark hardware. The per-node figure above is estimated from the measured weight footprint. If KV room is too tight for your use, E224 is the better fit for two Sparks. Reports from Spark owners are very welcome in the Community tab.

### Request format

* Thinking is always on and comes back in the `reasoning` / `reasoning_content` field.
* `reasoning_effort`: `low`, `high` or `max` (the default). Pass it as the top-level OpenAI field or via `chat_template_kwargs`.
* Recommended sampling: `temperature=1.0, top_p=0.95`. Greedy decoding can make long thinking loop on hard prompts.
* Use `low` for chat, Q&A, tool calls, MCQ and vision. Use `max` with `max_tokens` ≥ 131,072 for competition math and GPQA-level science.
* Tools: `--tool-call-parser glm47 --enable-auto-tool-choice` returns structured `tool_calls`.
* Images and videos use standard OpenAI multi-part content (`image_url` / `video_url`).

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
r = client.chat.completions.create(
    model="glm53-flash-e256",
    messages=[{"role": "user", "content": "用三句话解释量子纠缠。"}],
    temperature=1.0, top_p=0.95, max_tokens=16384,
    extra_body={"reasoning_effort": "low"},
)
print(r.choices[0].message.content)
```

## Evaluation protocol

| Benchmark | Data | Prompt / extraction |
|---|---|---|
| GPQA-Diamond | `fingertap/GPQA-Diamond`, 198 questions | "Think step by step, then give your final answer as 'ANSWER: X'"; extracted after `</think>` |
| AIME 2025 | `math-ai/aime25`, 30 problems × 4 samples | integer answer after `</think>`; pass@1 averaged over samples |
| HumanEval | `openai/openai_humaneval`, 164 problems | final ```` ```python ```` block after `</think>`, prompt header prepended, executed against the canonical tests |
| C-Eval | `ceval/ceval-exam` val, 1,606 questions | "答案：X" after `</think>` |
| MMMU | `MMMU/MMMU` val, 900 questions | images inlined as base64 at their `<image i>` positions (≤1,024 px); "ANSWER: X" |
| BFCL v4 | `bfcl-eval` | OpenAI-compatible FC handler, AST / state-based scoring |

## Limitations

* This is an unofficial derivative, not produced or endorsed by Z.ai or NVIDIA.
* **Memory-tight on a single 180 GB GPU** (6.5 GiB KV without MTP). It doesn't fit a single DGX Spark or an H200.
* Multi-turn tool use (BFCL Multi-Turn Base) is lower than E224's. Prefer E224 for long agent traces.
* MTP serving needs the vLLM patch in `scripts/`.
* Vision was validated on MMMU only.
* Like the base model, it can produce inaccurate, biased or unsafe content. Evaluate it for your use case before deploying.

License: MIT (inherited from the base model).
