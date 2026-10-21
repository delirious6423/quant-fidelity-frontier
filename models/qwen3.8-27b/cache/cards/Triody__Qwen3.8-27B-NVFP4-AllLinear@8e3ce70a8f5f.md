---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: Model Optimizer
pipeline_tag: text-generation
inference: false
tags:
- ModelOpt
- Qwen3.8
- quantized
- NVFP4
- fp4
- sglang
- speculative-decoding
---

# Qwen3.8-27B-NVFP4-AllLinear

## Model Overview

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with every large linear layer of the language model quantized to NVFP4, built for fast single-stream decoding with [SGLang](https://github.com/sgl-project/sglang) on Blackwell GPUs. Made by [Triody](https://triody.ink/).

| | |
|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Format | NVFP4 (W4A4, group size 16, FP8 scales), NVIDIA Model Optimizer 0.47.0 |
| Size | 17.0 GB (vs 21.9 GB for [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)) |
| Input / Output | Text / Text (language model only; no vision tower, no MTP head) |
| Engine | SGLang ≥ 0.5.19 (measured on 0.5.20) |
| Hardware | NVIDIA Blackwell with FP4 tensor cores. Measured on one RTX PRO 6000 (96 GB, SM120); smaller cards need their own memory settings |
| Release date | 2026-09-30 |
| License | Apache 2.0 |

**Result.** On one RTX PRO 6000, single stream, with SGLang's own recipe for that card: 302 / 252 / 196 tokens/s on GSM8K / HumanEval / ShareGPT, **+12% to +19%** over the same recipe with NVIDIA's checkpoint, at the same accuracy.

## Quickstart

Install SGLang (0.5.19 or later) and start the server with SGLang's cookbook recipe for RTX PRO 6000, using this repository as the model:

```bash
pip install "sglang>=0.5.19"

python3 -m sglang.launch_server --trust-remote-code --model-path Triody/Qwen3.8-27B-NVFP4-AllLinear \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 --attention-backend flashinfer --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8 \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 3.38
```

Then call it with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")
resp = client.chat.completions.create(
    model="Triody/Qwen3.8-27B-NVFP4-AllLinear",
    messages=[{"role": "user", "content": "Write a Python function that checks whether a number is prime."}],
    temperature=1.0, top_p=0.95, max_tokens=2048,
)
print(resp.choices[0].message.content)
```

> **For best performance:** [sgl-project/sglang#41827](https://github.com/sgl-project/sglang/pull/41827) lets the draft model read int8 weights (about 6% faster decode, same outputs). It is under review; once it is in your SGLang, set `SGLANG_SPEC_DRAFT_INT8_WEIGHTS=1` before starting the server. All speed numbers for this checkpoint below were measured with it.

## Read This First

- **It writes more on code.** At the same sampling settings it produced 21% more output tokens than NVIDIA's checkpoint on 50 HumanEval prompts; its median greedy HumanEval reply is 713 tokens against 599. Faster per token, not per code task. On math and chat the lengths are equal.
- **Made for SGLang.** `config.json` is the nested configuration SGLang loads; without the vision weights it is not a complete model for `transformers`.
- **Self-measured**, on one kind of GPU, one run per configuration.

## Post-Training Quantization

| Part of the language model | This checkpoint | nvidia/Qwen3.8-27B-NVFP4 |
|---|---|---|
| MLP (`gate_proj`, `up_proj`, `down_proj`), 64 layers | NVFP4 | NVFP4 |
| Output head (`lm_head`) | NVFP4 | NVFP4 |
| Attention projections, 16 layers | NVFP4 | FP8 |
| Gated DeltaNet projections, 48 layers | NVFP4 | FP8 |
| Gated DeltaNet `conv1d`, `in_proj_a`, `in_proj_b` | bf16 | bf16 |
| Token embeddings | bf16 | bf16 |

`NVFP4_DEFAULT_CFG` with the output head enabled, max calibration. `hf_quant_config.json` lists the 145 modules left out.

**Calibration dataset:** 512 texts of up to 512 tokens: GSM8K train (160), MBPP train (112), ShareGPT conversations (240). None of them overlaps with the evaluation or speed prompts below.

## Evaluation

### Speed

One RTX PRO 6000 Blackwell Server Edition, SGLang 0.5.20, concurrency 1, 50 prompts per set, up to 2048 output tokens, temperature 1.0, top-p 0.95, top-k 20, reasoning effort xhigh. All rows in the same session on the same machine.

| decode, median, tokens/s | GSM8K | HumanEval | ShareGPT | ms per speculative step |
|---|---|---|---|---|
| nvidia/Qwen3.8-27B-NVFP4, SGLang's recipe | 270 | 213 | 164 | 19.8 |
| nvidia/Qwen3.8-27B-NVFP4, int8 draft | 286 | 226 | 177 | 18.7 |
| **This checkpoint, int8 draft** | **302** | **252** | **196** | **16.9** |

Tokens accepted per speculative step are the same in all three rows (3.6); the gain is the time of a step. Time to first token: 45 ms for short prompts, 78 ms at 958 input tokens, 513 ms at 7,370 (SGLang's recipe: 47–54 ms, 90 ms, 573 ms).

### Accuracy

Greedy, thinking on, at most 4096 output tokens. GSM8K: first 400 test questions. HumanEval: 164 problems, tests executed. *Strict* counts a reply cut off by the token limit as wrong; *lenient* reads the answer from the reasoning text.

| | GSM8K lenient | GSM8K strict | HumanEval lenient | HumanEval strict |
|---|---|---|---|---|
| nvidia/Qwen3.8-27B-NVFP4, SGLang's recipe | 97.25% | 95.25% | 90.85% | 86.59% |
| **This checkpoint, int8 draft** | **97.25%** | **96.00%** | **91.46%** | **86.59%** |
| Qwen/Qwen3.8-27B, bf16, no speculation (SGLang 0.5.17) | 97.75% | 96.75% | 94.51% | 92.07% |

Item by item against NVIDIA's checkpoint, no difference is measurable under either score (sign tests, p ≥ 0.55). Both NVFP4 checkpoints lose about five points of strict HumanEval against bf16 (p = 0.012), because more replies hit the limit.

## Model Limitations

- Evaluated on GSM8K and HumanEval only; other tasks, languages, tool calling and long contexts are untested.
- Inputs up to 8k tokens were measured.
- Speculative decoding does not produce the same text as decoding without it, even with greedy sampling.
- The draft model, [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), has its own terms.

## Files

| File | Size | sha256 |
|---|---|---|
| `model-00001-of-00002.safetensors` | 9,959,955,840 | `c496111cc6ebae4568d90ed571138ff98b9cd285f7927fd05fd0889e7077a303` |
| `model-00002-of-00002.safetensors` | 7,034,658,960 | `ced69b940d43690febdbf5861a0f8afaae5b4738c053fe5fb7c5a0e377824bde` |

`config.causallm.json` is the text-only configuration Model Optimizer exported, kept for reference.

## About

Made by [Triody](https://triody.ink/). Questions, results on other hardware and bug reports are welcome in the Community tab of this repository.

## License

Apache 2.0, the license of the base model, whose `LICENSE` file is included unchanged. These weights are a modification of Qwen/Qwen3.8-27B by Alibaba Cloud: the linear layers listed above were quantized to NVFP4.
