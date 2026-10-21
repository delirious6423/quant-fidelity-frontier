---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - vllm
  - compressed-tensors
  - llm-compressor
  - quantized
  - qwen3_5
---

# Qwen3.8-27B-W4A16-AWQ-GPTQ

Quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a
Qwen3.5-family hybrid-attention VLM, produced with
[llm-compressor](https://github.com/vllm-project/llm-compressor) and served with
[vLLM](https://github.com/vllm-project/vllm).

## Recipe

- **Scheme:** INT4 weight-only (W4A16), pack-quantized, group size 128, AWQ then GPTQ.
- **Calibration:** HuggingFaceH4/ultrachat_200k, 512 samples at 2048 tokens.
- **What is quantized:** the text-decoder `Linear` layers only. The vision tower
  (`re:.*visual.*`), the hybrid linear-attention mixers (`re:.*linear_attn.*`),
  and `lm_head` stay in bf16. The full VLM (with `vision_config`) is saved in the
  [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) format,
  and the base MTP predictor is preserved for speculative decoding.
- **Hardware:** Runs on any CUDA GPU with a Marlin or compressed-tensors int4 kernel.

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

MODEL_ID = "Qwen/Qwen3.8-27B"
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# ... load 512 ultrachat_200k samples, chat-templated, at 2048 tokens ...

recipe = [
    AWQModifier(duo_scaling="both", n_grid=20),
    GPTQModifier(targets=["Linear"], scheme="W4A16", block_size=128,
                 dampening_frac=0.01, actorder="static",
                 ignore=["lm_head", "re:.*visual.*", "re:.*linear_attn.*"]),
]
oneshot(model=model, dataset=ds, recipe=recipe,
        max_seq_length=2048, num_calibration_samples=512)
model.save_pretrained("Qwen3.8-27B-W4A16-AWQ-GPTQ", save_compressed=True, save_original_format=False)
tokenizer.save_pretrained("Qwen3.8-27B-W4A16-AWQ-GPTQ")
```

## Serving (vLLM)

```bash
vllm serve soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ
```

```python
from vllm import LLM, SamplingParams

llm = LLM(model="soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ")
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=32))
print(out[0].outputs[0].text)
```

## Performance

Recovery vs. the bf16 base, evaluated through the vLLM backend with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(OpenLLM v1) and [lighteval](https://github.com/huggingface/lighteval)
(generative reasoning at temperature 0.6, top_p 0.95, up to 32k tokens).

### OpenLLM Leaderboard v1

| Benchmark | Qwen3.8-27B | W4A16-AWQ-GPTQ | Recovery |
| :-- | :--: | :--: | :--: |
| ARC-Challenge (25-shot), acc_norm | 50.68 | 50.09 | 98.8% |
| HellaSwag (10-shot), acc_norm | 71.99 | 71.94 | 99.9% |
| TruthfulQA-mc2 (0-shot), acc | 61.25 | 60.33 | 98.5% |
| Winogrande (5-shot), acc | 76.87 | 76.64 | 99.7% |
| **Average** | **65.20** | **64.75** | **99.3%** |

MMLU and GSM8K are omitted. Qwen3.8-27B is a reasoning model, so under the
OpenLLM v1 protocol GSM8K has its chain of thought truncated and MMLU's
loglikelihood is measured where the model wants to emit its think block, both of
which collapse to a harness artifact rather than a real score. Math and knowledge
are captured by the generative reasoning suite instead.

### Reasoning suite (generative)

| Benchmark | Qwen3.8-27B | W4A16-AWQ-GPTQ | Recovery |
| :-- | :--: | :--: | :--: |
| AIME-24, avg@4 | 95.00 | 90.83 | 95.6% |
| AIME-25, avg@4 | 93.33 | 85.00 | 91.1% |
| MATH-500, pass@1 | 82.00 | 80.80 | 98.5% |
| **Average** | **90.11** | **85.54** | **94.9%** |

AIME-24 and AIME-25 are reported as avg@4 (mean accuracy over 4 samples per problem, temperature 0.6, up to 32k tokens, stderr about 4 points) to average out the large single-sample variance of a 30-problem test. The base and the quantized model are scored in the same run under identical settings, so the small remaining gap reflects the int4 quantization rather than sampling noise. MATH-500 (500 problems) is pass@1.


## Community results

User-reported serving results, not verified by me.

**4x RTX 3090 24GB, vLLM, TP4** (from [discussion #1](https://huggingface.co/soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ/discussions/1), thanks to @mwyborski):

- vLLM selected the Marlin int4 kernel on all four ranks, with FlashAttention and bf16 KV. Serving window 256K tokens, no YaRN.
- Decode about 66.3 tok/s. Cold prefill 1,439 tok/s at 33.5K and 1,340 tok/s at 100.4K. Warm startup 191 s.
- GPU KV capacity about 846K tokens, roughly 3.31x a full 256K context.
- TP2xPP2 alternative: decode about 46.0 tok/s, cold prefill 2,376 tok/s at 33.5K. TP4 was kept because decode is about 44% faster, which wins long coding turns.
- Prefix caching works (a repeated 1,092-token probe fell from 1.34s to 0.22s, 71.8% cache hits). OpenAI/Anthropic tool calling, tool-result round trips, MCP schemas, and reasoning extraction all passed.
