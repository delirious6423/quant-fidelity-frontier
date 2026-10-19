---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- think-lite
- reasoning
- efficient-reasoning
- qwen3.8
- gguf
---

# think-lite-v2-qwen3.8-27b-full-25

A concise-reasoning fine-tune of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B): the same model, thinking
with fewer tokens. At the same reasoning effort it produces **20% fewer output
tokens** than the base on the benchmark suite below.

## How it was made

- **Data:** the base model's own verified-correct answers (1,370 training rows, 1 epoch) across math, code, structured
  output, instruction following, science, tool use, grounded and multi-turn tasks. Redundant reasoning was removed with
  deletion-only edits (restated prompts, repeated checks, abandoned detours, restart loops); nothing was rewritten. Every
  shortened trace was checked: the answer is unchanged, and the model still solves the task when it continues from the
  shortened reasoning.
- **Training:** DPO + SFT with LoRA (rank 32), shortened trace preferred over the original.
- **Weights:** update-selective merge: per weight, the largest 25% of the adapter's changes, applied at 0.7x (DLER, arXiv 2510.15110), merged into the bf16 base.

## Benchmarks

Suite 0.91.1, vLLM 0.30.0, bf16 weights, FP8 KV cache, Qwen's thinking-mode sampling; identical prompts, caps and
scoring for every column. Accuracy in %; reasoning = mean reasoning tokens per answer.

| Benchmark | Base (xhigh) | Base (low) | **This model (low)** | Base reasoning (low) | This model reasoning (low) | Reasoning change |
|---|---|---|---|---|---|---|
| GSM8K (1,319) | 97.2 | 97.4 | **97.3** | 183 | 147 | -20% |
| AIME 2025 (30 x 2, avg@2) | 93.3 | 90.0 | **78.3** | 12,125 | 8,618 | -29% |
| MMLU-Pro (1,000-question stratified subset) | 86.3 | 84.4 | **84.7** | 1,103 | 748 | -32% |
| IFEval (541, prompt-level strict) | 91.1 | 86.3 | **83.9** | 659 | 416 | -37% |
| LiveCodeBench v6, 2025-02 to 2025-04 (131, pass@1) | 84.0 | 71.8 | **67.2** | 10,741 | 7,581 | -29% |
| BFCL v4 non-live core (1,240) | 88.7 | 90.8 | **90.4** | 96 | 76 | -21% |

Total output tokens over all 4,291 requests at low effort: base 4,915,895, this model 3,915,800
(-20%).

## Usage

The chat template is the base model's; choose the reasoning effort with `reasoning_effort` (`low`, `medium`, `xhigh`).

**vLLM**

```bash
vllm serve Mintzs/think-lite-v2-qwen3.8-27b-full-25 --max-model-len 135168 --kv-cache-dtype fp8
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
reply = client.chat.completions.create(
    model="Mintzs/think-lite-v2-qwen3.8-27b-full-25",
    messages=[{"role": "user", "content": "How many primes are there below 100?"}],
    extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
)
print(reply.choices[0].message.content)
```

**llama.cpp (GGUF)**

```bash
llama-server -m Think-Lite-v2-Qwen3.8-27B-full-25-Q4_K_M.gguf --mmproj mmproj-Think-Lite-v2-Qwen3.8-27B-full-25-F16.gguf --jinja -c 32768 \
  --chat-template-kwargs '{"reasoning_effort": "low"}'
```

## Files

- `model-*.safetensors`: merged bf16 weights (same layout and config as the base).
- `Think-Lite-v2-Qwen3.8-27B-full-25-Q4_K_M.gguf`: Q4_K_M GGUF of the language model (llama.cpp).
- `mmproj-Think-Lite-v2-Qwen3.8-27B-full-25-F16.gguf`: the vision projector for llama.cpp.

## License

Apache-2.0, as the base model.
