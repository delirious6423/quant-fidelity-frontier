---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- thinklite
- reasoning
- efficient-reasoning
- qwen3.8
- gguf
---

# Thinklite-27B

**Qwen3.8-27B that thinks with fewer tokens.** Thinklite-27B is trained to drop the parts of its reasoning that do not
help it reach the answer: restating the question, repeating a step it already did, re-checking a result that was
already checked, and exploring approaches it then abandons. Answers arrive sooner, with the same model, the same chat
template and the same hardware cost per token.

| Version | Output tokens vs Qwen3.8-27B | Time to full answer (TTFA) |
|---|---|---|
| **Thinklite-27B Low** | 14% fewer | 1.12x faster |
| **Thinklite-27B High** | 20% fewer | 1.22x faster |
| **Thinklite-27B Max** | 33% fewer | 1.43x faster |

Measured on six benchmarks at `reasoning_effort=low` against Qwen3.8-27B at the same setting (details below). TTFA is
the time a single user waits for the complete answer; Thinklite-27B decodes at exactly the base model's speed, so the
whole speedup comes from writing fewer tokens.

## Versions

All three versions are full models (merged bf16 weights, drop-in replacements for Qwen3.8-27B) in this repository:

| Version | Branch | What it is | Choose it for |
|---|---|---|---|
| **Thinklite-27B High** | `main` | Trained on the full dataset; the weights keep the largest 25% of each layer's fine-tuning changes, applied at 0.7x | the balance of savings and accuracy |
| **Thinklite-27B Low** | `low` | Trained on a smaller dataset, merged the same way as High | the closest accuracy to the base model |
| **Thinklite-27B Max** | `max` | Trained on the full dataset, with the full fine-tuning applied | the largest token savings |

"Low", "High" and "Max" name the versions, not reasoning settings: every version supports `reasoning_effort` `low`,
`medium` and `xhigh`, exactly like Qwen3.8-27B. A Q4_K_M GGUF of Thinklite-27B High is on `main`; the LoRA adapters are
on the `adapters` branch.

## Objective

Reasoning models spend much of their output on thinking that does not change the answer. The goal of Thinklite is to
remove that overhead **without teaching the model new behaviour**: every training target is the model's own reasoning,
shortened only by deleting redundant parts, so the model learns to reach the same answers by a shorter route rather
than imitating a different model's style. Reasoning that is genuinely needed, for example on hard problems, is kept.

## Method

**1. Self-generated, verified answers.** Qwen3.8-27B answered a broad prompt set covering math, code, structured output
(JSON / YAML / TOML / tool calls), instruction following, science, tool use, grounded question answering and
multi-turn conversations. Only answers verified as correct were kept: exact checkers for math, executed unit tests for
code, schema and parser checks for structured output, rule checks for instructions, and an answer-only judge for
open-ended questions.

**2. Trimming and deletion-only editing.** Each correct reasoning trace was split into units.
- Everything after the point where the model had committed to its final answer was cut.
- Redundant units were deleted: restatements of the prompt, repeated steps, redundant checks, abandoned detours and
  filler. Nothing was rewritten, paraphrased or added.
- Every edit had to pass mechanical checks: the final answer is unchanged, every value the answer uses is still
  derived, and no dependency is broken.
- A **continuation test** let the model continue from the shortened reasoning and required it to still solve the task.
- Traces from the model's high-effort mode were kept at full length as anchors, so its `xhigh` behaviour is preserved.

**3. Training: SFT + DPO with LoRA and quantization-aware training.**
- LoRA (rank 32, alpha 64) on all attention, MLP and linear-attention projections.
- The objective is DPO (beta 0.1) plus a supervised term on the preferred answer. The shortened trace is preferred over
  the original; rows without a shorter variant are supervised directly.
- Quantization-aware training: during training the frozen base layers ran fake-quantized to int4 (group size 32), so
  the learned changes hold up when the model is quantized, as in the GGUF.
- Learning rate 5e-5. The full dataset is 1,370 rows (1 epoch); Thinklite-27B Low was trained on a smaller set of 474
  rows (2 epochs).

**4. Update-selective merge (High and Low).** Instead of adding the whole LoRA update to the base weights, each weight
matrix keeps only the largest 25% (by magnitude) of its fine-tuning changes, and those are applied at 0.7x
([DLER](https://arxiv.org/abs/2510.15110)). The model keeps most of the efficiency gain while staying closer to the
base model on hard problems. Thinklite-27B Max applies the full update.

## Benchmarks

**Setup:**
- One frozen suite: identical prompts, chat template, token caps and scoring for every model.
- vLLM 0.30.0, bf16 weights, FP8 KV cache, Qwen3.8's thinking-mode sampling, 64 concurrent requests.
- One NVIDIA RTX PRO 6000 Blackwell (96 GB).
- All rows use `reasoning_effort=low` except the reference "Qwen3.8-27B (xhigh)" column.
- Answers that hit their cap were rerun once with a 131,072-token cap. Truncated or unparseable answers count as
  incorrect.

- **GSM8K**: 1,319 problems
- **AIME 2025**: 30 problems x 2 samples, avg@2
- **MMLU-Pro**: 1,000-question stratified subset
- **IFEval**: 541 prompts, prompt-level strict
- **LiveCodeBench v6**: 131 problems (2025-02 to 2025-04), pass@1
- **BFCL v4**: 1,240 non-live function-calling cases

### Accuracy (%)

| Benchmark | Qwen3.8-27B (xhigh) | Qwen3.8-27B (low) | Thinklite-27B Low | Thinklite-27B High | Thinklite-27B Max |
|---|---|---|---|---|---|
| GSM8K | 97.2 | 97.4 | 96.9 | 97.3 | 96.7 |
| AIME 2025 | 93.3 | 90.0 | 88.3 | 78.3 | 65.0 |
| MMLU-Pro | 86.3 | 84.4 | 84.7 | 84.7 | 82.6 |
| IFEval | 91.1 | 86.3 | 84.3 | 83.9 | 83.0 |
| LiveCodeBench v6 | 84.0 | 71.8 | 72.5 | 67.2 | 61.8 |
| BFCL v4 | 88.7 | 90.8 | 90.4 | 90.4 | 90.2 |
| **Average** | | **86.8** | **86.2** | **83.6** | **79.9** |

### Reasoning tokens per answer (mean; change vs Qwen3.8-27B at low)

| Benchmark | Qwen3.8-27B (low) | Thinklite-27B Low | Thinklite-27B High | Thinklite-27B Max |
|---|---|---|---|---|
| GSM8K | 183 | 163 (-11%) | 147 (-20%) | 124 (-32%) |
| AIME 2025 | 12,125 | 8,374 (-31%) | 8,618 (-29%) | 7,867 (-35%) |
| MMLU-Pro | 1,103 | 934 (-15%) | 748 (-32%) | 554 (-50%) |
| IFEval | 659 | 464 (-30%) | 416 (-37%) | 326 (-51%) |
| LiveCodeBench v6 | 10,741 | 9,152 (-15%) | 7,581 (-29%) | 5,639 (-48%) |
| BFCL v4 | 96 | 80 (-17%) | 76 (-21%) | 70 (-27%) |
| **All output tokens, 4,291 requests** | 4,915,895 | 4,244,014 (-14%) | 3,915,800 (-20%) | 3,285,976 (-33%) |

### Time to full answer, single user (seconds, median / p95; speedup at the median)

| Benchmark | Qwen3.8-27B (low) | Thinklite-27B Low | Thinklite-27B High | Thinklite-27B Max |
|---|---|---|---|---|
| GSM8K | 11.3 / 23 | 10.4 / 21 (1.08x) | 10.3 / 20 (1.10x) | 9.7 / 17 (1.16x) |
| AIME 2025 | 181.9 / 1612 | 156.9 / 1109 (1.16x) | 147.6 / 1227 (1.23x) | 122.7 / 1163 (1.48x) |
| MMLU-Pro | 28.9 / 169 | 25.4 / 118 (1.14x) | 24.3 / 88 (1.19x) | 20.9 / 68 (1.38x) |
| IFEval | 25.1 / 110 | 19.3 / 85 (1.30x) | 18.9 / 80 (1.33x) | 15.8 / 69 (1.59x) |
| LiveCodeBench v6 | 129.4 / 1764 | 125.0 / 1588 (1.04x) | 89.4 / 1542 (1.45x) | 67.6 / 1323 (1.91x) |
| BFCL v4 | 5.6 / 19 | 5.4 / 17 (1.04x) | 5.4 / 16 (1.04x) | 5.4 / 16 (1.04x) |

TTFA is computed from each answer's length, the prompt prefill time and the single-user decode speed measured on the
same GPU (about 28 tokens/s for every model). Thinklite-27B Low was timed in a separate run on the same GPU type; its
times are shown relative to the base model timed in that run (same speedup, common base timing).

## Usage

**vLLM**

```bash
vllm serve Mintzs/thinklite-27B --max-model-len 135168 --kv-cache-dtype fp8                    # Thinklite-27B High
vllm serve Mintzs/thinklite-27B --revision low --max-model-len 135168 --kv-cache-dtype fp8     # Thinklite-27B Low
vllm serve Mintzs/thinklite-27B --revision max --max-model-len 135168 --kv-cache-dtype fp8     # Thinklite-27B Max
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
reply = client.chat.completions.create(
    model="Mintzs/thinklite-27B",
    messages=[{"role": "user", "content": "How many primes are there below 100?"}],
    extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
)
print(reply.choices[0].message.content)
```

**Transformers**

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained("Mintzs/thinklite-27B", revision="main", dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("Mintzs/thinklite-27B", revision="main")
```

Use `revision="low"` or `revision="max"` for the other versions.

**llama.cpp (Thinklite-27B High, GGUF)**

```bash
llama-server -hf Mintzs/thinklite-27B:Q4_K_M --jinja -c 32768 --chat-template-kwargs '{"reasoning_effort": "low"}'
# or with local files:
llama-server -m Thinklite-27B-High-Q4_K_M.gguf --mmproj mmproj-Thinklite-27B-High-F16.gguf --jinja -c 32768
```

**Adapters** (`adapters` branch): the LoRA adapters for the full-dataset training (`adapters/full`) and the
smaller-dataset training (`adapters/small`). Loading the full-dataset adapter with PEFT gives Thinklite-27B Max; High
and Low are those adapters with the update-selective merge above, so use the merged weights for them.

## Files

- `main`: Thinklite-27B High, merged bf16 safetensors with tokenizer, chat template and processor files; plus
  `Thinklite-27B-High-Q4_K_M.gguf` and `mmproj-Thinklite-27B-High-F16.gguf` (the vision projector).
- `low`: Thinklite-27B Low, merged bf16 safetensors.
- `max`: Thinklite-27B Max, merged bf16 safetensors.
- `adapters`: the LoRA adapters.

## Limitations

- Evaluated at `reasoning_effort=low`. Other effort levels are supported but are not part of the tables above.
- The vision encoder is unchanged from Qwen3.8-27B; training data was text only.
- As with any fine-tune, check the model on your own tasks before relying on it.

## License

Apache-2.0, the license of the base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
