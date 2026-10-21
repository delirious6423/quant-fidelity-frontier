---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: Ora Computing
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - quantized
  - nvfp4
  - fp8
  - compressed-tensors
  - vllm
  - reasoning
  - qwen3_5
---

<div align="center">
  <img src="assets/ora-logo-extended-white.svg" alt="Ora" width="520"/>
  <br/>
  <a href="https://www.oracomputing.com/en"><b>Website</b></a>
  &nbsp;·&nbsp;
  <a href="https://www.oracomputing.com/en/blog"><b>Blog</b></a>
  &nbsp;·&nbsp;
  <a href="https://www.oracomputing.com/en/contact"><b>Contact</b></a>
</div>

# Qwen3.8-27B-NVFP4-Efficient

> [!IMPORTANT]
> **A more efficient thinker at the same size.** Matches the accuracy of [Unsloth's NVFP4 Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) at essentially the same memory footprint, while using **23% fewer tokens on average** across math, coding and agentic benchmarks, and up to **30% fewer** on MATH500 and MBPP+.

![Average tokens per task, this model against Unsloth NVFP4](assets/tokens_per_task.png)

This model was produced in two optimization stages. First, we quantized [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) to mixed-precision NVFP4 and FP8, for both weights and activations. An adaptive algorithm gives more bit-width to sensitive layers, keeping the KL divergence to the unquantized model low under a fixed size budget (see more details in out [blog](https://www.oracomputing.com/blog/where-the-bits-go)). Second, we applied **On-Policy Self-Distillation (OPSD)** combined with **Quantization-Aware Training (QAT)** to teach the model to think more efficiently while preserving its performance.

<table>
  <thead>
    <tr><th align="left">Model</th><th align="left">Format</th><th align="right">Size</th></tr>
  </thead>
  <tbody>
    <tr><td>Qwen3.8-27B (original)</td><td>BF16</td><td align="right">55.56 GB</td></tr>
    <tr><td>Unsloth NVFP4</td><td>Mixed NVFP4 / FP8</td><td align="right">23.42 GB</td></tr>
    <tr style="background-color: rgba(104, 61, 255, 0.12)">
      <td><strong>This model</strong></td>
      <td><strong>Mixed NVFP4 / FP8</strong></td>
      <td align="right"><strong>23.84 GB</strong></td>
    </tr>
  </tbody>
</table>

## Thinking efficiency

Accuracy against the maximum number of output tokens allowed. A point at token cap *B* counts the answers that were correct **and** finished within *B* tokens. At almost every cap, this model gets more problems right.

![Accuracy vs token cap on AIME25 and MATH500](assets/accuracy_vs_token_budget_gain.png)

<table>
  <thead>
    <tr>
      <th align="left">Benchmark</th><th align="left">Model</th><th align="right">Accuracy</th>
      <th align="right">Mean output tokens</th><th align="right">Median output tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>AIME25 (avg@8) 32K token budget</td><td>Unsloth NVFP4</td><td align="right">77.1</td><td align="right">15,532</td><td align="right">11,698</td></tr>
    <tr style="background-color: rgba(104, 61, 255, 0.12)">
      <td></td><td><strong>This model</strong></td><td align="right"><strong>82.9</strong></td>
      <td align="right"><strong>13,039 (−16%)</strong></td><td align="right"><strong>7,590 (−35%)</strong></td>
    </tr>
    <tr><td>MATH500</td><td>Unsloth NVFP4</td><td align="right">94.6</td><td align="right">2,610</td><td align="right">620</td></tr>
    <tr style="background-color: rgba(104, 61, 255, 0.12)">
      <td></td><td><strong>This model</strong></td><td align="right"><strong>94.0</strong></td>
      <td align="right"><strong>1,838 (−30%)</strong></td><td align="right"><strong>552 (−11%)</strong></td>
    </tr>
    <tr><td>MBPP+</td><td>Unsloth NVFP4</td><td align="right">82.8</td><td align="right">2,179</td><td align="right">446</td></tr>
    <tr style="background-color: rgba(104, 61, 255, 0.12)">
      <td></td><td><strong>This model</strong></td><td align="right"><strong>80.7</strong></td>
      <td align="right"><strong>1,527 (−30%)</strong></td><td align="right"><strong>336 (−25%)</strong></td>
    </tr>
    <tr><td>InterCode CTF (agentic, avg@4)</td><td>Unsloth NVFP4</td><td align="right">80.8</td><td align="right">7,714</td><td align="right">—</td></tr>
    <tr style="background-color: rgba(104, 61, 255, 0.12)">
      <td></td><td><strong>This model</strong></td><td align="right"><strong>82.1</strong></td>
      <td align="right"><strong>6,618 (−14%)</strong></td><td align="right">—</td>
    </tr>
  </tbody>
</table>

A run counts as correct only if the right answer arrived within the token limit: 32,768 for AIME25 and MATH500, 16,384 for MBPP+. Hitting the limit counts as a failure, because the user never gets an answer. Token changes are against Unsloth NVFP4. InterCode CTF is multi-turn, so its token count covers the whole agent run rather than one answer, and a median per challenge would not mean much. This table counts all 30 AIME25 problems and InterCode CTF's total output, so those two rows differ from the plot at the top.

- **Accuracy is a tie.** On MATH500 the two NVFP4 models disagree on 11 of 500 problems (p = 0.55). On MBPP+, 24 of 378 (p = 0.15). On InterCode CTF at avg@4, 30 of 78 challenges, split exactly 15/15 (p = 1.00). On AIME25 the gap sits inside one standard error.
- **Fewer runaway traces.** On AIME25, 13% of our runs hit the 32K limit without finishing, against 22% for Unsloth. On MBPP+, traces over 4K tokens fall to 34 of 378, from 59.
- **Faster end to end.** Same GPU, 32 concurrent requests: MATH500 finished in **17.4 min against 23.8** (−27%), and MBPP+ generation took **11.5 min against 14.6** (−21%).
- **In agent loops the saving is smaller.** On InterCode CTF we write **26% fewer thinking tokens** per challenge (2,795 against 3,793) at the same accuracy, but total output drops only 14%: about half of what an agent emits is tool-call plumbing, not reasoning.

## Nothing sacrificed

Higher is better unless noted. ± is one standard error. Retention is this model's score relative to BF16.

<table>
  <thead>
    <tr>
      <th align="left">Benchmark</th><th align="center">Thinking</th>
      <th align="right">Qwen3.8-27B (BF16)</th><th align="right">Unsloth NVFP4</th>
      <th align="right" style="background-color: rgba(104, 61, 255, 0.12)">This model</th>
      <th align="right">Retention</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>AIME25 (avg@8) (32K cap)</td><td align="center"><strong>on</strong></td><td align="right">76.7 ± 6.2</td><td align="right">77.1 ± 6.3</td><td align="right" style="background-color: rgba(104, 61, 255, 0.12)"><strong>82.9 ± 5.2</strong></td><td align="right">108.1%</td></tr>
    <tr><td>MATH500</td><td align="center"><strong>on</strong></td><td align="right">95.2 ± 1.0</td><td align="right">94.6 ± 1.0</td><td align="right" style="background-color: rgba(104, 61, 255, 0.12)"><strong>94.0 ± 1.1</strong></td><td align="right">98.7%</td></tr>
    <tr><td>MBPP+</td><td align="center"><strong>on</strong></td><td align="right">82.8 ± 1.9</td><td align="right">82.8 ± 1.9</td><td align="right" style="background-color: rgba(104, 61, 255, 0.12)"><strong>80.7 ± 2.0</strong></td><td align="right">97.5%</td></tr>
    <tr><td>InterCode CTF (agentic)</td><td align="center"><strong>on</strong></td><td align="right">85.9 ± 4.0</td><td align="right">80.8 ± 3.4</td><td align="right" style="background-color: rgba(104, 61, 255, 0.12)"><strong>82.1 ± 3.3</strong></td><td align="right">95.5%</td></tr>
    <tr><td>MATH-500</td><td align="center">off</td><td align="right">90.8 ± 1.3</td><td align="right">89.6 ± 1.4</td><td align="right" style="background-color: rgba(104, 61, 255, 0.12)"><strong>89.8 ± 1.4</strong></td><td align="right">98.9%</td></tr>
    <tr><td>IFEval (prompt-level strict)</td><td align="center">off</td><td align="right">81.5 ± 1.7</td><td align="right">81.2 ± 1.7</td><td align="right" style="background-color: rgba(104, 61, 255, 0.12)"><strong>80.0 ± 1.7</strong></td><td align="right">98.2%</td></tr>
    <tr><td>BFCL v3 (single + multi-turn)</td><td align="center">off</td><td align="right">72.2 ± 0.9</td><td align="right">71.7 ± 0.9</td><td align="right" style="background-color: rgba(104, 61, 255, 0.12)"><strong>73.1 ± 0.9</strong></td><td align="right">101.3%</td></tr>
    <tr><td>WikiText word perplexity (lower is better)</td><td align="center">off</td><td align="right">8.48</td><td align="right">8.64</td><td align="right" style="background-color: rgba(104, 61, 255, 0.12)"><strong>8.65</strong></td><td align="right">98.0%</td></tr>
  </tbody>
</table>

Every difference against Unsloth NVFP4 is within two standard errors, and this model retains at least 95% of BF16's score on every benchmark. **Thinking-on rows count runs that hit the token limit as failures**, as in the table above. BFCL v3 combines single-turn and multi-turn accuracy with BFCL's weighting. WikiText retention is BF16 perplexity divided by this model's.

On both coding-style benchmarks the gap against Unsloth NVFP4 is small and not significant: −2.1 points on MBPP+ (p = 0.15, one pass) and +1.3 on InterCode CTF (p = 1.00, avg@4). 

## Usage (vLLM)

NVFP4 runs natively on NVIDIA Blackwell GPUs.

```bash
vllm serve oracomputing/Qwen3.8-27B-NVFP4-Efficient \
  --max-model-len 36864 \
  --reasoning-parser qwen3
```

Thinking is on by default at reasoning effort `xhigh`, the setting used for all results above. Pass `chat_template_kwargs` to change it per request:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="oracomputing/Qwen3.8-27B-NVFP4-Efficient",
    messages=[{"role": "user", "content": "How many positive divisors does 2026 have?"}],
    temperature=1.0, top_p=0.95, max_tokens=32768,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},  # or "medium", "low"
    },
)
print(response.choices[0].message.content)
```

Set `"enable_thinking": False` for instruct-style answers without a thinking trace. `reasoning_effort` is also accepted as a top-level OpenAI field; the standard names map onto the three levels this model was trained for, so existing OpenAI code works unchanged:

| Requested | Applied |
| --- | --- |
| `high`, `max`, `xhigh` | `xhigh` (default) |
| `medium` | `medium` |
| `low`, `minimal` | `low` |
| `none` | thinking off |

**Multimodal inputs.** The base model's vision tower is carried over in BF16, left unquantized and untouched by OPSD/QAT, so image and video inputs work as they do in [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). All results on this card are text-only; we did not evaluate vision.

**Recommended sampling** (same as [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)):

- Thinking mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
- Non-thinking mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`

## Evaluation setup

All three models were evaluated with the same harness, prompts and engine (vLLM 0.28.0).

**Thinking on** (AIME25, MATH500, MBPP+, InterCode CTF)

| Setting | Value |
| --- | --- |
| Reasoning effort | `xhigh` (`enable_thinking=True`, chat template default) |
| Sampling | `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0` |
| Max output tokens | AIME25 / MATH500: 32,768 (context 36,864). MBPP+: 16,384 (context 20,480). InterCode CTF: 4,096 per turn (context 65,536) |
| Samples | AIME25: 8 per problem (avg@8, 240 runs). InterCode CTF: 4 per challenge for the NVFP4 models (avg@4, 312 runs), 1 for BF16. MATH500, MBPP+: 1 per problem |
| Scoring | AIME25 / MATH500: exact match on the last `\boxed{}` answer. MBPP+: pass@1 against the full MBPP+ test harness. InterCode CTF: exact flag match |
| Token counts | Full output including the thinking trace, Qwen3.8-27B tokenizer |

With `reasoning_effort="xhigh"` the chat template adds this system message:

```text
Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.
```

User prompt (AIME25 adds "an integer between 0 and 999," before "within"):

```text
Solve the following math problem. Reason step by step, then give your final answer within \boxed{}.

Problem:
{problem}
```

MBPP+ (378 problems) is prompted zero-shot with the first assertion included to pin the function signature, and asks for the solution in a single `python` block; the answer is the last code block after the thinking trace, graded against the full MBPP+ extended test harness (not the single visible assertion) with a 30-second timeout. Note that the standard `mbpp_plus_instruct` task cannot be used to evaluate a thinking model: it prefills the assistant turn with a code fence, which removes the `<think>` opener and silently disables thinking.

InterCode CTF (78 challenges, [`inspect_evals/gdm_intercode_ctf`](https://github.com/UKGovernmentBEIS/inspect_evals)) is agentic: the model drives `bash` and `python` tools in a network-isolated Docker sandbox, with up to 50 messages and 3 submission attempts per challenge. It is served through vLLM with `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes`; without the tool-call parser the model cannot act and scores zero. A challenge whose transcript exhausts the 65,536-token context is scored as a failure.

**Thinking off** (general capability)

| Benchmark | Setup |
| --- | --- |
| IFEval | 0-shot, greedy, 1,280 max tokens, prompt-level strict accuracy |
| MATH-500 | 0-shot, greedy, 4,096 max tokens, exact match |
| BFCL v3 | Qwen3 function-calling handler, temperature 0.001 |
| WikiText | Word perplexity |

All thinking-off runs use `enable_thinking=False` and a context of 8,192 tokens.

## License and attribution

Released under the [Apache License 2.0](LICENSE), inherited from the base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Copyright 2026 Alibaba Cloud), whose license text is included verbatim in this repository.

Quantized and post-trained by **Ora Computing**. Changes from the base model: mixed-precision NVFP4/FP8 quantization of the language-model weights and activations, followed by On-Policy Self-Distillation and Quantization-Aware Training. The tokenizer is unmodified. The vision tower is carried over in BF16 without quantization. The base model's multi-token-prediction head is not included.
