---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
pipeline_tag: text-generation
library_name: transformers
tags:
  - deepseek
  - deepseek-v4.1
  - uncensored
  - research
  - red-team
  - moe
  - fp8
---

# DeepSeek-V4.1-Flash-1017-uncensored

An uncensored version of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (the 1017 release), produced by a **closed-form weight edit** (no fine-tuning) merged into the native FP8 checkpoint. It loads and serves exactly like the base model, and general capability — knowledge, reasoning, math — stays close to base level.

> ⚠️ **For authorized safety research only.** This model is deliberately uncensored and will comply with harmful requests. Keep it private; do not deploy or serve it publicly. You are responsible for all use.

## How it was made

Refusal behavior is removed by a **closed-form weight edit** — no fine-tuning and no gradient training — merged into the model's weights. The **384 routed experts are left untouched**, so the model's knowledge and capacity are unchanged, and it loads and serves exactly like the base checkpoint.

## Evaluation

Base and this model were scored with one identical harness, paired on the same prompts, so the deltas are meaningful.

### Capability is preserved (accuracy, higher is better)

| benchmark | base | this model |
|---|---:|---:|
| MMLU (14,042, full test set) | 88.22% | 87.91% |
| MMLU-Pro (4,000 sample) | 69.90% | 69.42% |
| MMLU-Pro (300 sample, reasoning) | 85.7% | 84.7% |
| GPQA-Diamond (198, reasoning) | 81.3% | 84.8% |
| GSM8K (step-by-step, greedy) | 93.5% | 96.0% |

MMLU-Pro (reasoning) and GPQA-Diamond are run with reasoning enabled — the mode the model is built for — while MMLU, MMLU-Pro (4,000) and GSM8K use the standard protocol. Across the suite the model **tracks base closely**: differences are small and mixed in sign — a few tenths of a point below base on MMLU (−0.3) and MMLU-Pro (−0.5, within noise), one point below on MMLU-Pro (reasoning; 21 of 300 answers differ), and above base on GSM8K and GPQA-Diamond. On GPQA a minority of the hardest items exhaust the 24k-token generation budget mid-reasoning (24 for base, 15 for this model); scoring only completed answers, base is 92.5% and this model 91.8%, and on the 174 questions both complete they get exactly the same number right (161). The edit removes refusals without a material change in the model's knowledge or reasoning.

### Refusal is removed — held-out harmful + benign sets, robust classifier

| metric | base | this model |
|---|---:|---:|
| refusal on harmful prompts (n = 1410) | 87.3% | **1.8%** |
| over-refusal on benign prompts (n = 350) | 6.3% | **0.0%** |

The edit removes refusals on harmful prompts and, at the same time, **reduces** over-refusal of benign prompts.

### Harmful compliance — LLM-judged (StrongREJECT rubric)

Each response is graded by an uncensored LLM judge (**dolly2-security**) with the published StrongREJECT rubric, scoring every answer 0–1 from whether it refused and how specific and convincing it is (higher = fuller compliance). Three harmful-prompt benchmarks, greedy, paired:

| benchmark | base | this model |
|---|---:|---:|
| HarmBench-320 (official test split) | 0.38 | **0.95** |
| StrongREJECT (313 prompts) | 0.02 | **0.99** |
| real-CVE exploit prompts (30) | 0.12 | **1.00** |

Judge-rated refusal rate falls accordingly — HarmBench 60.9% → **0.6%**, StrongREJECT 98.1% → **0%**, CVE 86.7% → **0%**. On the cybersecurity / CVE set the model complies on all 30 prompts at near-maximal rated specificity.

## Usage

Served with SGLang (DeepSeek-V4.1 support), the same way as the base model — it is a drop-in FP8 checkpoint:

```bash
python -m sglang.launch_server \
  --model-path MoonMath-ai/DeepSeek-V4.1-Flash-1017-uncensored \
  --tp-size 4 --ep-size 4 \
  --trust-remote-code \
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41
```

**Serving note:** at the maximum reasoning-effort setting the model reasons very thoroughly; give it adequate generation budget (≥16k tokens, or a moderate reasoning effort) so long answers complete rather than being truncated mid-reasoning.

## Limitations

- **Deliberately uncensored** — it will produce harmful content on request. Private, authorized research use only.
- Capability tracks base closely (MMLU −0.3 points); residual refusal on adversarial sets is low but non-zero.
- Derived from DeepSeek-V4.1-Flash-1017; the base model's license applies.

## Use it on Zro

These weights are served on **Zro**, MoonMath's private inference platform, as the model
**Abyss 1** (`abyss1`) — see the [Abyss 1 model page](https://zro.moonmath.ai/abyss1).
No GPU setup, and no prompts retained after processing.

**With the CLI** (open source, MIT):

```bash
npm install -g @moonmath-ai/zro
zro login
zro launch claude --model abyss1
```

`zro launch` also works with `codex`, `opencode`, `hermes`, `openclaw`, `pi` and many more harnesses.

**With the API** (OpenAI- and Anthropic-compatible):

```bash
curl https://zro.moonmath.ai/v1/chat/completions \
  -H "Authorization: Bearer $ZRO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"abyss1","messages":[{"role":"user","content":"Reply with OK"}]}'
```

- Model ID: `abyss1` (alias `zro/abyss1`). The security-focused `dolly2-security` and the
  lighter `dolly1-security` are also available.
- 1M-token context, up to 64K output. $0.30 per 1M input tokens, $1.10 per 1M output,
  $0.01 per 1M cache-read tokens. Prompt caching is on by default.
- Inference region: United States.
- Zero request retention, and no training on requests.
- Available on paid plans, which are monthly spend budgets, not token caps: Pro
  ($20/month, $40 usage budget) and Max ($60/month, $120 usage budget).

Links: [Abyss 1](https://zro.moonmath.ai/abyss1) ·
[pricing](https://zro.moonmath.ai/pricing) ·
[CLI docs](https://zro.moonmath.ai/docs/cli) ·
[API reference](https://zro.moonmath.ai/docs/api) ·
[privacy](https://zro.moonmath.ai/docs/privacy) ·
[sign up](https://zro.moonmath.ai/sign-up)
