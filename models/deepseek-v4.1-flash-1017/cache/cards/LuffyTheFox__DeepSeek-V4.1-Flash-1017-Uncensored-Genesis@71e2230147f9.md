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

# 🌟 Coming soon...DeepSeek-V4.1-Flash-1017-Uncensored-Genesis

> ⚡ If you like this Genesis LLM release you can donate to me via:

> USDT (TRC20): `TGa4KTwHfF6zDBsLUBEjd1f1KdeAFwYUks`

> USDT (ERC20): `0x93F4019E0aa85d8078F56B3D3176Fab5Dfa79924`

> USDT (SOL): `BorkkyPDG4aRN2op8c5SQF5NithX38U4sh7wDAWQhYyX`

> and support future Genesis LLM development. Details below:

<details>
❤️ Support Genesis — Help Independent Open AI Development
  
If Genesis models have improved your local AI experience, consider supporting their continued development.

Genesis is an independent AI development project focused on exploring ways to improve the reliability, efficiency, and capabilities of open-weight language models.

Over nearly six months, I've been developing and refining Genesis with the help of AI, mathematical analysis, extensive experimentation, and limited computing resources.

My goal is simple: make powerful local AI more accessible, useful, and enjoyable for everyone.

I share my work with the community because I believe meaningful AI innovation should not be limited to organisations with enormous computing budgets.

🚀 What Your Support Helps Make Possible

Your voluntary contributions can help me:

🔬 Continue researching and improving Genesis tensor calibration and processing techniques.

⚡️ Experiment with faster, more efficient local AI models.

🧠 Explore improvements in coding, reasoning, tool use, and long-context reliability.

🖥 Access better computing resources for development, testing, and validation.

📦 Release more models, quantisations, configurations, and technical updates for the community.

🌍 Keep sharing my experiments and releases openly.

Every contribution, no matter how small, helps support the time, research, experimentation, and resources behind this project.

🤝 Can't Donate? You Can Still Help!

Downloading models, sharing your results, reporting bugs, contributing benchmarks, and recommending Genesis to other developers are all valuable ways to support the project.

The models are available regardless of whether you donate. Donations are completely optional.

Thank you for experimenting with Genesis, sharing feedback, and helping independent AI development move forward.

— LuffyTheFox 🦊

Independent research. Open models. Community-powered progress.
</details>

> ⚡ **Why Genesis project exists?** During training, **ALL** models don't just learn knowledge - they also accumulate random noise in their tensors. This noise builds up and creates something I call the **Noise Gate** - a fundamental barrier that stops LLM models from learning further and makes them unstable, verbose, and prone to hallucinations. My approach reduces this noise. It repairs the signal in tensors without touching the learned knowledge and gradient using [Marchenko–Pastur distribution](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution) as a core criteria. The result is a model that consistent in performance, context clarity and following instructions, because it's no longer fighting its own internal chaos.

> **What is Genesis?** Genesis is post training data regeneration and calibrarion algorythm for neural networks (LLM) in GGUF format that I made with AI help during almost half a year of development. It's optimized, architecture independent, works with any model in GGUF format and based on mathematical statistics. I don't train or finetune models, I repair **purity of signal** in them instead on Google Collab Free on Tesla T4 GPU via Python based on how models learns information. On first stage I scan ssm_conv1d tensors in model, they handle long context memory. I repair balance between heads in them. On second stage, I scan blocks in model via chunks via 3 parameters and pick best one that fits to weight distribution in tensor. Best picked chunk replaces zero chunks in broken tensor without touching learned structure in model. On third stage I scan model and detect noise in tensors via custom SVD. During scanning I exclude token_embd.weight, output.weight, 1D tensors, bias and norms. Then I reduce training noise in tensors via custom SVD based on [Marchenko–Pastur law](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution) with preserved training data, 99% of siginal and learned gradient.

## Any questions?

> Contact: luffythefox@mail.ru, azakharchenko92@gmail.com

> My Telegram: @LuffyTheFox

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