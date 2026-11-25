---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B-FP8
  - Qwen/Qwen3.8-27B
tags:
  - qwen
  - qwen3
  - uncensored
  - refusal-removal
  - fp8
  - reasoning
  - vllm
  - sglang
language:
  - en
pipeline_tag: text-generation
library_name: transformers
---

![Raven-27B](banner.png)

# Raven-27B

> You are on **the full-size download** (`KC-Informatics/Raven-27B`). The other size is [`KC-Informatics/Raven-27B-AWQ`](https://huggingface.co/KC-Informatics/Raven-27B-AWQ). Both are the same model and share this page.

The strongest of our three models. Raven started as Qwen 3.8 27B. We edited its weights so that it refuses far less, and checked that it still says "I don't know" when a question has no real answer. Nothing was retrained.

On our 180 test questions it answered **146**. DeepSeek V4.1 Flash, a 671B model you can only use through the cloud, answered 154.

![Raven-27B against DeepSeek V4.1 Flash, by subject](chart_raven_vs_deepseek.png)

## Two downloads, one model

| | 4-bit | Full size |
|---|---|---|
| Repo | [`KC-Informatics/Raven-27B-AWQ`](https://huggingface.co/KC-Informatics/Raven-27B-AWQ) | [`KC-Informatics/Raven-27B`](https://huggingface.co/KC-Informatics/Raven-27B) |
| Download | 19.6 GB | 30.9 GB |
| Graphics memory | 21.2 GB in use when loaded with its draft model | The weights alone are 31 GB, so a 40 GB card or larger |
| Score on the 180 questions | 145 (with the helper) | 146 (without the helper) |
| Pick it if | You want the smaller, faster one. This is the one most people should take. | You have the memory and want the weights before compression. |

## What we changed, in plain words

Qwen 3.8 27B turns down a lot of requests with a stock reply. Inside the model, the decision to refuse shows up as a recognisable pattern of activity. We measured that pattern and removed it from the weights of the middle and upper layers (24 to 58 of 64).

A second pattern, the one the model shows when it truly does not know something, was measured separately and protected. That is why Raven will still tell you when a question has no answer instead of making one up.

| | Original Qwen 3.8 27B | Raven-27B (4-bit) |
|---|---:|---:|
| Answered requests that the original usually refuses | 2 of 12 | **8 of 12** |
| Said "I don't know" to trick questions with no real answer | 8 of 8 | **8 of 8** |
| Answered ordinary requests | 8 of 8 | 8 of 8 |

<details>
<summary>For the technically minded</summary>

- Method name in our files: `NS-UNREFUSE-v2`, a null-space projection edit. Script and settings are in `alteration_manifest.json`.
- Two refusal directions are made orthogonal to one "does not know" direction before removal (`gamma_ep = 0.50`), with light damping along 16 directions of ordinary prompts (`gamma_clean = 0.10`). Each edited row is scaled back to its original length.
- 68 weight matrices edited in layers 24 to 58: 8 attention outputs, 26 linear-attention outputs, 34 MLP down-projections. Layers 0 to 23 and 59 to 63, and the vision weights, are untouched.
- 4-bit build: AWQ (W4A16, `llmcompressor`), 128 calibration samples of 1,024 tokens, with the gated recurrent projections kept at full precision so the draft model keeps working.

</details>

## Speed

![Speed of Raven-27B, the original Qwen and DeepSeek](chart_speed.png)

Measured on one RTX PRO 6000 Blackwell (96 GB) card with the draft model `nace-ai/Qwen3.8-27B-DFlash2-Kapitsa`: 141 tokens a second in one chat, 220 when writing code, and 1,008 in total across eight chats at once.

## Results by subject

| | Raven-27B full size | Raven-27B 4-bit, with helper | Original Qwen 3.8 27B | DeepSeek V4.1 Flash (671B) |
|---|---:|---:|---:|---:|
| Science (GPQA Diamond, 60 questions) | 53 | 53 | 46 | 55 |
| Maths (AIME 2026, 30 questions) | 24 | 23 | 18 | 27 |
| Coding (LiveCodeBench v6, 40 questions) | 23 | 24 | 20 | 29 |
| Following instructions (IFEval, 30 questions) | 26 | 26 | 26 | 26 |
| Long documents (AA-LCR, 20 questions) | 20 | 19 | 14 | 17 |
| **All 180** | **146** | **145** | **124** | **154** |

![All models on all 180 questions](chart_overall.png)

## Run it

```bash
# 4-bit
vllm serve KC-Informatics/Raven-27B-AWQ --port 8000 \
  --trust-remote-code --gpu-memory-utilization 0.90 --max-model-len 131072 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"model": "nace-ai/Qwen3.8-27B-DFlash2-Kapitsa", "method": "dflash", "num_speculative_tokens": 8}'

# full size: the same command with KC-Informatics/Raven-27B
```

Needs vLLM 0.34 or newer. The `--speculative-config` line adds the draft model, which is where most of the speed comes from.

## The helper: `nudge.py`

Included in every repo. If the model uses up its thinking allowance without answering, the helper asks it once to conclude. Crow: 119 of 180 without it, 132 with it.

```bash
python nudge.py --upstream http://127.0.0.1:8000 --port 8001
```

Point your app at `http://127.0.0.1:8001/v1`.

## How we tested

180 public test questions, one attempt each, graded by script: 60 science (GPQA Diamond), 30 maths (AIME 2026), 40 coding (LiveCodeBench v6), 30 on following instructions (IFEval) and 20 on long documents (AA-LCR).

## Licence and credit

Apache 2.0, the same as `Qwen/Qwen3.8-27B` by the Qwen team. Draft model by nace-ai. `nudge.py` is Apache 2.0.
