---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
  - exl3
  - exllamav3
  - quantized
  - moe
  - deepseek
  - multimodal
  - dgx-spark
---

# DeepSeek-V4.1-Flash · EXL3 · 2.86 bpw · Hessian-traced

A quality-first quant of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) for **two NVIDIA DGX Sparks**.
Only the 384 routed experts are compressed (2–5 bits per expert, 2.86 bpw average). Attention, shared experts, head, drafter
and vision stay exactly as DeepSeek shipped them.

**Hessian-traced:** EXL3 shapes each expert's compression with a Hessian (which inputs matter most to that matrix).
Here the Hessians were re-traced on the quantized model's own activations over 1.03M tokens, so every one of the 384×40
experts was calibrated on the path it actually serves. Experts that calibration never reached went from 56 to 0.

This card focuses on **how the quant was evaluated against the parent model**: closeness of its next-token predictions,
knowledge and reasoning, long generation, stability and speed.

<img src="assets/x-1-launch.jpg" alt="Quality and speed" width="480">

## Evaluation at a glance

| What | How | Size | Result |
|---|---|---|---|
| Drift from the parent, agent sessions | Served KLD vs a full-model reference | 70,570 tokens | 0.168 mean KLD |
| Drift from the parent, general text | Same, public test text | 113,451 tokens | 0.088 mean KLD |
| Drift at agent decision points | Same, only where a run turns (tool or answer, which tool, another call, stop) | 285 tokens | 0.034 mean KLD |
| Same next token as the parent | Same runs | 184,021 tokens | 90.7% / 93.1% |
| Perplexity vs the parent | Same runs | 184,021 tokens | +1.4% / +1.9% |
| MMLU | All test questions, greedy, thinking off | 14,042 | 88.0% |
| GSM8K | Thinking on, greedy | 100 | 97% |
| Long generation (3D game in three.js) | 8 tries, headless play test | 8 games | 8/8 work, 33.5k tokens median |
| Stability | 6 h soak, mixed traffic, 4 streams | 3,822 replies | 0 failures |
| Speed | TensorFold, two Sparks | | 69–76 tok/s code, 1,582–2,072 tok/s prefill |

## 1. Closeness to the full model (KLD)

KL divergence measures how far the quant's next-token distribution is from the full model's at each position. It catches
damage that task scores average away. Zero means identical.

**The reference.** The full model needs 286 GiB of GPU memory; two Sparks have about 240. So we ran it one layer at a time from
disk, in bf16 from DeepSeek's native weights (commit `df42c10`), and stored its top-20 distribution at every position.
We first checked this layer-by-layer forward against DeepSeek's reference inference code. That check found a RoPE bug in our
loader (one tensor never initialized), which had put the reference perplexity at 1,916–4,966. After the fix: 4.01.

**The text.** Two holdouts, never used for calibration:

| Holdout | Content | Tokens | Notes |
|---|---|---|---|
| In-domain | Real agent sessions: chat, tool calls, thinking, rendered in the V4.1 chat format | 70,606 (18 rows) | Private; split from the calibration corpus by session, so no session is in both. sha256 `a4e5657c…` |
| General | Prose and long documents (WikiText-103), code (HumanEval), maths (GSM8K), multilingual (Belebele, 8 languages) | 115,383 (18 rows) | Public test splits, raw text, 13-gram overlap with calibration removed. sha256 `a520c06a…` |

**The scoring.** Each quant was served (vLLM with an EXL3 overlay; see Credits), fed the holdouts teacher-forced at 2,048-token windows, and
compared position by position with the stored reference.

| | Full model | This quant |
|---|---|---|
| **Agent sessions** (70,570 positions) | | |
| Mean KLD | 0 | 0.168 |
| Median KLD | 0 | 0.0065 |
| 90th percentile | 0 | 0.379 |
| 99th percentile | 0 | 2.65 |
| Same top-1 token | 100% | 90.7% |
| Full model's top-20 found in quant's top-20 | 100% | 83.8% |
| Perplexity | 4.010 | 4.068 (+1.4%) |
| **General text** (113,451 positions) | | |
| Mean KLD | 0 | 0.088 |
| Median KLD | 0 | 0.0036 |
| 90th percentile | 0 | 0.242 |
| 99th percentile | 0 | 1.13 |
| Same top-1 token | 100% | 93.1% |
| Full model's top-20 found in quant's top-20 | 100% | 85.1% |
| Perplexity | 2.339 | 2.383 (+1.9%) |

KLD is the headline metric rather than perplexity: a quant can drift and still land near the parent's perplexity on a given
text, but it cannot drift and keep a low KLD.

<img src="assets/x-8-vs-parent.jpg" alt="Against the full model" width="480">

### Decision points in agent sessions

The agent-session KLD above averages every token. An agent run is usually derailed by a few of them: whether to call a
tool or answer, which tool, whether to make another call, and when to stop. This scores only those tokens, from the same
served runs and the same full-model reference:

| Decision | Positions | This quant: KLD, same top-1 | Mia-AiLab 2.9 bpw: KLD, same top-1 | KLD ratio (95% CI) |
|---|---|---|---|---|
| Call a tool, or answer (the tokens after `</think>`) | 105 | 0.059, 96.2% | 0.091, 94.3% | 0.65 (0.38–0.87) |
| Which tool (every token of the name) | 96 | 0.032, 93.8% | 0.107, 90.6% | 0.30 (0.12–0.70) |
| Which tool (first token of the name) | 42 | 0.052, 88.1% | 0.197, 85.7% | 0.26 (0.09–0.70) |
| Another call, or done | 41 | 0.003, 95.1% | 0.010, 92.7% | 0.30 (0.22–0.40) |
| Stop (end of the turn) | 48 | 0.004, 100% | 0.005, 100% | 0.75 (0.32–1.71) |
| **All decision points** | **285** | **0.034, 95.8%** | **0.071, 93.7%** | **0.47 (0.21–0.73)** |
| Every position (the headline) | 70,570 | 0.168, 90.7% | 0.248, 88.4% | |

"Same top-1" is how often the quant's most likely token is the full model's. The ratio is this quant's mean KLD over the
reference's on the same positions; its interval resamples whole sessions. The same-top-1 gaps are small, so most of the
difference is in how sure each quant is, not in which token wins. Both are near-identical on stopping.

It is a small sample (18 sessions, 42 tool calls) and teacher-forced, so it measures single decisions, not whole agent
runs. The script is [`tools/dsv41/decision_kld.py`](https://github.com/Kristianaaron/dsv41-tensorfold/blob/main-cebu/tools/dsv41/decision_kld.py);
its every-position row reproduces the headline KLD exactly.

## 2. Knowledge and reasoning

| Test | Setup | Result |
|---|---|---|
| MMLU | All 14,042 test questions, greedy, thinking off; unparsed or cut-off answers count as wrong | 88.0% (12,359) |
| GSM8K | 100 questions, greedy, thinking on | 97% |

Release check on the final build (seed 20261004, fixed 200 MMLU + 100 GSM8K subset): MMLU 86.0% thinking off / 91.5% on
(6 answers hit the 8,192-token cap and count as wrong); GSM8K 95% off / 97% on. No request errors.

The full model has no task scores on this hardware: it does not fit. We do not copy DeepSeek's published scores into this
table because they come from a different harness.

## 3. Long generation

Short questions hide drift that compounds over long outputs, so we added a long coding task.

- **Prompt:** one request for a complete 3D Flappy Bird in a single HTML file with three.js. Thinking on.
- **Tries:** 8 (greedy plus seeds 1–7 at temperature 0.6, top-p 0.95), up to 98,304 tokens each, 4 requests at once.
- **Pass test:** a headless browser loads each game and checks that it renders with WebGL, flaps, crashes, restarts and resizes
  with no JavaScript errors.

| | Result |
|---|---|
| Working games | 8 / 8 |
| Tokens per game, median | 33.5k |
| Finished inside 32,768 tokens | 4 / 8 |
| Longest try | 43.8k tokens |
| Full game drafts written inside the reasoning, mean | 2.1 |
| Minutes to a working game, median (4 requests at once) | 17.5 |

## 4. Stability

6-hour soak of the release build: 3,822 replies, 0 failures, 2.07M completion tokens. Mixed code, prose, structured output and
reasoning; greedy and sampled; thinking on and off; 4 streams. Lowest free memory during the run: 4.7 GiB on the head node and
6.1 GiB on the worker (guard floor 3 GiB).

## 5. Speed (the cost)

Measured on [TensorFold](https://github.com/ashhart/TensorFold) (our fork of bertholomus's DeepSeek-V4.1 port), two Sparks, vision on, 262k context, native DSpark drafter:

| | tok/s |
|---|---|
| New code, 1 stream | 69–76 |
| Structured output (JSON), 1 stream | 85–105 |
| Prose, 1 stream | 41–42 |
| Code edits (drafts from the prompt) | 137–138 |
| Prefill, 8k / 32k tokens | 1,680 / 1,582 |
| Prefill, 131k / 245k tokens | 1,726 / 1,706 |
| 4 streams at once, total | 104 |

Ranges are two prompt sets (512- and 384-token replies), median of 3 runs each, release build, 2026-10-08. Prefill varies from run to run, because a request's first chunk is slow: warm runs of the release build reach 2,044–2,072 tok/s at 8k–32k.

Each decode step reads the backbone once, and the original FP8 backbone is about 4.3 GiB per node. Compressing it would make
fresh text faster; this build keeps it as DeepSeek shipped it, for quality.

<img src="assets/x-7-speed.jpg" alt="Speed on TensorFold" width="480">

## 6. How this build was chosen

Every candidate was scored with the same served-KLD harness (in-domain / general, lower is better).

Mean KLD was not the only thing that mattered:

- **Both domains.** A change had to help agent sessions *and* general text. Mean-shift correction cut general error by 30% but raised in-domain KLD by 14%, so it was dropped.
- **The worst tokens.** The 99th-percentile KLD (2.65 / 1.13) is where drift turns into visible mistakes, so we tracked the tail, not just the mean.
- **No capability cuts.** All 384 experts, vision, 262k context and the native DSpark drafter stay on. Memory for extra bits came from the budget, never from removing features.
- **Long generations.** Tokens spent and full redrafts on a long coding task (section 3), because short-answer metrics hide compounding drift.
- **Stability.** A 6-hour soak with a 3 GiB free-memory floor on each node (section 4).

| Build | Expert bpw | What changed | KLD |
|---|---|---|---|
| v3 | 2.63 | Per-layer bits from a sensitivity scan | 0.209 / 0.131 |
| v4 | 2.41 | Bits per expert, not per layer; a speed cap left 7 GiB per node unused | 0.198 / 0.129 |
| v4.1 | 2.66 | Speed cap removed, full memory budget, floor for rarely used experts | 0.176 / 0.104 |
| **v5 (this)** | **2.86** | Calibration recaptured on the quantized model's own activations (1.03M tokens); 262k context frees memory for bits | **0.168 / 0.088** |

Calibration coverage for v5: experts that calibration never reached went from 56 to 0, and thinly covered experts from
1,548 to 47.

Tried and rejected, by measurement:

| Idea | Result vs the build it would replace |
|---|---|
| Asymmetric calibration | +8% in-domain / +22% general KLD (worse) |
| One shared bit map for all layers | +3.4% / +3.6% (worse) |
| Mean-shift correction | −30% general error, but +14% in-domain (rejected: in-domain matters most) |
| Newer encoder (exllamav3 1.5.4) | +0.9% / +0.5% (worse) |
| Speed cap on expert bits | Spent only 12.8 of 20 GiB per node |

<img src="assets/x-12-builds-card.jpg" alt="Every build, measured the same way" width="480">

## 7. Numerical checks

- The FP8 backbone kernels used in serving were compared with fp64 reference maths on 624 projection and batch-shape cases
  (decode and prompt paths): in every case at least 99.7% of outputs equal a correctly rounded bf16 result; relative L2 error at most 0.0018.

## What we did not evaluate

- KLD was measured through a vLLM serve. The TensorFold engine cannot output probabilities yet, so KLD was not re-measured there.
- No code benchmark pass rates (HumanEval+, MBPP+), no tool-calling benchmark (only the decision-point KLD above), and no
  long-context retrieval test yet.
- The in-domain holdout is private (real sessions), so others can reproduce the general-text numbers but not the in-domain ones.
- The long-generation test is one prompt with 8 tries.
- The full model has no task or speed numbers on this hardware.

## Model details

| | |
|---|---|
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), commit `df42c10` |
| Routed experts | EXL3 (trellis, MCG codebook), 2.86 bpw average |
| Bit split (30,720 expert matrices) | 2-bit 7,575 · 3-bit 15,697 · 4-bit 6,877 · 5-bit 571 |
| Attention, shared experts | original FP8 |
| Output head | original bf16 |
| DSpark drafter, vision tower | original |
| Engram tables | not in this repo: use the base model's shards 47 and 48 |
| Size | 214 GB (199 GiB), 47 safetensors shards |
| Target | 2× DGX Spark (GB10, 128 GB each), tensor parallel 2, 262k context, vision on |

## Download

```bash
hf download Krishando/DeepSeek-V4.1-Flash-EXL3-2.86bpw \
  --local-dir ./DeepSeek-V4.1-Flash-EXL3-2.86bpw

# Engram tables (FP8, unchanged, ~95 GiB each) from the base model
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --include "model-00047-of-00048.safetensors" \
  --include "model-00048-of-00048.safetensors" \
  --local-dir ./DeepSeek-V4.1-Flash-engram
```

## Run

Runtime: [Kristianaaron/dsv41-tensorfold](https://github.com/Kristianaaron/dsv41-tensorfold), a TensorFold build with native FP8
backbone and mixed-width EXL3 expert support. Full steps are in its `recipes/dsv41-2spark/README.md`; in short:

```bash
git clone https://github.com/Kristianaaron/dsv41-tensorfold && cd dsv41-tensorfold
docker build -f recipes/dsv41-2spark/Dockerfile -t tensorfold-dsv41:2.86 .
docker save tensorfold-dsv41:2.86 | ssh spark-2 docker load
PACK=~/models/dsv41-2.86 ENGRAM=~/models/dsv41-engram SRC=$PWD WORKER=spark-2 HEAD_IP=10.0.0.1 \
  recipes/dsv41-2spark/launch.sh start
```

Tested setup: 2× DGX Spark, tensor parallel 2 over the QSFP link, 262k context, vision on, native DSpark drafter
(5 drafts) plus suffix drafts, 4 parallel requests. Start with at least 110 GiB free on each Spark.

## Credits and provenance

The quant (choice of calibration data, Hessian re-tracing, per-expert bit allocation, encode and the evaluation harness) is
original work by Kristian Aaron ([Krishando](https://huggingface.co/Krishando)). It stands on the work below: everything it was
built from, made with or measured with.

### Model, format and code

| What | Who | Used for | Licence |
|---|---|---|---|
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), commit `df42c10` | DeepSeek ([@deepseek_ai](https://x.com/deepseek_ai)) | The weights: FP8 backbone, routed experts (the part quantized here), DSpark drafter, vision tower, Engram tables. Its reference inference code was used to check our full-model reference | MIT |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | turboderp | The EXL3 format, the encoder that made this quant, and the proxy-error measure behind the bit allocation | MIT |
| [QTIP](https://arxiv.org/abs/2406.11235) | Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa | The trellis quantization method EXL3 builds on | Paper |
| [DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark) and [sfxnz/DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3) | Sufyan Shah ([sfxnz](https://huggingface.co/sfxnz)) | An earlier 2-Spark EXL3 pack and recipe for this model. This quant's encoder started from its `quantize_experts_exl3.py` and `pack_meta.py` (stream the official shards, EXL3 for the routed experts, every other tensor kept as stored, Engram tables read from disk); the Hessian-traced calibration and per-expert widths were added on top | MIT |
| [TensorFold](https://github.com/ashhart/TensorFold) | ashhart ([@ashxhart](https://x.com/ashxhart)) and the [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors) | The inference engine. Every speed on this card is TensorFold's | Apache-2.0 (MIT before 0.6.0) |
| [`deepseek_v41` family for TensorFold](https://github.com/bertholomus/TensorFold) at `808eb4a` | BertholomusAI, Albert Lee ([@bertholomusai](https://x.com/bertholomusai)) | DeepSeek-V4.1 tensor-parallel over two GB10s; the runtime builds on it | Apache-2.0 |
| [Transformers](https://github.com/huggingface/transformers) | Hugging Face | The DeepSeek-V4.1 model code behind the layer-by-layer full-model reference and the Hessian capture | Apache-2.0 |
| [vLLM](https://github.com/vllm-project/vllm) | The vLLM project | Serving for the KLD measurements | Apache-2.0 |
| vLLM EXL3 overlay from the DeepSeek-V4.1 serving kit | MiaAI-Lab | Running EXL3 packs inside vLLM for the KLD measurements | AGPL-3.0 |
| PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3` | NVIDIA | The run-time image (not included) | NVIDIA's container licence |

### Calibration data (1.03M tokens)

55% is the author's own agent sessions (private; rendered in the V4.1 chat format with thinking and tool calls). The rest:

| Dataset | Who | Domain | Licence |
|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT) | Hugging Face | Prose | ODC-By 1.0 |
| [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) (train) | Salesforce (Merity et al.) | Prose | CC BY-SA 3.0 |
| [codeparrot-clean-valid](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid) | CodeParrot | Code | See its dataset card |
| [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | ise-uiuc (Magicoder authors) | Code | MIT |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | Paster et al. | Maths | ODC-By 1.0 |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | NVIDIA | Maths | CC BY 4.0 |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (train) | OpenAI | Maths | MIT |
| [Hermes Function Calling v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Nous Research | Tool calls | Apache-2.0 |
| [PG-19](https://huggingface.co/datasets/emozilla/pg19) | Google DeepMind (Rae et al.) | Long documents | Apache-2.0 |
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) 20231101 (ar, de, es, fr, hi, ja, ru, zh) | Wikimedia | Multilingual | CC BY-SA 3.0 |

### Evaluation

| Dataset or tool | Who | Used for | Licence |
|---|---|---|---|
| [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) (test) | Salesforce (Merity et al.) | General holdout: prose and long documents | CC BY-SA 3.0 |
| [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) | OpenAI | General holdout: code | MIT |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (test) | OpenAI | General holdout: maths; the GSM8K score | MIT |
| [Belebele](https://huggingface.co/datasets/facebook/belebele) (8 languages) | Meta AI | General holdout: multilingual | CC BY-SA 4.0 |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | Hendrycks et al. | The MMLU score | MIT |
| [three.js](https://github.com/mrdoob/three.js) | The three.js authors | The long-generation task (a 3D game) | MIT |
| [Playwright](https://github.com/microsoft/playwright) with Chromium | Microsoft; the Chromium project | Headless play test of the generated games | Apache-2.0; BSD-3-Clause |

### Reference builds measured during development

Measured on the same harness while choosing this build:
[Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) and
[0xSero/DeepSeek-V4.1-Flash-Spark](https://huggingface.co/0xSero/DeepSeek-V4.1-Flash-Spark) (served with its launcher,
[0xSero/DeepSeek-V4.1-Flash-Two-Sparks](https://github.com/0xSero/DeepSeek-V4.1-Flash-Two-Sparks) at `ee68818`). Their weights
are not part of this quant.

### Names

"DeepSeek" and "DeepSeek-V4.1-Flash" belong to DeepSeek; "DGX Spark" and "GB10" belong to NVIDIA. This project is not
affiliated with or endorsed by DeepSeek, NVIDIA, or any of the people and projects above.

The runtime's git history keeps every upstream commit, and its `NOTICE` lists each file it adds or changes.

This is my first quant. Questions, issues and your own eval results are welcome in the Community tab.

## Licence

MIT, the same as the base model.
