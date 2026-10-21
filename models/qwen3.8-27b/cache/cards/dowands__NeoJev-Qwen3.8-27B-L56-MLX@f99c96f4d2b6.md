---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: text-classification
tags:
- mlx
- neojev
- jev
- system-one
- typed-decisions
- classification
- calibration
- apple-silicon
---


# NeoJev-Qwen3.8-27B-L56-MLX

**NeoJev** gives you typed, probabilistic decisions from Qwen3.8-27B in **one forward pass**, on Apple Silicon (MLX) or NVIDIA GPUs (PyTorch). You pass any text (the *state*) and a set of questions; you get back one of the options you supplied, a probability for every option, and a confidence. Nothing is generated and nothing is parsed, so an answer outside your schema is impossible by construction.

NeoJev is an open, local take on the "System One model" idea popularized by TypeSafe AI's Jev (typed decisions instead of text). It is an independent project and is **not** affiliated with or endorsed by TypeSafe AI.

![NeoJev on JevBench public items: accuracy, median latency and memory against open Jev-style models](assets/neojev_benchmark.png)

Accuracy is on JevBench's 231 public items through its own `TypeSafeAdapter`, not the official score; peer latencies come from JevBench's runs on other hardware, mostly over the internet, so compare them by order of magnitude. Details in [Compared with other open Jev-style models](#compared-with-other-open-jev-style-models-jevbench).

| backend | code | weights | measured on | mean accuracy | median / p99 latency |
|---|---|---|---|---|---|
| Apple Silicon (MLX) | `decider.py` | [dowands/NeoJev-Qwen3.8-27B-L56-MLX](https://huggingface.co/dowands/NeoJev-Qwen3.8-27B-L56-MLX) (4-bit, 56 layers) | M5 Pro 64 GB | 0.727 | 310 / 571 ms |
| NVIDIA GPU (PyTorch) | `decider_cuda.py` | [dowands/NeoJev-Qwen3.8-27B-L56](https://huggingface.co/dowands/NeoJev-Qwen3.8-27B-L56) (FP8, 56 layers) | L20 48 GB | 0.724 | 107 / 133 ms |

Both backends share the same readout head (`decider_head.npz`); it transferred from MLX 4-bit to CUDA FP8 without re-training.

## Quick start (Apple Silicon)

This repository is the **MLX build** (4-bit, 56 layers, 13.4 GB). For NVIDIA GPUs use [dowands/NeoJev-Qwen3.8-27B-L56](https://huggingface.co/dowands/NeoJev-Qwen3.8-27B-L56).

```bash
pip install mlx==0.32.2 mlx-lm==0.31.3 numpy huggingface_hub
huggingface-cli download dowands/NeoJev-Qwen3.8-27B-L56-MLX decider.py --local-dir .
```

```python
from decider import Decider
d = Decider("dowands/NeoJev-Qwen3.8-27B-L56-MLX")          # downloads the weights and the readout head on first use
d.ask(
    "Customer emailed twice about a failed refund for order #4417 and says they will dispute the charge tomorrow.",
    {
        "queue":   {"type": "choice", "options": ["Billing and Payments", "Technical Support", "Sales"],
                    "prompt": "Which team should handle this ticket?"},
        "urgency": {"type": "score", "levels": ["low", "medium", "high"], "prompt": "How urgent is it?"},
        "churn":   {"type": "noul", "statement": "The customer is likely to leave."},
    },
)
# {"queue": {"value": "Billing and Payments", "probs": {...}, "confidence": 0.87}, "urgency": {..., "expected_index": 1.37},
#  "churn": {"value": 0.41, ...}, "_tokens": 43}
```

### Serving (TypeSafe-compatible API)

`server.py` exposes `POST /v1/systemone` in the TypeSafe request/response shape, so clients written for that API (and JevBench's `TypeSafeAdapter`) work unchanged:

```bash
python server.py --backend mlx --model dowands/NeoJev-Qwen3.8-27B-L56-MLX --port 8000
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{"state": "My card was charged twice.",
  "questions": {"route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "technical": "bugs"}}}}'
```

It serves one request at a time. `--layout state_first` (the default, used for the JevBench numbers) caches nothing; `--layout schema_first` caches the question prefix per schema (LRU, 64 schemas), which is faster when the same questions repeat.

Question types:

| type | fields | returns |
|---|---|---|
| `choice` | `options` (2 to 16), `prompt` | the chosen option, per-option probabilities |
| `score` | `levels` (ordered), `prompt` | the chosen level, per-level probabilities, `expected_index` |
| `noul` | `statement` | probability that the statement is true |

`cache_schema=True` (default) puts the questions first and caches them across calls with the same question set; use it when you ask the same questions about many texts. Use `cache_schema=False` for one-off questions whose options change every call (for example multiple-choice QA).

## How it works

1. **Layout.** The questions are prefilled once per schema and cached. Each call only runs the state plus one short trigger per question (`Q1:`, `Q2:`, ...), all in a single forward pass. The answer for each question is read at its trigger position.
2. **Early exit.** Only the first 56 of the 64 decoder layers are kept (13.4 GB of weights instead of about 16 GB), which cuts compute by 12.5%.
3. **Label-free readout head.** Reading the option logits directly from layer 56 is poor (the plain "logit lens" gets 0.663 mean accuracy on our evaluation set). A small head, `h + (h U) V + b + c[position]` (rank 1024, followed by the model's own final norm and LM head), maps the layer-56 state to the option distribution. It was trained by **self-distillation**: the target is the full 64-layer model's option distribution in the original prompt layout, on 4,000 texts x randomly generated questions (10,035 questions). **No labels are used**, so the head is task-agnostic and you do not need any data to use it.
4. **Calibration.** One global temperature (T = 1.373), fitted once on public labels from the distillation sources.
5. **Token budget.** Each call is capped at 160 tokens by default; longer states keep their first 40% and last 60% of tokens.

Distillation sources (texts only, none overlap the evaluation tasks below): SetFit bbc-news, 20_newsgroups, enron_spam, subj, sst5, toxic_conversations, amazon_counterfactual_en, student-question-categories, CR, rte, mrpc, qnli; cais/mmlu; allenai/qasc; ehovy/race; openlifescienceai/medmcqa.

## Evaluation

These 11 tasks are held out from training: none of their texts, questions or labels were used. There are 500 test items per task and 5,500 calls in total. The baseline is the same base model with all 64 layers, reading the option logits zero-shot in the natural layout (state first, one question at a time).

| task / question | baseline acc / ECE | this model acc / ECE |
|---|---|---|
| customer-support-tickets / queue (10-way) | 0.264 / 0.480 | 0.322 / 0.060 |
| customer-support-tickets / priority | 0.420 / 0.341 | 0.380 / 0.251 |
| customer-support-tickets / type | 0.584 / 0.207 | 0.454 / 0.138 |
| yelp_review_full / stars (1-5) | 0.592 / 0.240 | 0.534 / 0.079 |
| yelp_review_full / 4-5 stars (yes/no) | 0.916 / 0.020 | 0.850 / 0.189 |
| ag_news / topic | 0.870 / 0.096 | 0.874 / 0.071 |
| boolq | 0.854 / 0.069 | 0.804 / 0.098 |
| dair-ai/emotion | 0.596 / 0.250 | 0.588 / 0.073 |
| glue/mnli | 0.870 / 0.059 | 0.838 / 0.113 |
| commonsense_qa | 0.840 / 0.068 | 0.804 / 0.050 |
| openbookqa | 0.924 / 0.031 | 0.908 / 0.089 |
| ai2_arc (challenge) | 0.964 / 0.017 | 0.962 / 0.111 |
| sciq | 0.990 / 0.008 | 0.992 / 0.081 |
| tweet_eval / irony | 0.874 / 0.138 | 0.866 / 0.189 |
| **mean** | **0.754 / 0.145** | **0.727 / 0.114** |

The mean accuracy is 2.7 points lower than the baseline (paired bootstrap 95% CI: -3.6 to -1.9).

**NVIDIA backend** (same 11 tasks, 500 items each, `Qwen/Qwen3.8-27B-FP8` with the first 56 layers, NVIDIA L20 48 GB, torch 2.11 + cu128, transformers 5.16.1, flash-linear-attention, causal-conv1d): mean accuracy **0.724**, mean ECE 0.103, latency median 107 ms / p90 132 ms / p99 133 ms, 99.96% of calls under 500 ms, peak GPU memory 25.8 GB. Two of the 5,500 calls took 9 to 14 s, most likely Triton autotuning on a new input shape. With a 1,000-token budget (no truncation) the accuracy was 0.724 and p99 208 ms, so the 160-token default costs almost nothing here.

**Latency** was measured on an Apple M5 Pro with 64 GB, MLX 0.32.2 and mlx-lm 0.31.3, one call at a time, with a remote-desktop session running during the measurement:

| | median | p90 | p99 | under 500 ms |
|---|---|---|---|---|
| baseline (64 layers, natural layout) | 560 ms | | | |
| this model | 310 ms | 475 ms | 571 ms | 93.2% |

Latency is roughly linear in tokens per call. Short texts (tweets, one-line tickets) take 180 to 300 ms; the 160-token cap is what keeps long texts near 500 ms. Lower `token_budget` if you need a hard ceiling. Background GPU load (screen sharing, other ML jobs) moves these numbers noticeably, because the display and the model share the GPU.

## How it compares

All methods were run on the same machine and the same held-out items (the first 200 test items of each of the 11 tasks, 14 questions, 2,200 calls). Accuracy is averaged over the 14 questions.

| method | mean accuracy | mean ECE | median latency per call | format errors |
|---|---|---|---|---|
| Qwen3.8-27B, all 64 layers, option logits (upper reference) | 0.745 | 0.160 | 563 ms | 0% |
| **this model** | **0.721** | **0.117** | **310 ms** | 0% (by construction) |
| Qwen3-Embedding-8B, similarity to each option | 0.599 | 0.165 | 3,472 ms* | 0% |
| Qwen3.5-0.8B, option logits | 0.554 | 0.181 | 28 ms | 0% |
| facebook/bart-large-mnli, HF zero-shot pipeline | 0.488 | 0.153 | 292 ms | 0% |
| always the most common answer | 0.357 | | | |

A second comparison adds the way most people use an LLM for this, which is to ask the full model to write a JSON answer. That is slower, so it ran on the first 100 items per task:

| method (100 items/task) | mean accuracy | median latency | format errors |
|---|---|---|---|
| Qwen3.8-27B, 64 layers, writes JSON (greedy, thinking off) | 0.739 | 1,757 ms | 0% |
| Qwen3.8-27B, 64 layers, option logits | 0.736 | 562 ms | 0% |
| **this model** | **0.711** | **309 ms** | 0% |

Reading the tables:

- This model is 2.4 to 2.8 points below the full model (depending on the item subset; 2.7 on all 500 items per task), and 5.7 times faster than asking the full model for JSON.
- It is 12 to 23 points above the other models that fit on a laptop and answer arbitrary questions zero-shot.
- The customer-support-tickets labels are synthetic and noisy: every method sits near the majority baseline there.
- bart-large-mnli was used through the generic zero-shot pipeline (options as labels in a template), not its native NLI format, so its MNLI score (0.33) does not reflect its NLI ability.

\* The embedding baseline was served by a local llama.cpp server (Q4_K_M), one request per call; its latency is not optimized.

On standard public benchmarks within the held-out set (500-item zero-shot subsets, this model): ARC-Challenge 0.962, OpenBookQA 0.908, CommonsenseQA 0.804, SciQ 0.992, BoolQ 0.804, MNLI 0.838, AG News 0.874. These are not comparable to leaderboard numbers that use few-shot prompts, chain of thought or the full test sets.

## Compared with other open Jev-style models (JevBench)

[JevBench](https://github.com/fstandhartinger/jevbench) (MIT, Benchmark Heaven) is the independent benchmark most Jev-class projects report on. We ran NeoJev on its **231 public items** the way JevBench runs TypeSafe-compatible systems: JevBench's own, unmodified `TypeSafeAdapter` sent every item to NeoJev's `server.py`, and JevBench's scoring code graded the answers (`jevbench_via_server.py` in this repo). The table below compares NeoJev with widely used open peers that JevBench measured. Peer numbers come from JevBench's published v1.4 results (repo commit `26eb72d`, 24 Sep 2026).

| system | base | weights (memory) | public accuracy (231) | p50 / p95 latency | where the latency was measured |
|---|---|---|---|---|---|
| **NeoJev (this model), CUDA** | Qwen3.8-27B, 56 layers | 25.8 GB peak GPU (FP8) | **0.861** | **0.132 / 1.33 s** | NVIDIA L20, local HTTP, serial |
| **NeoJev (this model), MLX** | Qwen3.8-27B, 56 layers | 13.4 GB (4-bit) | **0.853** | 0.52 / 9.7 s | Apple M5 Pro, local HTTP, serial |
| Jev 1.13.0 (TypeSafe AI, closed) | closed | n/a | 0.866 | 0.65 / 0.72 s | production API over the network |
| decider-2b (Mapika) | Qwen3.5-2B | 3.8 GB | 0.710 | 0.26 / 0.28 s | JevBench server, network |
| openJev Verdict 1.4 | ModernBERT-base 151M | about 0.6 GB (est.) | 0.576 | 0.31 / 0.92 s | CPU, 4 threads |
| Winnow-12B Q8 | Gemma 4 12B | about 13 GB (est., Q8) | 0.857 | 0.23 / 0.41 s | JevBench server, network |
| Bespoke Nimble 9B | Qwen3.5-9B + LoRA | about 18 GB (est., BF16) | 0.797 | 0.39 / 0.65 s | JevBench server, network |
| JevK5 v0.2 | Qwen3.5-4B | 8.4 GB | 0.853 | not published |  |
| JevOne (Juspay) | Qwen3.6-35B-A3B | about 70 GB (est., BF16) | 0.896 | 0.087 / 0.145 s | RTX PRO 6000 96 GB, local |
| kev 8B | Qwen3-8B + LoRA | about 16 GB (est., BF16) | 0.714 | 0.59 / 1.15 s | JevBench server, network |
| SemIf (formerly OpenJev) | Qwen3.5-4B frozen | about 8 GB (est., BF16) | 0.810 | 0.20 / 0.32 s | RTX PRO 4500 32 GB |
| SimpleJev Qwen3.8-27B | Qwen3.8-27B frozen | about 54 GB (est., BF16) | 0.866 | 1.01 / 1.88 s | JevBench server, network |
| OpenJev (razorback16, NVFP4) | DiffusionGemma 26B-A4B | about 15 GB (est., NVFP4) | 0.818 | 0.24 / 0.31 s | RTX PRO 4500 32 GB |
| openjev-sglang | Qwen3.6-35B-A3B | about 70 GB (est., BF16) | 0.853 | 0.68 / 0.73 s | JevBench server, network |

How to read it:

- **Accuracy.** On the public items NeoJev (0.861 CUDA, 0.853 MLX) is close to Jev 1.13.0 (0.866) and to the other Qwen3.8-27B systems (0.862 to 0.870). It is above the small models in the table (0.58 to 0.85; decider-2b 0.71, JevK5 0.85). Among open systems, only JevOne and NInfer Qwen3.8-Flash-Next (both 0.896, on 35B-class MoE models) and thinking-mode systems score higher. Frontier APIs (GPT, DeepSeek, Gemini) score higher as well.
- **Speed and memory.** Among the 27B-class systems, NeoJev's p50 is 0.13 s on an L20. SimpleJev and reflex-27b show 1.0 s and 1.9 s in JevBench's runs (with a network hop, on other hardware, so treat this as an order of magnitude only). NeoJev also needs about half the memory of a BF16 27B, because it runs 56 of 64 layers of the FP8 checkpoint. Its p95 is dominated by the long documents of the hard tier; the easy and standard tiers take 0.11 s.
- **Option order matters.** JevBench sends each choice question's options as a label-to-description map sorted by label, and NeoJev keeps the order it receives. When the options were instead put in each item's label order, NeoJev scored 0.866 (CUDA) and 0.870 (MLX); of the 119 choice items whose two orders differ, 8 flipped on MLX and 5 on CUDA. The table uses the JevBench path. Small or frozen decision models are generally order-sensitive, and JevBench documents this for other entrants too.
- **Latency is not directly comparable across rows.** Many JevBench rows include a network round trip from JevBench's server, and every row ran on different hardware.
- **This is not the official JevBench score.** The official score also needs the held-out and sealed items (which only JevBench runs; v1.4 adds 308 fresh sealed decisions), a judge tier, calibration and a cost figure.
- **Where the memory figures come from.** They are the published repository size where one exists (decider-2b, JevK5); otherwise they are an estimate of parameters × precision, marked "est.".

## Limitations

- **2.7 points below the full model on average**, and worse on some questions: up to -13 points on the third question of a three-question schema (`customer-support-tickets / type`). Questions placed third or later in one call read less reliably, and answers change with question order (the same question can get a different confidence in position 1 and position 2). Keep the order fixed for a given use case, put the most important question first, or split long schemas across calls.
- **Yes/no statements are weaker than multiple choice** in this layout (-5 to -7 points on boolq and yelp 4-5 stars).
- **Calibration is not uniform.** The mean ECE is 0.114, but some questions sit at 0.19 to 0.25. If you have a few hundred labelled examples for your task, fit your own temperature on them.
- Choice questions support at most 16 options. The model was evaluated on English only.
- Long inputs are truncated to the token budget, so detail in the middle of a long text is lost.
- Needs Apple Silicon (MLX, about 15 GB of free unified memory) or an FP8-capable NVIDIA GPU with about 26 GB of memory. On the NVIDIA side the FP8 kernel path was tested on an L20 (Ada) only.

## License

Apache-2.0, derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) via `mlx-community/Qwen3.8-27B-4bit`. See `LICENSE_NOTICE.md`.
