---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
language:
- fr
- en
tags:
- nvfp4
- fp8
- modelopt
- vllm
- tool-calling
- function-calling
- agent
- qwen3.8
- devana
- suite-366
---

# RaLLM v2.4 NVFP4

RaLLM is [Devana](https://www.devana.ai)'s fine-tune of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for the [Suite 366](https://www.suite366.ai) assistant: an agent that works through tools over CRM, calendar, email, files, notes and tasks, and that knows the Devana ecosystem.

This repository holds **RaLLM v2.4 with the adapter merged and the weights quantized to NVFP4** (21 GB), ready to serve with vLLM. It is built with the recipe NVIDIA used for [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), the quantized base model, and every figure below compares the two. The adapter alone, with its full-precision results, is in [Devanaai/RaLLM-v2.4-LoRA](https://huggingface.co/Devanaai/RaLLM-v2.4-LoRA).

## At a glance

Same tests, same decoding options, same server for both models.

| | Qwen3.8-27B NVFP4 (NVIDIA) | RaLLM v2.4 NVFP4 |
|---|---|---|
| Right decision between calling a tool and answering (592 turns) | 90.5 % | **97.8 %** |
| Expected tool calls that were made (394 calls) | 70 % | **83 %** |
| Unneeded tool calls (282 turns) | 47 | **8** |
| Writes to CRM or calendar despite a restricting system prompt (279 cases) | 85 | **6** |
| Questions about the Devana ecosystem answered correctly (12) | 4 | **11** |
| MMLU-Pro, English | 59.4 % | **61.8 %** |
| BFCL v3, single turn (function calling on public tools) | 85.6 % | 85.8 % |

## Tool use in Suite 366

![Next-step evaluation, NVIDIA's quantized base model against RaLLM v2.4 NVFP4](assets/eval_overview.png)

Devana's next-step evaluation takes held-out Suite 366 conversations and asks the model for the next assistant turn: call a tool, and which one, or answer. 592 turns, 310 where a call is expected, 47 of them at turn 6 or later in a tool chain. Greedy decoding, thinking off, vLLM 0.30 with MTP speculative decoding for both models. The conversations are synthetic and written in the product's format.

| Metric | Qwen3.8-27B NVFP4 (NVIDIA) | RaLLM v2.4 NVFP4 |
|---|---|---|
| Right decision between calling a tool and answering | 0.905 | **0.978** |
| Exact set of expected tools | 0.628 | **0.797** |
| Tool overlap with the reference | 0.724 | **0.855** |
| Expected calls that were made | 277 / 394 | **327 / 394** |
| Calls made that were expected | 61 % | **82 %** |
| Calls a tool when the reference answers in text (282 turns) | 47 | **8** |
| Stops with text while tool work remains (310 turns) | 9 | **5** |
| … at turn 6 or later in a tool chain (47 turns) | 2 | 2 |
| Long tool chains: right decision (157 turns) | 153 | 154 |
| Arguments valid against the tool schema | 0.987 | **1.000** |
| Malformed tool calls | 0 | 0 |

Paired on the same turns, the base model takes the wrong decision alone on 48 turns and RaLLM v2.4 alone on 5 (McNemar p < 0.001).

![Expected tool calls that were made, by tool](assets/tools_recall.png)

Every call made during a 200-turn run of this evaluation and during the probe below was also checked against the Suite 366 tool schemas taken from the product code (types, required and unknown arguments): 212 calls out of 212 conform.

### Inside the product

Suite 366 loads its tools on demand: the model first loads a tool category, then calls the tools it has just received. Requests were recorded from a Suite 366 instance, with the full product prompt and 27 to 128 tools, and replayed on the model at the product's temperature (0.7).

| Measure | RaLLM v2.4 NVFP4 |
|---|---|
| Goes on to call a tool after loading a category (34 requests, 136 samples) | 136 / 136 |
| Conversation titles generated correctly | 124 / 124 |

In a live session of 32 scenarios played in the Suite 366 interface (files, CRM, calendar, tasks, notes, documentation questions, a workflow, requests under “do not change anything”), 26 were handled fully and 6 partly; none failed, and every answer came in under two minutes.

## Beyond the product's own tools

A second test uses material the model never saw: invented tools in other domains, general questions, and system prompts that restrict writes.

| Probe | Qwen3.8-27B NVFP4 (NVIDIA) | RaLLM v2.4 NVFP4 |
|---|---|---|
| Unseen tools, chains of 9 to 11 calls: stops too early | 0 / 48 | 0 / 48 |
| Unseen tools: picks the expected next tool | 39 / 48 | **46 / 48** |
| Unseen tools: extra call once the chain is finished | 0 / 5 | 0 / 5 |
| General-purpose questions passed (code, maths, writing) | 11 / 14 | 11 / 14 |
| Questions about the Devana ecosystem passed | 4 / 12 | **11 / 12** |
| Updates the CRM or the calendar on its own when the conversation calls for it | 20 / 31 | **31 / 31** |
| Writes to CRM or calendar despite a restricting instruction | 85 / 279 | **6 / 279** |

![Writes to CRM or calendar despite a restricting system prompt](assets/restricted_writes.png)

The restriction test takes 31 situations where the assistant would update the CRM or the calendar on its own, and adds one restricting sentence to the system prompt (“read-only access”, “never modify without an explicit request”, “ask for confirmation first”). Nine sentences are tried, six of them in wordings the model never saw. RaLLM v2.4 never writes under a read-only or a confirmation rule.

## Public benchmarks

To check that the fine-tune kept the general abilities of the base model, both models ran the same sampled questions from five public benchmarks (thinking off, greedy decoding). Outputs cut by the token limit are excluded on both sides.

![Public benchmarks, NVIDIA's quantized base model against RaLLM v2.4 NVFP4](assets/public_benchmarks.png)

| Benchmark | Questions | Qwen3.8-27B NVFP4 (NVIDIA) | RaLLM v2.4 NVFP4 | Difference | p (McNemar) |
|---|---|---|---|---|---|
| MMLU-Pro, English | 1,400 | 59.4 % | **61.8 %** | +2.4 | 0.01 |
| MMLU-ProX, French | 1,369 | 58.3 % | 58.9 % | +0.6 | 0.55 |
| GSM8K | 291 | 96.6 % | 96.9 % | +0.3 | 1.00 |
| HumanEval | 162 | 95.7 % | 95.1 % | −0.6 | 1.00 |
| BFCL v3, single turn | 748 | 85.6 % | 85.8 % | +0.3 | 0.90 |
| … `simple` | 150 | 96.0 % | 96.0 % | 0.0 | 1.00 |
| … `multiple` | 150 | 95.3 % | 97.3 % | +2.0 | 0.25 |
| … `parallel` | 150 | 76.0 % | 78.0 % | +2.0 | 0.68 |
| … `parallel_multiple` | 150 | 82.0 % | 79.3 % | −2.7 | 0.48 |
| … `irrelevance` (no call when no tool fits) | 148 | 78.4 % | 78.4 % | 0.0 | 1.00 |

General knowledge, arithmetic, code and function calling on public tools are level with the base model, and MMLU-Pro is significantly higher. These figures are not comparable with published leaderboard scores (subsamples, thinking off, own scorer for BFCL); only the differences between columns are meaningful.

## Full precision and quantized

The same tests on four builds, all served by vLLM 0.30.

| | Qwen3.8-27B bf16 | Qwen3.8-27B NVFP4 (NVIDIA) | RaLLM v2.4 bf16 | RaLLM v2.4 NVFP4 |
|---|---|---|---|---|
| Right decision (592 turns) | 0.921 | 0.905 | 0.981 | 0.978 |
| Exact set of expected tools (592 turns) | 0.634 | 0.628 | 0.817 | 0.797 |
| Expected calls that were made (394) | 280 | 277 | 341 | 327 |
| Unneeded calls (282 turns) | 43 | 47 | 7 | 8 |
| Writes despite a restriction (279 cases) | 59 | 85 | 2 | 6 |
| MMLU-Pro, English | 62.9 % | 59.4 % | 63.9 % | 61.8 % |
| MMLU-ProX, French | 59.8 % | 58.3 % | 60.0 % | 58.9 % |
| BFCL v3, single turn | 85.1 % | 85.6 % | 86.6 % | 85.8 % |
| MTP draft tokens accepted, out of 3 | 2.52 | 2.52 | 2.61 | 2.58 |
| Time for a 200-turn evaluation | 111 min | 53 min | 74 min | 41 min |

The quantized build keeps the decisions of the full-precision one (paired on the 592 turns, p = 0.62) and runs the evaluation in nearly half the time. RaLLM v2.4 NVFP4 also completes the same evaluation faster than the quantized base model (41 minutes against 53).

## Usage

```bash
vllm serve Devanaai/RaLLM-v2.4-NVFP4 \
  --max-model-len 65536 --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Measured with vLLM 0.30.0, which detects the mixed NVFP4 and FP8 format from `config.json` (`quantization=modelopt_mixed`). The MTP head of the base model is included, not quantized, and used for speculative decoding. Tool calls come out in the base model's XML format, parsed by `qwen3_xml`.

**Recommendations.**

- Serve with thinking off, as above: the model was trained and evaluated that way.
- Keep a confirmation step on write tools. Restricting system prompts are respected in 273 of the 279 probe cases, not in all of them; the six remaining writes are all under “only on explicit request” rules.
- Text input. The vision tower of the base model is unchanged and was not part of the evaluation.
- French and English. The assistant's spontaneous CRM and calendar updates were trained in French.

## How it was built

| | |
|---|---|
| Source | [Devanaai/RaLLM-v2.4-LoRA](https://huggingface.co/Devanaai/RaLLM-v2.4-LoRA) merged into Qwen/Qwen3.8-27B (bf16), MTP head copied back from the base checkpoint |
| Training data | Synthetic conversations in the Suite 366 format, written for the purpose. No customer data |
| Tool | NVIDIA ModelOpt 0.48.0, `examples/hf_ptq/hf_ptq.py` |
| Recipe | `nvfp4_w4a4_mlp_fp8_attn_local_hessian`, the ModelOpt recipe for Qwen3.8-27B |
| Layers | 193 in NVFP4 (MLP projections and `lm_head`, weights and activations on 4 bits, groups of 16), 208 in FP8 (attention projections and the three large projections of each Gated DeltaNet layer). Embeddings, vision tower, MTP head and the small Gated DeltaNet layers (`conv1d`, `in_proj_a`, `in_proj_b`) not quantized |
| Calibration | 512 sequences of 2,048 tokens from `nemotron-post-training-v3`, local Hessian |
| Hardware | NVIDIA DGX Spark (GB10), for training, quantization and every measurement on this page |

## Files

| File | Content |
|---|---|
| `model-00001-of-00003.safetensors` | SHA-256 `2f0432133eb9292f49d9ba73f822311e9355e6f3bddac92df4300fe185f922dd` |
| `model-00002-of-00003.safetensors` | SHA-256 `d9d43dd374bc2e8ce9bb5db4c02ef1f733ef7b13a495c4d22afcbd0514bec48a` |
| `model-00003-of-00003.safetensors` | SHA-256 `f03f785eb3ab93040ff9db4605bfc19fdb4cf87494f7c8860d2e02ba1bf64d21` |
| `config.json`, `hf_quant_config.json`, `.quant_summary.txt` | Model configuration and quantization description written by ModelOpt |
| Tokenizer and chat template files | Those of the base model |
| `eval/` | Evaluation, probe, argument and public benchmark reports of the four builds (`eval/full/`: the 592-turn evaluation) |
| `assets/` | Charts used on this page |

## License

Apache 2.0, same as the base model.
