---
language:
- en
- de
- fr
- es
- ja
- ko
- zh
- ar
library_name: mlx
license: apache-2.0
pipeline_tag: text-generation
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
tags:
- mlx
- jang
- jang-8m
- quantized
- apple-silicon
- lfm2
- lfm2_moe
- moe
- reasoning
- thinking
- tool-use
- agent
- gptq
---

<p align="center">
  <a href="https://osaurus.ai"><img src="./osaurus-x-banner.png" alt="Osaurus AI"></a>
</p>

> **Runtime:** Osaurus **0.25.21 or newer** (`osaurus.json`: `required_osaurus_version`).

# OsaurusAI/LFM2.5-24B-A2B-JANG_8M

A JANG bundle of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B): Liquid AI's 24B mixture-of-experts reasoning model with 2B active parameters (40 layers: 30 short-convolution and 10 full-attention, 64 routed experts with 4 active, two dense layers), built for on-device agents. It always thinks before answering and calls tools in Liquid's Pythonic format. 44.7 GiB of bf16 weights in **23.7 GiB**.

## What is quantized how
| part | format |
|---|---|
| routed experts (64 x 38 layers) | affine 8-bit, group 64, GPTQ per expert on that expert's own routed-input statistics |
| dense MLP (layers 0-1), attention q/k/v/out, short-conv in/out projections | affine 8-bit, GPTQ |
| embeddings (tied lm_head) | affine 8-bit |
| routers, expert bias, depthwise conv kernels, norms | source values |

Every quantized matrix with an input Hessian was rounded with GPTQ. The Hessians come from the model's **own** outputs: the bf16 model wrote its thinking and answers for math, finance, code in eleven languages, terminal and coding-agent sessions (with real tool results), multi-step tool calls, networking, cybersecurity, science, office tasks, long documents and multilingual chat at its native sampling settings, and those sequences were the calibration data. Per-expert Hessians use only the tokens routed to that expert. Affine quantization is MLX's own grid (bf16 scales and biases).

## Fidelity vs the bf16 model
232,094 teacher-forced positions on held-out sequences the bf16 model generated itself (never seen in calibration), scored against the bf16 model: top-128 KL, top-1 agreement, and the change in log-likelihood of the real next token (dNLL).

This model routes chaotically: changing the bf16 embeddings by a single bf16 ULP already moves top-1 on 1.8% of positions. That run is the floor row below; read the quantized rows against it.

| | size | KL median | KL mean | top-1 | dNLL |
|---|---|---|---|---|---|
| bf16 model, embeddings perturbed by 1 ULP (the floor) | 44.7 GiB | 1.1e-04 | 0.0036 | 98.24% | -0.0000 |
| **JANG_8M** | **23.7 GiB** | **1.2e-04** | **0.0039** | **98.14%** | **+0.0001** |

By prompt family (KL median, top-1):

| family | KL median | top-1 |
|---|---|---|
| multi-step tool agent | 0.0003 | 97.4% |
| tool use (single turn) | 0.0005 | 98.0% |
| code: cpp | 0.0001 | 98.1% |
| code: go | 0.0000 | 98.5% |
| code: javascript | 0.0000 | 98.3% |
| code: python | 0.0000 | 98.5% |
| code: rust | 0.0003 | 98.3% |
| code: shell | 0.0003 | 97.8% |
| code: swift | 0.0003 | 97.7% |
| code: typescript | 0.0001 | 98.2% |
| creative writing | 0.0009 | 97.6% |
| cybersecurity | 0.0011 | 97.3% |
| finance | 0.0001 | 98.4% |
| general / multilingual / structured output | 0.0003 | 97.8% |
| coding agent (native tools) | 0.0001 | 98.4% |
| coding agent, next turn | 0.0002 | 98.6% |
| long documents | 0.0005 | 97.1% |
| math | 0.0000 | 98.8% |
| networking | 0.0001 | 98.2% |
| office tasks | 0.0001 | 98.6% |
| PyTorch code | 0.0001 | 98.2% |
| science | 0.0000 | 98.6% |
| SWE-agent | 0.0018 | 96.5% |
| terminal agent (shell) | 0.0001 | 98.0% |
| turn after a tool result | 0.0004 | 97.5% |

Decisions: the quantized model picks the same token as bf16 at **99 / 99** `</think>` closes (floor run: 99 / 99) and at **24 / 24** tool-call starts (floor run: 24 / 24).

## Model notes
- Text only. 131,072-token context.
- Reasoning: always on. The chat template opens `<think>` for every answer; there is no off switch and no effort levels. `preserve_thinking` keeps earlier turns' thinking in the context.
- Tools: tool definitions go into the system prompt; calls come back as `<|tool_call_start|>[name(arg='value')]<|tool_call_end|>` (a Python-literal list, parsed as data). Tool results use the `tool` role.
- Sampling (`generation_config.json`, mirrored in `jang_config.json`): temperature 0.1, top-k 50, repetition penalty 1.05, EOS `<|im_end|>`.
- Router: sigmoid scores with the expert bias used only to pick the top-4; the expert weights are the unbiased scores. The bundle's `jang_config.json` documents this; a softmax router produces wrong output on this model.
- No draft model exists for the 24B, so there is no speculative decoder in this bundle.

## Source
- `LiquidAI/LFM2.5-24B-A2B` revision `1769516ac588`, Apache-2.0.
- Quantization and bundle by Jinho Jang (eric@osaurus.ai), 2026-10-13.
