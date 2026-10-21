---
language:
- en
library_name: mlx
license: apache-2.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- jang
- quantized
- apple-silicon
- vision
- video
- reasoning
- thinking
- agent
- tool-use
- dflash
- gptq
- imatrix
- qwen3_5
- jangh
---

<p align="center">
  <a href="https://osaurus.ai"><img src="./osaurus-x-banner.png" alt="Osaurus AI"></a>
</p>

> **Runtime:** Osaurus **0.25.20 or newer** (`osaurus.json`: `required_osaurus_version` 0.25.20, `model_version` 1).

# OsaurusAI/Qwen3.8-27B-JANGH2

**Qwen3.8-27B in 14.25 GiB** with every attention-type layer, embeddings, `lm_head` and the vision tower kept at
8-bit, image and video input, and a bundled DFlash2 drafter.

| part | format |
|---|---|
| MLP gate / up / down | **JANGH 2-bit** (Hadamard-rotated codebook, fp16 row scales, GPTQ with act-order) |
| full attention (q / k / v / o) and gated-delta-net projections | affine **8-bit** |
| `lm_head`, token embeddings, vision tower | affine **8-bit** |
| norms, convolutions, gates | bf16 |

## Quality vs the bf16 model
| set | median KL | top-1 |
|---|---|---|
| mixed held-out text | 0.037 | 80.1 % |
| agentic and tool conversations | 0.061 | 75.3 % |
| reasoning | 0.008 | 85.4 % |
| own bf16 reasoning traces (low / medium / xhigh) | 0.008 / 0.002 / 0.026 | 81.2 / 88.1 / 81.9 % |

Think-block close decisions agree with bf16 on 40 / 40 held-out traces; tool-call vs answer decisions flip on 32 / 263
agentic positions. For near-reference quality use the JANG_4D bundle.

## Quality by topic
Median KL and top-1 agreement vs the bf16 model per topic of the held-out text set (both bundles, same rows):

| topic | tokens scored | JANG_4D KL | JANG_4D top-1 | JANGH2 KL | JANGH2 top-1 |
|---|---|---|---|---|---|
| coding (many languages) | 39,155 | 0.0002 | 98.4 % | 0.0092 | 89.5 % |
| math | 9,067 | 0.0001 | 98.7 % | 0.0068 | 90.8 % |
| office / finance | 13,159 | 0.0001 | 97.7 % | 0.0087 | 86.9 % |
| reasoning | 14,004 | 0.0001 | 94.5 % | 0.0081 | 81.0 % |
| science | 4,156 | 0.0010 | 98.0 % | 0.0460 | 87.1 % |
| long context + attention recall | 79,784 | 0.0007 | 96.6 % | 0.0446 | 80.4 % |
| cybersecurity | 15,683 | 0.0008 | 94.9 % | 0.0368 | 80.5 % |
| networking | 6,458 | 0.0011 | 93.7 % | 0.0244 | 81.9 % |
| terminal / shell | 42,987 | 0.0013 | 92.5 % | 0.0738 | 73.5 % |
| agentic + tool calls | 34,245 | 0.0026 | 89.8 % | 0.0975 | 71.2 % |
| creative writing | 10,029 | 0.0028 | 96.5 % | 0.1989 | 74.2 % |
| multi-turn agentic set | 199,746 | 0.0014 | 92.1 % | 0.0615 | 75.3 % |

Own bf16 reasoning traces, by topic (thinking + answer tokens):

| topic | JANG_4D KL | JANG_4D top-1 | JANGH2 KL | JANGH2 top-1 |
|---|---|---|---|---|
| coding (many languages) | 0.00007 | 98.4 % | 0.00508 | 89.4 % |
| math | 0.00001 | 98.6 % | 0.00061 | 91.6 % |
| office / finance | 0.00000 | 98.5 % | 0.00007 | 91.2 % |
| science | 0.00010 | 98.1 % | 0.00779 | 87.8 % |
| networking | 0.00000 | 98.7 % | 0.00009 | 92.8 % |
| cybersecurity | 0.00135 | 97.3 % | 0.08516 | 81.1 % |
| agentic + tool calls | 0.00093 | 89.7 % | 0.03968 | 72.6 % |

## Reasoning, tools, sampling
- Thinking is **on** by default (the template opens `<think>`); `enable_thinking: false` prefills a closed block.
- `reasoning_effort`: **low**, **medium**, **xhigh** (default); the template rejects anything else. Earlier thinking
  is kept in history by default (`preserve_thinking`).
- Reasoning parser `qwen3`; tool calls use the XML function format (parser `qwen3_coder`).
- Sampling presets from the model card: thinking T 1.0 / top_p 0.95 / top_k 20; instruct T 0.7 / top_p 0.8 / top_k 20 /
  presence 1.5.

## DFlash2 drafter
`dflash2/` holds the [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) block-diffusion
drafter (bf16, block size 8) for speculative decoding of text; image and video requests decode without it. The native
MTP head is not included.

## Calibration
One capture feeds the importance matrix, AWQ and GPTQ, merged by token count: 1.21M text tokens (cybersecurity, coding in
many languages, agentic and tool conversations, terminal, long reasoning, attention recall, math; all three efforts),
0.43M image and video tokens (191 media items through the vision tower), and 0.31M tokens of the model's own
full-precision reasoning traces at low / medium / xhigh. Held-out sets are disjoint from calibration.

## What's in the bundle
`model-*.safetensors` + one index (64-byte aligned), `config.json` (per-module quantization), `jang_config.json` (format,
capabilities, reasoning and tool contract, sampling, evaluation), `generation_config.json`, tokenizer, chat template,
image and video processor configs, `dflash2/`.

Quantized and validated by **Jinho Jang** — eric@osaurus.ai
