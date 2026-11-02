---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- ornith
- qwen35moe
- moe
---

# Ornith-1.5-35B-A3B, Q4_K experts / Q8_0 dense (GGUF)

A mixed quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
(revision `10fbf86f`): the routed experts in **Q4_K**, every other matrix in **Q8_0**, the output head in Q6_K.

| tensors | type |
|---|---|
| routed experts (`ffn_{gate,up,down}_exps`, 120 tensors) | Q4_K, under bartowski's importance matrix |
| DeltaNet and attention projections, shared experts, `attn_k`/`attn_v`, `ssm_alpha`/`ssm_beta`, `token_embd` | Q8_0 |
| `output` | Q6_K |
| norms, `ssm_a`, `ssm_dt`, `ssm_conv1d`, the router and the shared expert's gate | F32 |

20.67 GB. The experts are 92% of the weights but a decode token reads only 8 of each layer's 256, while every token
goes through every dense weight; so the bits go to the dense matrices, and the experts get Q4_K's per-sub-block
scales and minimums at the same 4.5 bits as Q4_0. The MTP/NextN block is left out.

**Chat template:** this file carries Ornith's own `chat_template.jinja`, which re-renders earlier turns' reasoning.
Ornith's repository also ships stock Qwen3.6's `tokenizer_config.json` (stock's template inside), which
llama.cpp's converter prefers, so a GGUF converted from the repository as is carries stock's template instead; the
official GGUFs carry an edited one. This one was converted with that field removed.

## Fidelity

KL divergence of the next-token distribution against the model's own Q8_0 (`llama-perplexity --kl-divergence`,
llama.cpp `ea0c4c2c`): wikitext-2 test at 4,096 context (20 chunks) and one long document at 32,768 (2 chunks).

| file | size | mean KL, 4k / 32k | 99th pct KL, 4k / 32k | same top token, 4k / 32k |
|---|---|---|---|---|
| **this file** | 20.7 GB | **0.0354 / 0.0511** | **0.373 / 0.565** | **92.7% / 92.0%** |
| bartowski Q4_K_M | 21.9 GB | 0.0460 / 0.0548 | 0.476 / 0.583 | 91.5% / 91.7% |
| bartowski Q4_K_S | 21.1 GB | 0.0511 / 0.0661 | 0.504 / 0.752 | 91.1% / 91.0% |
| bartowski IQ4_XS | 19.3 GB | 0.0582 / 0.0675 | 0.616 / 0.753 | 90.6% / 90.8% |
| bartowski Q4_0 | 20.4 GB | 0.0774 / 0.0796 | 0.771 / 0.885 | 88.9% / 89.7% |
| ornith-ai Q4_K_M (official) | 21.7 GB | 0.0596 / 0.0812 | 0.592 / 0.909 | 90.3% / 89.8% |

## How it was made

llama.cpp at `ea0c4c2c85fb183ec0fbd30a1af4e4275a18a064`: `convert_hf_to_gguf.py --outtype bf16 --no-mtp` (from a copy
of the checkpoint whose `tokenizer_config.json` has no `chat_template`), then
`llama-quantize --imatrix Ornith-1.5-35B-A3B-imatrix.gguf` (bartowski/Ornith-1.5-35B-A3B-GGUF @ `64b0493d34`) with
`--tensor-type ffn_(gate|up|down)_exps=q4_k --output-tensor-type q6_k --token-embedding-type q8_0` and Q8_0 as the
base type. sha256 `25a681297b42b5e5c0088d01fb1116148ce4eced2fe8dd5792bf03022eefa1a3`.

It is a standard GGUF (`qwen35moe`): llama.cpp loads it as is. It was made for
[eklow](https://github.com/ewtodd/eklow), a serving engine for one Intel Arc A770M, which runs Q4_K experts as fast
as Q4_0 ones. For speculative decoding, the DFlash 2 drafts
[incoai/Qwen3.6-35B-A3B-DFlash2](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2) and
[jzinno/Ornith-1.5-35B-A3B-DFlash2](https://huggingface.co/jzinno/Ornith-1.5-35B-A3B-DFlash2) accept about the same
on Ornith at block 8 in llama.cpp (5.21 / 5.01 / 4.01 and 5.23 / 5.00 / 3.92 tokens per verify on GSM8K / HumanEval /
MT-Bench at temperature 1, measured on the same mix with Q4_0 experts).
