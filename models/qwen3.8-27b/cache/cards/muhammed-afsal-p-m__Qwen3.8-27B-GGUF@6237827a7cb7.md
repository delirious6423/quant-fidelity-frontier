---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
library_name: gguf
quantized_by: muhammed-afsal-p-m
language:
- en
- zh
- hi
- ml
- ta
- te
- bn
- ar
- fr
- de
- es
- pt
- ru
- ja
- ko
- vi
- id
- th
- tr
tags:
- gguf
- qwen3.8
- qwen
- quantized
- imatrix
- thinking
- reasoning
- vision
- multimodal
- tool-calling
- multilingual
- llama.cpp
- conversational
---

# Qwen3.8-27B GGUF (imatrix)

Quantized GGUF versions of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), with **thinking mode,
vision (image input) and tool calling**. Every quant was **measured against the original BF16 model**
(perplexity + KL divergence) and **functionally tested** (thinking on/off, tool calls, vision).

**Pipeline code:** https://github.com/muhammedafsalpm/qwen-gguf-quantization

## Quick start

```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf --mmproj mmproj-Qwen3.8-27B-F16.gguf -ngl 99 -c 16384 --jinja
```

## Why these quants

- **Importance matrix from a near-lossless reference:** computed on the Q8_0 quant (mean KLD 0.0007 against BF16) using 164K tokens of chat, code, math, reasoning traces (thinking mode) and multilingual text, rendered with the model's own chat template.
- **Every file measured against BF16:** KL divergence, perplexity and same-top-token rate, so you can choose a size by measured quality.
- **Vision included:** the multimodal projector works with every quant, and image input was tested on each one.
- **I-quants included** (IQ4_XS, IQ3_M) for 12–16 GB GPUs, next to the standard K-quants.
- **Functionally tested:** thinking on, thinking off, tool calling and vision on every file, plus 46 graded questions per file (reasoning, math, code, SQL, image questions, Malayalam and Hindi).
- **Measured per language:** accuracy against BF16 on Malayalam, Hindi and Python code, not only English.

## Files

| File | Type | Size | Bits/weight | VRAM* | PPL ↓ | Mean KLD ↓ | Same top token ↑ | Notes |
|---|---|---|---|---|---|---|---|---|
| [Qwen3.8-27B-Q8_0.gguf](./Qwen3.8-27B-Q8_0.gguf) | Q8_0 | 29.1 GB | 8.50 | ~32 GB | 6.4399 | 0.0007 | 98.86% | Near-lossless. Best accuracy |
| [Qwen3.8-27B-Q6_K.gguf](./Qwen3.8-27B-Q6_K.gguf) | Q6_K | 22.4 GB | 6.57 | ~25 GB | 6.4312 | 0.0020 | 97.91% | Excellent quality |
| [Qwen3.8-27B-Q5_K_M.gguf](./Qwen3.8-27B-Q5_K_M.gguf) | Q5_K_M | 19.6 GB | 5.75 | ~23 GB | 6.4544 | 0.0041 | 97.10% | Very high quality |
| [Qwen3.8-27B-Q4_K_M.gguf](./Qwen3.8-27B-Q4_K_M.gguf) | Q4_K_M | 17.0 GB | 4.98 | ~20 GB | 6.4644 | 0.0121 | 95.09% | **Recommended default.** Best balance |
| [Qwen3.8-27B-IQ4_XS.gguf](./Qwen3.8-27B-IQ4_XS.gguf) | IQ4_XS | 15.5 GB | 4.55 | ~18 GB | 6.5045 | 0.0164 | 94.66% | Best choice for 16 GB GPUs |
| [Qwen3.8-27B-Q3_K_M.gguf](./Qwen3.8-27B-Q3_K_M.gguf) | Q3_K_M | 13.7 GB | 4.01 | ~17 GB | 6.6493 | 0.0468 | 90.81% | 16 GB GPUs with longer context |
| [Qwen3.8-27B-IQ3_M.gguf](./Qwen3.8-27B-IQ3_M.gguf) | IQ3_M | 13.0 GB | 3.80 | ~16 GB | 6.7204 | 0.0563 | 90.33% | 12–16 GB GPUs |

**Vision:** also download [`mmproj-Qwen3.8-27B-F16.gguf`](./mmproj-Qwen3.8-27B-F16.gguf) (0.9 GB); it works with every quant above.

\*VRAM is approximate: model + vision projector + ~8K context. Longer context needs more.

**How to read the accuracy columns** (all measured against the original BF16 model, perplexity **6.4334**):
- **Mean KLD** (KL divergence): how far the quant's word probabilities are from the original's. Lower is better; 0 means identical. This is the most reliable measure.
- **Same top token:** how often the quant picks the same most-likely next word as the original.
- **PPL** (perplexity): how well the model predicts held-out text. Lower is better. The measurement
  uncertainty is about ±0.08, so differences smaller than that (for example Q6_K vs BF16) are noise.

## Which file should I use?

- **32 GB+ GPU:** Q8_0 (practically identical to the original)
- **24–32 GB GPU (RTX 3090/4090/5090):** Q6_K or Q5_K_M
- **20–24 GB:** Q4_K_M (the usual default)
- **16 GB:** IQ4_XS or Q3_K_M
- **12 GB:** IQ3_M

## How to run

> **Compatibility:** these files need **llama.cpp build b12345 or newer**. Recent llama.cpp
> stores some of this model's tensors in a combined layout (`attn_qkv_gate`, `ssm_beta_alpha`) that older
> runtimes cannot read. **Ollama 0.40.16 and older cannot load them** (error: `tensor 'blk.0.ssm_beta.weight'
> not found`). Other apps work once their bundled llama.cpp is at least build b12345.

**llama.cpp** (chat with image support; tested):
```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf --mmproj mmproj-Qwen3.8-27B-F16.gguf -ngl 99 -c 16384 --jinja
```

**Ollama:** not supported yet (see Compatibility). For Ollama, use its own library model instead:
`ollama run qwen3.8:27b`.

**LM Studio and other llama.cpp-based apps:** load a quant once the app's llama.cpp runtime is build
b12345 or newer.

**Thinking mode** is on by default (reasoning effort `xhigh`). Per request, with llama-server or any
OpenAI-compatible API:
- turn thinking off: `"chat_template_kwargs": {"enable_thinking": false}`
- shorter thinking: `"chat_template_kwargs": {"reasoning_effort": "low"}` (`xhigh`, `medium` or `low`)

## Functional tests

Run through `llama-server` with the vision projector loaded (temperature 0):

| Quant | Thinking on | Thinking off | Tool call | Vision | Generation speed (tok/s, RTX 6000 Ada) |
|---|---|---|---|---|---|
| Q8_0 | ✅ | ✅ | ✅ | ✅ | 27.7 |
| Q6_K | ✅ | ✅ | ✅ | ✅ | 35.5 |
| Q5_K_M | ✅ | ✅ | ✅ | ✅ | 39.6 |
| Q4_K_M | ✅ | ✅ | ✅ | ✅ | 43.7 |
| IQ4_XS | ✅ | ✅ | ✅ | ✅ | 46.9 |
| Q3_K_M | ✅ | ✅ | ✅ | ✅ | 48.6 |
| IQ3_M | ✅ | ✅ | ✅ | ✅ | 50.2 |

**Scope of the evaluation:** the functional tests are smoke tests (one prompt per feature) that confirm each
feature works; they do not score quality. Accuracy (perplexity, KL divergence) is measured on English
Wikipedia text at 512-token context, so math, long context and image quality are not separately measured (accuracy on Malayalam, Hindi and
code is in the next section).

## Accuracy by language

Mean KL divergence against BF16 (lower is better) on held-out text in each language
(80 chunks × 512 tokens; English is the wikitext-2 test set above):

| Quant | English | Malayalam | Hindi | Python code |
|---|---|---|---|---|
| Q8_0 | 0.0007 | 0.0008 | 0.0007 | 0.0009 |
| Q6_K | 0.0020 | 0.0024 | 0.0018 | 0.0021 |
| Q5_K_M | 0.0041 | 0.0047 | 0.0039 | 0.0041 |
| Q4_K_M | 0.0121 | 0.0153 | 0.0115 | 0.0124 |
| IQ4_XS | 0.0164 | 0.0205 | 0.0168 | 0.0169 |
| Q3_K_M | 0.0468 | 0.0587 | 0.0471 | 0.0439 |
| IQ3_M | 0.0563 | 0.0633 | 0.0524 | 0.0528 |

## Question tests

Every quant answered the same questions at temperature 0 with
[`compare_quants.py`](https://github.com/muhammedafsalpm/qwen-gguf-quantization/blob/main/tools/compare_quants.py).
Automatic checks were confirmed by reading the answers.

| Quant | Basic (13) | Extended (19) | English / Malayalam / Hindi (14) |
|---|---|---|---|
| Q8_0 | 13 | 19 | 14 |
| Q6_K | 13 | 19 | 14 |
| Q5_K_M | 13 | 19 | 14 |
| Q4_K_M | 13 | 19 | 14 |
| IQ4_XS | 13 | 19 | 14 |
| Q3_K_M | 13 | 19 | 14 |
| IQ3_M | 13 | 19 | 14 |

- Every quant answered every graded question correctly, including the image questions (two charts and an
  invoice) and asking for approval before sending an email in the agentic planning question.
- These are small samples (one run per question); they show that no quant breaks, not small quality differences.

## How these quants were made

1. Original weights: `Qwen/Qwen3.8-27B` at commit `37561816e66b5b9153b1b64c10968c18c1e6f4a0` (SHA256-verified).
2. Converted to BF16 GGUF with llama.cpp `b12345`; vision projector exported as F16.
3. Importance matrix computed on the Q8_0 quant using **163,541 tokens** of calibration data,
   rendered with the model's own chat template (including `<think>` reasoning traces):

| Source | Dataset | Samples | Tokens |
|---|---|---|---|
| general | `Salesforce/wikitext` | 55 | 32,012 |
| chat | `HuggingFaceH4/ultrachat_200k` | 25 | 32,181 |
| math | `openai/gsm8k` | 127 | 24,253 |
| code | `iamtarun/python_code_instructions_18k_alpaca` | 131 | 24,431 |
| reasoning | `nvidia/OpenCodeReasoning` | 13 | 33,040 |
| multilingual | `CohereLabs/aya_dataset` | 73 | 17,624 |

4. All quants except Q8_0 use the importance matrix. The file [`Qwen3.8-27B.imatrix.gguf`](./Qwen3.8-27B.imatrix.gguf) is included.
5. Evaluated on the held-out **wikitext-2 test** split (150 chunks × 512 tokens),
   with KL divergence against BF16 logits. Calibration used the train split only.

Checksums: [`SHA256SUMS.txt`](./SHA256SUMS.txt).

## License and credits

The model weights are licensed under **Apache 2.0**, the same as the original
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). All credit for the model goes to the Qwen team.
Quantized with [llama.cpp](https://github.com/ggml-org/llama.cpp).
