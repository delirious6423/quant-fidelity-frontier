---
base_model: zai-org/GLM-5.3
license: other
license_name: glm-5.3
license_link: LICENSE
tags:
  - glm
  - exl3
  - tensorfold
  - quantized
  - dgx-spark
---

# GLM-5.3 EXL3 2.75bpw for TensorFold, by Mia's AI Lab

Mia's AI Lab's own EXL3 quantization of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (the full model,
753B parameters), sized for three DGX Sparks.

It runs on three DGX Sparks with TensorFold:
[GLM-5.3 EXL3 on 3x DGX Sparks with TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold).

## Speed on three DGX Sparks

Measured with the recipe's defaults (RoCE all-gathers, the
[RedHatAI DSpark speculator](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark) for drafts, FP8 KV cache, one
request at a time), each figure with the boot it came from:

| | Measured | Boot |
| --- | --- | --- |
| Decode, code | 39.5-40.4 tok/s | p17, 163,840-token window, sampled, 3 runs |
| Decode, prose | 29.1-30.2 tok/s | p17, same |
| Prefill | 655 / 656 / 638 tok/s at 8k / 16k / 32k tokens | p10 |
| Exactness | drafted and concurrent replies equal serial ones, 12/12 | every boot |

Speculative decoding runs faster on predictable text (lists, tables, worked arithmetic), and slower on free prose.

**Long-context mode (experimental):** context parallelism (each Spark keeps every third token's cache), a 4-bit KV
cache and captured decode graphs give a 655,360-token window at the same decode speed: code 39.9-40.1, prose
29.8-30.4 tok/s (boot fp4g655, 3 runs each), a 16-key recall test correct at 39k, 157k and 314k tokens, prefill
~220-245 tok/s. It is experimental until the recipe trims its prompt-time memory peak.

## Fidelity

![KL divergence to the original model by test set](https://huggingface.co/datasets/Mia-AiLab/model-card-assets/resolve/main/glm-5.3-exl3-2.75bpw-tensorfold/kld_by_set.png)

KL divergence measures how far a quant's next-token probabilities drift from the original model's (lower is better).
At 2.75 bits a weight on average, the routed experts stay closest where precision matters most: math 0.012 and code
0.035 with only the experts quantized, 0.024 and 0.061 as TensorFold actually serves it with its own 4-bit dense
weights on top.

![Top-1 agreement with the original model by test set](https://huggingface.co/datasets/Mia-AiLab/model-card-assets/resolve/main/glm-5.3-exl3-2.75bpw-tensorfold/top1_by_set.png)

The quant's most likely next token is the original model's 96.7% of the time on math and 95.4% on code (92.8% over
all sets; 90.8% as served).

| Test set | KL divergence, experts only | Top-1, experts only | KL divergence, as served | Top-1, as served |
| --- | ---: | ---: | ---: | ---: |
| Math (thinking) | 0.012 | 96.7% | 0.024 | 95.3% |
| Code (thinking) | 0.035 | 95.4% | 0.061 | 92.8% |
| Chat | 0.043 | 93.5% | 0.069 | 91.1% |
| Wiki | 0.126 | 88.8% | 0.164 | 87.1% |
| Tool calls | 0.201 | 90.7% | 0.289 | 88.7% |
| All sets | 0.086 | 92.8% | 0.124 | 90.8% |

## How it was measured

- Reference: the unquantized model (`zai-org/GLM-5.3-BF16`, rev `9d2398f4`), full vocabulary, temperature 1; reference
  log-probabilities stored at fp16, KL computed in float32.
- Text: 64 x 2,048 tokens: wikitext-2 test (16 rows) and a workload set of math, code, tool-call and chat
  conversations in GLM's chat template (12 rows each, separate from the calibration data).
- KL divergence = mean over all positions of sum_v p(v) (log p(v) - log q(v)); top-1 = agreement with the reference's
  most likely token.
- "As served" replaces the dense layers with TensorFold's own q4 quantization, as the serving recipe loads them.

## What is in the checkpoint

| Part | Format |
| --- | --- |
| Routed experts, layers 3-77 | EXL3, mixed widths per expert (2, 3 or 4 bits a weight per projection), averaging exactly 2.75 bits a weight in every layer; mcg codebook, output scales |
| MTP layer 78 experts | EXL3, 4 bits a weight |
| Everything else (attention with the DSA indexer, shared experts, dense layers 0-2, routers, embeddings, head) | BF16, unchanged from [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16) |
| Chat template | Z.ai's `chat_template.jinja` from the release |
| Size | 273 GiB, 137 weight files |

Each expert's widths are listed in `quantization_config.json` (`expert_bits`).

The average width, 2.75 bits a weight, is `bits` in `quantization_config.json` (with every expert's widths); `config.json` names the format (`quant_method: exl3`) without it, so Hugging Face's config check stays quiet.

Source: the BF16 release (`zai-org/GLM-5.3-BF16`, rev `9d2398f4`), calibrated for how TensorFold serves it.

## Fit on three DGX Sparks

With TensorFold quantizing the BF16 dense layers to 4 bits at load and the layers split across three Sparks, the
weights take about 84 GiB per Spark. Measured on the first boots (FP8 KV cache, 64k-token window, one request at a
time): the lowest free memory was 16.7 / 20.1 / 13.3 GiB on the three Sparks. The recipe's default window is 163,840
tokens with an FP8 KV cache; its memory plan checks every Spark's free memory before it starts.

## Credits

- Base model: Z.AI, [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), under the GLM-5.3 License (in `LICENSE`).
- Quantization format and converter: [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT).
- Serving engine: [TensorFold](https://github.com/ashhart/TensorFold).
