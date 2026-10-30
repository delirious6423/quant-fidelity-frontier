---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - 4-bit
  - gptq
  - calibrated
  - dense
  - qwen3_5
---


# Qwen3.8-27B-MLX-4bit-calib

A **calibrated 4-bit MLX** conversion of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (dense,
27B parameters).

Same file format as `mlx_lm.convert -q` (affine 4-bit, group size 64, bf16 scales and biases, embedding at 4 bits
too), so it loads anywhere a standard MLX 4-bit model loads and runs at the same speed. The difference is how the
4-bit numbers are chosen: GPTQ with a calibration set instead of plain rounding. Result: **about half the KL
divergence** of the standard 4-bit conversion, at the same size.

## Quality: KL divergence against the bf16 model

Next-token KL divergence against the bf16 model, over 36,569 tokens that were never used for calibration. Lower is
better.

| Variant | Text weights | KL mean | KL p99 | Top-1 agreement | KL, generic split | KL, private split | KL, agent split |
|---|---|---|---|---|---|---|---|
| Standard 4-bit (`mlx_lm.convert`, round-to-nearest) | 15.13 GB | 0.126 | 1.85 | 90.0 % | 0.055 | 0.181 | 0.618 |
| [ddalcu/Qwen3.8-27B-MLX-Serve-4bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit) (4-bit, bf16 embedding) | 16.96 GB | 0.123 | 1.81 | 90.0 % | 0.054 | 0.177 | 0.603 |
| **This model (GPTQ, calibrated)** | **15.13 GB** | **0.064** | **0.94** | **93.4 %** | **0.027** | **0.094** | **0.335** |

- Against the ddalcu conversion: **−0.059 mean KL (95 % CI −0.110 to −0.030)**, bootstrap over sequences, i.e.
  −48 %, while being 1.8 GB smaller. Against the standard conversion: −49 % (no confidence interval computed for
  this pair).
- The most downloaded MLX 4-bit conversions (not in the table above),
  [lmstudio-community/Qwen3.8-27B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit) and
  [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit), are uniform affine
  4-bit with group size 64 according to their `config.json` (checked 2026-10-29), the same recipe as the standard
  row. **We did not evaluate their files directly.**
- **Splits.** *Generic*: public text (oasst1 conversations, Wikipedia, Catalan and Chinese text), which anyone can
  reproduce. *Private*: Spanish and English technical text, source code and coding-agent sessions from our own
  use; it is not published, so treat those columns as indicative. *Agent* is the coding-agent part of the private
  split (long tool-calling transcripts, the hardest case for every variant).
- Not measured: downstream benchmarks, long-context quality beyond the evaluation sequences, the MTP layer
  (see below) and the vision tower (not included).

### Compared with other MLX 4-bit quantizations of the same base

We also measured the other published MLX quantizations of Qwen3.8-27B that we found, with the same method: next-token
KL divergence against the bf16 model over the same 36,569 held-out tokens (a text that was not used for
calibration; "generic" is the public part: oasst1 conversations, Wikipedia, Catalan and Chinese text). Lower is better.

| Repo | Revision | Size | Avg. bits | KL mean | KL, generic | Top-1 |
|---|---|---|---|---|---|---|
| **This model** | | **15.13 GB** | **4.50** | **0.064** | **0.027** | **93.4 %** |
| [WaveCut/Qwen3.8-27B-MLX-4bit-DWQ](https://huggingface.co/WaveCut/Qwen3.8-27B-MLX-4bit-DWQ) | `f74a4351` | 15.13 GB | 4.50 | 0.123 | 0.089 | 89.4 % |
| [AtomicChat/Qwen3.8-27B-MLX-AD-4.50bpw-DWQ-16.0GB](https://huggingface.co/AtomicChat/Qwen3.8-27B-MLX-AD-4.50bpw-DWQ-16.0GB) | `17a33d66` | 15.13 GB | 4.50 | 0.116 | 0.052 | 90.3 % |
| [Jundot/Qwen3.8-27B-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp) | `7dd276b7` | 16.27 GB (+7.5 %) | 4.84 | 0.063 | **0.022** | 93.9 % |
| [AtomicChat/Qwen3.8-27B-MLX-AD-3.80bpw-DWQ-13.7GB](https://huggingface.co/AtomicChat/Qwen3.8-27B-MLX-AD-3.80bpw-DWQ-13.7GB) | `9fa7ab0c` | 12.77 GB (−15.6 %) | 3.80 | 0.257 | 0.223 | 82.6 % |

Jundot's oQ4e, which is 7.5 % larger, **is better on generic text** (0.022 against 0.027; 95 % CI of the difference
−0.007 to −0.002) and ties on the overall mean (0.063 against 0.064; CI −0.009 to +0.008); it gets that from more
bits on some modules (5, 6 and 8 bits instead of 4), not from a better method at equal size. At exactly our size,
the two DWQ conversions are clearly worse. We did not measure OptiQ (19.4 GB) or the UD3 conversion.
The WaveCut and AtomicChat 4.50 files are the same format as this model (4-bit, group size 64, 4.50 bits on average).

## Size

- **15.13 GB** of text weights, without the MTP layer.
- **With the MTP layer: ~15.45 GB** (the layer adds ~0.32 GB).
- Text only: the vision tower of the original is **not included**. Image input does not work.
- Memory needed: weights plus KV cache plus runtime overhead. **Minimum Mac: 24 GB** (16K context, no drafter, with ingot's memory plan; 32 GB for 64K with the drafter). For mlx-lm the estimate is also 24 GB (weights + 8K KV cache ≈ 15.7 GB, estimated, not measured).

### MTP layer

`model-mtp.safetensors` holds the 29 tensors of the model's multi-token-prediction layer, **copied unchanged from
the ddalcu conversion** (their quantization, not calibrated by us), for servers that use MTP for speculative
decoding. mlx-lm loads the model without using it.

## How it was made

1. **Calibration data (~260K tokens):** the same mix as our other calibrated models, subsampled for speed (a dense
   27B model is slow to calibrate): about 41 % public generic text (the `calibration_datav3` set by bartowski and
   Project Gutenberg book excerpts in English, Spanish, French, German and Chinese), about 31 % conversations
   generated from oasst1 prompts in several languages and code tasks, with and without reasoning, by a model of the
   same family and tokenizer (Qwen3.6-35B-A3B), and about 29 % private text from our own use (technical notes,
   code, coding-agent sessions), which is not published. No evaluation text is used for calibration.
2. **Sequential GPTQ, layer by layer:** each layer is calibrated on the outputs of the already-quantized layers,
   streaming the bf16 weights from disk (the bf16 model does not need to fit in memory).
3. **Group parameters:** the scale and bias of each group of 64 are chosen by a search weighted by the Hessian
   diagonal and rounded to bf16 inside the loop, so the propagated error is the real one. Embedding: 4 bits with an
   MSE search per group; `lm_head`: GPTQ.

The conversion tool is not public. The output is a plain MLX model; nothing in it is specific to the tool.

## How to run it

**mlx-lm** (Apple Silicon):

```sh
pip install mlx-lm
mlx_lm.generate --model ingot-mlx/Qwen3.8-27B-MLX-4bit-calib --prompt "Hello" --max-tokens 256
mlx_lm.server --model ingot-mlx/Qwen3.8-27B-MLX-4bit-calib   # OpenAI-compatible API
```

Speed with mlx-lm: **not measured**. A dense 27B model reads all its weights for every token, so expect it to be
several times slower than a ~3B-active MoE of similar size.

**[ingot](https://github.com/ingot-mlx/ingot)**, a Swift/Metal inference engine for Apple Silicon (needs an M5 or later and macOS 27), runs this
model, with speculative decoding when the
[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) drafter (apache-2.0, not
included here) is placed in a `drafter/` folder next to the model. In one run on an Apple M6 (12-core GPU, 32 GB),
2026-10-20, this model and the ddalcu conversion took the same time per speculative round (~120 ms) and the same
9.8 tok/s without a drafter. That is a single run with an older version of the engine: we make no speed claim
beyond "same format, same speed".

## Files

`config.json`, tokenizer and chat template are those of the ddalcu conversion used as a reference (with the MTP
quantization entries); only the model weights are new. The weights are modified (quantized) from the original; see
`LICENSE` (Apache 2.0, copied from Qwen/Qwen3.8-27B).

## Credits

- [Qwen](https://huggingface.co/Qwen) for Qwen3.8-27B (Apache 2.0).
- [ddalcu](https://huggingface.co/ddalcu) for the MLX-Serve conversion whose MTP layer, config and template are
  reused here (Apache 2.0).
- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm); GPTQ
  (Frantar et al., 2022); bartowski for `calibration_datav3`; OpenAssistant for oasst1.
