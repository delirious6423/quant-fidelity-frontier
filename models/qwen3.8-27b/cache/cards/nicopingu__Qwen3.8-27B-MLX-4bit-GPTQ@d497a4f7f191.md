---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
tags:
- mlx
- gptq
- 4-bit
- apple-silicon
---

# Qwen3.8-27B, MLX 4-bit, GPTQ

A 4-bit MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) made with GPTQ instead of plain round-to-nearest.

Same format and same size (15 GB) as [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit): MLX affine 4-bit, group size 64, bf16 scales. It is a drop-in replacement for any tool that reads that checkpoint (mlx-lm, mlx-vlm). Since the layout and kernels are identical, decoding speed should be the same; it has not been measured separately yet. Only the 64 text decoder layers were re-quantized (1,488 tensors); the vision tower, embeddings and output head are copied unchanged from the mlx-community 4-bit checkpoint.

## Results

KL divergence against the 8-bit checkpoint ([mlx-community/Qwen3.8-27B-8bit](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit)) over the full vocabulary, mean over 8,192 tokens per text, and the share of tokens whose top-1 prediction matches the reference. Lower KL is better.

| Text | RTN 4-bit (mlx-community) | This model (GPTQ) | Top-1 match RTN / GPTQ |
|---|---:|---:|---:|
| Technical docs and Python code (not related to the calibration set) | 0.0484 | **0.0333** (-31 %) | 91.6 % / 93.2 % |
| English, wikitext-2 test | 0.0386 | **0.0268** (-30 %) | 90.6 % / 92.5 % |
| French, Wikipedia articles | 0.0297 | **0.0195** (-34 %) | 91.7 % / 93.1 % |
| Code, GitHub files (repositories not in the calibration set) | 0.0286 | **0.0200** (-30 %) | 95.8 % / 96.5 % |
| Function calling dialogues (glaive format) | 0.994 | **0.400** (-60 %) | 75.4 % / 88.6 % |

About -30 % KL on every text, with no regression. The largest gain is on structured tool-call output, where round-to-nearest 4-bit loses the most.

Caveats, stated plainly:
- The reference is the 8-bit checkpoint, not bf16 (bf16 does not fit next to the test on a 64 GB machine). True losses against bf16 are slightly higher for both models.
- The English, French, code and function-calling texts are different documents from the same sources as the calibration set, so those gains are somewhat favourable; the first row is fully independent of the calibration data.
- 8,192 tokens per text. No downstream benchmark (MMLU, etc.) was run.

## Method

- Sequential, layer-by-layer GPTQ, starting from the 8-bit checkpoint dequantized one layer at a time; each layer is calibrated on the outputs of the already-quantized previous layers.
- Hessian H = XᵀX in float32, dampening 5 % of the mean diagonal, blocks of 64 columns, same per-column update as `mlx_lm/quant/gptq.py`. Output kept in the standard MLX affine 4-bit / group 64 layout.
- Calibration: 262,144 tokens (256 sequences of 1,024), interleaved in equal parts from four public sources: English ([wikitext-2](https://huggingface.co/datasets/Salesforce/wikitext) train), French ([Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) 20231101.fr), code ([codeparrot/github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean)) and function-calling dialogues ([glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)).
- A first attempt calibrated only on technical documentation and code cut KL by 43 % on that domain but made it 23 % worse on general English: a narrow calibration set overfits. The varied set above fixes it.
- Hardware: Mac Studio M1 Ultra 64 GB, MLX 0.32.3, mlx-lm 0.32.0; about 2 hours for the 64 layers.

## Use

```bash
pip install -U mlx-vlm
python -m mlx_vlm.generate --model nicopingu/Qwen3.8-27B-MLX-4bit-GPTQ --max-tokens 200 --prompt "Write a Python function that parses an ISO 8601 date."
```

Text-only with mlx-lm:

```python
from mlx_lm import load, generate
model, tokenizer = load("nicopingu/Qwen3.8-27B-MLX-4bit-GPTQ")
print(generate(model, tokenizer, prompt="Hello", max_tokens=100))
```

## Serving with PLI

I run this model with [PLI](https://github.com/nicopingu/pli) (Apache-2.0), an OpenAI- and Anthropic-compatible inference server for Apple Silicon, tuned for the M1 Ultra (fork of TensorFold). Any server that loads mlx-community 4-bit checkpoints should load this one unchanged.

## License

Apache-2.0, as the base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). This repository only changes the quantization of the weights; all credit for the model goes to the Qwen team.
