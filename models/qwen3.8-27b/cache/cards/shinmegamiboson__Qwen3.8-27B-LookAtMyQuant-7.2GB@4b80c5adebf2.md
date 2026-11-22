---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
- en
pipeline_tag: text-generation
tags:
- lookatmyquant
- quantization
- mlx
---

# Qwen3.8-27B in 7.2 GB

Qwen3.8-27B compressed from 53.7 GB to 7.2 GB (2.1 bits per weight) by the [LookAtMyQuant](https://github.com/ShinMegamiBoson/lookatmyquant) quantization compiler, in the packed format its Apple Silicon runtime reads. It solves as many of 540 exact-answer math problems as the original (401), and it scores above Unsloth's `UD-IQ2_XXS` GGUF of the same size on math, HumanEval, MMLU and GSM8K.

## Results

Share of each benchmark answered correctly:

| | This file | Unsloth `UD-IQ2_XXS` (7.2 GB) | Original (53.7 GB) |
|---|---|---|---|
| Math, 540 exact-answer problems | 74.3% | 64.4% | 74.3% |
| HumanEval | 73.8% | 67.1% | 77.4% |
| MMLU | 78.8% | 75.0% | 83.5% |
| GSM8K | 94.5% | 92.7% | 95.5% |
| ARC-Challenge | 58.9% | 59.2% | 59.2% |
| HellaSwag | 78.6% | 79.3% | 82.9% |

Against the GGUF at p < 0.05, this file is ahead on MMLU, GSM8K and math, level on ARC-Challenge and HumanEval, and behind on HellaSwag.

KL divergence from the original's next-token distribution, in nats per token (lower is better):

| | This file | Unsloth `UD-IQ2_XXS` |
|---|---|---|
| Chat | 0.154 | 0.206 |
| Code | 0.180 | 0.232 |
| English prose | 0.170 | 0.184 |
| WikiText | 0.258 | 0.233 |

## It is made for English, code and math

The bit allocation was measured on English prose and chat, code and math, and the file was calibrated and fine-tuned on the same text. In other languages it is further from the original than the GGUF is.

## Run it

On a Mac with Apple silicon, with `mlx` and `mlx_lm` installed:

```sh
git clone https://github.com/ShinMegamiBoson/lookatmyquant && cd lookatmyquant
hf download shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB --local-dir q38-27b
python -c "
import sys; sys.path.insert(0, 'runtime_mlx')
import fast_q35
from mlx_lm import generate
model, tok = fast_q35.load('q38-27b')
prompt = tok.apply_chat_template([{'role': 'user', 'content': 'What is 17 times 23? Show the steps.'}], add_generation_prompt=True, enable_thinking=False)
print(generate(model, tok, prompt, max_tokens=200))
"
```

On an M5 Max it generates about 17 tokens a second in 7.6 GB of memory.

## Files

- `weights.bin` and `artifact.json`: the packed model (format `opgraph-packed-1`)
- `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`: Qwen3.8-27B's own

## How it was made

The [LookAtMyQuant 2](https://lookatmyquant.pages.dev/2/) recipe, which that page describes on Qwen3.5-4B: each row of each matrix gets a bit rate from the Fisher information measured on this model, the weights are trellis-coded at those rates, and the coded model is fine-tuned for 7,600 steps to match the original on text the original wrote.
