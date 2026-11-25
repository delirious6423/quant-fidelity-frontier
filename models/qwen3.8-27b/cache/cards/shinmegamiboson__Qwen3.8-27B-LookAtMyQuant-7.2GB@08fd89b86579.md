---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
- en
pipeline_tag: text-generation
tags:
- lookatmyquant
- quantization
- trellis
- mlx
---

# Qwen3.8-27B in 7.2 GB

Qwen3.8-27B compressed from 53.7 GB to **7.2 GB** (2.1 bits per weight) by the [LookAtMyQuant](https://github.com/ShinMegamiBoson/lookatmyquant) quantization compiler, packed for the Apple Silicon MLX runtime. It solves 401 of 540 exact-answer math problems — matching the 53.7 GB original — and outperforms Unsloth's same-size `UD-IQ2_XXS` GGUF on math, HumanEval, MMLU, and GSM8K.

## Run it

Install the [LookAtMyQuant](https://github.com/ShinMegamiBoson/lookatmyquant) runtime on Apple Silicon:

```sh
pip install git+https://github.com/ShinMegamiBoson/lookatmyquant.git

lamq generate shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB \
    -p "What is 17 times 23? Show the steps." --stats

lamq chat shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB
```

Python API (`mlx_lm` compatible):

```python
from mlx_lm import generate
from runtime_mlx import load

model, tok = load("shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB")
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "What is 17 times 23? Show the steps."}],
    add_generation_prompt=True,
    enable_thinking=False,
)
print(generate(model, tok, prompt, max_tokens=200))
```

On an Apple M5 Max it generates about 17 tokens/s in 7.6 GB of memory.

## Results

Share of each benchmark answered correctly:

| Benchmark | This build (7.2 GB) | Unsloth `UD-IQ2_XXS` (7.2 GB) | Original (53.7 GB) |
|---|---:|---:|---:|
| Math (540 exact-answer problems) | **74.3%** | 64.4% | 74.3% |
| HumanEval | **73.8%** | 67.1% | 77.4% |
| MMLU | **78.8%** | 75.0% | 83.5% |
| GSM8K | **94.5%** | 92.7% | 95.5% |
| ARC-Challenge | 58.9% | 59.2% | 59.2% |
| HellaSwag | 78.6% | 79.3% | 82.9% |

Against the GGUF at p < 0.05, this build is ahead on MMLU, GSM8K, and math, level on ARC-Challenge and HumanEval, and behind on HellaSwag.

KL divergence from the original's next-token distribution in nats per token (lower is better):

| Source | This build (7.2 GB) | Unsloth `UD-IQ2_XXS` |
|---|---:|---:|
| Chat | **0.154** | 0.206 |
| Code | **0.180** | 0.232 |
| English prose | **0.170** | 0.184 |
| WikiText | 0.258 | **0.233** |

## Method

Built with the [LookAtMyQuant 2](https://lookatmyquant.com/2/) recipe: each row group receives a bit rate from per-row Fisher information measured on Qwen3.8-27B, weights are trellis-coded at those rates with e-means codebooks, and the coded model is proximally distilled for 7,600 steps to match the original on English prose, chat, code, and math.

## Known limitations

- **English, code, and math focus:** Bit allocation, calibration, and distillation targeted English prose, chat, code, and math; in other languages KL is higher than the GGUF.

## Files

- `weights.bin` and `artifact.json`: packed model (`opgraph-packed-1`).
- `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`: Qwen3.8-27B metadata and tokenizer.
