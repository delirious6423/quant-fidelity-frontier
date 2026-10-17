---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- quantized
- 2-bit
---
# Qwen3.8-27B-2bit-oQ2

2-bit MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), built from [mlx-community/Qwen3.8-27B-bf16](https://huggingface.co/mlx-community/Qwen3.8-27B-bf16) with [oMLX](https://github.com/jundot/omlx) (`omlx.oq.quantize_oq_streaming`, oQ level 2).

Read the Limitations section before using this in anything that matters. It fails on close to half of a general-purpose prompt battery.

## Method

Base precision is 2-bit, group size 64, affine mode. Per-layer bit width is set by a sensitivity pass against a reference model (`mlx-community/Qwen3.8-27B-4bit`): layers with higher measured sensitivity to quantization get more bits, up to 8-bit for `embed_tokens`. Resulting average: 3.08 bits/weight.

Quantization reads the source safetensors tensor-by-tensor and never materializes the full 51 GB bf16 model in memory (peak \~3-4 GB during conversion). That's a hard requirement on 36 GB unified memory: whole-model calibration methods such as DWQ or AWQ need the entire model resident for a forward pass and fail with an out-of-memory kill on this hardware before producing anything.

## Results

Evaluated against the official 4-bit release, same hardware, same settings. Perplexity on 40 held-out samples (`allenai/tulu-3-sft-mixture`, sequence length 256, seed 42), plus a 16-prompt battery across 8 categories — general knowledge, math/logic, code, translation, summarization, creative writing, multi-step instructions, technical explanation — 2 prompts each, greedy decoding, 300-token budget. A completion is marked failed if it hits the token limit mid-reasoning or falls into a repetition loop (detected via 4-gram redundancy).

|  | 4-bit (mlx-community) | oQ2 (this model) |
| --- | --- | --- |
| Size on disk | 15 GB | 11 GB |
| Bits/weight | 4.50 | 3.08 |
| Perplexity | 5.405 ± 0.159 | 8.124 ± 0.216 |
| Prompt battery (16 total) | 15 passed | 8 passed |
| Decode speed | 8.3 tok/s | 12.7 tok/s |

![Quality vs. size trade-off](chart_efficiency.png)![Prompt success by category](chart_heatmap.png)The prompt battery is where the gap actually shows up, more than the perplexity number suggests. 4-bit completed 15 of 16 prompts. This model completed 8 — the other 7 entered a repetition loop before producing an answer, same prompt and decoding settings, only the quantization differs.

By category: creative writing failed both prompts. General knowledge, math/logic, translation, summarization, and multi-step instructions each lost exactly one of two prompts to a loop. Technical explanation was the only category that passed cleanly (2/2); code failed once by running out of token budget rather than looping.

Full text of all 32 completions (16 prompts × 2 models) is in `benchmark_results.json` in this repo. An interactive version of the same comparison, with both models' output shown side by side per prompt, is [here](https://claude.ai/code/artifact/edd7664c-e2a8-48da-a4b8-b18294558f64).

## Limitations

This model is smaller and faster than the 4-bit release, and measurably worse: perplexity is 50% higher, and it fails on close to half of a general prompt battery, not on edge cases but on ordinary requests like translating a sentence or writing a haiku. It has only been evaluated on the battery above — 16 prompts, one language pair, one decoding setting. Results on your actual prompt distribution may differ in either direction.

Use it where the 15 GB / 4-bit footprint genuinely doesn't fit and you've confirmed the failure rate is acceptable for your prompts. Don't use it as a general-purpose drop-in for 4-bit.

### Representative pass — "What is the difference between TCP and UDP?" (technical)

> The differences between TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) lie in how data is transmitted and in typical use cases.
> ****1. Reliability vs. speed**
>
> - **TCP:** Connection-oriented. Guarantees reliable delivery (e.g. for HTTP or email).
> - **UDP:** Connectionless. Prioritizes speed (e.g. for VoIP or live streaming).

Reaches a proper close and a complete answer. One minor terminology slip ("Protokol" instead of "Protokoll") in the original German output.

### Representative failure — "Translate to English: 'Der frühe Vogel fängt den Wurm.'" (translation)

> Die Übersetzung lautet: **"Early bird catches the worm."**
> ****Bedeutung**:Der frühe Vogel fängt den Wurm = Der frühe Vogel fängt den Wurm Der frühe Vogel fängt den Wurm = Der frühe Vogel fängt den Wurm

Correct translation on the first line, then repeats the same line until the 300-token limit. Repetition ratio 0.69.

## Usage

```bash
pip install -U mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("abuarchive/Qwen3.8-27B-2bit-oQ2")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is the difference between TCP and UDP?"}],
    add_generation_prompt=True, tokenize=False,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=400))
```

## 