---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- gptq
- quantized
- qwen3.8
---

# Qwen3.8-27B, GPTQ-K GGUF (4.9 bits per weight, 15.4 GiB)

A 4-bit GGUF of Qwen3.8-27B for stock llama.cpp, Ollama and LM Studio. It is 15,763 MiB; unsloth's UD-Q4_K_M is
15,702 MiB, of which 335 MiB is a NextN / multi-token-prediction layer this file omits, so on the weights used for
inference this file is 2.6% larger. It is closer to the original model on code and multilingual text, and slightly
further from it on English wiki text. It contains the language model only (no NextN layer); for image input, use an
`mmproj` file for Qwen3.8-27B.

> **Fixed 13 Oct 2026.** The first upload declared 65 layers (`block_count` and `nextn_predict_layers` were copied
> from the reference file) while it contains 64, so stock llama.cpp and llama-cpp-python refused to load it
> (`tensor blk.64.attn_norm.weight not found`). The current file declares 64 layers and no NextN layer; the tensors
> are unchanged. If you downloaded it before, please download it again.

## Results

**Measured by M. Federico with a PyTorch evaluator:** KL divergence to the bf16 model, and how often both pick the same
top-1 next token. 6 windows of 2,048 tokens per held-out set. Lower KL is closer to the original.

| file | MiB | wiki KL / top-1 | code KL / top-1 | multilingual KL / top-1 |
|---|---|---|---|---|
| unsloth UD-Q4_K_M | 15,702 | **0.0080** / 95.98% | 0.0186 / 98.14% | 0.0078 / 96.11% |
| **this file** | 15,763 | 0.0087 / **96.01%** | **0.0144 / 98.49%** | **0.0075 / 96.54%** |

**Independent check in llama.cpp itself** (`llama-perplexity --kl-divergence`, CPU build, 6 × 2,048 tokens per text).
The reference is unsloth's Q8_0 file, because the bf16 model does not fit the test machine. Wiki and code are our own
held-out texts; multilingual is the set above.

| file | wiki KL / top-1 | code KL / top-1 | multilingual KL / top-1 |
|---|---|---|---|
| unsloth UD-Q4_K_M | **0.0104** / **95.89%** | 0.0163 / 98.75% | 0.0085 / 95.85% |
| **this file** | 0.0112 / 95.70% | **0.0125 / 99.02%** | **0.0084 / 96.40%** |

The two measurements agree:
- KL on code is 23% lower;
- multilingual KL is equal or slightly lower, with higher top-1 agreement;
- wiki KL is 7–9% higher.

The measurement errors reported by llama.cpp are about ±0.0005 (wiki), ±0.0015 (code) and ±0.0003 (multilingual) on
KL.

HumanEval (164 problems, pass@1, greedy, thinking on, 4,096-token context, llama.cpp on CPU): this file 150/164
(91.5%), unsloth UD-Q4_K_M 154/164 (93.9%). They disagree on 6 problems (5 vs 1); the difference is not statistically
significant (exact McNemar p = 0.22), so we read it as a tie.

```bash
llama-cli -hf dfed24/Qwen3.8-27B-GPTQ-GGUF
```

## How it was made

- **Weights:** sequential GPTQ with error feedback, MSE range search, a Lloyd grid fit, refinement passes and a
  ridge-damped refit. It runs under the llama.cpp K-quant constraint, so every block is exactly what llama.cpp's Q4_K /
  Q5_K / Q6_K / Q3_K kernels expect.
- **Which tensors get more bits:** taken from unsloth's UD-Q4_K_M. The per-tensor types were read out of their file and
  mapped to K-quants. **This allocation is unsloth's work.** It is responsible for most of the improvement over our own
  first allocation, and it is why the file is their size.
- **Calibration:** 524k tokens of mixed text (WikiText-2 train, Python source, chat-format GSM8K and MMLU questions,
  UltraChat, Wikipedia in 8 languages). unsloth's calibration data and importance matrix were *not* used; they did not
  help this method.
- **Other tensors:** token embeddings Q4_K, output head Q6_K, norms and the linear-attention state parameters in
  F32/F16.

The code and the full experiment log are in [github.com/dfed25/mlx-gptq/tree/main/port](https://github.com/dfed25/mlx-gptq/tree/main/port).
None of these techniques is new on its own; this release is a combination that gives a better file at the same size
on two of the three text types we measured.

## Credits

- The quantization method is by Domenic Federico (Cal Poly San Luis Obispo).
- The PyTorch port, the K-quant fitting, the 27B runs and the PyTorch evaluation are by M. Federico.
- The per-tensor allocation is by unsloth.
- The independent llama.cpp check is by Domenic Federico, built with Claude (Anthropic) as a coding and research
  assistant.
- The licence follows the base model (Apache 2.0).
