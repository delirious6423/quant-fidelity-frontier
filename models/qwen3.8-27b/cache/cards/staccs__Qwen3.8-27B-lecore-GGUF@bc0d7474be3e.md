---
base_model: Qwen/Qwen3.8-27B
library_name: gguf
pipeline_tag: text-generation
tags: [gguf, llama.cpp, ollama, lm-studio, quantized, lecore]
---
# Qwen3.8-27B-lecore-GGUF

GGUF builds of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for llama.cpp, Ollama and LM Studio.
These are companions to the MLX export [staccs/Qwen3.8-27B-lecore-mlx](https://huggingface.co/staccs/Qwen3.8-27B-lecore-mlx), which has the measured perplexity numbers.

| File | Size | What it is |
|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | 16.55 GB | stock llama.cpp Q4_K_M |
| `Qwen3.8-27B-lecore-mix.gguf` | 15.48 GB | leCore per-tensor mix: each linear layer gets the GGUF K-quant matching the bits leCore measured for it (2→Q2_K, 3→Q3_K, 4→Q4_K, 8→Q8_0); embeddings and output follow the MLX export |

The `lecore-mix` file is quantized from the original bf16 weights with llama.cpp K-quants. The bit width of each tensor follows leCore's measured (KL-damage) allocation, so its numbers are close to, but not identical to, the MLX export.
Some tensor widths are not multiples of 256, so llama.cpp may fall back to a compatible type for those tensors.

**Note:** every linear tensor was allocated 4 bits, so `lecore-mix` is in practice a uniform Q4_K build (with a Q6_K output layer). This build is text-only: no vision mmproj is included.

## Usage
**Ollama**: download a file, then
```bash
printf 'FROM ./Qwen3.8-27B-Q4_K_M.gguf
' > Modelfile && ollama create qwen3.8-27b-lecore-gguf -f Modelfile && ollama run qwen3.8-27b-lecore-gguf
```
or pull straight from the Hub with `ollama run hf.co/staccs/Qwen3.8-27B-lecore-GGUF:Q4_K_M`

**LM Studio**: search for `staccs/Qwen3.8-27B-lecore-GGUF` in the model browser and pick a file. On a 36 GB Mac, the `lecore-mix` file is the one that fits.

**llama.cpp**
```bash
llama-cli -hf staccs/Qwen3.8-27B-lecore-GGUF -p "Hello" -n 256
```

## Benchmarks (lm-eval-harness 0.4.9, run on the shipped weights dequantized to bf16 on GPU)
MMLU 5-shot on the first 10 questions of each of the 57 subjects; GSM8K 5-shot on the first 100 test problems (greedy, 320 new tokens, no chat template). These are **subsets**, so expect about ±2–4 points of sampling noise.

| Model | MMLU (5-shot) | GSM8K strict | GSM8K flexible |
|---|---|---|---|
| **leCore MLX export** (numbers measured on the MLX weights; these GGUF files were not benchmarked) | 84.9% | 91.0% | 89.0% |
| stock 3-bit affine, group 64 (simulated) | 81.4% | 60.0% | 62.0% |

## License
Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), released by the Qwen team under the **Apache License 2.0**. Only quantization was done here, with no retraining. **Text-only:** the vision tower and the MTP head are not included in this export. The hybrid Gated-DeltaNet layers are quantized per tensor like every other linear layer.

