---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
language:
  - en
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - qwen3_5
  - qwen3.8
  - autoround
  - compressed-tensors
  - w4a16
  - int4
  - llm-compressor
  - vllm
  - xpu
  - reasoning
---

# Qwen3.8-27B-W4A16-AutoRound (iters=500)

This is a **W4A16 (4-bit weight, 16-bit activation)** quantization of [Qwen/Qwen3.8-27B](https://huggingface.co), calibrated using Intel's **[AutoRound](https://github.com)** with **500 optimization iterations** (`--iters 500`) and exported in the `auto_round:llm_compressor` (`compressed-tensors`) format for seamless inference with **vLLM** (both Intel XPU and NVIDIA CUDA) and **Transformers**.

### Key Highlights
* **Virtually Lossless Reasoning:** Retains **99.1%** of the official BF16 accuracy on GPQA-Diamond (**88.4%** vs. **89.2%**).
* **Architecture-Aware Preservation:** Keeps the vision encoder (`model.visual.*`), linear-attention projections (`linear_attn.in_proj_a`, `linear_attn.in_proj_b`), and `lm_head` in unquantized `bfloat16` to preserve multimodal support and long-chain reasoning stability.
* **Accessible VRAM Footprint:** Requires ~17–19 GB VRAM for model weights, running comfortably on a single 24 GB GPU (e.g., Intel Arc Pro B70, NVIDIA RTX 3090/4090) or across multi-GPU setups.

## Benchmark Evaluation (`GPQA-Diamond`)

Evaluated using **[EvalScope](https://github.com/modelscope/evalscope)** on the full 198-question **GPQA-Diamond** PhD-level science benchmark (physics, chemistry, and molecular biology) with thinking/reasoning enabled:

| Model | Precision / Format | GPQA-Diamond Accuracy | Correct / Total | Retention vs. BF16 |
| :--- | :--- | :---: | :---: | :---: |
| **[Qwen/Qwen3.8-27B](https://huggingface.co) (Official)** | `BF16` | 89.2% | 177 / 198 | 100.0% |
| **Denzien/Qwen3.8-27B-W4A16-AutoRound** | **INT4 (`W4A16`, `g128`, `iters=500`)** | **88.4%** | **175 / 198** | **99.1%** |

### Evaluation Details & Verification Artifacts
* **Evaluation Framework:** `evalscope` (`openai_api`, streaming enabled)
* **Sampling Parameters:** `temperature=1.0`, `top_p=0.95`, `top_k=20`, `presence_penalty=0.8`, `max_tokens=49152`
* **Average Output Length:** 14,195 tokens/question (including `<think>` reasoning traces)
* **Hardware Tested:** Dual Intel Arc Pro B70 GPUs (`vLLM` with `--max-model-len 65536`, batch size 16)
* **Raw Evaluation Logs:** Full predictions, grading reviews, config, and logs are included in [`./evals/gpqa_diamond/`](./evals/gpqa_diamond/).

## Quantization Recipe

Quantized from the unquantized `Qwen/Qwen3.8-27B` weights using `auto-round` on Intel XPU:

```bash
auto-round \
  --model "Qwen3.8-27B" \
  --scheme W4A16 \
  --iters 500 \
  --nsamples 256 \
  --seqlen 2048 \
  --batch_size 4 \
  --format "auto_round:llm_compressor" \
  --device xpu:0 \
  --low_gpu_mem_usage \
  --output_dir "./Qwen3.8-27B-int4-xpu"
```

*(Note: While quantized on Intel XPU hardware, the `auto_round:llm_compressor` `W4A16` checkpoint uses standard `compressed-tensors` formatting—symmetric INT4, group size 128—and runs on both Intel XPU and NVIDIA CUDA backends.)*

### Recommended Sampling Settings
For complex reasoning, math, and science tasks:
* `temperature`: `1.0`
* `top_p`: `0.95`
* `top_k`: `20`
* `presence_penalty`: `0.8` (helps prevent repetitive `<think>` loops on long chains)
* `max_tokens`: `32768` – `49152`

## License
This model inherits the **Apache 2.0** license from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
