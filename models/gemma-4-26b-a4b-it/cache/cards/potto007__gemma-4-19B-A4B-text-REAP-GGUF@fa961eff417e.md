---
quantized_by: potto007
pipeline_tag: text-generation
tags:
- MOE
- pruning
- compression
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
---

# Gemma-4-19B-A4B-text-REAP

## ✨ Highlights

Introducing **Gemma-4-19B-A4B-text-REAP**, a **memory-efficient compressed variant** of Gemma-4-26B-A4B-it that maintains near-identical performance while being **25% lighter**.

This model was created using **REAP (Router-weighted Expert Activation Pruning)**, a novel expert pruning method that selectively removes redundant experts while preserving the router's independent control over remaining experts. Key features include:

- **Near-Lossless Performance**: Maintains almost identical accuracy on code generation, agentic coding, and function calling tasks compared to the full 26B model
- **25% Memory Reduction**: Compressed from 26B to 19B parameters, significantly lowering deployment costs and memory requirements
- **Preserved Capabilities**: Retains all core functionalities including code generation, agentic workflows, repository-scale understanding, and function calling
- **Drop-in Compatibility**: Works with vanilla vLLM - no source modifications or custom patches required
- **Optimized for Real-World Use**: Particularly effective for resource-constrained environments, local deployments, and academic research

Original model: https://huggingface.co/google/gemma-4-26B-A4B-it

## Llamacpp Quantizations of gemma-4-26B-A4B-it by google

Using <a href="https://github.com/ggml-org/llama.cpp/">llama.cpp</a> release <a href="https://github.com/ggml-org/llama.cpp/releases/tag/b9352">b9352</a> for quantization.

Vision tower (VLM) stripped for lighter local operation. Stripped using forked version of REAP (https://github.com/potto007/reap/tree/gemma4-reap-support).

## Download a file (not the whole branch) from below:

| Filename | Quant type | File Size | Split | Description |
| -------- | ---------- | --------- | ----- | ----------- |
| [gemma-4-19B-A4B-text-REAP-f16.gguf](https://huggingface.co/potto007/gemma-4-19B-A4B-text-GGUF/blob/main/gemma-4-19B-A4B-text-REAP-f16.gguf) | bf16 | 37GB | fals | Full BF16 weights. |
| [gemma-4-19B-A4B-text-REAP-Q8_0.gguf](https://huggingface.co/potto007/gemma-4-19B-A4B-text-GGUF/blob/main/gemma-4-19B-A4B-text-REAP-Q8_0.gguf) | Q8_0 | 20GB | false | Extremely high quality, generally unneeded but max available quant. |
| [gemma-4-19B-A4B-text-REAP-Q6_K.gguf](https://huggingface.co/potto007/gemma-4-19B-A4B-text-GGUF/blob/main/gemma-4-19B-A4B-text-REAP-Q6_K.gguf) | Q6_K | 17GB | false | Uses Q8_0 for embed and output weights. Very high quality, near perfect, *recommended*. |
| [gemma-4-19B-A4B-text-REAP-Q5_K_M.gguf](https://huggingface.co/potto007/gemma-4-19B-A4B-text-GGUF/blob/main/gemma-4-19B-A4B-text-REAP-Q5_K_M.gguf) | Q5_K_M | 14GB | false | High quality, *recommended*. |
| [gemma-4-19B-A4B-text-REAP-Q4_K_L.gguf](https://huggingface.co/potto007/gemma-4-19B-A4B-text-GGUF/blob/main/gemma-4-19B-A4B-text-REAP-Q4_K_L.gguf) | Q4_K_L | 13GB | false | Uses Q8_0 for embed and output weights. Good quality, *recommended*. |
| [gemma-4-19B-A4B-text-REAP-Q4_K_M.gguf](https://huggingface.co/potto007/gemma-4-19B-A4B-text-GGUF/blob/main/gemma-4-19B-A4B-text-REAP-Q4_K_M.gguf) | Q4_K_M | 13GB | false | Good quality, default size for most use cases, *recommended*. |

## Downloading using huggingface-cli

<details>
  <summary>Click to view download instructions</summary>

First, make sure you have hugginface-cli installed:

```
pip install -U "huggingface_hub[cli]"
```

Then, you can target the specific file you want:

```
huggingface-cli download potto007/gemma-4-19B-A4B-text-REAP-GGUF --include "gemma-4-19B-A4B-text-REAP-Q8_0.gguf" --local-dir ./
```

You can either specify a new local-dir (gemma-4-19B-A4B-text-REAP) or download them all in place (./)

</details>

## 🧩 Model Creation

This GGUF was created by applying the **REAP (Router-weighted Expert Activation Pruning)** method uniformly across all Mixture-of-Experts (MoE) blocks of **gemma-4-26B-A4B-it**, with a **25% pruning rate**.

### How REAP Works

REAP selects experts to prune based on a novel **saliency criterion** that considers both:
- **Router gate values**: How frequently and strongly the router activates each expert
- **Expert activation norms**: The magnitude of each expert's output contributions

This dual consideration ensures that experts contributing minimally to the layer's output are pruned, while preserving those that play critical roles in the model's computations.

### Key Advantages

- **One-Shot Compression**: No fine-tuning required after pruning - the model is immediately ready for deployment
- **Preserved Router Control**: Unlike expert merging methods, REAP maintains the router's independent, input-dependent control over remaining experts, avoiding "functional subspace collapse"
- **Generative Task Superiority**: REAP significantly outperforms expert merging approaches on generative benchmarks (code generation, creative writing, mathematical reasoning) while maintaining competitive performance on discriminative tasks

📚 For more details, refer to the following resources:

- [🧾 arXiv Preprint](https://arxiv.org/abs/2510.13999)
- [🧾 REAP Blog](https://www.cerebras.ai/blog/reap-one-shot-pruning-for-trillion-parameter-mixture-of-experts-models)
- [💻 REAP Codebase (GitHub)](https://github.com/CerebrasResearch/reap)

---

## ⚖️ License

This model is derived from
**[`gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)**
and distributed under the **Apache 2.0 License**.

🔗 [View License File →](https://ai.google.dev/gemma/docs/gemma_4_license)

---

## 🧾 Citation

If you use this GGUF, please cite the REAP paper:

```bibtex
@article{lasby-reap,
  title={REAP the Experts: Why Pruning Prevails for One-Shot MoE compression},
  author={Lasby, Mike and Lazarevich, Ivan and Sinnadurai, Nish and Lie, Sean and Ioannou, Yani and Thangarasa, Vithursan},
  journal={arXiv preprint arXiv:2510.13999},
  year={2025}
}
```
