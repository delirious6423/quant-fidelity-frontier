---
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- gsq
- gumbel-softmax
- quantization
- ptq
- qwen
- vllm
- humming

---

# Qwen3.8-27B — 3-bit GSQ

Low-bit quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
produced with **GSQ** (Gumbel-Softmax Quantization).

This checkpoint applies GSQ-based post-training quantization to the model
weights, reducing precision while preserving the original model's reasoning,
coding, multilingual, long-context, and agentic capabilities.

The transformer weights are quantized to **3-bit GSQ** with **group size 128**.
The embedding layer and LM head are quantized separately to **4-bit RTN** with
**group size 64** to preserve output quality and embedding fidelity.

- Paper: [GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling](https://arxiv.org/abs/2604.18556) (arXiv:2604.18556)
- Paper page on HF: [https://huggingface.co/papers/2604.18556](https://huggingface.co/papers/2604.18556)
- Code: [https://github.com/IST-DASLab/GSQ](https://github.com/IST-DASLab/GSQ)
- Collection: [https://huggingface.co/collections/ISTA-DASLab/gsq](https://huggingface.co/collections/ISTA-DASLab/gsq)

# Evaluation Results

We evaluate the quantized checkpoint against the original
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). Both models were evaluated with **xhigh thinking** enabled.

## Accuracy

| Benchmark | Base Model | 3-bit GSQ |
|---|---:|---:|
| AIME 2025 | **100.00** | **100.00** |
| GPQA Diamond | 89.90 | **91.41** |

## Generation Statistics

| Benchmark | Base Model | 3-bit GSQ |
| --------- | ----------------: | ---------------: |
| AIME 2025 | **0.603M** | 0.615M |
| GPQA Diamond | 3.721M | **3.705M** |

> **Note:** Due to the stochastic nature of these tasks, benchmark results can exhibit significant variance across runs. The reported scores correspond to a single evaluation run and should not be interpreted as definitive estimates of model performance.

# Calibration dataset

The GSQ quantization calibration dataset was constructed to represent a broad
range of LLM workloads, including reasoning, coding, scientific tasks,
multilingual understanding, long-context processing, and agentic behaviour.

The calibration mixture consists of:

| Category | Percentage |
| -------- | ---------: |
| Math | 13.5% |
| Code | 17.5% |
| Science | 20.0% |
| General | 12.5% |
| Multilingual | 12.5% |
| Long context | 14.0% |
| Agentic trajectories | 10.0% |

# Serving with vLLM

Serving this checkpoint requires a patched vLLM installation.

Requirements:

- **vLLM 0.27.1**
- The included patch:

```text
patch_vllm_qwen35_embedding.py
```

This patch enables vLLM support for quantized embedding weights.

Install vLLM:

```bash
pip install vllm==0.27.1
```

Apply the patch in the same Python environment:

```bash
python patch_vllm_qwen35_embedding.py
```

Then serve the model:

```bash
vllm serve ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
```

> **Important:** If vLLM is reinstalled or the environment is recreated, run
> the patch again before serving the checkpoint.

## Model size / text-only usage

The full checkpoint size is approximately **11.83 GB** when deployed with
vision capabilities enabled.

The quantization calibration dataset used for this release did **not include
vision samples**. Therefore, while the vision components are preserved in the
checkpoint and can be loaded, they were not calibrated using multimodal
calibration data.

For text-only deployment, the vision components are not required. The model can
be loaded with the `--language-model-only` option:

```bash
vllm serve ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --language-model-only
```

This removes the vision-related components from the loaded model and reduces
the checkpoint size to approximately **10.90 GB**.

This release does **not currently support speculative decoding**. The MTP
(Multi-Token Prediction) components have been removed from the published
checkpoint and are not available for MTP-based inference.

Actual VRAM usage during serving will be higher than the raw checkpoint size
and depends on:

- KV-cache allocation
- context length
- batch size
- vLLM runtime overhead
- GPU architecture

# Citation

```bibtex
@article{gsq2026,
  title  = {GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling},
  author = {Dadgarnia, Alireza and Tabesh, Soroush and Nikdan, Mahdi and Helcig, Michael and Kurti{\'c}, Eldar and Kleinegger, Max and Alistarh, Dan},
  journal= {arXiv preprint arXiv:2604.18556},
  year   = {2026},
  url    = {https://arxiv.org/abs/2604.18556}
}
```
