---
license: apache-2.0
tags:
- hlwq
- gemma4
- moe
- quantized
- vision
- multimodal
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: image-text-to-text
arxiv: '2603.29078'
---

> [!IMPORTANT]
> **Naming notice (2026-04-10).** The "HLWQ" technique used in this model is being rebranded to **HLWQ (Hadamard-Lloyd Weight Quantization)**. The change is only the name; the algorithm and the weights in this repository are unchanged.
>
> The rebrand resolves a name collision with an unrelated, earlier KV cache quantization method also named HLWQ ([Han et al., arXiv:2502.02617, 2025](https://arxiv.org/abs/2502.02617)). HLWQ addresses **weight** quantization with a **deterministic Walsh-Hadamard rotation** and Lloyd-Max scalar codebook; Han et al.'s HLWQ addresses **KV cache** quantization with a **random polar rotation**. The two methods are technically distinct.
>
> Existing loaders that load this repository by ID continue to work without changes. Future model uploads will use the HLWQ name.
>
> Reference paper for this technique: [arXiv:2603.29078](https://arxiv.org/abs/2603.29078) (v2 in preparation; v1 still uses the old name).

# 🧊 Gemma-4-26B-A4B-it-HLWQ-Q5

**25.2B MoE (3.8B active) + Vision** — PQ5 quantized weights including ALL MoE experts.

Download: **26.9 GB** (vs 51.6 GB BF16 original)

| Metric | Value |
|---|---|
| **Download** | 26.9 GB (1.9x smaller) |
| **Quantized** | 427 linear + 7,680 MoE experts |
| **Architecture** | 30 layers, 128 experts (top-8) |
| **Vision** | ✅ Image+Text → Text |
| **Routers** | FP16 (exact expert selection) |

## 📊 Charts

![Download Size](download_size.png)
![Quantization Coverage](quant_coverage.png)
![Gemma Family](gemma_family.png)

## 🚀 Quick Start

### Expert Offloading (8.6 GB GPU — best for consumer GPUs)

```bash
# vLLM MoE expert cache is in PR #37190 (still open). Use the fork until it lands:
pip install git+https://github.com/caiovicentino/vllm-expert-offload.git
```

```python
from vllm import LLM, SamplingParams
llm = LLM('google/gemma-4-26B-A4B-it', dtype='bfloat16',
          moe_expert_cache_size=8, enforce_eager=True,
          kernel_config={'moe_backend': 'triton'})
```

### Streaming Loader (PQ5 dequant + INT4)

See [POLARQUANT_GEMMA4_26B_A4B_VISION.ipynb](POLARQUANT_GEMMA4_26B_A4B_VISION.ipynb)

## 🏆 GPU Support

| GPU | Method | VRAM |
|---|---|---|
| **T4 (16 GB)** | Expert offloading | 8.6 GB |
| **RTX 4090 (24 GB)** | Expert offloading | 8.6 GB |
| **A100 (80 GB)** | Full load + PQ5 dequant | ~50 GB |

## 📓 Notebooks

| Notebook | Description |
|---|---|
| [MoE Quantize](POLARQUANT_GEMMA4_26B_MOE_QUANTIZE.ipynb) | PQ5 quantize all experts + save codes |
| [Vision Inference](POLARQUANT_GEMMA4_26B_A4B_VISION.ipynb) | Multimodal streaming loader |
| [Expert Offload](GEMMA4_26B_EXPERT_OFFLOAD_FORK.ipynb) | vLLM fork, 14.8 tok/s |

## 🔧 Technical Details

- **MoE experts**: 3D `nn.Parameter` (128, out, in) — each expert quantized independently
- **gate_up_proj**: (128, 1408, 2816) per layer
- **down_proj**: (128, 2816, 704) per layer
- **128 experts × 2 params × 30 layers = 7,680 expert quantizations**
- **Quantization time**: 50 seconds on A100
- **PQ5 codes**: int8 + fp16 norms + Hadamard rotation + Lloyd-Max centroids

## 📖 Citation

```bibtex
@article{polarquant2025,
  title={HLWQ: Hadamard-Rotated Lloyd-Max Quantization for LLM Compression},
  author={Vicentino, Caio},
  journal={arXiv preprint arXiv:2603.29078},
  year={2025}
}
```

📄 [Paper](https://arxiv.org/abs/2603.29078) · 💻 [GitHub](https://github.com/caiovicentino/polarengine-vllm) · 📦 [pip install polarquant](https://pypi.org/project/polarquant/)


---

## 🚀 Quick Start

### Install
```bash
pip install git+https://github.com/caiovicentino/polarengine-vllm.git
```

### Load & Generate (1 line!)
```python
from polarengine_vllm import HLWQModel

model = HLWQModel.from_pretrained("caiovicentino1/Gemma-4-26B-A4B-it-HLWQ-Q5")
print(model.generate("Hello, how are you?", max_new_tokens=100))
```

### With KV Cache Compression (5.3x more context)
```python
model = HLWQModel.from_pretrained("caiovicentino1/Gemma-4-26B-A4B-it-HLWQ-Q5", kv_cache_nbits=3)
# KV cache now uses 5.3x less memory — fit longer conversations!
print(model.generate("Explain quantum computing in detail.", max_new_tokens=500))
```

### Benchmark
```bash
polarquant bench caiovicentino1/Gemma-4-26B-A4B-it-HLWQ-Q5 --ppl --chart
```

### Gradio Demo
```bash
polarquant demo caiovicentino1/Gemma-4-26B-A4B-it-HLWQ-Q5 --share
```

## 📦 Method: HLWQ

**Hadamard Rotation + Lloyd-Max Optimal Centroids**

Unlike GGUF (uniform quantization), HLWQ places quantization levels where weight density is highest — mathematically proven optimal for Gaussian-distributed neural network weights.

```
HLWQ Q5 (cos_sim > 0.996) > GGUF Q5_K_M (~0.99) at same size
```

## 🔗 Links

- 📄 [Paper — arXiv:2603.29078](https://arxiv.org/abs/2603.29078)
- 💻 [GitHub — HLWQ-Engine](https://github.com/caiovicentino/polarengine-vllm)
- 📦 [PyPI — `pip install polarquant`](https://pypi.org/project/polarquant/)
