---
library_name: mlx
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- image-text-to-text
- qwen3_5_text
- oq
- oQe
- enhanced
- mixed-precision
- quantized
- dense
- vision-language
- multimodal
- conversational
- apple-silicon
track_downloads: true
---

# Qwen3.8-27B-oQ5e-fp16-mtp

This is an **oQe (Enhanced Quantization)** build of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), converted and optimized for Apple Silicon using **oMLX**.

It is a **multimodal (vision-language)** model — the vision tower is preserved at `float16`, unchanged, so image and video understanding remain fully intact alongside the quantized language model.

> [!NOTE]
> **Card generation:** This model card was drafted with the assistance of an AI coding interface. All technical content — quantization details, sampling parameters, output-length guidance, and the citation — is reproduced from the official [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) model card and the quantization configuration.

## Quantization Details

* **Base Model:** [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
* **Quant Scheme:** oQe enhanced — imatrix-enhanced dynamic mixed-precision with sensitivity-driven bit allocation
* **Effective Bits-per-Weight:** ~5.9 bpw (oQ5e)
* **Output File Size:** 19.76 GiB (21.21 GB)
* **Context Length:** 262,144 tokens natively; up to 1,000,000 with YaRN (RoPE) scaling
* **Non-Quantized Precision:** `float16` (chosen for ~20% faster prefill on M1/M2; M3 and newer default to bfloat16, which is natively supported at full speed)
* **Vision Encoder:** Preserved at `float32`
* **MTP Support:** Preserved (Multi-Token Prediction heads intact)
* **Format:** MLX `safetensors` (compatible with oMLX, `mlx-lm`, and LM Studio MLX runtime)

## Compatibility & Runtimes

This model works out-of-the-box on Apple Silicon with:

1. **oMLX Engine:** Native support for MTP draft heads and SSD-tiered KV prompt caching.
2. **`mlx-lm` / `mlx-vlm`:** Standard Apple Silicon python runtime (use `mlx-vlm` for image input).
3. **LM Studio:** Load via the native MLX engine (`mlx-engine`).

## Recommended Usage (from the Base Model Card)

Because this is a quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), the base model's own recommended settings apply directly to this quant. These settings come from the base model's official [Best Practices](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices) and are recommended for optimal performance:

### Recommended Sampling Parameters

| Mode | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|------|-------------|-------|-------|-------|------------------|-------------------|
| Thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Instruct (non-thinking) | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

Thinking mode is on by default; reasoning depth can be tuned with `reasoning_effort` (default `xhigh`; also `medium` and `low`). For supported frameworks, `presence_penalty` can be raised up to 2 to reduce repetition.

### Output Length

>The base model recommends allocating sufficient output length for agentic tasks. For frameworks that support *separate* token limits for reasoning and final output, it suggests (within the 1M context length): **Reasoning Content — 262,144 tokens**, **Final Response — 131,072 tokens**.

> [!Note]
> Local runtimes (oMLX, `mlx-lm`/`mlx-vlm`, LM Studio) expose a single `max_tokens` limit that covers both reasoning and the final answer, so these budgets cannot be set separately. On Apple Silicon the native context window is 262,144 tokens; the 1M extension requires RoPE (YaRN) scaling, which is supported by serving frameworks like vLLM/SGLang/TokenSpeed but is not exposed in oMLX itself.

## Citation

If you use this quantization, please cite the base model:

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```