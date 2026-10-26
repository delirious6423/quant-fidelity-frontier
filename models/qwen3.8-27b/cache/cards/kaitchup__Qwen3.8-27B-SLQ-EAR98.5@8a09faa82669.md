---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- qwen3_5
- slq
- mixed-precision
- humming
- vllm
- gptq
- autoround
- quantization
---

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/64b93e6bd6c468ac7536607e/mj6xac74jHGLqymiovObc.png" alt="The Kaitchup — AI on a Budget" width="100%" />
  <p><a href="https://kaitchup.substack.com/subscribe"><strong>Subscribe to my newsletter</strong></a> · <a href="https://buymeacoffee.com/kaitchup"><strong>Support</strong></a></p>
</div>

# Qwen3.8-27B-SLQ-EAR98.5

This is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), compressed with a **customized version of SLQ (Statistically-Lossless Quantization)** and packed for Humming in vLLM. The search combines several quantization sources and chooses precision separately for individual projections, including selected attention heads.

- **EAR target:** 98.5%; measured packed-model EAR: **98.502656%**.
- **Weight files:** 25.609 GB (25,609,379,336 bytes). This is file size, not total serving memory.
- Mixed integer precisions with selected original BF16 weights; BF16 activations.
- Original MTP weights retained unchanged in BF16.
- **Made by:** [The Kaitchup](https://kaitchup.substack.com/).
- **License:** [Apache 2.0](LICENSE).

Method: [SLQ paper](https://arxiv.org/abs/2605.02404) · [SLQ repository](https://github.com/IST-DASLab/SLQ).


## Candidate sources

Candidate weights came from **GPTQ (INT2–INT8)**, generated locally from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); **[kaitchup/Qwen3.8-27B-hum-high-MTP-17.7GB](https://huggingface.co/kaitchup/Qwen3.8-27B-hum-high-MTP-17.7GB)**, quantized with [AutoRound](https://github.com/intel/auto-round); and **[ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ)**. Original BF16 weights were also available to the search.

## Precision by layer

![Stored bitwidth for every decoder projection](precision.svg)

The map shows actual stored bitwidth. Open the [SVG](precision.svg) to zoom in.

## EAR comparison

**EAR means Expected Acceptance Rate.** It measures the overlap between the original BF16 model's next-token probabilities and those of the compared model. Here, both distributions are renormalized over the BF16 reference's ten most likely tokens. An EAR of 0.99 means 99% average probability overlap on that measured support; it does not mean 99% benchmark accuracy or identical generated answers.

All measured rows use the same reference and metric. The execution column distinguishes packed/native inference from measurements of dequantized weights.

| Model | EAR ↑ | Weight files (GB) | Measured execution |
|---|---:|---:|---|
| [Original BF16](https://huggingface.co/Qwen/Qwen3.8-27B) | 100.000000% | 55.563 | Original BF16 reference |
| [Official FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | 97.705903% | 30.867 | Native FP8 weights and activations |
| [Official FP8 — weights only](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | 98.101249% | 30.867 | Dequantized weights, BF16 execution |
| [GSQ 3-bit](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ) | 89.425738% | 11.828 | Complete target weights dequantized; BF16 execution |
| [Kaitchup hum-high (17.7 GB)](https://huggingface.co/kaitchup/Qwen3.8-27B-hum-high-MTP-17.7GB) | 94.335032% | 17.695 | Complete target weights dequantized; BF16 execution |
| [Unsloth UD-Q4_K_XL GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | 97.444147% | 17.559 | GGUF weights inverse-mapped to BF16; Transformers execution |
| [Custom SLQ EAR>98](https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-EAR98) | 98.000775% | 23.732 | Packed vLLM / Humming runtime |
| [Custom SLQ EAR98.5](https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-EAR98.5) | 98.502656% | 25.609 | Packed vLLM / Humming runtime |
| [Custom SLQ EAR99](https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-EAR99) | 99.002514% | 30.518 | Packed vLLM / Humming runtime |

SLQ's “statistically lossless” objective concerns approximate fidelity, not identical numerical outputs. These releases meet their stated measured EAR targets; the 98% version deliberately uses a looser target. Statistical equivalence on downstream tasks has not been established for these releases.

## Instructions: vLLM

Tested for text loading, generation and EAR with **vLLM 0.31.0**, **humming-kernels 0.1.16**, and an **RTX PRO 6000 Blackwell**. Use a vLLM installation with Humming support. The original vision weights and processors are retained; image/video inference and speculative decoding were not included in this validation.

Start an OpenAI-compatible text server:

```bash
vllm serve kaitchup/Qwen3.8-27B-SLQ-EAR98.5 \
  --quantization humming \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

For reasoning parsing and automatic tool calls, add these server options:

```bash
--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

Send a chat request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kaitchup/Qwen3.8-27B-SLQ-EAR98.5",
    "messages": [{"role": "user", "content": "Explain expected acceptance rate in one paragraph."}]
  }'
```

GPU memory must accommodate the weights, runtime buffers and KV cache. The measured scores describe the tested execution path; other runtimes can differ slightly.

## Support

If these models are useful, [support the work](https://buymeacoffee.com/kaitchup) or [subscribe to The Kaitchup](https://kaitchup.substack.com/subscribe).
