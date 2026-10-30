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
- compressed-tensors
- quantization
---

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/64b93e6bd6c468ac7536607e/mj6xac74jHGLqymiovObc.png" alt="The Kaitchup — AI on a Budget" width="100%" />
  <p><a href="https://kaitchup.substack.com/subscribe"><strong>Subscribe to my newsletter</strong></a> · <a href="https://buymeacoffee.com/kaitchup"><strong>Support</strong></a></p>
</div>

# Qwen3.8-27B-SLQ-EAR98-v2

This is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), compressed with a **customized version of SLQ (Statistically-Lossless Quantization)** and packed for Humming in vLLM. The search combines integer quantization candidates and chooses precision and group size separately for individual projections.

- **EAR target:** >98%; measured packed-model EAR: **98.004750%**.
- **Weight files:** 22.350 GB (22,349,593,352 bytes). This is file size, not total serving memory.
- Mixed integer precisions with BF16 embeddings and activations.
- Original MTP weights retained unchanged in BF16.
- **Made by:** [The Kaitchup](https://kaitchup.substack.com/).
- **License:** [Apache 2.0](LICENSE).

Method: [SLQ paper](https://arxiv.org/abs/2605.02404) · [SLQ repository](https://github.com/IST-DASLab/SLQ).
These checkpoints are released by The Kaitchup.

## Candidate sources

Candidate weights came from **GPTQ**, generated locally from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and existing integer refinement variants. Original BF16 weights were also available to the search.

## Precision by layer

![Stored bitwidth and group size for every decoder projection](precision.svg)

The map shows actual stored bitwidth and group size. Full-attention Q/K/V share a row. Open the [SVG](precision.svg) to zoom in.

## EAR comparison

**EAR means Expected Acceptance Rate.** It measures the overlap between the original BF16 model's next-token probabilities and those of the compared model. Here, both distributions are renormalized over the BF16 reference's ten most likely tokens. An EAR of 0.99 means 99% average probability overlap on that measured support; it does not mean 99% benchmark accuracy or identical generated answers.

The plot compares these checkpoints with the previous EAR releases using packed-runtime EAR@10.

![Packed EAR@10 versus model size](ear-vs-size.svg)

SLQ's “statistically lossless” objective concerns approximate fidelity, not identical numerical outputs. These releases meet their stated measured EAR targets; the 97.8% version deliberately uses a looser target. Statistical equivalence on downstream tasks has not been established for these releases.

## Instructions: vLLM

Tested for text loading, generation and EAR with **vLLM 0.31.0**, **humming-kernels 0.1.16**, and an **NVIDIA GH200**. Use a vLLM installation with Humming support. The original vision weights and processors are retained; image/video inference and speculative decoding were not included in this validation.

Start an OpenAI-compatible text server:

```bash
vllm serve kaitchup/Qwen3.8-27B-SLQ-EAR98-v2 \
  --quantization compressed-tensors \
  --linear-backend humming \
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
    "model": "kaitchup/Qwen3.8-27B-SLQ-EAR98-v2",
    "messages": [{"role": "user", "content": "Explain expected acceptance rate in one paragraph."}]
  }'
```

GPU memory must accommodate the weights, runtime buffers and KV cache. The measured scores describe the tested execution path; other runtimes can differ slightly.

## Support

If these models are useful, [support the work](https://buymeacoffee.com/kaitchup) or [subscribe to The Kaitchup](https://kaitchup.substack.com/subscribe).
