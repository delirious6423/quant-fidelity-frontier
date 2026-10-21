---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
tags:
  - qwen3.8
  - nvfp4
  - abliterated
  - uncensored
  - vllm
  - dgx-spark
  - compressed-tensors
library_name: transformers
---

# Qwen3.8-27B-abliterated-NVFP4

NVFP4 (`compressed-tensors`) checkpoint of an **abliterated** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), packaged for **vLLM on NVIDIA Blackwell / DGX Spark (GB10)**.

Abliteration orthogonalizes the refusal direction out of the residual stream. It is a weight-space edit, not a fine-tune. Multi-token prediction (MTP) and the vision tower are left unchanged.

**Research only:** safety alignment has been substantially removed. This checkpoint is intended for interpretability, refusal-mechanism study, red-teaming, and controlled experiments — not for production or end-user deployment.

## ⚠️ Disclaimer — read before use

This model has had its safety alignment substantially removed via abliteration (orthogonalizing the refusal direction out of the residual stream). As a direct consequence:

- It will comply with harmful, unethical, offensive, or illegal requests that the original **Qwen3.8-27B** would refuse. It has no meaningful built-in guardrails.
- It is released strictly for **legitimate research** — interpretability, AI-safety and refusal-mechanism study, red-teaming, robustness evaluation, and controlled experiments.
- You assume **full responsibility and liability** for how you use it and for everything it generates. Do not deploy it to end users or in production without adding your own safety, moderation, and abuse-prevention layers.
- Use must comply with the **Apache 2.0 License** inherited from the base model, and all laws and regulations that apply to you.
- The authors and uploaders accept **no liability** for any misuse or harm. Its outputs do not reflect the views of the uploaders or of Qwen / Alibaba.

**By downloading or using this model you acknowledge and accept the above.**

## Provenance

| Step | Detail |
| --- | --- |
| Base | `Qwen/Qwen3.8-27B` |
| Edit | Abliteration on transformer layers 18–51 (other layers, MTP, and vision unmodified) |
| Quantization | NVFP4 via `llm-compressor` oneshot; calibration `ultrachat-200k` (`train_sft`), 256 samples, seq 4096 |
| Format | `compressed-tensors`, ~18 GB |
| Target | vLLM on Blackwell (`sm_121a`) |

## vLLM (DGX Spark / GB10)

```bash
export CUTE_DSL_ARCH=sm_121a
vllm serve GoldfingerCH/Qwen3.8-27B-abliterated-NVFP4 \
  --quantization compressed-tensors \
  --host 0.0.0.0 --port 8000 \
  --served-model-name Qwen3.8-27B-abliterated-NVFP4 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.70 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --trust-remote-code \
  --api-key internal
```

On Spark, FP8 KV cache leaves on the order of ~1.9M tokens of cache (~7× concurrency at 262k).

## License

Apache 2.0, inherited from the base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Abliteration does not change the underlying license obligations.

## Credits

- Qwen Team (Alibaba) — Qwen3.8-27B architecture and weights
- Quantization and Spark packaging — GoldfingerCH
