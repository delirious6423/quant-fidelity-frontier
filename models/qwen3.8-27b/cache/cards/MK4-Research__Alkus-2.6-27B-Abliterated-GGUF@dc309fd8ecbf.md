---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
tags:
- gguf
- qwen3
- coding
- reasoning
- cybersecurity
- security
- vulnerability-detection
- code-analysis
- abliterated
- llama.cpp
pipeline_tag: text-generation
---



![Screenshot 2026-11-11 at 19.33.37](https://cdn-uploads.huggingface.co/production/uploads/692c9b3b93121a11649cec08/QQYdGYU4wlTNjLrkUhMhW.png)

# Alkus 2.6 27B (Abliterated) - GGUF

Alkus 2.6 is a **reasoning model** — a coding-, cybersecurity-, and persistence-focused fine-tune of
Qwen3.8-27B (abliterated base), trained as a LoRA adapter on hand-written reasoning data and converted
to GGUF for llama.cpp / LM Studio. It thinks in explicit chain-of-thought and has found real,
previously-unknown vulnerabilities in production code.

- Base: Qwen3.8-27B (abliterated)
- Type: **reasoning model** — explicit `<think>` chain-of-thought, verification-forward, with tool use
- Fine-tune: QLoRA, LoRA rank 16, 2 epochs on a single A100 (train_loss 0.968 / eval_loss 0.917)
- Security: has found real, previously-unknown vulnerabilities in production code (responsibly disclosed; specifics withheld)
- Note: the base is abliterated (reduced refusal behavior). Use responsibly and at your own risk.

## Files

| File | Quant | Size |
| --- | --- | --- |
| Alkus-2.6-27B-Abliterated-Q4_K_M.gguf | Q4_K_M | ~15 GB |
| Alkus-2.6-27B-Abliterated-BF16.gguf | BF16 | ~50 GB (full precision) |

## Benchmarks

![Alkus 2.6 benchmark results](benchmarks.png)

Evaluated with the EleutherAI lm-evaluation-harness (canonical few-shot, no chat template),
with the base loaded as NF4 4-bit plus the LoRA adapter.

| Task | Setting | Metric | Alkus 2.6 |
| --- | --- | --- | --- |
| ARC-Easy | 0-shot | acc | 84.0 (acc_norm 83.5) |
| PIQA | 0-shot | acc | 81.0 (acc_norm 82.3) |
| WinoGrande | 5-shot | acc | 77.3 |
| ARC-Challenge | 25-shot | acc | 69.3 (acc_norm 74.2) |

PIQA is the clean apples-to-apples comparison (both 0-shot lm-eval): the published
Qwen3.8-27B base scores about 81.6, and Alkus 2.6 scores 81.0 - effectively a tie. The
coding- and persistence-oriented fine-tune did not degrade general-benchmark performance.

GSM8K and MMLU were not completed for this release.

## Recommended runtime

- Context: up to 131072
- Flash Attention: on
- K/V cache: Q8_0
- Fits comfortably in 24 GB+ VRAM at Q4_K_M; BF16 is for full-precision use.

## Usage (llama.cpp)

```bash
llama-cli -m Alkus-2.6-27B-Abliterated-Q4_K_M.gguf \
  -c 131072 -fa -ctk q8_0 -ctv q8_0 \
  -p "Write and test a Python function that merges overlapping intervals."
```

## License

Inherits the license of the Qwen base model.
