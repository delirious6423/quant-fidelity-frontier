---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- qwen3.8
- quantized
- mixed-precision
- task-aware-quantization
- tak
- reasoning
- llama.cpp
---

# Qwen3.8-27B — TAK reasoning winner

This repository contains the TAK reasoning winner for
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
It is a reasoning-specialized, mixed-precision GGUF built at the `UD-IQ2_S` byte target. No fine-tuning or model merge was applied.

Part of the [TAK Reasoning Champions](https://huggingface.co/collections/ByteOtter/tak-reasoning-champions-6a9f09c206557492a82905f3) collection.

## File

| File | Size | SHA-256 |
|---|---:|---|
| `qwen3.8-27b-tak-reasoning.gguf` | 8.372 GB | `251f7c9be422340b65159f69de204d504622c9108832e169cf041c14c0fcbad8` |

## Benchmark results

The headline category is reasoning. Other suites are diagnostics, not product claims. The matched low-bit comparator is Unsloth
[`UD-IQ2_S`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-IQ2_S.gguf),
also at 8.372 GB. TAK and Unsloth completed all 960 evaluation items with zero failed items. The BF16 source completed all
128 reasoning items with zero failed items.

### Reasoning headline

| Artifact | Size | Reasoning | TAK difference |
|---|---:|---:|---:|
| TAK reasoning winner | 8.372 GB | **82.81** | — |
| Unsloth `UD-IQ2_S` | 8.372 GB | 77.34 | +5.47 |
| BF16 source | 54.658 GB | 83.59 | -0.78 |

The last column is TAK minus the listed artifact. Positive values favor TAK.

### Full suite diagnostics

| Evaluation suite | TAK | Unsloth | TAK delta |
|---|---:|---:|---:|
| Coding | 53.13 | 53.91 | -0.78 |
| Math | 25.00 | 20.31 | +4.69 |
| **Reasoning** | **82.81** | **77.34** | **+5.47** |
| Knowledge QA | 78.13 | 78.13 | 0.00 |
| Instruction following | 35.94 | 26.56 | +9.38 |
| Summarization and extraction | 79.69 | 75.00 | +4.69 |
| Structured output | 64.06 | 59.38 | +4.69 |
| General fidelity | 35.59 | 37.83 | -2.24 |
| Coherence | 100.00 | 75.00 | +25.00 |
| Stability | 38.01 | 35.19 | +2.82 |
| Context | 95.83 | 96.88 | -1.04 |

Scores are percentages from the same frozen MLab v1 evaluation pack. Full-suite deltas are TAK minus Unsloth in percentage
points. The comparator artifact has SHA-256
`7897d2c5a5cee46aef50895141b2c8a0803c1185f3d03c4fda4cd137a7ad77fe`.

## Process overview

1. Start from the upstream BF16 instruction checkpoint.
2. Build a reasoning-oriented calibration sample.
3. Evaluate mixed-precision candidates under the comparator’s byte ceiling.
4. Freeze the strongest validated candidate and run the full evaluation pack.

TAK changes weight precision. It does not fine-tune the model, merge weights, or replace the tokenizer or chat template.
Internal scoring weights, search heuristics, candidate rules, and the tensor-level allocation recipe are not published.

## Usage

Use a recent `llama.cpp` build:

```bash
llama-cli \
  -hf ByteOtter/Qwen3.8-27B-TAK-Reasoning-GGUF \
  -p "Explain your answer carefully: If all A are B and no B are C, can any A be C?" \
  -n 512
```

## Limitations

- This model targets reasoning. The off-target scores only describe diagnostic behavior.
- Quantization can change outputs, accuracy, and stability across prompts and runtimes.
- This repository contains the text-generation GGUF only. It does not include a multimodal projector.
- Review the base model card and validate the model for your application, especially for high-stakes use.

## Related research

- [You Had One Job: Per-Task Quantization Using LLMs’ Hidden Representations (TAQ)](https://openreview.net/pdf/1ef27c381ab87b7ec64c70d14c69cc19a40fb9f9.pdf)
- [Beyond Activation Alignment: The Alignment-Diversity Tradeoff in Task-Aware LLM Quantization (TASA)](https://arxiv.org/abs/2607.00908)

TAK was developed independently and is not an official implementation of TAQ or TASA.

## Support

If this model is useful to you, you can [support ByteOtter on Buy Me a Coffee](https://buymeacoffee.com/byteotter).
You can also [follow ByteOtter on X](https://x.com/byteotter).

## License and attribution

This quantized derivative follows the base model’s
[Apache License 2.0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE).
See `LICENSE` and `NOTICE` for the included license and attribution notice.
