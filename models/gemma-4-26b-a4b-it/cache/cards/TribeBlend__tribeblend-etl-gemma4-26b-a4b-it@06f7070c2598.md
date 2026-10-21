---
license: other
language:
  - en
tags:
  - text-to-sql
  - data-chat
  - gguf
  - google
  - gemma4
base_model: google/gemma-4-26B-A4B-it
---

# tribeblend-etl-gemma4-26b-a4b-it

Gemma 4 26B A4B IT adapted for high-quality local analyst workflows with about 4B active parameters per token.

Direct **base-model** GGUF (Q4_K_M) of `google/gemma-4-26B-A4B-it`, published for
TribeBlend's local Data Chat runtime. TribeBlend grounds answers with Knowledge
Graph context at prompt time and a model-aware agent harness, so the base
instruction/reasoning model ships as-is (no fine-tuning).

- **Base model**: google/gemma-4-26B-A4B-it
- **Provider / family**: google / gemma4
- **Local runtime arch**: gemma4
- **Recommended profile**: expert
- **Quantization**: Q4_K_M
- **Native context window**: 262144

## Usage

Designed for [TribeBlend](https://github.com/TribeBlend/tribeblend) Data Chat,
loaded via llama-cpp-2.

## License

Inherits the upstream base-model license (other); verify upstream
terms before redistribution.
