---
license: other
license_name: qwen-community
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-27B
- catonooka/Qwen3.8-27B-Heretic-Abliterated
library_name: exllamav3
tags:
- exl3
- abliterated
- uncensored
- qwen3.8
- quantized
---

# Qwen3.8-27B-Heretic-Abliterated-EXL3-4.0bpw-h8

EXL3 quantization (4.0 bpw, head_bits 8) of
[catonooka/Qwen3.8-27B-Heretic-Abliterated](https://huggingface.co/catonooka/Qwen3.8-27B-Heretic-Abliterated)
(bf16 source). See that card for the abliteration method, trial parameters and evals.

## Quant recipe

- exllamav3 converter, bits 4.0, head_bits 8, embed_bits 16, codebook mul1
- mtp_bits 4 (MTP head quantized in-pack for speculative decode), vision_bits 0 (vision
  tower dropped in this pack; use the bf16 repo if you need vision)
- cal 250 rows x 2048 cols, hessians_reg 0.025, shard 8192
- Wall clock on 4x RTX 3090: 42 min, 672 modules

## Speculative decode acceptance (DSpine 5-layer drafter, EXL3 5.0bpw-h8, pretrained on STOCK)

Same drafter that serves the stock model, no retraining:

| config | accept | t/s |
|---|---|---|
| story think ndt7 | 26.0% (stock pack baseline: 19.2%) | 57.9 |
| code think ndt7 | 73.0% | 121.6 |
| code think ndt15 | 47.7% | 155.2 |

Abliteration did not hurt draft acceptance; the gate to retrain the drafter never triggered.

## Thinking levels (same prompt, temp 0)

All four template levels work on this build: xhigh 383 think-words, medium 79, low 57,
off 0 (no leakage, clean prose). Control thinking ONLY via chat-template kwargs
(`enable_thinking: false`, `reasoning_effort: xhigh|medium|low`). Hard-banning the
`<think>` token in a raw prompt makes this build stop instantly.

## Loader

exllamav3 (`Config.from_directory`). Standard EXL3 pack layout with tokenizer and
quantization_config included.
