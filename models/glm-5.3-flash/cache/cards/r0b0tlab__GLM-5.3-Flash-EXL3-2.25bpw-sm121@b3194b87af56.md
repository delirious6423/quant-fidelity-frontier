---
license: mit
base_model: zai-org/GLM-5.3-Flash
library_name: exllamav3
pipeline_tag: text-generation
inference: false
tags:
- base_model:zai-org/GLM-5.3-Flash
- base_model:quantized:zai-org/GLM-5.3-Flash
- exl3
- exllamav3
- quantization
- moe
- dgx-spark
- gb10
- sm121
- speculative-decoding
- dflash2
- long-context
language:
- en
---

# GLM-5.3-Flash EXL3 2.25bpw (GB10 / SM121 runtime)

EXL3 (ExLlamaV3 trellis) quantization of
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)
(320B total / ~18B active MoE) by **r0b0tlab**.

- **Format:** EXL3, ~2.25 bpw average, `mul1` codebook, mixed K3/K4 tiles,
  output scales on, 6-bit head.
- **Size:** 98.5 GB (91.8 GiB) across 31 safetensors shards, plus
  `quantization_config.json` (full module/tensor manifest) and tokenizer
  files.
- **Calibration:** mixed corpus, tap-K3 selection.

## Verified 256K long-context result (ExLlamaV3 runtime, 2026-09-27)

On a single GB10, our standalone ExLlamaV3 + TabbyAPI runtime
([`r0b0tlab/glm53-flash-exl3-exllamav3-gb10`](https://github.com/r0b0tlab/glm53-flash-exl3-exllamav3-gb10))
with our EXL3 DFlash2 draft
([`r0b0tlab/GLM-5.3-Flash-DFlash2-EXL3-3.00bpw`](https://huggingface.co/r0b0tlab/GLM-5.3-Flash-DFlash2-EXL3-3.00bpw))
served this pack at **262,144 tokens (FP16 KV, batch size 1, DFlash2 K=5)**
and passed a token-exact needle-in-a-haystack retrieval at
**259,993 actual prompt tokens** — clean answer, exact client/server token
parity, guarded by an independent 6,144-MiB host-memory floor with no NVRM
allocation failure.

Not qualified on that profile: multi-key output formatting, concurrency > 1,
throughput at 256K, long-run stability. Details and sanitized evidence live in
the runtime repository.

## Serving

Two runtimes are available for this pack:

1. **ExLlamaV3 / TabbyAPI (256K verified):**
   [`r0b0tlab/glm53-flash-exl3-exllamav3-gb10`](https://github.com/r0b0tlab/glm53-flash-exl3-exllamav3-gb10) —
   native ARM64 CUDA 13 container, guarded 256K serve.
   Published image (ARM64/SM121):
   `docker pull ghcr.io/r0b0tlab/glm53-flash-exl3-exllamav3-gb10:b25c467`.

2. **vLLM (32,768-token published profile):**
   [`r0b0tlab/glm53-flash-exl3-dflash2-sm121`](https://github.com/r0b0tlab/glm53-flash-exl3-dflash2-sm121) —
   host-native vLLM v0.30.0rc1 + EXL3 plugin; DFlash2 K=5 measured
   structured 50.2 / code 43.4 / prose 19.5 tok/s single-stream on one GB10.
   The two engines' results are independent; do not transfer between them.

## Q200v2 quality kit on the ExLlamaV3 runtime (2026-09-27)

Text-180: **168/180 (93.33%)** — GSM8K 98.75%, HumanEval 97.5%, IFEval 77.5%,
hard_reasoning 95.0%. BFCL-hard20: 6/20. **Full 200: 174/200 (87.0%)**.
Serial execution, effort=low, zero transport errors. (The vLLM runtime's
2026-09-21 kit on its own 32K profile: 170/180 text, 10/20 BFCL — engines are
independent; scores do not transfer.) Evidence:
`evidence/glm53-exl3-q200v2-20260927/` in the ExLlamaV3 runtime repo.

## Known limits

- Single GB10, TP=1. Vision verified by image Q&A smoke (image requests are
  drafted text-only under DFlash2).
- C≥2 tool-call loads not exercised on the 256K profile.

## Provenance

Quantized by r0b0tlab from `zai-org/GLM-5.3-Flash` (MIT). Weights unchanged
beyond quantization. EXL3 format follows exllamav3 (MIT).
