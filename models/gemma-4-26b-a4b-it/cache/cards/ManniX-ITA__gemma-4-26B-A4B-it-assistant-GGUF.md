---
license: gemma
base_model:
- google/gemma-4-26B-A4B-it
library_name: gguf
pipeline_tag: image-text-to-text
language:
- en
tags:
- draft-model
- gemma4
- gguf
- mtp
- opencoti
- speculative-decoding
model_name: gemma-4-26B-A4B-it-assistant-GGUF
author: ManniX-ITA
architecture: Gemma4
active_parameters: 4B
format: GGUF
quantizations:
- F16
- Q4_K_M
- Q4_K_S
- Q5_K_M
- Q8_0
features:
- MTP
- MoE
- draft-model
- speculative-decoding
- vision
---

# Gemma-4 26B-A4B IT Assistant — MTP draft head (llama.cpp `gemma4-assistant` format)

MTP / speculative-decoding **draft head** for `gemma-4-26B-A4B-it`, in the upstream
llama.cpp `gemma4-assistant` GGUF data-plane format (arch `gemma4-assistant`, with
`nextn.*` (+ `masked_embd_*` where applicable) tensors and `embedding_length_out`).

Use it as the drafter for self-speculative decoding against the matching
`gemma-4-26B-A4B-it` target:

```
--mtp-head gemma-4-26B-A4B-it-assistant.Q8_0.gguf --spec-type draft-assistant -ngld 99
```

**Quants:** F16, Q8_0, Q5_K_M, Q4_K_M, Q4_K_S.

Adapted from [AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF). Weights byte-identical to the source GGUFs — only the GGUF metadata/tensor names were rewritten to the upstream `gemma4-assistant` data-plane format. The Gemma license applies.
