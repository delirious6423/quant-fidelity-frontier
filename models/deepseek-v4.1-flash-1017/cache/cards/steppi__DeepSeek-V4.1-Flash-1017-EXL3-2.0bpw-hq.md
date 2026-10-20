---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags: [exl3, exllamav3, deepseek, moe, quantized, vision, 2-bit]
---

# DeepSeek-V4.1-Flash-1017 — EXL3 2.0 bpw (hq) + vision

EXL3 quantization of `deepseek-ai/DeepSeek-V4.1-Flash-1017` (rev 1bae0e10) sized to fit **8×RTX 3090 24 GB**
(~139 GiB, 262k context, vision working).

- 2.0 bpw target (`-b 2.0 -hq`), head 6 bpw, MTP head 4 bpw, vision BF16, 250×2048 calibration rows.
- Engram tables are kept in their original FP8 (not quantized).
- Requires the vcruz305/exllamav3 fork with V4.1 support + the patches linked below (upstream exllamav3 does not support V4.1).

## Measured (8×RTX 3090, TabbyAPI, ctx 262144, no drafter)
- Prefill ~640 t/s @ 14.7k tokens (native), decode ~22-23 tok/s.
- Vision end-to-end tests 5/5 (OCR, counting, object recognition).
- No perplexity/KLD or benchmark numbers yet.

## Known limitations
- Layers 10-13 were calibrated with a local window only; ~8 experts of ~15k received no calibration data.
- `max_batch_size` must be 1; speculative drafter (DSpark) is slow at long context and was left off.

## Credits
DeepSeek-AI (base, MIT), turboderp (exllamav3), vcruz305 (V4.1 fork), Mia-AiLab (chat template).
