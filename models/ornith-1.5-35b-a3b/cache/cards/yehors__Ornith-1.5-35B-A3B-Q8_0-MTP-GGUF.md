---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - mtp
  - speculative-decoding
  - moe
---

# Ornith-1.5-35B-A3B — Q8_0 GGUF with MTP head

Q8_0 GGUF quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
that **keeps the NextN / MTP prediction head** (`blk.40`), so self-speculative decoding runs against the file
itself via `--spec-type draft-mtp`.

- **Base:** ornith-ai/Ornith-1.5-35B-A3B (MIT), ~35B MoE, ~3B active per token, hybrid SSM + attention, 256K context.
- **This file:** `Ornith-1.5-35B-Q8_0-MTP.gguf`, ~36 GB, 8.52 BPW. Weights are Q8_0 (near-lossless vs BF16);
  the MTP head is quantized to Q8_0 as well.

## Why this exists

Stock `llama-quantize` rejects the extra `blk.40` (NextN) block for the `qwen35moe` architecture
(`Bad layer 40 ... Must be in [0, 40)`), so the official uniform Q8 and other tools drop the MTP head.
This quant was produced with a one-line local patch that treats the NextN block as the final layer during
type selection, preserving `blk.40.nextn.*` so speculative decoding stays available at Q8.

## Run (llama.cpp)

```bash
llama-server \
  -m Ornith-1.5-35B-Q8_0-MTP.gguf \
  -ngl 999 --flash-attn on \
  --ctx-size 65536 -b 2048 -ub 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --reasoning-format deepseek --reasoning-budget 0 \
  --jinja
```

Sampling (coding): `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0`.

Note: the base model is a hybrid (SSM + attention) model, so `--cache-reuse` does not take effect
(prompts are reprocessed each turn).

## Measured (AMD Radeon 8060S iGPU, Vulkan, ~140 GB/s)

| test | t/s |
|------|-----|
| pp512 | ~936 |
| tg128 (no speculation) | ~47.6 |

With `--spec-type draft-mtp` on top, TG is higher depending on draft acceptance. No speculative throughput
benchmark is published here.

## License

MIT, inherited from the base model. Credit: ornith-ai for the base model.
