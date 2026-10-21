---
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
base_model: ornith-ai/Ornith-1.5-35B-A3B
pipeline_tag: text-generation
tags:
- mlx
- quantization
- mtp
- mixture-of-experts
- apple-silicon
---

# Ornith-1.5-35B-A3B · MLX 8-bit (oQ8e) + MTP

MLX quantization of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — a Mixture-of-Experts model with ~3B active parameters — in 8-bit (`oQ8e`) with **Multi-Token Prediction** (MTP, speculative depth 3), built to run on Apple Silicon via a local oMLX server.

> [!IMPORTANT]
> This quantization was re-uploaded on **2026-09-04**; weights are complete and validated.

## Quantization details

- **Base model**: [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (`qwen3_5_moe` architecture, MoE with ~3B active parameters per token)
- **Quantizer**: [oQ](https://github.com/jundot/omlx) — oMLX v0.6.4, mixed-precision
- **Vision encoder preserved** — runs as a full VLM
- **Bits**: 8 · **Group size**: 64 · **Format**: MLX safetensors (~35 GB, 8 shards)
- **Report**: `oq_imatrix_report.json` in this repo

## Benchmarks on Apple Silicon (M1 Max 64 GB)

Measured on 2026-09-05 with the oMLX server on identical seed-fixed prompts (same-day runs): TurboQuant KV cache at 4 bits, DFlash2 speculative decoding (community draft [`jzinno/Ornith-1.5-35B-A3B-DFlash2`](https://huggingface.co/jzinno/Ornith-1.5-35B-A3B-DFlash2)), greedy decoding. Decode throughput, medians of two runs.

Decode speed (tokens/s) vs context length:

| Context | oQ4e (4-bit) | oQ6e (6-bit) | oQ8e (8-bit) |
|---|---|---|---|
| ~4k | 76–91 | 73–79 | 73–79 |
| ~8k | 77–78 | 75–78 | 75–76 |
| ~15k | 71–73 | 75–78 | 74–82 |
| ~23k | 49–72 | 72–73 | 66–69 |

- Speculative decoding acceptance: 87–92% on generation tasks (~+70% tokens/s vs plain autoregressive decoding)
- Quality probes: needle-in-a-haystack recall 4/4 with zero false positives on all three variants (same fixed codes, ~23k context); factual, math and Italian-language checks all correct
- Agentic checks (headless runs of the [pi agent](https://github.com/badlogic/pi-mono), verified programmatically): 6/6 completed — bug-fix (3 planted bugs, unittest suite passes: 46/42/58 s for oQ4e/oQ6e/oQ8e) and multi-file analysis (5 CSVs into a verified summary.json: 33/42/52 s)

Takeaway: from 4-bit to 8-bit the throughput stays in the same band and retrieval recall is identical. On 64 GB machines the 6-bit variant is the sweet spot.

## Sibling variants

- [Ornith-1.5-35B-A3B-oQ4e-mtp](https://huggingface.co/LookUpMark/Ornith-1.5-35B-A3B-oQ4e-mtp) — smallest footprint
- [Ornith-1.5-35B-A3B-oQ6e-mtp](https://huggingface.co/LookUpMark/Ornith-1.5-35B-A3B-oQ6e-mtp) — recommended on 64 GB machines
- [Ornith-1.5-35B-A3B-oQ8e-mtp](https://huggingface.co/LookUpMark/Ornith-1.5-35B-A3B-oQ8e-mtp) — highest fidelity

## Credits

All credit for the base model goes to the **Ornith team ([ornith-ai](https://huggingface.co/ornith-ai))**:

- Base model: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (MIT — see [LICENSE](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE)), which extends Ornith-1.0 (built on Qwen3.5 and Gemma4) via an end-to-end self-improvement loop
- Blog: [deep-reinforce.com/ornith.html](https://deep-reinforce.com/ornith.html) · [ornith.ai/ornith_1_5.html](https://ornith.ai/ornith_1_5.html)
- Quantization tooling: [oMLX](https://github.com/jundot/omlx)
- Speculative decoding draft: [`jzinno/Ornith-1.5-35B-A3B-DFlash2`](https://huggingface.co/jzinno/Ornith-1.5-35B-A3B-DFlash2)

This repo is only a quantization for local Apple Silicon inference. Same MIT terms apply.

## Intended use

Local inference on Apple Silicon Macs (oMLX / mlx-lm). For base-model benchmarks and details, see the upstream card.
