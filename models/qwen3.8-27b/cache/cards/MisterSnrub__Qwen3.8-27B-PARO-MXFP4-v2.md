---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags: [mxfp4, paroquant, gptq, vllm, rocm, gfx1201]
---

# Qwen3.8-27B PARO-MXFP4 v2

Qwen3.8-27B in MXFP4 for one AMD Radeon AI PRO R9700 (32 GB). Needs GGZ14's vLLM-Radiance fork plus our patches
(`quant_method: paroquant_mxfp4`; stock vLLM and transformers can't load it). Setup:
[github.com/mtstanfield/vllm-mxfp4, branch `r9700-tp1`](https://github.com/mtstanfield/vllm-mxfp4/tree/r9700-tp1), `R9700-HANDOFF.md`. Serving image: [MisterSnrub/vllm-radiance-mxfp4-image](https://huggingface.co/MisterSnrub/vllm-radiance-mxfp4-image).

## Measured on one R9700 (TP=1, 300 W)

| | |
|---|---|
| Decode, 1 stream (BetterBench 0.6.0, weighted) | 102.3 tok/s (median over all tasks 108.1) |
| Prefill | 3,357 tok/s at 5.9k tokens, 3,000 at 47k |
| 4 concurrent streams | 278 tok/s total, 85 per stream |
| 8 concurrent streams (64k context variant) | 426 tok/s total |
| KV cache | 375,633 tokens fp8, max context 262,144 |
| Served perplexity (fp8 KV) | wikitext-2 5.970, code 2.304 |
| KL vs bf16 on agent transcripts, vs plain round-to-nearest MXFP4 | 64-68% lower |

MTP speculative decoding with 4 draft tokens. Full BetterBench report and setup: `R9700-HANDOFF.md` on the branch.

## Recipe

- Decoder linears: MXFP4 (e2m1, e8m0 scale per 32) in a rotated basis, using [ParoQuant](https://github.com/z-lab/paroquant)
  pairwise rotations per 128-channel group, retrained for the MXFP4 grid from z-lab's int4 rotations.
- GPTQ onto the MXFP4 grid in the rotated basis, with per-block scale search.
- Calibration: 70% coding-agent transcripts (private) + 30% wikitext-2 train.
- `lm_head` fp8 per-channel, `embed_tokens` fp8 per-row, MTP head fp8. `extras/qwen38-paro-v2-mtp-gptq-a05.pt` is the MTP head
  requantized to MXFP4 with GPTQ (goes in `$CACHE/mtp_gptq/`).
- Served as W4A8: per-token fp8 activations.

Credits: Qwen (base model), z-lab (ParoQuant), GGZ14 (vLLM-Radiance), StillDeadcode (libr4d).
