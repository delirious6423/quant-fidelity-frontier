---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- oq
- quantized
- qwen3.8
---

# Qwen3.8-27B-oQ6e-mtp

6-bit MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (hybrid Gated DeltaNet + Gated Attention, native vision, 262K context, thinking mode with preserved reasoning), sized for **64 GB-class Apple Silicon**. Made with [oQ](https://github.com/jundot/omlx) (oMLX v0.6.0.dev1) mixed-precision quantization.

**Sibling repos:**
- [Qwen3.8-27B-oQ4e-fp16-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ4e-fp16-mtp) — 4-bit / fp16, ~16.7 GiB, tuned for M1/M2; ANE-eligible in oMLX 0.6.1 (+21% prefill measured on M2 Ultra)
- [Qwen3.8-27B-oQ8e-fp16-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ8e-fp16-mtp) — 8-bit / fp16, ~27.5 GiB, higher fidelity if you have the memory
- [Qwen3.8-27B-oQ4e-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ4e-mtp) — 4-bit / bf16, ~17 GB, for 32 GB-class machines or long-context use on 64 GB
- [Qwen3.8-27B-oQ8e-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ8e-mtp) — 8-bit / bf16, ~27.9 GiB; same quantized tensors and imatrix as the oQ8e-fp16 build, bf16 floating tensors for M3/M4+ numerical safety

## Quantization details

- **Model type**: qwen3_5
- **Bits**: 6 (effective **6.8 bpw** mixed precision), group size 64 — ~22 GiB weights, leaving KV-cache headroom under a 64 GB machine's default GPU-memory limit
- **Enhanced quantization (oQe)**: imatrix-calibrated (1024 samples) — affine quantization weighted by activation importance
- **MTP weights preserved** (`mtp.*` tensors + config) — multi-token-prediction / Lightning MTP works after quantization
- **Non-quant weight dtype**: bfloat16 (matches the base model; the safe choice on M3/M4/M5)
- **Vision components included** (not a text-only strip)
- **Format**: MLX safetensors

## Recommended sampling (per the Qwen3.8 model card)

| Mode | temperature | top_p |
|---|---|---|
| Thinking (default) | 1.0 | 0.95 |
| Instruct (non-thinking) | 0.7 | 0.80 |

Thinking controls via `chat_template_kwargs`:

- `enable_thinking` (default `true`)
- `preserve_thinking` (default `true`) — keeps reasoning traces across multi-turn history
- `reasoning_effort`: `xhigh` (default) / `medium` / `low` — in our testing (on the 8-bit sibling), `medium` reduced thinking volume ~25% with no loss on agentic tasks

## Tested

The 8-bit sibling of this quant was validated 2026-08-14 on an M2 Mac Studio: 0 stalled turns in multi-turn agentic tool use with thinking ON and thinking blocks fed back into history (synthetic battery + a 12-turn live session at 27 messages of history) — see the sibling repo's card for details. This 6-bit build uses the same imatrix calibration; it has not yet been independently run through the same battery.
