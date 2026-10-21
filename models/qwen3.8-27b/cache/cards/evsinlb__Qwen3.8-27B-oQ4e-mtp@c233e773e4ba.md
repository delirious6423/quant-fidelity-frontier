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

# Qwen3.8-27B-oQ4e-mtp

4-bit MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (hybrid Gated DeltaNet + Gated Attention, native vision, 262K context, thinking mode with preserved reasoning). The smallest build of the family: fits **32 GB-class Apple Silicon**, and on 64 GB machines it's the **long-context option** — the ~5 GB saved vs. the 6-bit goes to KV cache headroom under the default GPU wired-memory limit. Made with [oQ](https://github.com/jundot/omlx) (oMLX v0.6.0.dev1) mixed-precision quantization.

**Sibling repos:**
- [Qwen3.8-27B-oQ4e-fp16-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ4e-fp16-mtp) — 4-bit / fp16, ~16.7 GiB, tuned for M1/M2; ANE-eligible in oMLX 0.6.1 (+21% prefill measured on M2 Ultra)
- [Qwen3.8-27B-oQ6e-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ6e-mtp) — 6-bit / bf16, ~22 GB, for 64 GB-class machines
- [Qwen3.8-27B-oQ8e-fp16-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ8e-fp16-mtp) — 8-bit / fp16, ~27.5 GiB, highest fidelity
- [Qwen3.8-27B-oQ8e-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ8e-mtp) — 8-bit / bf16, ~27.9 GiB; same quantized tensors and imatrix as the oQ8e-fp16 build, bf16 floating tensors for M3/M4+ numerical safety

## Quantization details

- **Model type**: qwen3_5
- **Bits**: 4 (effective **\~4.9 bpw** mixed precision), group size 64 — \~17 GB weights
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

## Measured context windows (64 GB M5 Pro, oMLX, default wired-memory limit)

Nonce-cold ladder, 2026-08-14: **64K OK** (\~207s cold prefill), **131K OK** (\~457s), **160K crashed the oMLX server** (allocation spike during prefill; the server did not gracefully reject). Bisect found \~152K safe; **131,072 is the recommended client window** on 64 GB machines, leaving margin below the crash edge.

Two practical warnings: (1) real windows on this hybrid architecture are **prefill-overhead-bound, not KV-bound** — KV itself is cheap (only 16 of 64 layers are full attention, ~64 KB/token in bf16), but allocation peaks during long prefill run well above weights+KV, so don't size your window from KV math; (2) **cap the context client-side** — an over-window request may crash the server rather than return an error. The full 262,144 native window was validated with the 8-bit sibling on a Mac Studio.

## ANE prompt processing note (oMLX 0.6.2, experimental)

This q4 layout is eligible for oMLX's experimental ANE/GPU prefill split. The 0.6.2 built-in tuner, run on a 64 GB M5 Pro with this build, measured a best case of **+9.2% (498.6 prompt tok/s)** at MLP 35% / GDN 45% — modest, because single-die chips have one ANE (the dual-ANE M2 Ultra measured +37.6% at the same tuner shape with the fp16 sibling; see its card for full measurements). **On 64 GB machines we recommend leaving it off anyway**: the ANE path keeps ~4 GB extra resident, which comes out of exactly the headroom behind the 131,072 recommended window — and an over-margin prefill hard-crashes the server rather than returning an error. If you enable it, re-run a context ladder before trusting any large window. Real-world gains are also below the tuner figure, which uses an ideally block-aligned 4,097-token shape.

## Fidelity note

4-bit is the aggressive end for a 27B reasoning model; the imatrix calibration mitigates but does not eliminate low-bit degradation, so expect the largest quality gap of the family on long reasoning chains. That said, **this exact build passed the same multi-turn agentic battery as the 8-bit sibling** (2026-08-14, M2 Mac Studio): 3/3 scenarios completed with correct answers, 0 stalled turns across 8 agentic tool-call turns with thinking fed back into history, and thinking volume comparable to the 8-bit (~80% of its reasoning chars). The battery's scenarios are short; the caveat about very long reasoning chains stands. If your machine fits the 6-bit at your target context length, prefer it.
