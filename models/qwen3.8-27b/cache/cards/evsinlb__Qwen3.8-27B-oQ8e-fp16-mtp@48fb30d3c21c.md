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

# Qwen3.8-27B-oQ8e-fp16-mtp

8-bit MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (hybrid Gated DeltaNet + Gated Attention, native vision, 262K context, thinking mode with preserved reasoning). Made with [oQ](https://github.com/jundot/omlx) (oMLX v0.6.0.dev1) mixed-precision quantization.

**Sibling repos:**
- [Qwen3.8-27B-oQ4e-fp16-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ4e-fp16-mtp) — 4-bit / fp16, ~16.7 GiB, tuned for M1/M2; first ANE-eligible build of the family (oMLX 0.6.1, +21% prefill on M2 Ultra). As of oMLX 0.6.3 this 8-bit build is ANE-eligible too — see the measured section below.
- [Qwen3.8-27B-oQ6e-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ6e-mtp) — 6-bit / bf16, ~22 GB, for 64 GB-class Apple Silicon
- [Qwen3.8-27B-oQ4e-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ4e-mtp) — 4-bit / bf16, ~17 GB, for 32 GB-class machines or long-context use on 64 GB
- [Qwen3.8-27B-oQ8e-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ8e-mtp) — 8-bit / bf16, ~27.9 GiB; same quantized tensors and imatrix as the oQ8e-fp16 build, bf16 floating tensors for M3/M4+ numerical safety

## Quantization details

- **Model type**: qwen3_5
- **Bits**: 8 (effective **8.6 bpw** mixed precision), group size 64 — ~27.5 GiB weights
- **Enhanced quantization (oQe)**: imatrix-calibrated (1024 samples) — affine quantization weighted by activation importance
- **MTP weights preserved** (`mtp.*` tensors + config) — multi-token-prediction / Lightning MTP works after quantization
- **Non-quant weight dtype**: float16 — ~20% faster prefill on M1/M2 Apple Silicon (native fp16). On M3/M4, prefer the bf16 sibling repo for numerical safety.
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
- `reasoning_effort`: `xhigh` (default) / `medium` / `low` — in our testing, `medium` reduced thinking volume ~25% with no loss on agentic tasks

## Tested

Validated 2026-08-14 on an M2 Mac Studio (this exact quant), served via an OpenAI/Anthropic-compatible gateway:

- **Multi-turn agentic tool use with thinking ON and thinking blocks fed back into history**: 0 stalled turns across a 3-scenario synthetic battery (~8 tool-call turns) and a 12-turn live session reaching 27 messages of history — tool-call emission stayed reliable throughout. This is a notable improvement over Qwen3-generation models, where returning thinking blocks in history degraded tool calling; Qwen3.8's preserved-thinking design handles it.
- Reasoning depth flexes with task difficulty (observed ~100 chars of thinking on routine turns, ~4,000 on decision-heavy turns).
- Thinking mode costs roughly 25–35% wall time vs. instruct mode on agentic workloads.
- Clean `reasoning_content` separation on `/v1/chat/completions`; proper thinking + `tool_use` blocks over an Anthropic-style `/v1/messages` bridge.
- **Context**: validated at the full native 262,144 on the M2 Mac Studio (oMLX built-in context test); client deployments use a 229,376 window (262,144 − 32,768 output headroom). On smaller machines, prefill-time allocation — not KV — is the binding constraint; see the 4-bit sibling's card for measured 64 GB numbers.

## Measured: ANE prompt processing (oMLX 0.6.3)

oMLX 0.6.3 extended its experimental Qwen ANE prefill path to q6/q8 layouts (bit-aware affine, replacing the 0.6.1/0.6.2 INT8 approximation that was q4-only), so this build is now ANE-eligible — earlier revisions of this card correctly called it ineligible; that ended with 0.6.3. The 0.6.3 path also adds an FP16 CPU branch (AMX work sharing), which this fp16 variant feeds natively.

Measured on an **M2 Ultra Mac Studio** (128 GB), oMLX 0.6.3, built-in benchmark (Code/Python context, ANE-aligned prompts **off** — i.e. realistic ragged prompt lengths), one run per leg, same day, identical settings apart from ANE:

| Prompt depth | GPU only (pp tok/s) | ANE + CPU split (pp tok/s) | Gain |
|---|---:|---:|---:|
| 1,024 | 307.3 | 306.8 | 0% |
| 4,096 | 313.3 | 425.0 | +35.7% |
| 65,536 | 231.2 | 276.2 | **+19.5%** |

Split (found by the 0.6.3 Tuner V2, dual-ANE machine): MLP 50% on ANE, GDN 37.5% on ANE (48-layer limit), CPU shares 10% gate / 20% down-projection / 10% GDN with 8 workers, tail padding from 1,436 tokens. The tuner's own shape measured +42.7% (438.3 pp tok/s); the gain compresses with depth as the non-ANE full-attention layers claim a growing share of prefill — same pattern as the 4-bit sibling.

Notes before enabling:

- **Prompts at or under ~1K gain nothing** — they never fill the 2,048-token ANE program block and ride the GPU path unchanged.
- **Decode is untouched at depth** (26.6 → 26.4 tg tok/s at 64K). One watch item: our single 1K leg showed ~8% slower decode with ANE resident — beyond typical ±3.5% run variance but unreplicated; re-measure on your machine if short-turn latency matters.
- **Cost: ~16 GB extra resident memory** for the model's lifetime (measured peak 44.0 → 60.0 GB at 64K) — more than the 4-bit sibling's ~11 GB on the same path. On machines without clear headroom above the model + KV, leave it off.
- The gain applies to **cold prefills only**; prefix-cached turns are unaffected, and 0.6.3 itself improved prefix-cache reuse in long agentic sessions.
- The first request after a load carries extra warm-up on top of the eager ANE compile; subsequent requests match steady state.
- **Quality-checked with ANE on** (2026-08-27, this machine): the family's multi-turn agentic battery passed 3/3 scenarios, 0 stalled turns across 8 tool-call turns, thinking fed back into history verbatim. The battery is short-context; 0.6.3's recurrent-safe ANE design (token-local rows only on the approximate path) addresses the earlier long-prompt divergence reports upstream.
