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

# Qwen3.8-27B-oQ8e-mtp

8-bit MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (hybrid Gated DeltaNet + Gated Attention, native vision, 262K context, thinking mode with preserved reasoning). Made with [oQ](https://github.com/jundot/omlx) (oMLX v0.6.4) mixed-precision quantization.

This is the **bf16 sibling** of [Qwen3.8-27B-oQ8e-fp16-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ8e-fp16-mtp): identical quantized tensors and calibration, with the non-quantized (floating) tensors left in bfloat16 instead of converted to float16. Pick this one on **M3/M4** or wherever numerical safety matters more than M1/M2 fp16 prefill speed; pick the fp16 build on M1/M2 if you want the oMLX ANE + CPU work-sharing path.

**Sibling repos:**
- [Qwen3.8-27B-oQ8e-fp16-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ8e-fp16-mtp) — 8-bit / fp16, ~28.8 GiB; same recipe, fp16 floating tensors for M1/M2 and the oMLX CPU-sharing prefill path (measured +19.5% prefill at 64K on an M2 Ultra)
- [Qwen3.8-27B-oQ6e-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ6e-mtp) — 6-bit / bf16, ~22 GB, for 64 GB-class Apple Silicon
- [Qwen3.8-27B-oQ4e-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ4e-mtp) — 4-bit / bf16, ~17 GB, for 32 GB-class machines or long-context use on 64 GB
- [Qwen3.8-27B-oQ4e-fp16-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ4e-fp16-mtp) — 4-bit / fp16, ~16.7 GiB, tuned for M1/M2 ANE prefill

## Quantization details

- **Model type**: qwen3_5
- **Bits**: 8, group size 64, affine, uniform across quantized tensors (no per-tensor overrides) — **27.94 GiB** of safetensors measured (quantized weights + bf16 floating tensors + MTP + vision)
- **Enhanced quantization (oQe)**: imatrix-calibrated — affine quantization weighted by activation importance. The imatrix is the same one collected for the fp16 sibling (same source checkpoint, reused from oQ's cache), so the two builds share importance weights exactly.
- **MTP weights preserved** (`mtp.*` tensors + config) — multi-token-prediction / Lightning MTP works after quantization
- **Non-quant weight dtype**: bfloat16 — the source model's native dtype; safer on M3/M4. (The fp16 sibling trades this for ~20% faster prefill on M1/M2, where fp16 is native.)
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

The quantized tensors are byte-identical to the fp16 sibling's; only the floating tensors differ in dtype. The behavioral validation below was run on that sibling (2026-08-14 / 2026-08-27, M2 Mac Studio, served via an OpenAI/Anthropic-compatible gateway) and is expected to carry over: <!-- If you re-run the battery on this artifact, replace this paragraph with the date and results. -->

- **Multi-turn agentic tool use with thinking ON and thinking blocks fed back into history**: 0 stalled turns across a 3-scenario synthetic battery (~8 tool-call turns) and a 12-turn live session reaching 27 messages of history — tool-call emission stayed reliable throughout. This is a notable improvement over Qwen3-generation models, where returning thinking blocks in history degraded tool calling; Qwen3.8's preserved-thinking design handles it.
- Reasoning depth flexes with task difficulty (observed \~100 chars of thinking on routine turns, \~4,000 on decision-heavy turns).
- Thinking mode costs roughly 25–35% wall time vs. instruct mode on agentic workloads.
- Clean `reasoning_content` separation on `/v1/chat/completions`; proper thinking + `tool_use` blocks over an Anthropic-style `/v1/messages` bridge.
- **Context**: validated at the full native 262,144 on the M2 Mac Studio (oMLX built-in context test); client deployments use a 229,376 window (262,144 − 32,768 output headroom). On smaller machines, prefill-time allocation — not KV — is the binding constraint; see the 4-bit sibling's card for measured 64 GB numbers.

## Performance (measured on this artifact)

**M5 Pro MacBook Pro (20-core GPU, 64 GB), oMLX 0.6.4, built-in benchmark (Code/Python context), Lightning MTP on, ANE off:**

| Test | TTFT (ms) | pp tok/s | tg tok/s | Peak mem |
|---|---:|---:|---:|---:|
| pp1024/tg128 | 2,532 | 404.5 | 14.0 | 33.7 GB |
| pp4096/tg128 | 9,573 | 427.9 | 10.9 | 35.2 GB |
| pp16384/tg128 | 43,510 | 376.6 | 18.6 | 37.2 GB |

Continuous batching (pp1024/tg128): 2x = 18.6 aggregate tg tok/s (1.33x), 4x = 33.4 (2.39x). Decode is bandwidth-bound single-stream, so concurrent requests recover real throughput; note Lightning MTP is a single-stream feature and doesn't stack with batching.

Notes:
- The tg spread (10.9-18.6) across depths is MTP-acceptance variance on single runs, not a context trend.
- Peak memory stays under the macOS default \~38 GB wired limit through 16K on a 64 GB machine; plan a wired-limit raise (or quantized KV) beyond \~32K, where model + KV (\~230 KB/token) passes it.
- For scale: the fp16 sibling (same quantized tensors) measured 307/313/231 pp and 37/34/26.6 tg at 1K/4K/64K on an M2 Ultra Mac Studio (128 GB, GPU only, oMLX 0.6.3) with 1.5-2.1x of that decode coming from Lightning MTP. The M5's *higher* prefill on a much smaller GPU reflects the newer GPU architecture; its lower decode tracks its ~1/3 memory bandwidth.

## ANE prompt processing (oMLX 0.6.3+)

oMLX's experimental Qwen ANE prefill path is bit-aware affine and works on q8 layouts, so this build is ANE-eligible for the **ANE portion** of the split. The **CPU work-sharing branch** ("Share MLP work with CPU") requires fp16 floating tensors and is only available on the fp16 sibling — the +19.5% at 64K measured there includes that branch, so expect less here. Two things to know before enabling it on either build:

- **Memory cost is large**: on the M2 Ultra the fp16 sibling loaded at 55.7 GB with the full ANE + CPU-sharing stack resident vs 28.5 GB with it off. On a machine that also hosts other models or a prefix cache, that difference is the whole headroom.
- **Prompts at or under ~1K gain nothing** and decode is untouched; the gain is cold-prefill only at depth.

If you want the ANE path, the fp16 sibling's card carries the measured split and tuner settings.
