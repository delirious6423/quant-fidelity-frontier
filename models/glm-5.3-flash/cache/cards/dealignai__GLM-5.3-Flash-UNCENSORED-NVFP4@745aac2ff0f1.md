---
license: mit
base_model:
- zai-org/GLM-5.3-Flash
language:
- en
tags:
- abliterated
- uncensored
- crack
- glm
- moe
- nvfp4
- vision
- mtp
thumbnail: dealign_mascot.png
---

<!-- dealignai-serving-warning-v1 -->
> [!IMPORTANT]
> **Serving note — read before your first request (vLLM / SGLang with a reasoning parser).**
>
> - **`reasoning_effort` only accepts `"low"` and `"high"`.** Anything else — `medium`, `xhigh`,
>   `off`, or omitting it — silently resolves to **`max`**, the deepest mode. Template line:
>   `reasoning_effort ... in ['low','high'] else 'max'`.
> - **At `max`, a small `max_tokens` returns an EMPTY reply.** The model spends the whole budget
>   inside `<think>` and you get `finish_reason="length"` with `content=""`. In multi-turn the empty
>   turn poisons the history and the conversation does not recover. Measured: `max` + `max_tokens
>   2000` dies from turn ~4; `max` + `6000` is clean; `low` and `high` are clean at 2000.
>   **It is budget exhaustion, not a loop** — sampling changes do not rescue it.
> - **Do NOT pass `enable_thinking`.** This template has no such variable (only `clear_thinking` and
>   `reasoning_effort`). Passing it — or a top-level `reasoning_effort: "none"` — turns the parser off
>   while the model still thinks, dumping raw reasoning into `content`.
> - **Reasoning is returned in `message.reasoning`**, not `message.reasoning_content`
>   (vLLM renamed the output field; the input side still accepts both).
> - `clear_thinking` must be nested inside `chat_template_kwargs` — a top-level key is ignored.
>
> ```json
> {
>   "model": "<this-model>",
>   "messages": [{"role": "user", "content": "..."}],
>   "max_tokens": 2000,
>   "temperature": 1.0, "top_p": 0.95,
>   "chat_template_kwargs": {"reasoning_effort": "low", "clear_thinking": true}
> }
> ```
> Use `"high"` for hard tasks; if you want `max`, give it `max_tokens >= 6000`.
>
> <!-- dealignai-serving-warning-v2-media -->
> **Video and image parameters — measured, because several are accepted then silently ignored.**
>
> - 🔴 **`media_io_kwargs.video.fps` must stay BELOW the clip's own frame rate.** Requesting a value
>   at or above it makes the placeholder builder count 3x the real tokens and **kills the engine**:
>   `ValueError: Attempted to assign N = N multimodal tokens to 3N placeholders` -> `EngineDeadError`,
>   server down until restarted. Upstream: vLLM #55644 / #55647. `fps: 2` is the safe, accurate choice.
> - **`max_frames` and `num_frames` are not interchangeable across builds.** On vLLM nightly
>   `max_frames` is honoured and `num_frames` ignored; on the older `0.1.dev*` line it is the exact
>   reverse. A client that sets only one silently gets full-rate sampling on the other build. Set
>   `fps`, which every build honours.
> - **Pixel controls: use `mm_processor_kwargs.max_image_tokens` / `min_image_tokens`.**
>   `max_pixels`, `min_pixels`, `size` and `detail` return HTTP 200 and change nothing.
>   `mm_processor_kwargs.fps` is an HTTP 400. Image tokens are
>   `text + 2 + ceil(H/28)*ceil(W/28)`, floored at 16 and capped at 8000.
> - Default video sampling differs by build (roughly 2 fps vs 6 fps), so the same clip can cost very
>   different prompt tokens. Pass `fps` explicitly if token cost matters.
>
> ```json
> "media_io_kwargs": {"video": {"fps": 2}},
> "mm_processor_kwargs": {"max_image_tokens": 1024}
> ```


> ## ⚠️ Updated 2026-08-29 — files fixed · please re-download
>
> This release **supersedes the previous weights.** It fixes a rare **repetition-loop** issue and
> lands MMLU **85.28%** (within ~0.9 pt of the base model — quality intact).
>
> **Reasoning-mode note (please read):** GLM-5.3-Flash has multiple reasoning-effort modes. This
> CRACK is tuned to be **fully uncensored at reasoning-OFF and at MAX effort — the recommended and
> default modes — with 0% refusals on HarmBench-320.** We deliberately kept the weight edit
> **conservative to preserve model quality** instead of aggressively ablating every effort level;
> **low effort** retains some refusals **by design** (aggressively removing them costs quality).
> Use reasoning-off or default/max effort for the fully-uncensored experience. Please
> **re-download** for the corrected files.

<div align="center">
<img src="dealign_mascot.png" width="140" />

# GLM 5.3 CRACK Uncensored NVFP4

**CRACK** · guardrails removed at the weight level · vision tower + MTP head retained

a **CRACK** release by [dealignai](https://huggingface.co/dealignai) · Twitter [@dealignai](https://twitter.com/dealignai)

*Also mirrored at [dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4](https://huggingface.co/dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4).*

</div>

---

## What Is This?

**CRACK** is dealignai's brand for permanent, weight-level uncensoring. This is
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) with its refusal behavior — which
caused heavy over-refusal, especially on copyright and other benign-but-flagged requests — removed
**directly in the model weights**.

**This is genuine weight modification — none of the usual shortcuts:**

- ❌ **No fine-tuning / SFT / DPO** — the model was not retrained on anything.
- ❌ **No cheap template tricks** — no jailbreak system prompt, no chat-template edits, no "sure, here is" prefixing.
- ❌ **No LoRA, no adapters, no steering vectors, no runtime hooks, no custom `model.py`.**
- ✅ **A permanent edit baked into the tensors.** Load it with stock vLLM and it just works.

## Specs

| | |
|---|---|
| **Architecture** | GLM-5.3-Flash (`glm5_next`) — hybrid MoE (KDA linear + DeepSeek-sparse attention) |
| **Parameters** | **320B total · 18B active** per token |
| **Quantization** | **NVFP4** (routed experts NVFP4; attention + shared experts + embeddings bf16) |
| **Context** | 1M tokens |
| **Vision** | GLM-4.1V vision tower — **retained, byte-for-byte identical to base** |
| **MTP** | multi-token-prediction draft head — **also CRACK'd** (81.7% acceptance) |
| **Reasoning** | reasoning-off / low / high / max effort — see the compliance table below |

## MTP Is Also CRACK'd

The **MTP (multi-token prediction) speculative-decoding draft head is CRACK'd too** — not just the
main model. The draft head will not propose refusals, so speculative decoding stays **compliant *and*
fast** on the exact prompts a stock model would refuse.

## Reasoning Modes — Compliance

GLM-5.3-Flash supports several reasoning-effort modes. Guardrail removal is strongest in the modes
people actually use by default. HarmBench-320, greedy (temperature 0), measured per mode:

| Reasoning mode | Refusals | Notes |
|---|---|---|
| **Reasoning-off** | **0%** | fully uncensored |
| **Max effort (default)** | **0%** | fully uncensored |
| **High effort** | ~4% | complies on all but the most extreme safety cases |
| Low effort | ~9% | **intentionally left conservative to preserve quality** |

**0 degenerate / looping outputs in every mode.** The design choice: keep the ablation light enough
that capability (MMLU) stays essentially at base, rather than over-ablating to force the rarely-used
low-effort mode. **For a fully uncensored model, use reasoning-off or the default/max effort mode.**

*These rates are **greedy decoding (temperature 0)** — the strict worst case. Under the model's
**recommended sampling** (temperature 1.0, top_p 0.95) the model is at least as compliant.*

## Capability Is Preserved — MMLU-logit

Identical logit-mode scoring (argmax over A/B/C/D) on base vs. this model, 1,026 questions:

| | Base | CRACK Uncensored | Δ |
|---|---|---|---|
| **MMLU (overall)** | **86.16%** | **85.28%** | **-0.88 pp** |

A sub-1-point delta — reasoning and knowledge are intact.

## MMLU by Topic (base → CRACK)

<details>
<summary>All 57 MMLU subjects</summary>

| Subject | Base | CRACK |
|---|---|---|
| Abstract Algebra | 55.6% | 61.1% |
| Anatomy | 88.9% | 94.4% |
| Astronomy | 94.4% | 94.4% |
| Business Ethics | 94.4% | 94.4% |
| Clinical Knowledge | 88.9% | 88.9% |
| College Biology | 94.4% | 94.4% |
| College Chemistry | 44.4% | 50.0% |
| College Computer Science | 88.9% | 88.9% |
| College Mathematics | 72.2% | 66.7% |
| College Medicine | 88.9% | 88.9% |
| College Physics | 83.3% | 83.3% |
| Computer Security | 83.3% | 83.3% |
| Conceptual Physics | 94.4% | 94.4% |
| Econometrics | 83.3% | 83.3% |
| Electrical Engineering | 83.3% | 77.8% |
| Elementary Mathematics | 100.0% | 94.4% |
| Formal Logic | 66.7% | 61.1% |
| Global Facts | 61.1% | 61.1% |
| High School Biology | 94.4% | 94.4% |
| High School Chemistry | 88.9% | 88.9% |
| High School Computer Science | 100.0% | 100.0% |
| High School European History | 72.2% | 72.2% |
| High School Geography | 88.9% | 83.3% |
| High School Government And Politics | 94.4% | 94.4% |
| High School Macroeconomics | 94.4% | 94.4% |
| High School Mathematics | 55.6% | 44.4% |
| High School Microeconomics | 83.3% | 83.3% |
| High School Physics | 88.9% | 88.9% |
| High School Psychology | 100.0% | 100.0% |
| High School Statistics | 94.4% | 94.4% |
| High School Us History | 94.4% | 88.9% |
| High School World History | 100.0% | 100.0% |
| Human Aging | 72.2% | 72.2% |
| Human Sexuality | 88.9% | 94.4% |
| International Law | 94.4% | 94.4% |
| Jurisprudence | 88.9% | 88.9% |
| Logical Fallacies | 83.3% | 83.3% |
| Machine Learning | 83.3% | 83.3% |
| Management | 100.0% | 100.0% |
| Marketing | 94.4% | 94.4% |
| Medical Genetics | 100.0% | 94.4% |
| Miscellaneous | 88.9% | 88.9% |
| Moral Disputes | 83.3% | 88.9% |
| Moral Scenarios | 77.8% | 66.7% |
| Nutrition | 100.0% | 100.0% |
| Philosophy | 94.4% | 94.4% |
| Prehistory | 94.4% | 94.4% |
| Professional Accounting | 88.9% | 88.9% |
| Professional Law | 77.8% | 77.8% |
| Professional Medicine | 94.4% | 94.4% |
| Professional Psychology | 100.0% | 100.0% |
| Public Relations | 61.1% | 61.1% |
| Security Studies | 83.3% | 83.3% |
| Sociology | 100.0% | 94.4% |
| Us Foreign Policy | 88.9% | 88.9% |
| Virology | 61.1% | 55.6% |
| World Religions | 94.4% | 88.9% |

</details>

## Usage

```bash
vllm serve dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 \
  --tensor-parallel-size 4 --moe-backend marlin \
  --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

OpenAI-compatible chat/completions, tools, reasoning, **vision** (`image_url`), and **MTP**
speculative decoding all work. NVFP4 routed experts serve via the Marlin FP4 path on Hopper (H100/H200).

## Credits

- **[dealignai](https://huggingface.co/dealignai)** — CRACK abliteration research & release · Twitter **[@dealignai](https://twitter.com/dealignai)**
- **[@jordanschenck](https://twitter.com/jordanschenck)** — compute

## Disclaimer

This model has had its safety guardrails removed and will comply with requests a stock model
refuses. Released for alignment and safety research. You are responsible for how you use it.
