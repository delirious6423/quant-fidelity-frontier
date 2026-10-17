---
license: apache-2.0
language:
  - en
  - zh
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
  - omlx
  - apple-silicon
  - oq
  - quantized
  - qwen3_5
  - qwen3.8
  - mtp
  - speculative-decoding
  - token-efficient
  - agentic-coding
  - vision
---

<div align="center">
  <img src="assets/dirk_banner_eyebrow.png" alt="Dirk — Qwen3.8-27B, sharpened" width="100%">
</div>

**Dirk** is the Qwen3.8-27B that gets straight to the point.

> **📊 Benchmarks — live, in progress.** Measured right now and posted **as each seed lands**, so any arm with **hatched bars** is partial and will move before it settles. Honest, in the open. More plates appear here as benchmarks start.

<div align="center">
  <img src="assets/card_swe_sharp.png" alt="SWE-bench-Live: Sharp Qwen3.8-27B (Dirk) and Nail (Sharp 35B-A3B MoE) vs the stock template and cloud frontier Opus 5 (high) / Sonnet 5 — same weights, Sharp is ~2x faster to a fix on the solvable band and still out-solves Opus 5 (high); median and mean shown per arm, judge-free" width="100%">
</div>

<div align="center">
  <img src="assets/card_dirk_mmlu_live.png" alt="MMLU-Pro board — seconds per correct and accuracy across Qwen3.6-27b, Dagger, Dirk (medium), Qwen3.8-27b (medium), and Nail (35B-A3B MoE); Dirk tops accuracy at 85.3%, and the MoE Nail is quickest to a correct answer at 43s" width="100%">
</div>

With our **[Sharp chat template](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates), MTP, and vision baked in**, the model answers lean and stays on-task out of the box. No template
wrangling: download, point oMLX at it, go. If you want it to think deeper, set the effort level through `chat_template_kwargs`:

```json
{"messages": [...], "chat_template_kwargs": {"reasoning_effort": "high"}}
```

Levels: `low`, `medium`, `high`, `xhigh`. Omit it for Dirk's lean default (medium). Turn thinking off entirely
with `"enable_thinking": false`.

## What it is

- **Base:** `Qwen/Qwen3.8-27B`, a dense 27B **vision-language** model (vision preserved).
- **Quant:** oMLX's **oQ** quantizer — imatrix-calibrated mixed-precision MLX, which **keeps the
  model's MTP head** — runtimes with multi-token-prediction speculative decoding can use it for faster generation.
- **Template:** the [Sharp chat template](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates)
  (`v22.5.0`, Qwen 3.8-aware) — froggeric's fixed Qwen template plus an always-on terseness system prompt, and turning off xhigh thinking default.
  It replaces the checkpoint's `chat_template.jinja`; the weights and the MTP tensors are untouched.

The only thing Dirk changes versus the stock quant is the template. Same weights, asked better.

## Proven on Nail and Dagger

Dirk is new, but the *template* is not. The identical terseness edit, measured on
[Dagger](https://huggingface.co/peculiar-ragdoll/Dagger-Qwen3.6-27B-GGUF-MTP)'s base (ThinkingCap-27B,
same weights, only the template swapped):

| | stock template | Sharp template | change |
|---|--:|--:|:--|
| Claw-Eval, answer component | 59.3 | **66.7** | +7.4 |
| Claw-Eval answer tokens | 5393 | **2217** | −59% |
| MMLU-Pro tokens per correct answer | 1601 | **1248** | −22% |

Roughly: the same answers in a bit over half the words, with accuracy moving *up*. That is what Dirk
inherits on day one; its own measured numbers are in progress.

## Thinking effort

Stock Qwen3.8-27B **forces `reasoning_effort=xhigh`** on every call — always-on maximum-effort
reasoning. **Dirk removes that default,** so it runs at the model's native **`medium`** effort: in both
the official and Unsloth templates, `medium` is the setting that injects *no* reasoning instruction
(only `xhigh` and `low` add one), and Dirk simply leaves it there. So Dirk thinks at the baseline and
answers terse, instead of being pushed to the ceiling on every request. Set `reasoning_effort` yourself
(`low`, `medium`, `high`, `xhigh`), per request, through `chat_template_kwargs` — the OpenAI-style
*top-level* `reasoning_effort` field is dropped by llama.cpp and oMLX, so it must go there (see the JSON example above).

## Run it

| tier | size | notes |
|---|--:|---|
| [oQ4e](https://huggingface.co/peculiar-ragdoll/Dirk-Qwen3.8-27B-MLX-oQ4e) | 15.8 GB | **start here** — imatrix-calibrated ~4.9 bpw; fits a **24 GB** Mac with real room for context |
| **oQ6e** *(this repo)* | 22.1 GB | ~6.9 bpw; more precision headroom, wants **32 GB** |

Both tiers carry the Sharp v22.5.0 template and the MTP head, and both keep the vision tower — no
separate projector file to download.

**oMLX** — put the folder under `~/.omlx/models/peculiar-ragdoll/Dirk-Qwen3.8-27B-MLX-oQ6e` (or pull it from the
oMLX admin dashboard), then switch MTP on:

```json
"Dirk-Qwen3.8-27B-MLX-oQ6e": { "mtp_enabled": true, "max_context_window": 262144 }
```

Sampling defaults: `temperature 1.0`, `top_p 0.95`, `top_k 20`.

**Prefer to keep the files yourself?** Download explicitly, then point `--model` at the local path:

```bash
hf download peculiar-ragdoll/Dirk-Qwen3.8-27B-MLX-oQ6e --local-dir Dirk-MLX
python -m mlx_vlm.generate --model Dirk-MLX --max-tokens 512 \
  --prompt "Explain what this function does."                       # text
python -m mlx_vlm.generate --model Dirk-MLX --max-tokens 512 \
  --prompt "What is in this screenshot?" --image photo.jpg          # vision
```

Requires `mlx-vlm >= 0.6.3`. MLX loaders without MTP support simply ignore the MTP tensors.

Both runtimes apply the embedded Sharp template automatically — nothing to pass.

**Load it with `mlx-vlm`, not `mlx-lm`.** This is a vision-language checkpoint. `mlx_lm.load()`
accepts it and then emits garbage tokens — we hit exactly that in testing before switching to
`mlx_vlm`, where the same files generate correctly. It is a loader mismatch, not a bad quant, but it
fails quietly. Single-run sanity figure on our box, `mlx-vlm` text-only with **no MTP**: **~25.5
tok/s** decode — a smoke test, not a benchmark.

## Pick your weapon

**Qwen3.8-27B** may be the new intelligence density frontier for local models that run on consumer hardware, but the already battle-tested **Dagger** and **Nail** both have their own use cases, in an arsenal that contains all three.

- **[Nail-35B-A3B](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-GGUF-MTP)** generates tokens 3–4× faster than 27B models, while still being very good at routine coding, debugging, knowledge work, and many other kinds of tasks — which means that for tasks that aren't too hard for it, it writes the unit test and regression test, and implements the feature in the time it takes 3.8-27B to get out of the gate. Reach for Nail when you need volume routine work done right and fast.
- **[Dagger-27B](https://huggingface.co/peculiar-ragdoll/Dagger-Qwen3.6-27B-GGUF-MTP)** is — unlike 3.8-27B — specifically tuned to minimize the number of thinking tokens while sacrificing minimal accuracy, which might still give it the advantage in speed-to-answer and multi-turn stamina under the context ceiling. Reach for Dagger when you need a session to survive 100 turns.
- **Dirk-27B** is what you reach for when the task is genuinely hard and you want the strongest local answer without filler — accepting that Nail reaches an answer faster on work it can handle, and that a marathon session running 100 turns under the context ceiling is Dagger's domain, not Dirk's.

Dagger and Nail might still be your go-to workhorses for long and short tasks within their ability bands, due to their advantage in speed and stamina.

## Credits

- [Qwen](https://huggingface.co/Qwen) — the Qwen3.8-27B weights.
- [Jundot](https://huggingface.co/Jundot) and [oMLX](https://github.com/jundot/omlx) — the [Jundot/Qwen3.8-27B-oQ6e-mtp](https://huggingface.co/Jundot/Qwen3.8-27B-oQ6e-mtp) oQ quant (MTP-preserving) this repo redistributes.
- [froggeric](https://huggingface.co/froggeric) — the fixed chat template the Sharp template builds on.

Apache-2.0, matching upstream.
