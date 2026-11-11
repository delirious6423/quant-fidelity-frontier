---
license: apache-2.0
language:
  - en
  - zh
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - llama.cpp
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

<div align="center">
  <img src="assets/card_swe_sharp.png" alt="SWE-bench-Live, 25 settled tasks: Sharp Qwen3.8-27B (Dirk) and TielCoder (Sharp Ornith-1.5, a 4-bit 35B-A3B MoE) against the stock template and cloud frontier Opus 5 (high) / Sonnet 5 (medium) — same weights, Sharp reaches a fix in 37% of stock’s median time on the band both solve (2.7x) and out-solves Opus 5 (high) 15 to 14, one solve behind stock; median and mean shown per arm, judge-free" width="100%">
</div>

<div align="center">
  <img src="assets/card_dirk_mmlu_live.png" alt="MMLU-Pro board — seconds per correct and accuracy across Qwen3.6-27b, Dirk (medium), Qwen3.8-27b (medium), and Nail (35B-A3B MoE); Dirk tops accuracy at 85.3%, and the MoE Nail is quickest to a correct answer at 43s" width="100%">
</div>

With our **[Sharp chat template](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates), MTP, and vision baked in**, the model answers lean and stays on-task out of the box. No template
wrangling: download, point llama.cpp at it, go. If you want it to think deeper, set the effort level through `chat_template_kwargs`:

```json
{"messages": [...], "chat_template_kwargs": {"reasoning_effort": "high"}}
```

Levels: `low`, `medium`, `xhigh` — `high` is accepted but is an alias for `xhigh`, not a step below it. Omit it for Dirk's lean default (medium). Turn thinking off entirely
with `"enable_thinking": false`.

## What it is

- **Base:** `Qwen/Qwen3.8-27B`, a dense 27B **vision-language** model (vision preserved).
- **Quant:** Unsloth **UD** quants across the whole ladder, from their current
  [**Dynamic 3.1**](https://unsloth.ai/docs/models/qwen3.8) release (Nov 2026; Unsloth re-quantized
  only the tiers where 3.1 helps — see the note under the file table). Every tier from
  `UD-Q2_K_XL` up **keeps the model's MTP (`nextn`) head**: runtimes with multi-token-prediction
  speculative decoding can use it for faster generation. The two smallest (`UD-IQ2_XS`, `UD-IQ2_S`)
  ship without it upstream.
- **Template:** the [Sharp chat template](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates)
  (Qwen 3.8-aware) — froggeric's fixed Qwen template plus a terseness system prompt, and turning off the
  xhigh thinking default. Every tier supports the terseness opt-out
  (`chat_template_kwargs: {"terse": false}`). Every tier carries the same version, `v22.5.0`.
  It is byte-swapped into the GGUF metadata; the weights and the MTP tensors are untouched.

The only thing Dirk changes versus the stock quant is the template. Same weights, asked better.

## Proven before Dirk

Dirk is new, but the *template* is not. The identical terseness edit, measured on ThinkingCap-27B (same weights,
only the template swapped):

| | stock template | Sharp template | change |
|---|--:|--:|:--|
| Claw-Eval, answer component | 59.3 | **66.7** | +7.4 |
| Claw-Eval answer tokens | 5393 | **2217** | −59% |
| MMLU-Pro tokens per correct answer | 1601 | **1248** | −22% |

Roughly: the same answers in a bit over half the words, with accuracy moving *up*. That is what Dirk
inherits — and its own SWE-bench-Live and MMLU-Pro numbers, shown above, bear it out.

## Thinking effort

Stock Qwen3.8-27B **forces `reasoning_effort=xhigh`** on every call — always-on maximum-effort
reasoning. **Dirk removes that default,** so it runs at the model's native **`medium`** effort: in both
the official and Unsloth templates, `medium` is the setting that injects *no* reasoning instruction
(only `xhigh` and `low` add one), and Dirk simply leaves it there. So Dirk thinks at the baseline and
answers terse, instead of being pushed to the ceiling on every request. Set `reasoning_effort` yourself
(`low`, `medium`, `xhigh`; `high` maps to `xhigh`), per request, through `chat_template_kwargs` — the OpenAI-style
*top-level* `reasoning_effort` field is dropped by llama.cpp and oMLX, so it must go there (see the JSON example above).

## Run it

| file | size | notes |
|---|--:|---|
| `Dirk-Qwen3.8-27B-UD-IQ2_XS.gguf` | 7.6 GB | smallest tier; fits **12 GB** with context room; **no MTP head** |
| `Dirk-Qwen3.8-27B-UD-IQ2_S.gguf` | 8.4 GB | **the 12 GB pick** — 2-bit with room for context; **no MTP head** |
| `Dirk-Qwen3.8-27B-UD-Q2_K_XL.gguf` | 10.0 GB | largest 2-bit, and the smallest tier with MTP; for **12 GB** at shorter context |
| `Dirk-Qwen3.8-27B-UD-IQ3_XXS.gguf` | 10.9 GB | fits **16 GB** easily, or **12 GB** at short context |
| `Dirk-Qwen3.8-27B-UD-IQ3_S.gguf` | 12.0 GB | 3-bit; value pick if **16 GB** is your ceiling |
| `Dirk-Qwen3.8-27B-UD-Q3_K_XL.gguf` | 13.1 GB | 3-bit with **16 GB** headroom; prefer IQ4_XS unless you need the ~1 GB |
| `Dirk-Qwen3.8-27B-UD-IQ4_XS.gguf` | 14.3 GB | **the 16 GB pick** — 4-bit with real context room |
| `Dirk-Qwen3.8-27B-UD-Q4_K_S.gguf` | 15.4 GB | tight 4-bit; for when Q4_K_XL won't fit with your context |
| `Dirk-Qwen3.8-27B-UD-Q4_K_M.gguf` | 16.6 GB | conventional 4-bit; what a bare `-hf` resolves to — prefer Q4_K_XL if it fits |
| `Dirk-Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17.7 GB | **start here** — the 24 GB default; best size/quality balance |
| `Dirk-Qwen3.8-27B-UD-Q5_K_XL.gguf` | 21.0 GB | **the 24 GB quality pick** — leaves real room for context |
| `Dirk-Qwen3.8-27B-UD-Q6_K.gguf` | 22.0 GB | 6-bit; largest that still fits **24 GB**, tighter than Q5_K_XL |
| `Dirk-Qwen3.8-27B-UD-Q6_K_XL.gguf` | 25.3 GB | near-max quality; wants ~32 GB |
| `Dirk-Qwen3.8-27B-UD-Q8_K_L.gguf` | 28.0 GB | 8-bit near-lossless; fits **48 GB** with 256k context |
| `Dirk-Qwen3.8-27B-UD-Q8_K_XL.gguf` | 31.5 GB | 8-bit, effectively lossless |

Every file carries the Sharp template, every file from `UD-Q2_K_XL` up carries the MTP (`nextn`) head
(Unsloth's IQ2 files have none), and all share `mmproj-F16.gguf`
for vision — you need only one copy of it.

**Why Unsloth across the board.** Every tier comes from Unsloth's current
[Dynamic 3.1](https://unsloth.ai/docs/models/qwen3.8) release, which re-quantized `IQ2_XS`, `IQ2_S`,
`Q2_K_XL`, `IQ4_XS`, `Q4_K_M`, `Q4_K_XL`, `Q5_K_XL` and `Q6_K`. The rest — the 3-bit tiers (which
Unsloth found on par with 3.1), `Q4_K_S`, `Q6_K_XL` and both 8-bit tiers — are unchanged Dynamic 3.0
files that carry the same new template. The 2–3 bpw tiers used to come from IST-DASLab's GSQ-RCO quants, which led
Dynamic 3.0 there. Dynamic 3.1 now beats GSQ-RCO on coding benchmarks (Terminal Bench, DeepSWE,
LiveCodeBench), so the low end moved back to Unsloth too. **Those are Unsloth's measurements, not
ours** — we have re-templated their files, not re-benchmarked them.

**Let llama.cpp fetch it** — pass a `:quant` tag from the table (`:Q4_K_XL`, `:IQ4_XS`, `:Q6_K_XL`, …).
A tag is **recommended** so you get exactly the tier you want; a bare `-hf` with no tag now resolves to
`UD-Q4_K_M`, a sensible 4-bit default, rather than failing to find one. The `mmproj` rides along in the
manifest, so **vision works from the same tag** — no second download.

```bash
# text — auto-downloads to llama.cpp's own cache (24 GB-card default shown)
llama-server   -hf peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF:Q4_K_XL -ngl 99   # or llama-cli
# vision — same tag; the mmproj is pulled automatically
llama-mtmd-cli -hf peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF:Q4_K_XL -ngl 99 --image photo.jpg
```

**Prefer to keep the files yourself?** Download explicitly, then point `-m` at the local path:

```bash
hf download peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF Dirk-Qwen3.8-27B-UD-Q4_K_XL.gguf \
  mmproj-F16.gguf --local-dir Dirk
llama-cli      -m Dirk/Dirk-Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 99                              # text
llama-mtmd-cli -m Dirk/Dirk-Qwen3.8-27B-UD-Q4_K_XL.gguf --mmproj Dirk/mmproj-F16.gguf -ngl 99  # vision
```

llama.cpp applies the embedded Sharp template automatically — nothing to pass.

**Driving it from a coding agent?** Add `--reasoning-format deepseek` to `llama-server`. It returns
the model's `<think>` block in the OpenAI `reasoning_content` field instead of inline in `content`,
so the agent never sees raw thinking tokens in the text stream. Current llama.cpp already defaults
to this (`--reasoning-format auto` is defined as "same as deepseek"), so it is a no-op on a recent
build and insurance on an older one. Just don't pass `--reasoning-format none` — that is the one
that leaves the tags inline.

## Pick your weapon

**Qwen3.8-27B** may be the new intelligence density frontier for local models that run on consumer hardware, but the already battle-tested **Nail**, joined by the newer **TielCoder**, each have their own use cases, in an arsenal that contains all three.

- **[Nail-35B-A3B](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-GGUF-MTP)** generates tokens 3–4× faster than 27B models, while still being very good at routine coding, debugging, knowledge work, and many other kinds of tasks — which means that for tasks that aren't too hard for it, it writes the unit test and regression test, and implements the feature in the time it takes 3.8-27B to get out of the gate. Reach for Nail when you need volume routine work done right and fast.
- **[TielCoder-35B-A3B](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF)** is the dedicated coder: Nail's 35B-A3B speed class, rebuilt on Ornith-1.5 with the Sharp template and pointed at one job. It fixes 12 of 25 on SWE-bench-Live — level with Opus 4.6, four clear of Sonnet 5 (medium) — at the lowest mean time per attempt of the 35B-A3B family. It pays for that in general knowledge: 73.7 on MMLU-Pro against Nail's 84.0. Reach for TielCoder when the work is code; reach for Nail when the same session also has to know things.
- **Dirk-27B** is what you reach for when the task is genuinely hard and you want the strongest local answer without filler — accepting that Nail reaches an answer faster on work it can handle.

Nail and TielCoder might still be your go-to workhorses for tasks within their ability bands, due to their advantage in speed.

## Credits

- [Qwen](https://huggingface.co/Qwen) — the Qwen3.8-27B weights.
- [Unsloth](https://huggingface.co/unsloth) — the UD Dynamic 3.1 quants this repo redistributes.
- [froggeric](https://huggingface.co/froggeric) — the fixed chat template the Sharp template builds on.

Apache-2.0, matching upstream.
