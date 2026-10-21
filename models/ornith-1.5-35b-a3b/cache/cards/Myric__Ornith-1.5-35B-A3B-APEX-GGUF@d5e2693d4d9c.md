---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - moe
  - apex
  - quantized
  - llama.cpp
---

# Ornith-1.5-35B-A3B — APEX GGUF

Three banded-allocation GGUF tiers of
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) —
35.5 B total / ~3 B active MoE, 256 routed experts at top-8, hybrid gated-delta-net with
full attention every 4th layer, 262 K context, plus a 1-layer MTP head.

These differ from a flat `Q4_K_M` in three specific ways: routed experts are **banded** by
depth rather than given one type; the 60 recurrent state/gate coefficients are **pinned F32**;
and the **MTP head is retained** rather than dropped.

## Files

| File | Size | bpw | routed experts (edge / near / middle) |
|------|------|-----|---------------------------------------|
| `Ornith-1.5-35B-A3B-APEX-i-quality.gguf` | 21.67 GB | 4.88 | Q6_K / Q5_K / Q3_K |
| `Ornith-1.5-35B-A3B-APEX-i-compact.gguf` | 16.58 GB | 3.74 | Q4_K / IQ4_XS / IQ2_S |
| `Ornith-1.5-35B-A3B-APEX-i-mini.gguf` | 12.55 GB | 2.83 | Q3_K / IQ2_S / IQ2_XXS |

Shared experts are `Q8_0` and attention is `Q6_K` in every tier. bpw is over all 35.505 B
parameters, computed from the file size.

**Start with i-quality.** i-compact and i-mini exist for smaller memory budgets; only
i-quality has been benchmarked (below).

For the **vision projector**, use the one from the source publisher — we did not rebuild or
re-host it: [`mmproj-Ornith-1.5-35B-BF16.gguf`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/blob/main/mmproj-Ornith-1.5-35B-BF16.gguf).
It works with i-quality on stock llama.cpp — measured, see below.

## Running it

```
llama-server \
  --model Ornith-1.5-35B-A3B-APEX-i-quality.gguf \
  -ngl 999 -fa on -c 131072 \
  --jinja --temp 1.0 --top-k 20 --top-p 0.95
```

`temp 1.0 / top_k 20 / top_p 0.95` is the source model's own default, carried in the GGUF
metadata. Substituting another model's sampling changes results materially.

KV cache is cheap: only 10 of the 40 backbone layers are full attention (2 KV heads,
head_dim 256), so **20 KiB/token** — 2.5 GiB at 131072, 5.0 GiB at the full 262144. The other
30 layers hold a constant-size recurrent state.

## The MTP head

The source checkpoint's 1-layer multi-token-prediction block (`blk.40`) is **kept**, quantized
flat at `Q4_K` (11 tensors) with its norms at F32. Most GGUF conversions of this model family
drop it.

`Q4_K` is not arbitrary: on GLM-4.7-Flash, Q4_K measured the best size/speed trade, and `IQ2`
was measurably worse. Mainline llama.cpp consumes the head via `--spec-type draft-mtp`, passing
this same file as its own draft model:

```
llama-server -m Ornith-1.5-35B-A3B-APEX-i-quality.gguf \
  --spec-type draft-mtp -md Ornith-1.5-35B-A3B-APEX-i-quality.gguf \
  --spec-draft-n-max 1 \
  -ngl 999 -fa on -c 131072 --parallel 1 --temp 1.0 --top-k 20 --top-p 0.95
```

**Use `--spec-draft-n-max 1`.** Measured on i-quality, DGX Spark (GB10), stock llama.cpp,
300 tokens at temp 1.0, fixed seed, `cache_prompt: false`:

| drafter | tokens/s | draft acceptance | mean accepted length |
|---|---|---|---|
| none | 69.00 | — | — |
| `draft-mtp`, n_max 1 | **73.52** | 0.674 | 1.67 |
| `draft-mtp`, n_max 2 | 65.24 | 0.385 | 1.77 |
| `draft-mtp`, n_max 3 | 54.78 | 0.288 | 1.86 |
| `draft-mtp`, n_max 4 | 46.92 | 0.199 | 1.79 |
| `draft-mtp`, n_max 6 | 38.01 | 0.133 | 1.79 |
| none, repeated | 69.86 | — | — |

The two no-drafter runs bracket the run-to-run spread at about 1%. Acceptance falls from 0.674
to 0.133 as depth grows while mean accepted length stays near 1.7, so depths above 1 add draft
cost without adding accepted tokens. Depth does not transfer between models — measure it.

## What was measured

`i-quality`, agentic coding suite, frontier tier (9 tasks), one run, DGX Spark (GB10),
stock llama.cpp, sampling as above:

| | result |
|---|---|
| tasks solved | **9 of 9** |
| individual tests passed | **154 of 154** |
| tasks lost to harness limits (timeout or output cap) | **none** |
| mean output tokens per task | 4,050 |
| total wall time | 0.92 h |

### Vision

i-quality plus the publisher's projector, stock llama.cpp, five perception tasks put to the
server directly (read a placard, transcribe a handwritten spec, name a function and its buggy
line from a screenshot, read a value off a bar chart, count shapes):

| arm | passed |
|---|---|
| with the image | **5 of 5** |
| identical questions, image withheld | **0 of 5** |

The second row is the control: every answer is unguessable by construction (a coined phrase, a
coined identifier, an arbitrary count, a deliberately non-standard base case), so a blind pass
would mean the task was measuring text priors rather than the projector. None passed blind.

### Speed

Generation speed, DGX Spark (GB10), stock llama.cpp, 8192 ctx, single slot:

| tier | tokens/s | how |
|---|---|---|
| i-quality | 69.0 and 69.9 on two runs | server `/completion`, 300 tokens |
| i-compact | 73.9 | `llama-cli`, 200 tokens |
| i-mini | 80.8 | `llama-cli`, 200 tokens |

i-compact and i-mini were load-and-generate checked only; the two harnesses differ, so read
across tiers as approximate.

Read that as a floor, not a ranking: **this suite saturates.** A clean sweep means the quant
cleared the competence bar, not that it beats something else. The discriminating measurement
once score saturates is output tokens at equal score, and that needs several runs per arm —
this is one run, on a suite where individual tasks are known to flip between identical runs.

## Provenance and sources

- **Source weights:** `ornith-ai/Ornith-1.5-35B-A3B` safetensors, converted to BF16 GGUF here.
- **Importance matrix: not ours.** These quants were built against
  [bartowski](https://huggingface.co/bartowski)'s published imatrix for this model
  (sha256 `8d5b1693…c0228`), used unmodified. Credit for the calibration prior belongs there;
  what is ours is the allocation across tensors.
- **Allocation method:** structural banding by depth against a size target, **not** measured
  per-tensor sensitivity. Ornith's architecture has no sensitivity name-map in our pipeline, so
  the allocator falls back to the structural prior.
- **Vision projector:** the publisher's own file, linked above rather than re-hosted.
- Unofficial community conversion — **not affiliated with or endorsed by the source publisher.**
  Provided as-is, without warranty.

## Caveats

- **Vision is verified on i-quality only**, with the publisher's projector, on the five tasks
  above. i-compact and i-mini have not been checked for vision.
- **Structural allocation is a prior, not a measurement.** On at least one model where both
  paths could be compared, the structural prior did worse than stock allocation. It is used here
  because measured sensitivity is not yet available for this architecture.
- **i-compact and i-mini are unbenchmarked.** Both load and generate correct-looking code, and
  pass structural checks, but neither has been through the agentic suite.
- IQ2 tiers at 2.8 bpw are aggressive on a 3 B-active MoE; expect i-mini to degrade on long
  multi-step reasoning well before i-quality does.
