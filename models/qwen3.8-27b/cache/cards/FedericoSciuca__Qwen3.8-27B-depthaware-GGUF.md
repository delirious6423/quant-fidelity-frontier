---
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
- quantized
- dense
- llama.cpp
- quantprobe
library_name: gguf
pipeline_tag: text-generation
---

# Qwen3.8-27B - depth-aware 2-bit GGUF (12.5 GB)

**A 27B dense model compressed to 3.66 effective bits, built from a measurement rather than a
convention** - and the build that answered a question this project had been avoiding.

Most low-bit quants decide which layers to protect by rule of thumb. This one was built from a
**probe**: every depth band of this specific model was pushed to 2-bit one at a time and scored on
held-out text, and the protection went where the damage actually was (the last quarter of the
network, blocks 51-63).

## The result, and the prediction it broke

Every equal-bytes win behind this method had been measured on **mixture-of-experts** models. MoE
has a structural feature a dense model lacks - most of each layer's weight sits in routed experts
that move between memory tiers independently. So it was a live possibility that depth-aware
placement *only* works on MoE, and that a dense model gets nothing.

We tested it, and [pre-registered the prediction](https://github.com/FedericoTs/quantprobe/blob/master/preregistrations/2026-08-24-does-depth-aware-pay-on-a-dense-27b.md)
**before building anything**: dense would be *weaker* than MoE's 29.2%.

| build | placement | bytes | PPL (32 chunks, held-out WikiText-2) | excess over reference |
|---|---|---|---|---|
| reference (Q8_0) | 8.51-bit | 29,047,086,048 | 5.5950 | - |
| **this model** | measured band **51-63** | **12,483,292,128** | **6.0590** | **+0.4640** |
| control | 13 blocks spread evenly | **12,483,292,128** | 6.4136 | +0.8186 |

**43.3% of the control's excess loss removed, at byte-identical size.** Not "within a gate" -
*identical*: in a dense stack every decode block has the same FFN dimensions, so moving which
thirteen blocks get the better format changes placement and literally nothing else. The control is
the strongest form this comparison can take.

And the prediction was **wrong in the direction that matters**: dense came in at 43.3% against the
MoE's 29.2% - about **1.5x stronger**, not weaker. That is published as a staked miss, because the
reasoning behind it was backwards. On an MoE a "protected block" only upgrades its routed experts
(the always-active path is already protected everywhere), so it is a *weaker* intervention there
than here, where the whole FFN changes format. Concentration of a fixed byte budget matters more
when every protected block is read in full, every token.

## What this means for the method

**Depth-aware quantization is not a mixture-of-experts trick.** That is the headline, and it was
genuinely in doubt before this build. One dense model is one dense model - a Gemma or Llama arm is
the next honest test - but the MoE-only caveat is retired.

## How it was built

Base Q2_K everywhere, with: blocks 51-63 FFN at Q4_K (the measured fragile band), attention and
SSM at Q4_K, shared-expert and NextN tensors at Q8_0, token embedding at Q4_K. llama.cpp b10098,
no importance matrix on any arm (so the comparison isolates placement, matching the MoE study).

Full stake, scorer, and raw perplexity logs - including the reference that took 99 minutes to the
arms' 5 because it streams off disk at this size:
[prereg #112](https://github.com/FedericoTs/quantprobe/blob/master/preregistrations/2026-08-24-does-depth-aware-pay-on-a-dense-27b.md) ·
[`weights/data/prereg112.json`](https://github.com/FedericoTs/quantprobe/blob/master/weights/data/prereg112.json)

## Get it

```bash
pip install quantprobe
quantprobe fetch qwen3.8-27b ./models
```

Or use the Files tab above. Before downloading 12.5 GB, `quantprobe plan --model qwen3.8-27b`
reads your machine and tells you the speed and the binding constraint - no download involved.

## Honest limits

- **Perplexity is a proxy, not a task score.** 32 chunks of WikiText-2 ranks two builds at
  identical size; it does not tell you this model is good at your workload. It tells you this build
  is better than the same bytes spent evenly.
- **One dense model, one architecture family.** Both arms and the MoE comparison are Qwen-family.
  This retires the "MoE-only" caveat; it does not prove the effect on every dense architecture.
- **No speed number is published here.** Decode on the measurement box has a 15-24% run-to-run
  spread (it VOIDed two other pre-registrations), and a placement change moves speed by less than
  that, so a speed claim would be noise dressed as a result. The decode prediction is recorded as
  *unscored*, not as a pass.
- **This file carries a NextN/MTP head (block 64) that llama.cpp loads but never executes** during
  ordinary decode ([confirmed from source](https://github.com/FedericoTs/quantprobe/blob/master/FINDINGS.md), L-33/U-60).
  It costs ~2% of the file's capacity and no decode speed. Whether protecting it would buy anything
  is staked as U-61, untested here.

## Provenance

Built and measured by [quantprobe](https://github.com/FedericoTs/quantprobe), which predicts tok/s
from `tok/s = efficiency x bandwidth / bytes-per-token` and stakes every prediction before the
measurement. Source: `unsloth/Qwen3.8-27B-GGUF` (Q8_0, 8.51-bit). This repo is the depth-aware
build only; the atlas entry, the probe log, and the equal-bytes comparison are all linked above.
