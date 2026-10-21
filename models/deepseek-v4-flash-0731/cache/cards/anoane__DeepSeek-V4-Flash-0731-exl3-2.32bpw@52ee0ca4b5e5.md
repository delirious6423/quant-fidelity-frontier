---
license: mit
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
tags:
  - exl3
  - exllamav3
  - quantized
  - mixture-of-experts
  - moe
---

# DeepSeek-V4-Flash-0731 — exl3, 2.32 bpw uniform per-layer

An exl3 quantization of DeepSeek-V4-Flash-0731 at **2.32 bpw**, with one bit-width per layer
chosen by measured quantization hardness. It is the maximum-reach rung of a three-pack ladder
sized for a single 96 GB card, and the only one that runs **MTP speculative decoding and a
1M-token context at the same time**.

## The three packs, and why these bitrates

All three target one machine: a single **NVIDIA RTX PRO 6000 Blackwell, 96 GB**. The bitrates
are not round numbers — each is the largest that still clears a specific capability on that
card, with an 8-bit KV cache.

| | [**2.72 bpw** hybrid][h272] | [**2.54 bpw** uniform][u254] | [**2.32 bpw** uniform][u232] |
|---|---|---|---|
| goal | best quality | balanced | maximum reach |
| allocation | **per-expert** 2/3/4-bit | per-layer uniform | per-layer uniform |
| max context, MTP **off** | 256k | 1M | **1M** |
| max context, MTP **on** | **does not fit** | 320k | **1M** |
| **NLL** (held out) | **0.6385** | 0.6451 | 0.6485 |
| **perplexity** (held out) | **1.8936** | 1.9062 (+0.66%) | 1.9126 (+1.00%) |
| decode @8k | 86.71 tok/s | 91.99 | — |
| decode @262k | 86.44 tok/s | 91.64 | **92.15** |
| decode + MTP drafting @131k | *drafter does not fit* | 111.4 tok/s | **114.5 tok/s** |
| VRAM @262k | 96,573 MiB | 90,473 MiB | **83,553 MiB** |
| on disk | 103 GB | 92 GB | **85 GB** |
| shards | 16 + a 5.2 GB spliced MTP | 13, MTP in place | 12, MTP in place |
| expert bitrate | 2.7200 | 2.5349 | 2.3256 |
| layer split | per-expert 4430/5230/1348 | 20×K2, 23×K3 | 29×K2, 14×K3 |

- **2.72, per-expert** — every one of the 11,008 routed experts carries its own bit-width,
  allocated from measured routing traffic. Best quality of the three, and the reason the
  ANEMONE exllamav3 fork exists: stock exllamav3 asserts one bit-width per layer and will not
  load it. On this card it serves 256k with the drafter off; MTP does not fit alongside it.
- **2.54, per-layer uniform** — one bit-width per layer, chosen by measured usage-weighted
  quantization hardness rather than by position. Buys back the templated kernels and enough
  VRAM for either the MTP drafter at 256k, or a 1M context without it.
- **2.32, per-layer uniform** — the only one that runs the drafter *and* a 1M context at the
  same time.

### What the measurements actually say

All figures: one RTX PRO 6000 Blackwell, batch 1, 8-bit KV cache, `--no-rlp`, same engine
build and same harness for every pack, measured in the same sessions.

Perplexity and NLL are over **364,544 held-out tokens** spanning 24 engineering and software
domains, with each domain's calibrated extent computed from the calibration manifest and
excluded — text none of the packs were calibrated on. The corpus is not published, so the
absolute values are not comparable with anyone else's perplexity number; the **differences
between the packs** are the transferable part, and the token counts match exactly.

Three things are worth reading off the table:

**Quality degrades gently.** The full 14.5% bitrate cut from 2.72 to 2.32 costs **1.00%**
perplexity. The 2.54 → 2.32 step costs only 0.34%.

**The decode gain comes from the allocation style, not the bitrate.** Going per-expert →
per-layer is worth ~6% (86.4 → 91.6 tok/s) because it restores the compile-time-specialised
kernels; going 2.54 → 2.32 on top of that is worth another 0.6% (91.6 → 92.2). Decode here is
not strongly bandwidth-bound, so lower bitrate buys **context and headroom**, not speed.

**Drafting is the largest single speed lever, where it fits.** MTP drafting adds ~21% on 2.54
and ~24% on 2.32. Acceptance falls as the draft model gets coarser — 137 of 256 tokens
accepted on 2.54 against 104 on 2.32 — but the net is still faster. Every drafted token is
verified by the main model, so drafting changes speed and never output quality.

Not measured, so not claimed: prefill timings for the two uniform packs.

[h272]: https://huggingface.co/anoane/DeepSeek-V4-Flash-0731-exl3-hybrid-2.72bpw
[u254]: https://huggingface.co/anoane/DeepSeek-V4-Flash-0731-exl3-2.54bpw
[u232]: https://huggingface.co/anoane/DeepSeek-V4-Flash-0731-exl3-2.32bpw

## What this pack is

| | |
|---|---|
| base model | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (~300B MoE, 43 layers × 256 routed experts, top-6) |
| format | exl3 (exllamav3), `mul1` codebook, `out_scales: always` |
| expert bitrate | **2.3256 bpw** — 29 layers at K=2, 14 at K=3 |
| non-expert | attention and shared experts K=4, `lm_head` K=6 |
| MTP module | hybrid (routed experts K=2, attention/shared K=3), **built in place** |
| size | 85 GB, 12 shards |

### Which layers get the extra bit

Not the ends. The conventional heuristic promotes the first and last layers; this pack ranks
all 43 by usage-weighted quantization hardness measured from a prior conversion, and promotes
the hardest 14:

```
K=3   7, 8, 9, 24, 25, 27, 28, 29, 32, 33, 34, 35, 36, 38
K=2   0-6, 10-23, 26, 30, 31, 37, 39-42
```

Measured reconstruction error came out as the allocation intends — K=3 layers average about
half the error of K=2 layers — but the ranking is an imperfect predictor of which individual
layers reconstruct worst, and the spread *within* each class is roughly as large as the gap
between them. It is a better criterion than position, not a solved problem.

## Quality

Held-out perplexity **1.9126**, NLL **0.6485**, over 364,544 tokens across 24 engineering and
software domains — text excluded from calibration by computing each domain's calibrated
extent from the calibration manifest.

Against the 2.72 per-expert pack measured identically that is **+1.00%**, and against the
2.54 pack **+0.34%** — for a 14.5% and 8.4% cut in expert bitrate respectively. The
degradation is gentle, which is what makes this rung worth having.

## Serving

Quantized and tested with the [ANEMONE fork of
exllamav3](https://github.com/anoane/exllamav3-anemone).

```bash
hf download anoane/DeepSeek-V4-Flash-0731-exl3-2.32bpw --local-dir ds4-232

git clone https://github.com/anoane/exllamav3-anemone
cd exllamav3-anemone
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0" \
    python -m venv venv && ./venv/bin/pip install -e . --no-build-isolation
```

Build for plain `12.0` on Blackwell, **never** `12.0a` — a single-arch build is unloadable on
anything else.

Configuration these numbers were measured with:

```
max_seq_len / cache_size  1048576       (1M, with or without the drafter)
cache_mode                8,8           8-bit K and V
gpu_split                 [95.5]
max_batch_size            1
env                       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
                          EXL3_NO_RLP=1
```

`expandable_segments` is not optional at these context lengths: without it the allocator
fragments and the load fails tens of MiB short. A complete config, launcher and systemd unit
are in the [recipe repository](https://github.com/anoane/anemone-recipe).

**On the fork requirement:** the 2.72 pack genuinely needs the fork, because stock exllamav3
asserts a single bit-width per layer and refuses a per-expert pack. This pack *is* uniform
per layer, so that assertion is satisfied — but it has not been tested against a stock build,
and every number here was taken with the fork. Treat stock compatibility as plausible and
unverified.

### Serve it the way DeepSeek specify

Two settings outside this pack dominate how the model behaves, and both are easy to get wrong.

**There is no official Jinja chat template.** DeepSeek's model card says so and ships a Python
encoder (`encoding/encoding_dsv4.py`) instead, so any `chat_template.jinja` beside an exl3
pack is a community reconstruction. The reasoning-effort prompt is the mechanism that makes
effort levels work, and a paraphrase is not equivalent — DeepSeek's `max` prompt instructs the
model *not to stop reasoning until it has independently verified the solution from multiple
angles*, where the version that circulated with these packs merely said "think exhaustively
before answering". The `chat_template.jinja` here carries their verbatim strings. Valid levels
are `low`, `high`, `max` — there is no `medium`, and `low` injects no prompt at all.

**Sampling should be `temperature 1.0`, `top_p 1.0`** (`0.95` for agentic use), per both their
card and the model's own `generation_config.json` — and nothing else. `min_p` is not one of
their recommendations. Low temperature or aggressive truncation drives this model into
repetition loops during long reasoning: at `temperature 0.2` it will emit tens of thousands of
characters of self-checking and never answer. `serve/sampler_overrides/deepseek_v4.yml`
encodes the recommended values as TabbyAPI fallbacks (`force: false`, so a client that sends
its own values still wins).

**Give the reasoning room — and this pack can.** DeepSeek recommend a maximum output length of
**384K tokens** at `high` and `max` effort, and the model genuinely uses it: a hard
competitive-programming problem measured on the 2.72 pack spent ~45,000 tokens reasoning
before answering. The 2.72 pack's 262,144-token context is *less than that recommended budget
on its own*; **this pack reaches 1M, including with the MTP drafter**, so it can hold DeepSeek's recommended reasoning
budget with room to spare. That is a practical argument for the lower-bitrate packs beyond
raw speed.

If a reply ever comes back with a large `reasoning_content` and an empty `content`, it ran out
of context mid-deliberation — raise `max_tokens` or lower the effort level.

### The MTP drafter

The MTP module is built into the shards, so nothing extra needs downloading. Enable it with
`--mtp`. It drafts a block of tokens per verification round and the main model verifies every
one, so it changes speed and never output quality.

Measured at **96,801 MiB with a 1,048,576-token cache** — the drafter and a 1M context fit
together on one 96 GB card, which neither the 2.72 nor the 2.54 pack manages.

Acceptance is lower than on the higher-bitrate packs (104 of 256 generated tokens against 137
on 2.54), because the draft model is quantized alongside everything else. The net is still
faster: ~114 tok/s against ~92 without drafting.

## Reproducing

The full recipe — routing census, hardness ranking, calibration, conversion and gates — is at
[anoane/anemone-recipe](https://github.com/anoane/anemone-recipe), including the measured
usage counts this allocation derives from:

```bash
python gen_strategy_anemone.py 2.32
python gen_strategy_perlayer.py 2.32 --ref-strategy work/strategy_anemone_232.json \
                                     --out work/strategy_perlayer_232.json
STRATEGY=work/strategy_perlayer_232.json OUT=models/ds4-232 ./convert_anemone.sh 2.32
```

The conversion took **12 h 31 m** on one 96 GB card, all-or-nothing with no checkpoint resume,
and needs ~215 GB of free disk.

## Limitations

- **Lowest quality of the three packs.** If you do not need 1M context or the drafter, the
  2.54 or 2.72 pack is a better choice — see the table above.
- **Single-GPU fit assumes one process, batch 1, and an 8-bit KV cache.** With one sequence
  slot a second concurrent request waits rather than running.
- **Prefill timings are unmeasured** for this pack. Decode is characterised above; prefill is
  not, and should not be assumed identical to the other packs'.
- **Calibration domain.** Bit allocation derives from engineering and software prompts.
  Routing tracks the prompt distribution — including style and source, not only subject — so
  a very different workload would ideally re-profile.

## License

Inherits the license of the base model, DeepSeek-V4-Flash-0731. Quantization metadata and
scripts are MIT.
