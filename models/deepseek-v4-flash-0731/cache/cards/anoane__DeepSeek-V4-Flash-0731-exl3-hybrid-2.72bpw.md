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

# DeepSeek-V4-Flash-0731 — exl3 hybrid, 2.72 bpw

An **exl3 quantization of DeepSeek-V4-Flash-0731 in which every routed expert carries its
own bit-width.** 11,008 experts are individually assigned a 2-, 3- or 4-bit trellis code
according to how much each one is actually used, instead of the usual single bit-width per
layer. The result is 2.74 bpw overall — 103 GB — which serves the full 256k context on a
single 96 GB card.

| | |
|---|---|
| base model | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (~300B MoE, 43 layers × 256 routed experts, top-6) |
| format | exl3 (exllamav3), `mul1` codebook, `out_scales: always` |
| size | 103 GB, 16 shards + MTP |
| average bitrate | **2.74 bpw** (2.72 across the routed experts) |
| perplexity | **1.8938** on a 364,544-token held-out corpus (see *Quality*) |
| context | **256k** on one 96 GB card (RTX PRO 6000 Blackwell), 97,278 MiB peak |
| engine | requires the ANEMONE fork of exllamav3 (see *Serving*) |

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

## Why per-expert bit-widths

In a mixture-of-experts model the experts are not equally important: routing concentrates
traffic on a minority of them, and the rest fire rarely. Spending the same number of bits
on every expert therefore wastes capacity twice over — the hot experts are under-served and
the cold ones are over-served.

This pack allocates bits by **measured decode-time usage**. Experts were profiled over a
205-prompt generation campaign, then a greedy waterfill assigned each one 2, 3 or 4 bits
under a fixed total budget:

| bit-width | experts | share |
|---|---:|---:|
| K=2 | 4,430 | 40.2% |
| K=3 | 5,230 | 47.5% |
| K=4 | 1,348 | 12.2% |

Non-expert tensors are quantized conventionally: attention and shared experts at **K=4**,
`lm_head` at **K=6**, and the MTP module hybrid (its routed experts K=2, its attention and
shared tensors K=3).

**How much per-expert allocation is worth — measured, not asserted.** Against a uniform-K
pack of the same class this pack scores **0.32% better perplexity**, and it carries 0.83%
more bits, so part of even that margin is weight rather than allocation. (A convert-time
distortion proxy puts the same comparison at +90.9%; it over-predicts the outcome by about
two orders of magnitude, which is why the measurement is quoted here instead.) A per-layer arrangement at an identical bit budget is projected to cost
around **+0.15% perplexity while giving back ~5.7% of decode speed**.

Read plainly: per-expert allocation is worth a fraction of a percent of perplexity, and it
costs ~6% of decode throughput to collect. Its real value is that it lets a model of this
size reach 2.72 bpw and fit one card at 256k at all. If you would rather have the decode
speed, the recipe repository ships the per-layer generator too.

Two further checks on the allocation. Each additional bit reduces measured class error by
roughly 4× (0.0366 / 0.0091 / 0.0023 for K=2/3/4 — ratios of 4.02 and 3.96), which is the
marginal-gain assumption the waterfill itself makes, so hand-shifting the mix does not beat
it. And re-scoring by `usage × intrinsic difficulty` rather than usage alone moves 841
experts for a 1.48% improvement in total weighted distortion — usage spans orders of
magnitude across experts while intrinsic difficulty varies only about ±6.4%, so usage is
the axis that matters. (Both are convert-time proxy measures, computed from quantization
error rather than from perplexity.)

## Quality

Perplexity on a 364,544-token held-out corpus spanning 24 engineering and software domains,
one process per configuration, same harness throughout:

| configuration | perplexity | vs. recommended |
|---|---:|---:|
| **recommended config (`--no-rlp`)** | **1.8938** | — |
| `--hc-bf16` as well | 1.8939 | +0.004% |
| `--fp4-prefill` | 1.9263 | +1.7% |
| `--fp4-prefill --fp4-dense` | 2.0271 | +7.0% |

The recommended configuration enables no option that costs quality. `--no-rlp` reschedules
prefill chunking inside the generator path, which this harness never enters — so its row
is the number an all-options-off run produces, and this table prices the other flags rather
than that one.

Per-domain perplexity spans **1.10 to 2.97 across 23 of the 24 domains**. The 24th is an
outlier worth stating plainly: its held-out tail turned out to be encoded binary rather
than prose — 1.5 characters per token — and it scores **37.8**, contributing 26% of the
total loss from 4.5% of the tokens. Excluding it the same four configurations measure
1.6449 / 1.6449 / 1.6727 / 1.7620, and the flag costs move by under 0.1 percentage point,
which is why they are quoted from the full set. A 25th domain is dropped entirely because
calibration consumed its whole file. A workload concentrated in one domain will not see the
aggregate either way.

**What "held out" means here.** Calibration consumes a contiguous run of tokens from the
head of each domain file, sized by that domain's weight in the manifest. The harness
computes that extent per domain and evaluates only past it. This matters more than it
sounds: an earlier version skipped a flat 120,000 tokens, but at 2000 calibration rows the
heaviest domains give up ~242,000 tokens each, so most domains were being scored on their
own calibration text — which reported 2.0420 rather than the 1.8938 above. The corpus is
private, so the absolute number is not comparable with anyone else's; the configuration
deltas are the transferable part.

## Serving

```bash
hf download anoane/DeepSeek-V4-Flash-0731-exl3-hybrid-2.72bpw \
    --local-dir ds4-anemone-272-hybrid
```

This pack needs the ANEMONE fork of exllamav3 — upstream exllamav3 requires a single
bit-width per layer and will refuse to load it.

```bash
git clone https://github.com/anoane/exllamav3-anemone
cd exllamav3-anemone
python -m venv venv && ./venv/bin/pip install -e . --no-build-isolation
```

Build for plain `12.0` on Blackwell, **not** `12.0a` — a single-arch `120a` build makes the
extension unloadable on anything else:

```bash
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0" ./venv/bin/pip install -e . --no-build-isolation
```

Serving through TabbyAPI (max quality, 256k context, single tenant):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EXL3_NO_RLP=1          # one fewer forward pass per prompt; no measurable cost
python main.py --config config.yml
```

with `config.yml` setting `model_dir`, `model_name: ds4-anemone-272-hybrid`,
`max_seq_len: 262144`, `cache_size: 262144`, `cache_mode: 8,8`, `gpu_split: [95.5]` and
`max_batch_size: 1`. A complete config, launcher and systemd unit are in `serve/` beside
this file, and in the [recipe repository](https://github.com/anoane/anemone-recipe).

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not optional at this context: without
it the allocator fragments and the load fails a few tens of MiB short.

`cache_mode: 8,8` is an 8-bit KV cache — a lossy choice, and the one this pack's 256k fit
assumes. Measured through the real cached decode path it costs **+0.017%** perplexity,
which is below that harness's own 0.035% resolution: no measurable cost, rather than a
cost measured to be tiny. A 16-bit cache does not fit at 256k on this card.

TabbyAPI constructs the engine itself and does not forward ANEMONE CLI flags, so options
are set as environment variables (`EXL3_NO_RLP=1`, …). Against the exllamav3 CLI directly,
the equivalent flags (`--no-rlp`, …) work.

Options that cost quality are opt-in and off by default. Their measured prices on this pack:

| flag | buys | costs |
|---|---|---|
| `--hc-bf16` | −645 ms @128k, −1.40 s @260k prefill | +0.004% perplexity (1.8938 → 1.8939) |
| `--fp4-prefill` | ~1.4× paired; ~1.45× for the whole fast profile (see below) | +1.7% perplexity; ≤128k context |
| `--fp4-dense` | ~13 ms | +5.2% on top of `--fp4-prefill` — don't |

`--hc-bf16` is a reasonable trade if you serve long prompts: on held-out text its cost is
small enough to read as no measurable cost, and it is the only option that meaningfully
improves deep-context prefill. It is off here because this pack was tuned for maximum
quality, not because the trade is bad.

### Measured performance

One RTX PRO 6000 Blackwell (96 GB, compute 12.0), batch 1, 8-bit KV cache, this pack.

| prompt tokens | recommended (`--no-rlp`) | `--no-rlp --hc-bf16` |
|---:|---:|---:|
| 512 | 228.5 ms | 225.4 ms |
| 2,048 | 518.8 ms | 500.3 ms |
| 8,192 | 2,174 ms | 2,086 ms |
| 16,384 | 4,390 ms | 4,179 ms |
| 32,768 | 9,000 ms | 8,633 ms |
| 65,536 | 18,473 ms | 18,088 ms |
| 131,072 | 38,710 ms | 38,066 ms |
| 260,000 | 84,130 ms | 82,729 ms |

The fast-prefill profile — `--no-rlp --fp4-prefill --hc-bf16 --prefill-graphs`, capped at
128k — measures **354 ms @2k** and **1,492 ms @8k**. Those are the whole profile rather than
FP4 alone; the options were not measured one at a time at those depths, so do not read them
as the price of `--fp4-prefill` by itself.

`--no-rlp` is worth about −34% at 512 tokens and −20% at 2k, then fades to nothing past
~16k: it removes one extra forward pass per prompt, which only matters when the prompt is
short. `--hc-bf16` is the opposite — irrelevant when short, worth ~645 ms at 131k and
~1.4 s at 260k.

| | |
|---|---|
| decode @2k | 81.9 tok/s |
| decode @128k | 77.8 tok/s |
| peak VRAM at a full-depth 256k prefill | 97,278 MiB of 97,887 |
| headroom | ~609 MiB |
| max context | 256k (≤128k with `--fp4-prefill`) |

Decode barely moves with bitrate — a 6.4% heavier pack costs 0.38% throughput — and loses
only ~5% between 2k and 128k of context. The FP4 options do not touch it at all; `--hc-bf16`
nudges it up slightly (82.5 / 78.9 tok/s against profile A's 81.9 / 77.8).

**Long-context retrieval.** Needle-in-a-haystack at 256k, five needle depths: **5/5**. The
256k context is real, not nominal.

### Serve it the way DeepSeek specify

Two settings outside this pack dominate how the model behaves.

**There is no official Jinja chat template.** DeepSeek's model card says so and ships a Python
encoder (`encoding/encoding_dsv4.py`) instead, so any `chat_template.jinja` beside an exl3
pack is a community reconstruction. The reasoning-effort prompt is the mechanism that makes
effort levels work, and a paraphrase is not equivalent — DeepSeek's `max` prompt instructs the
model *not to stop reasoning until it has independently verified the solution from multiple
angles*. The template in `serve/` carries their verbatim strings. Valid levels are `low`,
`high`, `max`; there is no `medium`, and `low` injects no prompt at all.

**Sampling should be `temperature 1.0`, `top_p 1.0`** (`0.95` agentic), per both their card
and the model's `generation_config.json` — and nothing else. `min_p` is not one of their
recommendations. Low temperature or aggressive truncation drives this model into repetition
loops during long reasoning. `serve/sampler_overrides/deepseek_v4.yml` encodes the
recommended values as TabbyAPI fallbacks.

**Give the reasoning room.** DeepSeek recommend up to **384K output tokens** at `high` and
`max` effort, and the model uses it — a hard competitive-programming problem measured on this
pack spent ~45,000 tokens reasoning before answering, taking 9.3 minutes at ~82 tok/s. This
pack's 262,144-token context is *less than that recommended output budget on its own*. If a
reply comes back with a large `reasoning_content` and empty `content`, it ran out of context
mid-deliberation: raise `max_tokens`, lower the effort level, or use a pack that reaches 1M.

### The MTP module

`zz_mtp_hybrid.safetensors` (5.2 GB) holds the multi-token-prediction module, quantized
hybrid — routed experts at 2 bits, attention and shared tensors at 3. It is not resident
unless you enable speculative decoding, and on a single 96 GB card **it does not fit
alongside a 256k cache** — measured at the production split, not assumed. It is ~5.0 GB of
weights plus a draft cache, against roughly 1.3 GB of headroom. It ships so the pack is
complete and usable on larger machines.

If you want drafting on this card, that is what the [2.54 pack][u254] is for: it frees 6.1 GB
and runs the drafter up to 320k context.

Where it does fit, note that MTP quality affects the draft *acceptance rate* — that is,
speed — and not output quality, since every drafted token is verified by the main model.
Acceptance rate for this module has not been measured, because it cannot be run on the
hardware this pack was built and tested on.

## Reproducing

The full recipe — expert-usage profiling, strategy generation, the conversion command and
the verification gates — is published at
[ANEMONE/anemone-recipe](https://github.com/anoane/anemone-recipe), including the measured
usage census this pack's allocation was derived from. A conversion takes roughly 14–24 h on
a single 96 GB card with no checkpoint resume, and needs ~215 GB of free disk (the measured peak is ~206 GB, where the work
directory and the finished pack coexist, and the run cannot resume), peaks around 72 GB of VRAM — 46 GB of that
the calibration working set — and, with the bf16/tiered calibration options, about
62 GB of host RAM.

## Limitations

- **Engine-specific.** Only the ANEMONE exllamav3 fork loads per-expert mixed-K packs;
  transformers, vLLM and llama.cpp cannot read this format.
- **Single-GPU fit is tight.** 256k context assumes one process, batch 1, and no
  FP4-dense. With one sequence slot a second concurrent request waits rather than running;
  multi-tenant or batched serving needs a smaller context.
- **Calibration domain, and a caveat about what "domain" means.** Bit allocation was
  profiled on engineering and software prompts. Expert usage is broadly stable across
  *subjects* — top-64 overlap 0.61 between domains within one prompt batch — but the two
  batches merged into the shipped census overlap less with each other (0.525, with 31% of
  the hot set distinctive). Routing shifts with the *source and style* of the prompts, not
  only their subject. If your traffic looks stylistically unlike "engineering questions
  posed to a reasoning model", re-profiling is worth more than the numbers above suggest.
- **Mixed-K costs decode speed.** The MoE decode path runs ~6% slower than a uniform-K pack
  of the same bitrate (83.5 vs 89.0 tok/s in a controlled comparison). Roughly two thirds
  of that is the runtime-K kernel itself, where the extra register pressure forces the
  autotuner onto a lower-occupancy tile; the remainder is dispatch. Four attempted fixes measured negative; a fifth recovered
  58% of the kernel share, but only for packs built with two K values rather than three. This is the price of the format, paid on every token.
- **Speculative decoding is unavailable** on a single 96 GB card at this bitrate.

## License

Inherits the license of the base model, DeepSeek-V4-Flash-0731. Quantization metadata and
scripts are MIT.
