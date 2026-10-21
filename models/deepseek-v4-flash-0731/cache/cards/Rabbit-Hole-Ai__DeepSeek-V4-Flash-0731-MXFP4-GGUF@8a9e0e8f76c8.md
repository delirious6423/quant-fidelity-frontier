---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
- gguf
- deepseek
- deepseek-v4-flash
- ik_llama.cpp
- quantized
- mxfp4
- moe
- rtx-5090
---

# DeepSeek-V4-Flash-0731 — MXFP4-expert trunk GGUF for **one** RTX 5090 + a big-RAM workstation

A single-file GGUF of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731),
converted in-house and served at **196,608 context on one 32 GB RTX 5090** with the MoE experts on CPU.
**145.63 GiB / 4.400 BPW / 284.3 B trunk parameters.**

- **The routed experts are bit-exact to the release.** 0731 ships `"expert_dtype": "fp4"`; the conversion
  repacks those weights into GGUF MXFP4 losslessly. There is nothing left to quantize in the 94% of the
  model that matters, so there is no imatrix and no expert-fidelity trade to argue about.
- Everything else is **Q8_0** (attention/dense FFN/shared expert), with `token_embd` + `output` kept at
  **BF16** and norms at F32.
- **26.4 tok/s** single-stream generation, no drafter, **~28.3 GB peak VRAM**, on one 5090 + a 24-core CPU
  (measured 2026-08-17 on the pre-speculative-decoding configuration — see the "Update, 2026-09-02" box
  below and the Benchmarks section for what changed since).
- **Requires [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** for this configuration — the
  192K context only fits because of ik's `--swa-compress`. Stock llama.cpp will load the file but
  cannot reproduce the serving envelope. **Minimum ik build: upstream main `0ed847d3`** (reports
  `version: 4848 (0ed847d3)`) or later — raised from an earlier `c46ffaa5`/4821 minimum; see "Engine
  minimum, and why it moved" below.

> ### Update, 2026-09-02 — speculative decoding now works on this model
> An earlier revision of this card stated flatly that speculative decoding was impossible here, for
> three reasons, one of which was a hard engine-level conflict between MTP and `--swa-compress`. That
> conflict has since been fixed upstream. **Production now runs MTP speculative decoding and
> `--swa-compress` together**, using a separately-sourced MTP head that is **not part of this repo**.
> See "Speculative decoding" under Serving notes for the honest story — what changed, what you need,
> and the measured numbers (with their caveats).

## Who this is for

You have **one 32 GB GPU and a workstation with a lot of RAM** (this was solved on 255 GB; the weights
alone are ~146 GiB and the MoE tensors live in system memory). You want a 300B-class agentic model at
near-release fidelity and a real 192K context, and you are willing to accept ~26 tok/s (more with the
optional MTP head — see below) to get it.

If you have less than ~192 GB of RAM, or you are not running ik_llama.cpp, this file is not the right
pick — take a smaller-BPW community quant instead.

## What the quant actually is

| ggml type | tensors | size | what |
|---|---|---|---|
| **MXFP4** | 129 | **~137.1 GiB (94%)** | routed experts — bit-exact repack of the release's native FP4 |
| Q8_0 | 365 | ~5.8 GiB | all non-expert weights ("Q8-rest") |
| BF16 | 190 | ~2.6 GiB | `token_embd` + `output` |
| F32 | 641 | ~0.1 GiB | norms |
| I32 | 3 | ~0 | `ffn_gate_tid2eid` hash routing |

Three honest notes on the recipe:

1. **"Trunk" means the MTP/DSpark drafter tensors are not in this file.** 0731 replaced the Preview's
   1-block MTP head with a 3-block DSpark drafter; the conversion skips all 4,705 `mtp.*` tensors, and
   the arithmetic checks out exactly. **This is still true and unchanged** — this repo ships the trunk
   only. A separately converted DSpark drafter for *this* trunk was measured and **rejected** — draft
   acceptance collapses at block positions ≥ 2 for architectural reasons, not weight reasons (unchanged
   finding, see Serving notes). What *has* changed is that a **different, externally-sourced MTP head**
   (not DSpark, not converted from this trunk) now works against this file — see "Speculative decoding"
   below.
2. **`--outtype` only governs the non-expert tensors.** The converter writes MXFP4 experts
   unconditionally because the release already is FP4. So "Q8-rest" is a description of the 6% you can
   actually choose, not of the model.
3. **BF16 non-experts were tried and rejected.** BF16-rest costs **−14.7% generation throughput** and
   **+5.2 GB VRAM** for a KL divergence improvement of 0.005294 mean vs the Q8_0 trunk — real, but with a
   median of 0.000174 (half of all tokens untouched) and invisible in perplexity. Q8_0 is the ship.

**External correctness reference:** built with BF16 non-experts, this same conversion path produces a
tensor table **byte-identical to Unsloth's `UD-Q8_K_XL`** on all four tensor types (MXFP4 experts
147,169,738,752 bytes on both sides, 1,328 tensors). That is an independent check on the conversion, not
a self-graded one.

### ⚠️ The YaRN flags are REQUIRED, not optional

The conversion **lost all five `deepseek4.rope.scaling.*` metadata keys**. Without them YaRN is inert on
the compressed-attention path and long context is silently wrong. You must pass rope scaling on the
command line:

```
--rope-scaling yarn --rope-scale 16 --yarn-orig-ctx 65536 --yarn-beta-fast 32 --yarn-beta-slow 1
```

Verify it landed: the load banner's `llama_init_from_model: freq_scale` must read **0.0625**, not 1.
Nothing else in the log tells you.

### Engine minimum, and why it moved

The stated minimum ik build is now **upstream main `0ed847d3`** (`version: 4848`), up from an earlier
`c46ffaa5` (`version: 4821`). This is a **correctness bump, not a speed one**: commit `26113d1d`
(ik PR [#2347](https://github.com/ikawrakow/ik_llama.cpp/pull/2347)) fixes a real race in DeepSeek-V4
DSA (sparse attention) on CUDA — cuBLAS GEMMs racing the gather/softmax on `ctx.stream()`, which could
silently produce NaN scores. That is a correctness fix for exactly this model's attention path on
exactly this backend (CUDA), not a general-purpose patch, so it is called out here rather than left as
a footnote. Measured cost of the bump: a modest, consistent ~3–4% regression in both prefill and
generation vs. the prior build, accepted because the correctness fix was judged worth it (methodology
and numbers in the Benchmarks section).

`0ed847d3` is also well past the commit that makes speculative decoding possible at all on this
architecture (ik PR [#2309](https://github.com/ikawrakow/ik_llama.cpp/pull/2309), merged 2026-08-13,
`2cda8d2d`) — see "Speculative decoding" below. If you only care about MTP and not the NaN fix, `#2309`
is the older floor; if you want both (recommended), use `0ed847d3` or later.

## Serving

Tested configuration, ik_llama.cpp upstream main `0ed847d3` or later, CUDA, one RTX 5090. This is the
**current production recipe, including MTP speculative decoding**:

```bash
llama-server \
  -m ./dsv4f-0731-trunk-q8rest-mxfp4moe-00001-of-00004.gguf \
  -c 196608 --swa-compress \
  --ctx-checkpoints 2 --ctx-checkpoints-tolerance 5 --ctx-checkpoints-interval 0 \
  --jinja --chat-template-file ./dsv4f-0731-chat-template.jinja \
  --chat-template-kwargs '{"reasoning_effort":"high"}' \
  -t 24 --parallel 1 \
  --temp 1.0 --top-p 0.95 --top-k 0 --min-p 0.0 \
  -fa 1 -ctk f16 -ctv f16 -ngl 99 --n-cpu-moe 43 -b 7168 -ub 7168 \
  --rope-scaling yarn --rope-scale 16 --yarn-orig-ctx 65536 \
  --yarn-beta-fast 32 --yarn-beta-slow 1 \
  -md /path/to/your-deepseek4-mtp-head.gguf -ngld 99 --spec-type mtp:n_max=1
```

`-t 24` is the physical core count of the test machine — set it to yours. **The `-md ...` line needs an
MTP head GGUF that is not shipped in this repo** — see "Speculative decoding" immediately below for what
that file is, where it comes from, and why it's separate. If you don't have one, drop the last line and
add `--spec-type none`; everything else in the command is unchanged and this file still works exactly as
it always has, at the 26.4 tok/s figure in the "Who this is for" summary.

### Serving notes, in order of how much they matter

- **`--swa-compress` is the enabling flag, not a tweak.** Without it, this exact configuration loads to
  **31,856 of 32,607 MiB**, spills, and collapses to PP 93 t/s / TG 14.8 t/s. With it, the same
  configuration sits at **27,009 MiB** and runs normally. It compacts **all 43 layers**, not just the two
  sliding-window ones. Look for `SWA compress = 43 of 43 layers compacted` in the load banner; if the
  line is absent the flag did not land. Quality cost: greedy output was **byte-identical** with the flag
  on and off on the same binary, and the configuration was **deterministic 18/18** across three depths.
  (These specific numbers were measured without an MTP head loaded; with `-md` in play you will also see
  a *second*, separate banner line for the MTP companion, e.g. `43 of 44` — that is the companion's own
  compaction report and is expected, not a broken trunk.)

- **Speculative decoding (MTP) — now works, here's the honest story.** An earlier revision of this card
  said flatly that speculative decoding was off, for three reasons. Reasons 1 and 2 are still true.
  Reason 3 no longer is, and that's worth explaining rather than just correcting silently.

  1. *This file has no drafter — still true.* 0731 replaced the Preview's MTP head with a 3-block
     DSpark drafter, and the trunk conversion deliberately excludes all 4,705 `mtp.*` tensors. Nothing
     in this repo can draft on its own; you need an external head file (below).
  2. *A DSpark drafter converted from this trunk was measured and loses — still true.* Draft acceptance
     collapses at block positions ≥ 2 for architectural reasons. A 7-stage `--spec-type` matrix found
     the best drafter-free stage (`ngram-map-k`) gains +22–27% at 16K depth but degrades to +6.2% on
     copy-heavy work and −15.8% on deliberative reasoning at 90K (acceptance 0.48) — a net loss for a
     long-context reasoning workload. Not recommended.
  3. *"ik refuses MTP together with `--swa-compress`" — was true, is no longer true.* This was measured
     directly: on the ik build this card originally shipped against (`c46ffaa5`), loading a correctly
     architecture-tagged MTP head and enabling `--swa-compress` together produced a hard, code-level
     error at context creation (`--swa-compress is not supported together with MTP speculative
     decoding`), confirmed both by reading the source and by an actual failed load attempt. ik PR
     [#2309](https://github.com/ikawrakow/ik_llama.cpp/pull/2309) ("speculative: Allow --swa-compress
     with DeepSeek4 MTP", commit `2cda8d2d`, merged 2026-08-13) carved out an exception specifically for
     models whose `general.architecture` is `deepseek4` — i.e., this trunk. Once that fix reached a
     rebuilt binary, the previously-fatal combination started working, was qualified with a matched
     on/off control, and was promoted to production.

  **What you need to reproduce this:** a standalone MTP/"NextN" companion GGUF that (a) is compatible
  with this trunk's tensor shapes and (b) declares `general.architecture = deepseek4` in its GGUF
  metadata — that specific architecture tag is what makes ik's `-md` path accept it *and* what makes the
  `--swa-compress` exception above apply to it. An MTP head built for an older, fork-specific loading
  mechanism will not benefit from this fix even if it otherwise "works" — the architecture tag is what's
  load-bearing, not just the tensor content. **This card does not include such a head or a recipe to
  build one** — production sources it from a separate, already-published community GGUF.

  **The head we use, identified:** `DeepSeek-V4-Flash-MTP-3.93bpw.gguf` from
  [`ddh0/DeepSeek-V4-Flash-GGUF`](https://huggingface.co/ddh0/DeepSeek-V4-Flash-GGUF) — matched by
  filename *and* exact byte size (3,768,959,552 B) against the file we run. A 4.93 bpw head is also
  published there if you have the VRAM.

  ⚠️ **Note what that repo is: it is the *Preview*-era repo, not the 0731 one.** `ddh0`'s
  [`DeepSeek-V4-Flash-0731-GGUF`](https://huggingface.co/ddh0/DeepSeek-V4-Flash-0731-GGUF) contains
  **no MTP head at all**. So the configuration above is drafting 0731 trunk weights with a
  **Preview-generation MTP head** — deliberately, and it works. This is worth stating plainly because
  it inverts the obvious expectation: 0731's *own* successor drafter (the 3-block DSpark) was measured
  and rejected here (acceptance collapses at block positions >= 2), while the older 1-block Preview head
  it replaced still drafts usefully for it.

  Two things make that safe rather than reckless. First, speculative decoding verifies every drafted
  token against the target model, so a mismatched drafter costs **acceptance rate, not output quality** —
  the worst case is that it is slower, not that it is wrong. Second, a drafter with a differing vocabulary
  is rejected at load rather than silently degrading, so a head that loads has already cleared that bar;
  do not assume this for any other head/trunk pairing without checking.

  **Measured effect, on the current production binary (`0ed847d3`), matched on/off control, same
  `--n-cpu-moe 43`, n=3, greedy, `cache_prompt=false`:**

  | depth | MTP OFF (tok/s) | MTP ON (tok/s) | TG delta | PP delta | draft acceptance |
  |---|---|---|---|---|---|
  | ~16.6K tokens | 23.59 ± 0.38 | 26.35 ± 0.26 | **+11.7%** | −7.7% | 0.776 |
  | ~82.6K tokens | 21.95 ± 1.16 | 25.76 ± 0.06 | **+17.4%** | −8.0% | 0.889 |

  Notably, acceptance *rises* with depth (0.776 → 0.889) instead of collapsing, which is the opposite of
  every other speculative stage tried on this model (DSpark, ngram families) — that is the specific
  thing that makes MTP worth shipping here. **We are not publishing a single "X% faster" headline number**
  because the underlying evidence, read honestly, has real limits worth carrying forward with any number
  you quote from this table:
  - This is a **matched on/off A/B on the current engine**, not a marketing number — but it uses greedy
    sampling and `cache_prompt=false` (every request re-prefills from scratch). Production runs
    `temp 1.0` and `cache_prompt=true`; the trade in that regime has not been separately measured.
  - **No quality/KLD number exists comparing MTP-on output to MTP-off output for this model.** Output
    was checked for corruption (it passed — see below) but not for a subtle fidelity regression. Treat
    "works" as "does not crash or garble text," not as "verified lossless."
  - Minimum free VRAM under a deep load was **3,742 MiB** on the current build (clears the standard
    1,536 MiB safety floor with real margin, but with less headroom than an earlier engine build
    measured on the same `--n-cpu-moe 43` — re-check your own floor if you're tighter on VRAM than this
    rig).
  - A cross-arm check (same binary, greedy, MTP on vs. off) found outputs are **not byte-identical**
    between the two — but neither is toggling `--n-cpu-moe` by itself with MTP off entirely, which
    proves the difference is ordinary floating-point reduction-order sensitivity (a documented,
    benign effect of this stack), not an MTP-specific correctness problem.

- **`--n-cpu-moe` — moved from 39 to 43.** The prior `--spec-type none` recipe used
  `--n-cpu-moe 39` (all 39 MoE layers' experts on CPU; 4 dense + 39 MoE = 43 total trunk layers). The
  MTP-enabled recipe above runs `--n-cpu-moe 43`. **We cannot give you a clean isolated reason for this
  specific delta** — no report on disk re-measures `--n-cpu-moe 39` (or 40, 42) with the MTP head
  attached; the only two values actually tested together with MTP were 41 and 43. At 41, free VRAM
  bottomed out at 319 MiB under a deep load — 1.2 GB *under* the safety floor, i.e. spilling — which
  dragged prefill down across the board (a spill artifact that earlier internal notes had mistakenly
  attributed to MTP itself). At 43, the same load bottomed out with several GB of headroom to spare and
  both prefill and generation were *higher* than at 41. **43 is what's shipped because it's what was
  tested and clears the floor with margin, not because someone proved it's the exact minimum needed** —
  a value of 42 was estimated (not measured) to leave only a thin margin and was deliberately not risked
  without a real measurement. If you're VRAM-constrained, treat 43 as a known-good floor rather than a
  tuned optimum, and expect that the MTP head itself (a few GB) is most of why more layers moved to CPU
  versus the no-MTP recipe.
- **`-b/-ub 7168`** is where the `--swa-compress` saving was spent instead: **+31.1% prefill** at 90K
  depth (995 → 1,305 t/s) with generation flat, **as measured on the `c46ffaa5` build** — the same build
  the original 2026-08-17 benchmark table below used. **`-ub` tuning is not portable across ik engine
  builds**; the spill point for a given ubatch has been observed to move across engine bumps with no
  changelog entry calling it out. If you're on `0ed847d3` (or any later build), re-sweep `-ub` on your
  own binary before trusting this number rather than copying it — `-ub 8192` is faster still on the
  original build but leaves under 1.3 GB free, which is not safe if a desktop shares the card.
  - *Expected and benign:* changing `-ub` changes greedy output text. Prefill chunk size changes the
    floating-point reduction order. Each configuration is self-consistent (3/3 byte-identical across
    reloads); this is not non-determinism.
- **Context checkpoints are worth turning on** for interrupt-then-resend clients (agentic editors).
  `--ctx-checkpoints 2 --ctx-checkpoints-tolerance 5 --ctx-checkpoints-interval 0` takes exactly one
  checkpoint per turn (~350 ms). On a cancelled 16K generation followed by a divergent resend: **13.37 s
  of re-prefill → 0.98 s**, a ~13.6× speedup on that pattern. Cost measured at −0.8% to −4.1% prefill,
  generation flat to +3.6%, and a byte-identical compute buffer. This was re-confirmed (warm-vs-cold
  restore, non-vacuous — real restore log lines, not a meaningless cache-reuse counter) on the current
  MTP-enabled production binary as well.
- **Usable context was measured, not assumed — but not re-verified under MTP.** 4 mechanical pass/fail
  task types × 5 depths × 5 reps, **120/120 passed** out to a real `n_past` of **187,118**, on the
  no-drafter (`--spec-type none`) configuration. The 196,608 allocation is not decorative. This sweep has
  **not** been re-run with the MTP head loaded; the MTP companion's extra VRAM/KV usage is small enough
  relative to the compute buffer that we don't expect a material change, but "don't expect" is not the
  same as "measured," so flagging it as unverified rather than silently carrying the number forward.
- **KV cache f16.** q8_0 KV is smaller but adds graph splits; the KV cache is a rounding error next to the
  compute buffer here (311 MiB KV vs 5,254 MiB compute at this configuration, measured on the no-drafter
  config). The MTP companion carries its own small additional KV/compute footprint on top of this, not
  separately broken out here.

## Benchmarks — read them honestly

**Two configurations are documented below, and they are not directly comparable — do not read the
second table as "the new number that beats the first."**

### Table 1 — original release configuration (no drafter), measured 2026-08-17

This table describes the **pre-MTP, `--spec-type none` configuration** on an older engine build. It is
**not** what production serves as of 2026-09-02 (production now runs MTP — see above) but it remains the
right table if you are running this file *without* an MTP head, and it is the only table with the full
shallow-prompt + VRAM/power measurement methodology.

Measured on: RTX 5090 (32 GB) + AMD Threadripper PRO 9965WX (24c/48t) + 255 GB DDR5, Windows 11,
ik_llama.cpp `version: 4821 (c46ffaa5)`, `--n-cpu-moe 39`, `--spec-type none`, otherwise the same serving
flags as the block above.

| metric | value | conditions |
|---|---|---|
| **Generation** | **26.43 tok/s** | median of 3 (26.50 / 26.43 / 26.42), 512 tokens out |
| **Prefill** | **174.5 tok/s** | at ~420-token prompt depth — see the caveat below |
| **Prefill (deep)** | **~1,305 tok/s** | at 90K depth, batch-saturated, `-b/-ub 7168` |
| **TTFT** | 3,055 ms | ~420-token prompt, streamed, first non-empty content chunk |
| **Peak VRAM** | 28.31 GB | whole-card `nvidia-smi`, includes ~605 MiB desktop baseline |
| **Peak GPU power** | 126.7 W | max of 0.5 s samples during generation |
| **Speculative decoding** | none | `--spec-type none` |

`llama-bench` baseline for cross-checking, **ik defaults** (that binary supports neither `-c` nor
`--swa-compress` nor the YaRN flags, so it is *not* the serving configuration): **pp512 295.9 / tg128
27.3**, r=3.

**Methodology.** Per model: a `llama-bench` pp512/tg128 ×3 no-spec baseline, then a throwaway
`llama-server` using the exact serving flags **except sampling forced greedy (temp 0 / top-k 1 —
production serves temp 1.0; disclosed)**: a 96-token warmup on unrelated text, then 3 non-streamed
measured runs on 3 **distinct** ~420-token prompts (expository / narrative / technical), each run the
model's **first sight** of its prompt (`cache_prompt: false`, fresh server, no prompt reuse); medians
published. A single streamed TTFT run last, so it cannot warm the measured runs. Peak VRAM and power from
0.5 s `nvidia-smi` sampling during generation.

**Caveats that matter more than the headline:**

- **26.4 tok/s is single-stream at shallow depth.** At 90K depth generation measures **25.2 tok/s** —
  the decay is graceful, but it is a decay, and this number is one request at a time with
  `--parallel 1`. It is not a throughput figure.
- **The two prefill numbers are both true.** 174 t/s at a 420-token prompt is a *latency* measurement
  with a 7,168-token ubatch about 6% occupied; ~1,305 t/s at 90K depth is the same configuration
  saturated. Quoting either one alone misrepresents the model.
- **126.7 W is not a typo and not an efficiency claim.** Generation is bound by CPU expert GEMV over the
  memory fabric; the GPU is mostly idle during decode. Your CPU and memory bandwidth, not your GPU, set
  the generation speed of this file.
- **No quality benchmark is published here.** Fidelity is argued structurally (bit-exact experts,
  measured Q8_0-vs-BF16 KLD on the 6% that is quantized), not from a perplexity or eval sweep against
  other people's quants. Treat any comparison to other DeepSeek-V4-Flash GGUFs as unmeasured.

### Table 2 — current production configuration (MTP + `--swa-compress`), depth-loaded methodology

This is the on/off MTP comparison already given in full above, repeated here for completeness. It uses a
**different methodology** (greedy, `cache_prompt=false`, two fixed depths, no VRAM/power sweep, no TTFT)
than Table 1, so treat the two as answering different questions rather than as before/after on the same
ruler.

| depth | MTP OFF TG (tok/s) | MTP ON TG (tok/s) | MTP OFF PP (tok/s) | MTP ON PP (tok/s) | draft acceptance |
|---|---|---|---|---|---|
| ~16.6K tokens | 23.59 ± 0.38 | 26.35 ± 0.26 | 1167.44 ± 0.61 | 1077.50 ± 8.75 | 0.776 |
| ~82.6K tokens | 21.95 ± 1.16 | 25.76 ± 0.06 | 1238.75 ± 8.13 | 1139.81 ± 2.70 | 0.889 |

Both rows: ik_llama.cpp `version: 4848 (0ed847d3)`, `--n-cpu-moe 43`, `-md` pointed at the external MTP
head described above, `--spec-type mtp:n_max=1`, otherwise the serving block above. **No re-measurement
of the no-drafter trunk on this same (`0ed847d3`) build exists** — the closest no-drafter numbers we have
are Table 1's, on an older build with different `--n-cpu-moe`, so a direct "how much did the trunk itself
change across engine bumps, holding MTP off" comparison is not something we can currently cite.

## Chat template

**0731 ships no Jinja chat template** — the release uses a Python encoder (`encoding_dsv4.py`) instead.
Serving it with a Preview-era template is a real quality loss unrelated to quantization, so a template is
bundled here.

It is [tarruda's llama.cpp PR #26398 template](https://github.com/ggml-org/llama.cpp/pull/26398) — an
independent implementation validated by its author against the release's own encoder — with **one term
removed** (`or ns.has_tool_calls` in the `keep_reasoning` expression). The reference encoder keys reasoning
retention on a message carrying a tools *schema*; the upstream template also kept it when any message had
`role == "tool"`, a tool *result*, which diverges on conversations that replay tool results without
resending schemas.

Cross-checked against the reference encoder: **18/19** shared cases byte-identical, **4/4** on the gap
cases the shared suite does not reach (zero-argument tool calls, `## Response Format:` blocks,
`drop_thinking` as a real parameter, and the tool-result case above).

`reasoning_effort` is a real per-request parameter with the ladder `low | high | max`, passed via
`--chat-template-kwargs` or the chat kwarg. **Silent failure mode:** the `low` prompt is the empty string,
so a malformed value looks completely normal while your requested effort is quietly not applied. Verify
via `/apply-template` and confirm the expected reasoning-effort prefix is present.

## Files

| File | What it is |
|---|---|
| `dsv4f-0731-trunk-q8rest-mxfp4moe-0000N-of-00004.gguf` | the model in 4 shards (`llama-gguf-split`; 145.63 GiB, 4.400 BPW, 284.3 B trunk params). Point `-m` at shard `00001` — llama.cpp loads the rest automatically |
| `dsv4f-0731-chat-template.jinja` | the chat template described above |

This repo does **not** ship an MTP/speculative-decoding head. See "Speculative decoding" above if you
want to reproduce that part of the serving envelope.

## Credits

This is a repack and a serving envelope. The hard parts are other people's work:

- **DeepSeek** — DeepSeek-V4-Flash-0731, the model and its weights, released under the **MIT License**.
  The routed experts in this file are their FP4 weights, bit for bit.
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>
- **ggml-org / llama.cpp** — the GGUF format, `convert_hf_to_gguf.py`'s `DeepseekV4Model` and the MXFP4
  expert-repack path that produced this file.
  <https://github.com/ggml-org/llama.cpp>
- **Iwan Kawrakow (ikawrakow) and the ik_llama.cpp contributors** — the fork this configuration requires,
  including its DeepSeek-V4 CUDA path, MXFP4 support, and (as of 2026-08) the fix that lets
  `--swa-compress` and MTP speculative decoding run together on this architecture.
  <https://github.com/ikawrakow/ik_llama.cpp>
- **joelfarthing** — `--swa-compress` (ik_llama.cpp PR #2266, "deepseek4: compacted sliding-window KV
  cache"), the single flag that makes 192K context fit on a 32 GB card.
  <https://github.com/ikawrakow/ik_llama.cpp/pull/2266>
- **tarruda** — the DeepSeek-V4-Flash-0731 chat template (llama.cpp PR #26398), which the bundled
  template is derived from.
  <https://github.com/ggml-org/llama.cpp/pull/26398>
- **Unsloth** — their `UD-Q8_K_XL` GGUF served as the independent external reference that validated this
  conversion's tensor table byte-for-byte.
  <https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF>
- **`ddh0`** — the MTP head we pair with this trunk in production is theirs, not ours, and is not
  redistributed in this repo: `DeepSeek-V4-Flash-MTP-3.93bpw.gguf` from
  <https://huggingface.co/ddh0/DeepSeek-V4-Flash-GGUF> (matched by filename and exact byte size).
  Without it the speculative-decoding configuration above does not exist.

Converted, tuned and measured on a single RTX 5090 with a Threadripper PRO 9965WX, Windows 11, CUDA 13.3.

## License

MIT, inherited from [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
