---
language: en
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: radiance
pipeline_tag: text-generation
tags:
- gptq
- w5nl
- moe
- radiance
- rdna4
- gfx1201
- rocm
- amd
- speculative-decoding
- mtp
- qsa
- per-layer-embeddings
- vision
- quantization
---

# Qwen3.8-Flash-Next — GPTQ W5NL experts + BF16 steering trunk (radiance `.rad`)

A quantization of the official [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
BF16 checkpoint (335.3 GiB) for the
[radiance](https://github.com/StillDeadcode/radiance) engine on AMD RDNA4 (`gfx1201`), cooked on
2026-11-09 and serving in production the same day. The `.rad` container format loads in radiance
only — not GGUF, not safetensors. The checkpoint's vision tower rides inside the container, so
image input works at the shape documented below.

The recipe idea in one line: **spend bits where the error gets multiplied, and save them where it
doesn't.** The rows that decide *which* token state reaches *which* expert — the QSA attention
projections and every hyper-connection mixer — stay at the checkpoint's bf16, as does the 95.37 GiB
n-gram table behind the model's per-layer-embedding (PLE) path. Everything the experts actually store
moves to five-bit GPTQ codes — the widest rung the engine's non-linear expert kernel serves (u5 into
a 32-level table, `w5nl64a8h`) — on **both** projections rather than only `down`.

| | |
|---|---|
| File | `qwen38-w5nl64-qsa-hc-bf16-cd3.rad` (rename freely; the engine reads by content) |
| Size | 189,991,063,552 bytes (176.94 GiB) — **0.53×** the 335.3 GiB BF16 source, i.e. 1.90× smaller |
| SHA256 | `bdf7569b84e51c2375560c49eea560a65702d45c9ee268364cb1a30528024f59` — the deployed file and this upload are the same bytes |
| Engine | `stilldeadcode/radiance@sha256:c2acbaa373c89272ec47a22befa1cf04d2ddf1de51c2baf534f5273e517f7100` (1.17.0) |
| Architecture | `qwen4exp`, 48 trunk layers + 1 MTP draft layer, 512 experts on **49** layers (25,088 expert rows per projection), 262,144 native ctx |
| Proof hardware | 2× Radeon AI PRO R9700, TP=2 (`--tp-wire exact`), 4 × 200,000-token slots, fp8 KV |
| Calibrated with | the published Grams of [`gostar77`](https://huggingface.co/datasets/gostar77/qwen3.8-flash-next-radiance-q5-w5dn64-grams) — 1,006,978,080 routed rows (credit below) |

**In plain English:** this is the same model, smaller and slightly less exact, with the few parts
that steer the network kept exact and the enormous pile of expert weights kept "good enough". It
runs on two AMD pro GPUs in one box and answers at a 200,000-token window.

## What each group is stored as

Read back **off the artifact** with the engine's own `rad-info --verbose`, then asserted group by
group against the container this file replaced (`artifacts/rad-info-dtype-census.txt`, verdict
`PASS — candidate carries the ordered dtype map`; the full `rad-info --verbose` dumps — **51,489
tensor rows** each — are in `artifacts/`). `live` = the container it replaced. The ten groups below
itemise 50,611 of those rows; the last row names the 878 the census groups do not claim.

| group | live GiB | this GiB | delta | rows | encoding |
|---|---|---|---|---|---|
| QSA attention | 0.61 | 1.21 | **+0.59** | 52 | `cast dtype=bf16` |
| hyper-connection mixers | 0.62 | 1.18 | **+0.57** | 194 | `cast dtype=bf16` |
| shared expert | 0.23 | 0.23 | +0.00 | 98 | int8, group 128, searched bf16 scale |
| delta-net gates (`ssm_inz`/`ssm_out`) | 1.96 | 1.96 | +0.00 | 72 | int8, group 128, searched bf16 scale |
| `lm_head` | 0.60 | 0.60 | +0.00 | 1 | int8, group 128, searched bf16 scale |
| `token_embd` | 1.18 | 1.18 | +0.00 | 1 | checkpoint bf16 |
| PLE n-gram table | 95.37 | 95.37 | **+0.00** | 1 | `cast dtype=bf16`, plain; where it lives is a serve-time knob |
| MTP drafter, its non-expert rows | 0.19 | 0.21 | +0.02 | 16 | `u2` draft head, two `fp8_e4m3` mixing rows, bf16 mixers — its 512 experts ride the row below |
| experts, bf16 (protected) | 0.09 | 0.29 | **+0.20** | 64 | `cast dtype=bf16` |
| experts, quantized | 63.92 | 73.41 | **+9.49** | 50,112 | `w5nl64a8h`: u5 codes, GPTQ cd=3, fwht128 |
| rows itemised above | 164.78 | 175.64 | **+10.87** | 50,611 | |
| not itemised: vision tower 0.84, `ffn_gate_inp` routers, PLE side rows, norms/convs | 1.08 | 1.09 | +0.01 | 878 | mostly checkpoint bf16 |
| **container file** | **166.07** | **176.94** | **+10.87** | 51,489 | +0.21 of it is container sections and plane alignment |

### The expert rung

`w5nl64a8h` in the engine's served-encoding table — u5 codes into libquant's 32-level non-linear
table, an E4M3 scale per group of 64 with an f32 second level under a fixed 2⁻¹³, rotated by the
128-wide Hadamard. It is applied to **every routed expert of all 49 expert layers — the 48 trunk
layers and the MTP block — on both projections**: 50,112 rows once the 32 protected ones step out.
The trunk rows are coded by GPTQ with coordinate descent 3 (recipe rule `:173`); the MTP block's
1,024 expert rows are coded by `rtn` (rule `:172`) and named *ahead* of the expert rule, because layer
48 has no Gram in any recording and **asking GPTQ for a group with no Hessian does not fail loudly —
it falls back silently to an unreviewed plane.** First-match-wins is what makes that line's position
load-bearing. The published radiance rungs of this checkpoint lift only `down` to five bits, because
that rung was priced as a byte-for-error trade against prefill; this container buys the whole rung and
the drafter with it (+0.21 GiB of the bill) under the fleet's floor rule that nothing here sits at
four bits. The bill splits by projection the way the shipped `rad-info` dumps sum it: `gate_up`
39.49 → 49.13 GiB (+9.64) across its 25,088 tensors, 0.515625 → 0.640625 B/element; `down` was
already u5 and moves 24.52 → 24.57 GiB — the +9.49 GiB quantized plus +0.20 GiB protected of the
table above.

### 32 protected experts, derived not guessed

32 individual routed experts (64 weight rows — `gate_up` and `down` of each, one `cast dtype=bf16`
line covering both) sit at whole bf16, spread over 13 trunk layers and never more than four in one.
`scripts/derive-protected-experts.py` picks them off the published Grams: an expert is admitted when
its activation-energy share clears the bar **and** its energy is at least 3× its routing share (this
campaign's bar landed at 0.827 %), and `--target-count 32` then tops the set up by relaxing **only**
the ratio bar — the energy bar never relaxes, so asking for more protected experts cannot quietly buy
experts the census did not earn. Two engine laws bound the set:

- **Even on each rank's half.** A rank owns a contiguous 256-expert half of a layer and the four-bit
  GEMM takes that half's remainder as two equal tables, so an odd count is repaired by admitting the
  next most energetic candidate (`blk.4: half 1 odd -> added e424` in the derive report).
- **Eight per rank, enforced at load.** `kMaxProtected` (`arch/common/rad_block_moe_fp8.h:114`) caps a
  rank at 8, so a TP=2 layer may hold 16. `declare()` checks it when the container is *loaded*: a
  ninth line on a rank costs a full convert before anything refuses it. Deriving with the cap in front
  of you is the cheap side of that trade.

The lines that shipped are in `artifacts/recipe-as-shipped.recipe` under `PROTECTED-BEGIN`. The
committed recipe deliberately holds none — which is why the count is an argument to
`scripts/apply-protected-cast.sh <grams> <recipe> 32`, not a line in the file. The container this one
replaced protected **10** experts, in layers 34, 44, 46 and 47.

### Two laws that shaped the map

- **A group that eats the trunk's quantized activations carries the trunk's encoding.** The first
  convert of this campaign died six seconds in: `rad-convert` refused an fp8 shared expert — *"'
  blk.0.ffn_gate_up_shexp.weight' is not an int8 weight, and its input is fed int8 codes because
  this model's trunk is int8; quantise it with its neighbours."* Per-group kernel availability is not
  the constraint. So `shared expert` is int8, byte-for-byte the shipped container's encoding, and fp8
  survives here on exactly **two rows** — the MTP block's `fc_embedding`/`fc_hidden`. Every `*_hc_*`
  mixer is bf16, and that swap is the +1.16 GiB in the table above. The sentence is quoted from the
  campaign plan and the recipe header comment, not from a log: the driver writes each stage to one
  file, the successful run overwrote `convert.log`, and the attempt that refused left behind only
  `--plan-only failed (rc=1); nothing was written`. **Capture a refusal's stdout or its reason is
  gone.**
- **A gate that hard-codes an engine law is a second copy of it.** With the shared expert corrected to
  int8, the driver's own recipe readback refused the next attempt: `VERDICT FAIL stage=draft: the
  shared expert is not the fp8 e4m3 row`. The assertion was still demanding the encoding the engine
  had just rejected. It now compares the shared expert against *the trunk's* encoding, whatever that
  happens to be — the only form of the check that survives the next trunk change.
- **The PLE n-gram table cannot be dual-encoded.** `arch/common/rad_block_ple.h` picks its gather path
  from the tensor's encoding — the bf16 path only for a `plain` single-plane encoding. One tensor
  group, one encoding: bf16 plain it is. Where those 95.37 GiB *live* is a serve-time knob, not a byte
  in this file: `--ngram-placement disk` streams them and pins nothing, while `ram` — what this fleet
  runs — pins the whole of them (`the table is 95.37 GiB: as asked`; the boot prints `n-gram table
  copied into pinned memory the cards read, 95.37 GiB in 60.5 s`, then `pinned by this process 140.18 GiB`).

## Calibration provenance

The Hessians are not ours. This container is calibrated on the **published Grams** of
[`gostar77/qwen3.8-flash-next-radiance-q5-w5dn64-grams`](https://huggingface.co/datasets/gostar77/qwen3.8-flash-next-radiance-q5-w5dn64-grams):
24,768 RADGRAM2 files (39 GiB), `ready: true`, recorded from the exact original BF16 checkpoint —
1,006,978,080 routed `down` rows and 2,097,871 pooled `gate_up` rows. Coverage as the door verified
it: 23,553 of 24,576 per-expert `down` Grams carry ≥ 4,096 routed rows and 1,023 fall back to the
layer's pooled Hessian. That 4,096 is the door's *report*, not its gate; the engine's own fallback
floor is 2,560 routed rows and the convert log names every expert it moves — `'blk.47.ffn_down_exps.405'
was routed 2283 calibration rows, fewer than 2560; it reads the layer's Hessian`.
`scripts/grams-to-imatrix.py` re-voices the same recording for `--imatrix` (63,155,748 B, 10,243/4096
chunks, 41,957,420 rows, 288 entries across 48 layers and 24,576 experts). Because Grams are
activation statistics, every grid change re-converts and never re-taps.

**In plain English:** someone else already measured, on the exact full-precision model, how each
expert weight group correlates across a billion real tokens. We bought nothing there; we spent that
measurement at five bits instead of four.

## Accuracy evidence — and its honest limit

**No KL, top-1 or perplexity score was taken for this file.** That instrument needs a BF16 reference
recording over a scoring corpus, and this host has none; building it is ~17 h of a drained machine
plus two score passes. The design decisions were priced from the published accuracy ladder, the
measured per-group byte bills, and the pinned-RAM ledger, and the artifact's dtype map was then read
back off the file and asserted. So:

| claim | status |
|---|---|
| dtype map = the ordered map that was ordered | **measured** (`rad-info`, `PASS`) |
| byte bill per group | **measured** (table above) |
| solver reconstruction error per tensor | **measured** — the convert receipt's per-rule means: 5.057e-02 over the 49,088 GPTQ expert rows, 4.572e-02 over the 1,024 `rtn` MTP expert rows, 6.7e-03–8.7e-03 over the int8 trunk rows. Worst single expert 6.73 % raw / 6.72 % imatrix-weighted (`blk.5.ffn_down_exps.163`); the worst row in the file is the `u2` MTP draft head at 52 %, which is what two bits cost by design |
| mean KL / top-1 / PPL vs BF16 | **not measured — any number here would be a fabrication** |

The ladder it was priced against is the quantizer's own published measurement of **this checkpoint**
([`README-GPTQ-VALIDATION.md`](https://huggingface.co/gostar77/qwen3.8-flash-next-radiance-q5-w5dn64/blob/main/README-GPTQ-VALIDATION.md)),
66,793 full-vocabulary held-out positions against the uncalibrated radiance bootstrap:

| their container | mean KL | top-1 | PPL / BF16 PPL |
|---|---:|---:|---:|
| bootstrap, fp8 KV | 0.099725 | 92.468 % | 6.1510 / 6.3423 |
| IQ4R, fp8 KV | 0.106781 | 92.317 % | 6.2023 / 6.3423 |
| **their calibrated Q5, fp8 KV** | **0.093144** | **93.034 %** | 6.1807 / 6.3423 |
| bootstrap, bf16 KV | 0.096952 | 92.820 % | 6.3643 / 6.3423 |
| IQ4R, bf16 KV | 0.108551 | 92.384 % | 6.5391 / 6.3423 |
| their calibrated Q5, bf16 KV | 0.084917 | 93.288 % | 6.4987 / 6.3423 |

Two readings matter here. The fp8-KV column is the posture this fleet actually serves in, and their
Q5 beats both the bootstrap and IQ4R there. And the ladder has **no rung with five bits on both
projections** — the row this file is — so there is no published number to inherit: this design was
priced from those deltas plus the byte bill, and stops there. A ladder number is comparable only
inside its own table (their reference, their corpus, their percentile rule) and is not this fleet's
gate bar.

`scripts/kld-gate-qwen38.sh` is shipped: it compares two scored reports and refuses a bar whose
reference, corpus, document set, percentile rule or position count differs from the candidate's. If
you score this container, publish the report and this paragraph gets replaced.

## Serving shape (what it is proven on)

The live boot, at the shape below, with the n-gram table pinned. Load-bearing lines:

```
I STARTUP REPORT -- radiance 1.17.0
I   --ngram-placement                  ram                    [command line]
I                                      the table is 95.37 GiB: as asked
I   --embedding-placement              ram                    [auto]
I                                      the table is 606.25 MiB a rank: pinned in host memory and read
I                                      over the link in place, so its VRAM goes to the KV cache and
I                                      the experts
I   --vision-tower-placement           vram                   [auto]
I                                      the towers are 856.68 MiB a rank
I   vram split    weights 27.47 GiB (experts 22.90 GiB of it), KV cache 2.31 GiB, a card
I                 elastic 25.78 GiB   the experts keep what the host pool cannot hold; the cache
I                                     takes the rest
I   kv cache      2.31 GiB a card: 269484 tokens, 1.3 sessions at 200000
I   memory    ttm.pages_limit 200.00 GiB | pinned by this process 140.18 GiB
I   note experts-from-host: 13.97 GiB of weights a rank live in the pinned host pool and cross PCIe
   when they are used
```
The excerpt trims the ledger's column indentation. `artifacts/boot-ledger-ngram-ram.txt` is that boot
whole (826 lines); `artifacts/boot-ledger.txt` is the campaign's own boot at the same shape with
`--ngram-placement disk` (329 lines). The VRAM half of the split is identical in the two — that knob
moves host residency, not card bytes.

```bash
podman run -d --name frank-archon \
  --network host \
  --init \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 \
  --device /dev/kfd --device /dev/dri \
  -e RADIANCE_DISABLE_VMEM=1 \
  -v /path/to/models:/models:ro \
  -v /path/to/state/tune:/opt/radiance/share/radiance/tune \
  stilldeadcode/radiance@sha256:c2acbaa373c89272ec47a22befa1cf04d2ddf1de51c2baf534f5273e517f7100 \
  --model /models/qwen38-w5nl64-qsa-hc-bf16-cd3.rad \
  --host :: --port 7770 \
  --served-model-name frank-archon \
  --tp 2 --tp-wire exact \
  --max-num-seqs 4 \
  --max-model-len 200000 \
  --placement expert_tiered \
  --ngram-placement ram \
  --host-pool-mib 14336 \
  --gpu-headroom-mib 96 \
  --kv-cache-dtype fp8 \
  --prefix-cache-mib 12288 \
  --checkpoint-slots 32 \
  --num-speculative-tokens 3 \
  --max-num-batched-tokens 2048
```

That is the fleet's own serve line with its env values substituted, `FRANK_ARCHON_EXTRA_ARGS` empty.
The `tune` mount is where the kernel autotuner caches its picks — without it every boot re-benchmarks
the bands. `--ngram-placement disk` is the other placement this container has run in production: it
pins nothing and boots faster, and `ram` trades pinned host pages for never reading the table per
token. That choice belongs to your RAM, not to this file.

Sizing, from the engine rather than from a formula:

- The expert tier is tiered by design (`--placement expert_tiered`) and the ledger row above says who
  pays for the extra expert bytes: 22.90 GiB a card plus 13.97 GiB a rank in the pinned host pool is
  36.87 GiB, against 36.85 GiB — every routed expert byte in the file divided by two ranks. The
  resolver sizes that resident set against `--host-pool-mib` plus whatever the weights/KV split leaves
  in VRAM, **at the worst-case sequence's KV cost**. The 12,288 MiB pool the publisher validated for
  its lighter container came out ≈0.94 GiB a rank short for this one, and the resolver refused to
  allocate rather than boot starved, naming exactly two knobs: a bigger `--host-pool-mib` or a shorter
  `--max-model-len`. It did not offer to shrink the cache, which is why the window is what this fleet
  refused to trade — 14336 is what ships.
- The pinned ledger is `2 × host-pool + the prefix-cache host tier + max-num-batched-tokens + the
  checkpoint store + the n-gram table IN FULL while it is pinned (97,659 MiB — the 95.37 GiB the boot
  reports)`. That is 144,763 MiB, and the boot agrees: `pinned by this process 140.18 GiB`. On a box
  with a `ttm` pages cap the sum is the floor the pre-flight checks: 145,000 MiB under
  `--ngram-placement ram`, 48,000 under `disk`, where the table term drops out.
  `--max-num-batched-tokens 2048` sizes the prefill staging arena, which is expert bytes, not window.
- `seccomp=unconfined` and unbounded `memlock` are both load-bearing: the disk tier's expert reads
  go through io_uring and NUMA binding, and the pinned pool is charged against `RLIMIT_MEMLOCK`.
- A prompt over `--max-model-len` is refused, never truncated.

**No throughput figures are published here.** Nothing was benched beside the production service, and
a rate lifted from a first request after a boot is a warmup artifact. Decode is carried by the MTP
drafter plus prompt lookup — check the `speculation:` line of your own boot before believing anyone's
number, including this file's silence.

What *is* verified in production, on 2026-11-09 at the shape above: `/health` answers
`{"status":"ok","model":"frank-archon",…}`; one chat turn with the reasoning channel in
`reasoning_content`; a `tools` request returning a well-formed `tool_calls`; a solid-color image
probe answered `Red` through the OpenAI `image_url` part. `GET /v1/models` reports
`max_model_len: 200000` and `reasoning_efforts: [none, minimal, low, medium, xhigh]` — the template
in this container rejects `high` and `max` with a 400, so map your client onto that list.

## What this replaced, and why

| | the container it replaced | this one |
|---|---|---|
| file | `qwen4exp-w5dn64-bf16ple-gptq.rad` | `qwen38-w5nl64-qsa-hc-bf16-cd3.rad` |
| size | 178,306,117,632 B (166.07 GiB) | 189,991,063,552 B (176.94 GiB) |
| routed experts | `w5nl` **down** only, `w4nl` gate/up | **`w5nl` both projections** (+9.49 GiB) |
| protected experts | 10 (layers 34, 44, 46, 47) | **32** derived, whole bf16 (+0.20 GiB) |
| QSA attention + HC mixers | `fp8_e4m3` | **bf16** (+1.16 GiB stored; the rows are sharded, ≈ +0.58 GiB a rank) |
| PLE n-gram table | bf16 | bf16 (where it lives is a serve-time knob, not a byte in the file) |
| MTP layer 48 | `w4nl` experts, `fp8_e4m3` mixers, `u2` head | `w5nl` experts (inside the +9.49 GiB), bf16 mixers, `u2` head |
| recipe | 29 rules | 53 rules |

+10.87 GiB of stored container for the widest expert rung the engine serves, full precision on every
row that routes, and **no four-bit tensor anywhere in the file** — the container it replaced carries
25,078 of them. The steering rows are **sharded, not replicated**: the file holds 7.66 GiB of
non-expert, non-PLE weight where a card carries 4.57 GiB of it, and the ledger prints `token_embd` at
606.25 MiB a rank — exactly half — while the vision tower (856.68 MiB a rank) and `lm_head` are
replicated. So the bf16 steering costs about +0.58 GiB a rank, and it comes out of the elastic expert
tier: out of speed, never out of the window.

## Files

| File | What |
|---|---|
| `qwen38-w5nl64-qsa-hc-bf16-cd3.rad` | the container (176.94 GiB) |
| `artifacts/rad-info-dtype-census.txt` | the per-group bill and the `PASS` assertion (ten groups, 50,611 rows) |
| `artifacts/rad-info-candidate-tensors.txt` / `-live-tensors.txt` | full `rad-info --verbose` dumps, 51,489 tensor rows each — diff them |
| `artifacts/recipe-as-shipped.recipe` | the 53-rule recipe that wrote the file: `:172` codes the MTP block's experts by `rtn`, `:173` the trunk's by `gptq cd=3`, `PROTECTED-BEGIN` carries the 32 spliced cast lines |
| `artifacts/VERDICT.txt` | the campaign's own verdict block (including its appended correction) |
| `artifacts/boot-ledger.txt` | the campaign's boot at the shipped shape, `--ngram-placement disk` (329 lines) |
| `artifacts/boot-ledger-ngram-ram.txt` | the live boot with the table pinned, 2026-11-10 (826 lines) — same VRAM split, different host residency |
| `artifacts/grams-door.log`, `imatrix.log`, `checkpoint-verify.log` | the input gates: Grams coverage report, imatrix bill, the 131-shard BF16 verification |
| `artifacts/convert-arm.log`, `convert-campaign.log`, `draft-plan.log`, `payload.sha256` | the convert receipt (51,489 weights in 21,088 s, per-rule solver error), the MTP draft plan, the deployed file's hash |
| `artifacts/run-20261109-*.log` | every stage gate of every attempt: `021144` cut short before any verdict, `022013` refused at the Grams door (`GRAMS_INCOMPLETE`, healed by one `--fetch` pass), `030443` at the convert plan, `030930` at the driver's own stale assertion, `031015` produced this file |
| `LICENSE` | the base model's license text (qwen-community-1.0) |
| `scripts/` | the campaign driver, arm driver, protected-expert derivation, Grams→imatrix, dtype census, trunk biller, both fetch doors, the KLD gate door, the recipe and the trunk fragment |

## Reproduce it

```bash
scripts/download-qwen38-bf16.sh --fetch     # the vendor BF16 checkpoint, 131 shards, 335.3 GiB
scripts/download-qwen38-grams.sh            # the published Grams, reassembled and validated
CALIB=<grams-flat> scripts/convert-qwen38-w5nl64.sh --shape qsa-hc-bf16 --cd 3
```

`scripts/run-onequant.sh` is the unattended, CPU-only driver that chains those doors with a gate at
every stage (image digest must equal production's; BF16 shard sha256 vs the Hub's LFS record; Grams
manifest + validation; recipe `--dry-run` readback so a wrong dtype map is caught *before* hours of
converting; `--plan-only` byte bill and a free-space gate; then `rad-info` asserted group by group).
It never opens `/dev/kfd`, so a serving endpoint keeps serving while it runs. Disk peak: the plan
(176.81 GiB) plus room, and the receipt requires the artifact to land within 98–103 % of the plan.

## License chain

```
Qwen/Qwen3.8-Flash-Next                       license: qwen-community-1.0
  ├─ gostar77/…-radiance-q5-w5dn64-grams      the published Hessians this file was calibrated on
  └─ this .rad                                GPTQ w5nl, 32 derived bf16 experts, bf16 steering trunk
```

This repository carries the base model's `LICENSE` (qwen-community-1.0, the file from
`Qwen/Qwen3.8-Flash-Next`); its acceptable-use terms apply downstream to this container. The
quantization, the protected-expert derivation, the campaign scripts and the measurement artifacts here
are mine, released under the same terms as the base model. Credit to
[`gostar77`](https://huggingface.co/gostar77) for the published Grams and for
the radiance quantization rungs this campaign was priced against, and to
[StillDeadcode](https://github.com/StillDeadcode) for the engine.
