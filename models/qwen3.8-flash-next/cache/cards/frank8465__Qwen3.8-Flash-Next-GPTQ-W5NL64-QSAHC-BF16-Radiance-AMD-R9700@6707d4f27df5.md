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
projections and every hyper-connection mixer — stay at the checkpoint's bf16, as does the 95 GiB
per-layer-embedding n-gram table. Everything the experts actually store moves to five-bit GPTQ
codes, the widest rung of the four-bit class, on **both** projections rather than only `down`.

| | |
|---|---|
| File | `qwen38-w5nl64-qsa-hc-bf16-cd3.rad` (rename freely; the engine reads by content) |
| Size | 189,991,063,552 bytes (176.94 GiB) — 1.90× vs the 335.3 GiB BF16 source |
| SHA256 | `bdf7569b84e51c2375560c49eea560a65702d45c9ee268364cb1a30528024f59` — the deployed file and this upload are the same bytes |
| Engine | `stilldeadcode/radiance@sha256:c2acbaa373c89272ec47a22befa1cf04d2ddf1de51c2baf534f5273e517f7100` (1.17.0) |
| Architecture | `qwen4exp`, 48 trunk layers + 1 MTP draft layer, 24,576 routed experts, 262,144 native ctx |
| Proof hardware | 2× Radeon AI PRO R9700, TP=2 (`--tp-wire exact`), 4 × 200,000-token slots, fp8 KV |
| Calibrated with | the published Grams of [`gostar77`](https://huggingface.co/datasets/gostar77/qwen3.8-flash-next-radiance-q5-w5dn64-grams) — 1,006,978,080 routed rows (credit below) |

**In plain English:** this is the same model, smaller and slightly less exact, with the few parts
that steer the network kept exact and the enormous pile of expert weights kept "good enough". It
runs on two AMD pro GPUs in one box and answers at a 200,000-token window.

## What each group is stored as

Read back **off the artifact** with the engine's own `rad-info --verbose`, then asserted group by
group against the container this file replaced (`artifacts/rad-info-dtype-census.txt`, verdict
`PASS — candidate carries the ordered dtype map`; the full 50,611-row dumps of both containers are
in `artifacts/`). `live` = the container it replaced.

| group | live GiB | this GiB | delta | rows | encoding |
|---|---|---|---|---|---|
| QSA attention | 0.61 | 1.21 | **+0.59** | 52 | `cast dtype=bf16` |
| hyper-connection mixers | 0.62 | 1.18 | **+0.57** | 194 | `cast dtype=bf16` |
| shared expert | 0.23 | 0.23 | +0.00 | 98 | int8, group 128, searched bf16 scale |
| delta-net gates (`ssm_inz`/`ssm_out`) | 1.96 | 1.96 | +0.00 | 72 | int8, group 128, searched bf16 scale |
| `lm_head` | 0.60 | 0.60 | +0.00 | 1 | int8, group 128, searched bf16 scale |
| `token_embd` | 1.18 | 1.18 | +0.00 | 1 | checkpoint bf16 |
| PLE n-gram table | 95.37 | 95.37 | **+0.00** | 1 | `cast dtype=bf16`, plain, **disk-tiered** |
| MTP drafter | 0.19 | 0.21 | +0.02 | 16 | w5nl projections, fp8 mixing rows, `u2` draft head |
| experts, bf16 (protected) | 0.09 | 0.29 | **+0.20** | 64 | `cast dtype=bf16` |
| experts, quantized | 63.92 | 73.41 | **+9.49** | 50,112 | `w5nl64a8h`: u5 codes, GPTQ cd=3, fwht128 |
| **total container** | **164.78** | **175.64** | **+10.87** | 50,611 | |

### The expert rung

`w5nl64a8h` in the engine's served-encoding table — u5 codes into libquant's 32-level non-linear
table, an E4M3 scale per group of 64 with an f32 second level under a fixed 2⁻¹³, rotated by the
128-wide Hadamard, chosen by GPTQ with coordinate descent 3. Applied to **every routed expert of
all 48 trunk layers on both projections**, plus the MTP layer 48 draft expert. The published
radiance rungs of this checkpoint lift only `down` to five bits, because that rung was priced as a
byte-for-error trade against prefill; this container buys the whole rung. The bill is +9.49 GiB of
stored expert bytes, split by projection the way the shipped `rad-info` dumps sum it: `gate_up`
39.49 → 49.13 GiB (+9.64) across its 25,088 tensors, 0.515625 → 0.640625 B/element; `down` was
already u5 and moves 24.52 → 24.57 GiB. The two sum to the +9.49 GiB quantized plus +0.20 GiB
protected of the table above. Nothing comes out of the KV pool.

### 32 protected experts, derived not guessed

32 individual routed experts (64 weight rows: `gate_up` + `down` of each) are held at whole bf16.
The set comes from the published Grams' own statistics — energy share **and** a routing-share ratio
bar, admitted in pairs so each rank's 256-expert half stays even (the four-bit GEMM takes the rest
of a half as two equal tables by parity), under the engine's cap of eight protected experts per rank
per layer. `scripts/derive-protected-experts.py` produces the list; the exact lines that shipped are
in `artifacts/recipe-as-shipped.recipe` under `PROTECTED-BEGIN`. Upstream's own verdict on a *larger*
protected set is "tried without benefit: +4 % bytes for 7 % less error", so this file does not grow
it beyond what the census earned.

### Two laws that shaped the map

- **A group that eats the trunk's quantized activations carries the trunk's encoding.** The first
  convert of this campaign died six seconds in: `rad-convert` refused an fp8 shared expert — *"'
  blk.0.ffn_gate_up_shexp.weight' is not an int8 weight, and its input is fed int8 codes because
  this model's trunk is int8; quantise it with its neighbours."* Per-group kernel availability is not
  the constraint. So `shared expert` is int8, identical to the shipped container, and fp8 stays only
  where the producer is fp8 (the `*_hc_*` mixers, the MTP projections).
- **The PLE n-gram table cannot be dual-encoded.** `arch/common/rad_block_ple.h` picks its gather
  path from the tensor's encoding — the bf16 path only for a `plain` single-plane encoding. One
  tensor group, one encoding: bf16 plain it is, and it rides the **disk tier**
  (`--ngram-placement disk`) so the 95.37 GiB costs no pinned RAM at all.

## Calibration provenance

The Hessians are not ours. This container is calibrated on the **published Grams** of
[`gostar77/qwen3.8-flash-next-radiance-q5-w5dn64-grams`](https://huggingface.co/datasets/gostar77/qwen3.8-flash-next-radiance-q5-w5dn64-grams):
24,768 RADGRAM2 files (39 GiB), `ready: true`, recorded from the exact original BF16 checkpoint —
1,006,978,080 routed `down` rows and 2,097,871 pooled `gate_up` rows. Coverage as the door verified
it: 23,553 of 24,576 per-expert `down` Grams carry ≥ 4,096 routed rows, 1,023 fall back to the
layer's pooled Hessian (the engine prints each one). `scripts/grams-to-imatrix.py` re-voices the same
recording for `--imatrix` (63,155,748 B, 10,243/4096 chunks, 41,957,420 rows). Because Grams are
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
| solver reconstruction error per tensor | **measured** — the convert log's own figures: the ruled expert group averages 8.698e-03, the worst individual expert tensors land at 5.8–6.4 % relative |
| mean KL / top-1 / PPL vs BF16 | **not measured — any number here would be a fabrication** |

The ladder it was priced against, quoted from the quantizer's own published tables for this
checkpoint: the four-bit rung sits at mean KL 0.0789, the rung whose `down` is already u5 — the
container this one replaces — sits at 0.0717 with 93.34 % top-1, and the vendor's expected figure for
the full-five-bit move is stated as an *absolute* all-mean level of ≈0.066. Treat those as the
neighborhood, not as this file's score.

`scripts/kld-gate-qwen38.sh` is shipped: it compares two scored reports and refuses a bar whose
reference, corpus, document set, percentile rule or position count differs from the candidate's. If
you score this container, publish the report and this paragraph gets replaced.

## Serving shape (what it is proven on)

The engine's own boot ledger is in `artifacts/boot-ledger.txt`. The load-bearing lines:

```
I model: qwen4exp  arch=qwen4exp  quant=bf16  48 layers  262144 ctx
I device: the 2 cards reach each other's memory directly (peer to peer)
I speculation: 'mtp' drafts 3 token(s) a step in 3 serial pass(es); prompt lookup fills in for any sequence it declines
I   vram split    weights 27.47 GiB (experts 22.90 GiB of it), KV cache 2.31 GiB, a card
  elastic         25.78 GiB   the experts keep what the host pool cannot hold; the cache takes the rest
  kv cache         2.31 GiB   9% of it -- 34% backing of the 4 x 200000 worst case: 1.3 full-length
  host_pool       14.59 GiB budget,        0 B used,  14.59 GiB free
I   kv cache      2.31 GiB a card: 269484 tokens, 1.3 sessions at 200000
I     attn_paged      -> attn_decode_h256_gqa12_fp8kv
I     kv_store        -> kv_store_fp8
I     qsa_block_key   -> qsa_block_key_bf16
```
(column indentation trimmed to fit; the artifact carries the ledger whole, 329 lines.)

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
  --ngram-placement disk \
  --host-pool-mib 14336 \
  --gpu-headroom-mib 96 \
  --kv-cache-dtype fp8 \
  --prefix-cache-mib 12288 \
  --checkpoint-slots 32 \
  --num-speculative-tokens 3 \
  --max-num-batched-tokens 2048
```

That is the fleet's own serve line with its env values substituted, `FRANK_ARCHON_EXTRA_ARGS`
empty. The `tune` mount is where the kernel autotuner caches its picks — without it every boot
re-benchmarks the bands.

Sizing, from the engine rather than from a formula:

- The expert tier is disk-backed by design, and the engine's own ledger row says who pays:
  `elastic 25.78 GiB   the experts keep what the host pool cannot hold; the cache takes the rest`.
  The resolver sizes that resident expert set per rank against `--host-pool-mib` plus whatever the
  weights/KV split leaves in VRAM, **at the worst-case sequence's KV cost**. This payload's tier
  wanted ≈24.90 GiB a rank; the 12 GiB pool the publisher validated for its own lighter container
  came out ≈0.94 GiB short and the resolver refused to allocate rather than boot starved, naming
  exactly two knobs: a bigger `--host-pool-mib` or a shorter `--max-model-len`. The window is not
  negotiable on this fleet, so 14336 is what ships.
- The pinned ledger is `2 × host-pool + prefix host tier + max-num-batched-tokens`; on a box with a
  `ttm` pages cap, that sum is the floor the pre-flight checks. `--max-num-batched-tokens 2048`
  sizes the prefill staging arena, which is expert bytes, not window.
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
| protected experts | none | 32 derived, whole bf16 (+0.20 GiB) |
| QSA attention + HC mixers | quantized | **bf16** (+1.16 GiB, replicated per rank) |
| PLE n-gram table | bf16, disk | bf16, disk (unchanged) |
| MTP layer 48 experts | `w4nl` | `w5nl` (+0.02 GiB) |

+10.87 GiB of stored container for the widest expert rung the engine serves, full precision on every
row that routes, and no four-bit tensor anywhere in the file. The trunk is **replicated, not
sharded**, so that +1.16 Gi/rank comes out of the elastic split's expert tier — i.e. out of speed,
never out of window.

## Files

| File | What |
|---|---|
| `qwen38-w5nl64-qsa-hc-bf16-cd3.rad` | the container (176.94 GiB) |
| `artifacts/rad-info-dtype-census.txt` | the per-group bill and the `PASS` assertion |
| `artifacts/rad-info-candidate-tensors.txt` / `-live-tensors.txt` | full `rad-info --verbose` dumps of this container and the one it replaced — diff them |
| `artifacts/recipe-as-shipped.recipe` | the 53-rule recipe that actually wrote the file, with the 32 protected cast lines spliced in |
| `artifacts/VERDICT.txt` | the campaign's own verdict block (including its appended correction) |
| `artifacts/boot-ledger.txt` | the engine's STARTUP REPORT at the served shape |
| `artifacts/grams-door.log`, `imatrix.log`, `checkpoint-verify.log` | the input gates: Grams coverage, imatrix bill, 131-shard BF16 verification |
| `artifacts/convert-arm.log`, `convert-campaign.log`, `draft-plan.log`, `payload.sha256` | the convert receipt (51,489 weights, 21,088 s, per-tensor solver error), the MTP draft plan, the deployed file's hash |
| `artifacts/run-20261109-*.log` | every stage gate of every attempt, including the two that refused |
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

This repository carries the base model's license trio; the root model's qwen-community-1.0
acceptable-use terms apply downstream. The quantization, the protected-expert derivation, the
campaign scripts and the measurement artifacts here are mine, released under the same terms as the
base model. Credit to [`gostar77`](https://huggingface.co/gostar77) for the published Grams and for
the radiance quantization rungs this campaign was priced against, and to
[StillDeadcode](https://github.com/StillDeadcode) for the engine.
