---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: paiton
pipeline_tag: text-generation
inference: false
tags: [paiton, vllm, amd, rocm, rdna4, gfx1201, int3, 3-bit, w3a8, gptq, hadamard, moe, mtp, speculative-decoding]
---

# Qwen3.8-Flash-Next 3-bit (W3A8) for Paiton on two RDNA4 GPUs

These are 3-bit weights for [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(revision `de4b8e4d`), quantised with our own GPTQ run and served by Paiton's native RDNA4 runtime on two
AMD Radeon AI PRO R9700 cards. The 512 routed experts of all 48 MoE layers are stored at 3.125 bits per weight in
a rotated 8-level codebook format; everything else is 8-bit or bf16, so the whole language model is resident in
the two cards' 64 GB of VRAM with the model's n-gram embedding table in pinned system RAM. The repository is
self-contained: quantised experts, trunk, speculative (MTP) layer, draft head, n-gram table and the bf16 vision
tensors. Measured through the OpenAI-compatible server with BetterBench 0.6.0, a single request decodes at
**216.3 tok/s** (weighted over eight prompt categories), eight concurrent requests at **565.1 tok/s**, and a
64K-token prompt prefills at **8,161 tok/s**. Against the bf16 model, the full-vocabulary KL on text and assistant
positions is 0.060 (0.041 with the bf16 routing), with 90.7 % top-1 agreement.

**GPUs:** 2 × Radeon AI PRO R9700 (`gfx1201`, 32 GB each) · **Size on disk:** 108 GiB (116 GB) ·
**Context:** 262,144 tokens natively (benchmarked at 98,304 for decode and 200,000 for prefill) · **Text weights,
MTP layer and draft head; bf16 vision tensors included**

## What you get

All numbers are BetterBench 0.6.0 through the vLLM OpenAI-compatible server on two R9700 cards (tensor parallel 2,
experts split across the two cards), exact wire (no lossy activation transport), with BetterBench's defaults:
temperature 0.7, top-p 0.95, top-k 20, 20 measured passes per category after 3 warm-ups, a unique nonce per
request so the prefix cache is always cold, 48 completed requests per concurrency level, and 8 runs per prefill
depth. Decode, latency and concurrency lines come from one server configuration (MTP depth 3, 98,304-token
context window); the prefill line comes from a second configuration of the same build (MTP off, 200,000-token
context window, 2,048-token prefill chunks).

| Line | This release | Conditions |
|---|---:|---|
| Decode, 1 request, weighted over 8 categories | **216.3 tok/s** | release image rc8, run `rc8img-0937`, 10 Oct 2026 10:10 UTC |
| Aggregate throughput, 8 concurrent requests | **565.1 tok/s** (48 of 48 requests completed) | release image rc8, run `rc8img-0937`, 10 Oct 2026 10:10 UTC |
| Prefill, 64,000-token prompt (BetterBench 64K row) | **8,161 tok/s** | release image rc8, run `rc8img-0937`, 10 Oct 2026 10:10 UTC |
| Time to first token, short prompts (p50) | **117 ms** | release image rc8, run `rc8img-0937`, 10:10 UTC |
| Update interval p99 | **13.9 ms** per stream update | release image rc8, run `rc8img-0937`, 10:10 UTC; see the MTP note below |

Decode per category (tok/s): chat 202.1, code 214.3, file edit 236.0, JSON 259.2, math 255.5, prose 183.3,
reasoning 190.1, summarization 239.8; aggregate at 1 / 2 / 4 / 8 concurrent requests 204.6 / 315.1 / 449.2 / 565.1 tok/s (run `rc8img-0937`).

Prefill by prompt depth (tok/s, 2,048-token chunks): 2,000: 7,925 · 8,000: 8,321 · 16,000: 8,311 ·
32,000: 8,232 · 64,000: 8,161 · 128,000: 7,928. A separate probe with exactly 64,000 prompt tokens and a cold
prefix cache measured 8,212 tok/s over the whole prompt (8,204 tok/s steady state after the first chunk).

MTP note: with speculative decoding, the server emits several accepted tokens in one stream update (median 2.85
tokens per update in this run; 160 of 160 single-stream runs were classified as batched by BetterBench). Decode
tok/s and time to first token are unaffected by this; the update interval p99 is the p99 gap between stream
updates, not between tokens (about 5.3 ms per token).

Peak VRAM during the runs (`mem_info_vram_used`): 30.6 GiB and 30.5 GiB on the two cards for the decode
configuration, 31.4 GiB and 31.5 GiB for the 200,000-token prefill configuration.

## How to run

The weights are served by the Paiton container through the
[public Paiton plugin repository](https://github.com/Eliovp-BV/paiton-vllm-plugin), which pins the runtime image
and provides the launcher. Transformers, stock vLLM and llama.cpp cannot load this format.

The launcher lives in the plugin repository under `models/Qwen3.8-Flash-Next` and pins the runtime image
`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261010-r2`
(digest `sha256:ea84a147c6dd44f47a41836ace6b39864fab940095c4c0ba13e82537dd9e2a4b`). It downloads and verifies these
weights on first use, builds the runtime view next to them, and serves the OpenAI-compatible API as model name
`Qwen3.8-Flash-Next` at `http://127.0.0.1:18982/v1`.

```bash
git clone https://github.com/Eliovp-BV/paiton-vllm-plugin && cd paiton-vllm-plugin/models/Qwen3.8-Flash-Next
./run-flashnext.sh          # decode mode: 98,304-token window, speculative decoding on (the benchmark numbers)
./run-flashnext-200k.sh     # 200,000-token window, speculative decoding off
./run-flashnext-exact.sh    # exact-arithmetic prefill path, about 7 % slower prefill
./run-flashnext-cached.sh   # decode mode with the prefix-caching opt-in
./run-flashnext.sh --dry-run   # any wrapper: print the docker command only
```

The wrappers call `launch-flashnext.py`, which also takes `--mode decode|prefill-long|decode-nopf|prefill-long-nopf`, `--prefix-caching`,
`--port`, `--weights DIR`, `--cache DIR` and `--image REF` directly.

Both benchmark configurations ran the model through that launcher: tensor parallel 2 over the two R9700 cards, the
routed experts split across the cards by the routing counts shipped in `common/routing-counts.json`, the n-gram
(PLE) table in pinned system RAM, and the MTP layer in this repository as the speculative drafter (depth 3) for the
decode runs. The decode mode uses a 98,304-token context window; the prefill-long mode a 200,000-token window with
MTP off. Manual download (about 116 GB; the optional `contract-probe/` directory of 2.2 GiB can be skipped):

```bash
hf download EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4 --local-dir "$PAITON_FLASHNEXT_DIR"
(cd "$PAITON_FLASHNEXT_DIR" && sha256sum -c SHA256SUMS)
```

## Accuracy

Everything in this section was measured against the bf16 model on the same evaluation corpus with an
emulation of the served arithmetic (the same expert, trunk and head arithmetic the runtime executes); the
runtime's served-KL gate, which measures the running server with the same formula, passed against this
container (0.0616 against a threshold of 0.0600 + 0.002).

**What the KL number means.** KL t+a is the average Kullback-Leibler divergence from the bf16 model's next-token
distribution to this model's, over the full vocabulary, at the text and assistant positions of a
294,912-position evaluation corpus (170,884 such positions); zero would be identical, and two correct bf16
implementations already differ by about 0.008. "Free" means the quantised model chooses its own experts;
"routing locked" forces the bf16 model's expert choices, which isolates the arithmetic error from routing flips.
Top-1 is how often the most likely next token agrees with bf16.

### Format comparison on the evaluation corpus (full set)

| Weights | bits / weight (routed experts) | KL t+a, free | KL t+a, routing locked | top-1 agreement |
|---|---:|---:|---:|---:|
| bf16 (two correct implementations against each other: the floor) | 16 | 0.0077 | n/a | 96.77 % |
| **INT3 (this repository)** | 3.125 | **0.0604** | **0.0413** | **90.74 %** |
| INT3-hq (measured, not released) | 3.236 | 0.0490 | 0.0323 | 91.65 % |
| INT4 ([W4A8 repository](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4)) | 4.125 | 0.0288 | 0.0135 | 93.51 % |
| MXFP4, standard OCP round-to-nearest (reference measurement, no container) | 4.25 | 0.0515 | 0.0320 | 91.69 % |

The MXFP4 row is a reference measurement of the standard format (FP4 E2M1 elements, E8M0 block scale per 32
weights, no calibration) applied to the same experts with the same trunk and arithmetic, so the table isolates the
weight format. An INT3-hq variant we measured but do not ship yet (3-bit everywhere, 4-bit down projections in layers 24-39) matches
MXFP4 quality at 24 % fewer bits: free 0.0490 vs 0.0515, locked 0.0323 vs 0.0320, top-1 91.65 vs 91.69 %.
The INT4 cut is 44 % lower in free KL and 58 % lower in locked KL than MXFP4 with 3 % fewer bytes. Replicate
spread of the GPTQ recipe (two calibration replicates of this format: 0.0630 vs 0.0686 free on the confirmation
half) is larger than some of the differences in the table.

### Long context (long-v1: 44 held-out documents of 8K-32K tokens, 524,288 positions)

KL over all positions of each bucket by position in the document, free / routing locked, with the text+assistant
top-1 of the bucket in brackets; the last column is the document-level text+assistant KL free / locked / top-1.
The ≥ 8K bucket has 12 documents, the others 44.

| Weights | < 2K | 2-4K | 4-8K | ≥ 8K | doc-level t+a |
|---|---|---|---|---|---|
| **INT3 (this repository)** | 0.1554 / 0.1043 [89.46 %] | 0.6300 / 0.2931 [89.66 %] | 0.6408 / 0.2984 [90.06 %] | 0.4333 / 0.2150 [91.65 %] | 0.0561 / 0.0408 / 90.39 % |
| INT3-hq (measured, not released) | 0.1377 / 0.0838 [89.54 %] | 0.5612 / 0.2376 [90.24 %] | 0.5881 / 0.2475 [90.49 %] | 0.3953 / 0.1796 [92.11 %] | 0.0529 / 0.0347 / 90.80 % |
| INT4, cut A (the staged 4-bit container, not yet released) | 0.0872 / 0.0434 [92.42 %] | 0.4305 / 0.1363 [92.84 %] | 0.4717 / 0.1492 [93.08 %] | 0.3204 / 0.1049 [94.14 %] | 0.0267 / 0.0149 / 93.26 % |
| INT4, cut B (measured, not released) | 0.0786 / 0.0334 [93.06 %] | 0.4031 / 0.1162 [93.39 %] | 0.4433 / 0.1296 [93.62 %] | 0.3094 / 0.0944 [94.61 %] | 0.0225 / 0.0115 / 93.80 % |
| MXFP4 (reference) | 0.1313 / 0.0756 [90.84 %] | 0.5866 / 0.2618 [91.24 %] | 0.6108 / 0.2794 [91.64 %] | 0.4178 / 0.1979 [92.55 %] | 0.0395 / 0.0266 / 91.72 % |

No bf16 floor exists for this set (no second bf16 pass). Against the previous 3-bit container, the
routing-locked KL on 8K-token rows improves (0.206 vs 0.222 over all positions, 0.246 vs 0.261 beyond position
6,144) and the free KL is within noise (0.428 vs 0.424, 0.522 vs 0.515).

**Negative log-likelihood on the same long documents** (text+assistant positions, nats per token, bf16 → this
quantisation, with the paired difference; document-level column last). The quantised models lose 0.4-0.7 % of the
bf16 log-likelihood on assistant text at 3 bits and 0.2-0.3 % at 4 bits, and the loss does not grow with position.

| Weights | < 2K | 2-4K | 4-8K | ≥ 8K | doc-level t+a |
|---|---|---|---|---|---|
| bf16 reference | 1.8416 | 1.7573 | 1.6911 | 1.7858 | 1.7574 |
| **INT3 (this repository)** | 1.8498 (+0.0082) | 1.7682 (+0.0108) | 1.7020 (+0.0110) | 1.7942 (+0.0083) | 1.7671 (+0.0097) |
| INT3-hq (measured, not released) | 1.8538 (+0.0122) | 1.7690 (+0.0116) | 1.6944 (+0.0034) | 1.7920 (+0.0061) | 1.7646 (+0.0071) |
| INT4, cut A (staged, not yet released) | 1.8467 (+0.0051) | 1.7624 (+0.0050) | 1.6918 (+0.0008) | 1.7881 (+0.0022) | 1.7601 (+0.0027) |
| INT4, cut B (measured, not released) | 1.8451 (+0.0035) | 1.7602 (+0.0028) | 1.6869 (−0.0041) | 1.7888 (+0.0030) | 1.7580 (+0.0006) |

**fp8 KV cache (measured, not used by any released mode).** Both released modes keep the bf16 KV cache. An E4M3 KV cache with one scale per
token passed the accuracy gate below but gave no throughput gain at eight streams on this container, so it stays out. Measured against the same container with a bf16 KV cache on the long documents (text+assistant KL, free /
routing locked; positions 4K-6K, ≥ 6K and document level; top-1 change), the option costs:

| Weights | 4K-6K | ≥ 6K | doc-level | t+a top-1 |
|---|---|---|---|---|
| **INT3 (this repository)** | +0.0023 / −0.0021 | +0.0012 / +0.0005 | +0.0013 / +0.0000 | −0.05 pt |
| INT3-hq (measured, not released) | −0.0069 / +0.0009 | −0.0026 / +0.0007 | −0.0022 / +0.0007 | +0.00 pt |
| INT4, cut A (staged, not yet released) | +0.0053 / +0.0004 | +0.0011 / +0.0007 | +0.0017 / +0.0006 | −0.11 pt |
| INT4, cut B (measured, not released) | +0.0026 / −0.0003 | +0.0007 / +0.0004 | +0.0009 / +0.0003 | −0.15 pt |

Over all positions of the documents (not only assistant text) the same option adds 2-8 % to the bucket KLs of the
INT3 and INT4 cuts and nothing measurable to INT3-hq.

**Needle retrieval at 131,072 and 200,000 tokens:** 40 of 40 at both lengths on the served stack (10 Oct 2026, run NEEDLEQ5), identical answers to the bf16 model (which retrieves 40 of 40
needles at both lengths with the same prompts).

### Speculative decoding

The MTP layer's experts are INT4 (GPTQ, calibrated on the quantised target). Expected accepted drafts per step at
depth 3 (teacher-forced, greedy, text+assistant positions): 1.776 with the exact int8 head and 1.768 with the
int2 draft head shipped in `mtp/` (top-16 rescored with the exact head), against 1.789 for the bf16 MTP layer on
the bf16 model.

## Format

- **Routed experts (gate/up and down of all 48 MoE layers, 512 experts each, 120.8 billion weights).** 8-level
  codebook ±{1, 3, 5, 8} with an E4M3 scale per (row, 64 weights) times a per-matrix power of two: 3.125 bits
  per weight, 47.2 GB of expert weights. The codes are stored in a block-diagonal Hadamard-128 basis along the
  input dimension; the runtime applies the matching Walsh-Hadamard transform to the expert inputs online.
  Expert inputs are quantised to fp8 (E4M3) per token after the transform (the "A8"), the group sums are exact,
  and each expert's output is rounded to bf16 before the weighted combine. Calibration: sequential GPTQ on
  2 million tokens (best of two calibration replicates, selected on one half of the evaluation corpus and
  reported on the other).
- **Trunk (W8A16).** Attention q/o projections, Gated DeltaNet projections, the shared expert, the
  hyper-connection mixers, the n-gram key/value projections and the final mixer are E4M3 per output row (from
  GPTQ W8), multiplied with unquantised bf16 activations in fp32.
- **Output head.** int8 per (row, 128) with a searched signed bf16 scale and fp32 logits; E4M3 rows are shipped
  as a fallback head.
- **Router, attention, Gated DeltaNet, hyper-connections.** fp32 router logits (softmax, top-10, renormalised);
  everything else in bf16 with the checkpoint's semantics, including the KV cache in the emulation.
- **n-gram (PLE) table.** E4M3 codes with an fp32 scale per row, 320,001,536 rows × 160, read at layer 1
  (16 rows per token); kept in pinned system RAM (48.9 GiB), never on the GPU.
- **MTP layer and vision tower.** MTP experts INT4 g128 (signed bf16 scale per 128, same Hadamard-128 layout);
  int2 draft head (`mtp/draft-head-int2.*`, levels −2..1 with a bf16 scale per (row, 128)); all other MTP
  tensors and the vision encoder bf16 as in the checkpoint (`common/bf16-mtp-vision.safetensors`).
- **Container.** `manifest.json` (format `fnq-container-1`, name `RX-B3h-r2`) records every tensor's file,
  dtype, shape, format, level table, scale tensor, rotation and consumer activation rule, the bit-exactness checks
  of the export run, the KL of this exact container, and the sha256 of every file. `FORMAT.md` is the byte-level
  specification; `QUALITY.md` the accuracy summary of this container.

## Context, speculative decoding and vision

- **Context.** The base model's native context is 262,144 tokens. The launcher ships two modes: `decode` serves a
  98,304-token window with speculative decoding on (the decode numbers above), `prefill-long` serves a 200,000-token
  window with speculative decoding off (the prefill numbers above, measured up to 128,000 tokens). The default prefill
  path is the fast native one (the numbers above); an exact-arithmetic prefill mode is available as a launcher flag
  (about 7 % slower prefill, decode unchanged). The 200,000-token
  window was tested end to end on the served stack: needle retrieval at 200,000 tokens 40 of 40 with the same answers
  as the bf16 model, and prefix caching (opt-in flag) returns byte-identical outputs on cache hits, a repeated 128K
  prompt in 0.41 s instead of 20 s. Single-stream decode speed in the 200K mode (speculative decoding off): about 100 tok/s, flat with prompt length (105.3 / 100.6 / 99.9 tok/s after 32K / 100K / 190K-token prompts, 256-token continuations); time to first token for a 190K prompt 24.8 s.
  The next release adds the RAM and SSD KV-cache tier (prefix-cache offload outside the VRAM) as on the 27B release,
  which is what longer-than-VRAM contexts need; a 4-bit container follows.
- **Speculative decoding (MTP).** The repository ships the quantised MTP layer and the int2 draft head; the
  decode benchmark used depth 3 (about 2.9 accepted tokens per update).

- **Vision.** The vision encoder is shipped unchanged in bf16 (`common/bf16-mtp-vision.safetensors`). Image input is not validated in this release; the released modes are text-only.

## Hardware requirements

| Component | Requirement |
|---|---|
| GPUs | Two AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4, 32 GB each) in one host; both benchmark configurations peaked at 30.5-31.5 GiB per card |
| Model memory | 24.9 GiB of text weights per card plus 0.7 GiB for the MTP layer; the rest of each card is KV cache, GDN state and graphs |
| System RAM | The n-gram table is pinned in host RAM: 48.9 GiB in total, split across the two ranks, plus the runtime's own working set. The benchmark host has 251 GiB |
| Disk | 108 GiB (116 GB) for this repository; 2.2 GiB of it is the optional `contract-probe/` |
| Host | Linux x86-64, Docker, AMD GPU device access (`/dev/kfd`, `/dev/dri`); runtime image `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261010-r2` |

## Files and checksums

| Path | Size | What it is |
|---|---:|---|
| `layers/L00.safetensors` … `layers/L47.safetensors` (48 files) | 47.49 GiB | one decoder layer each: 3-bit routed experts (codes and scales), E4M3 trunk projections, bf16 keep-list tensors, under the checkpoint tensor names plus a format suffix |
| `model.safetensors` | 2.38 GiB | bf16 token embeddings, final hyper-connection mixer, int8 output head with its E4M3 fallback |
| `mtp/mtp-experts.safetensors`, `mtp/mtp-manifest.json` | 1.21 GiB | INT4 experts of the MTP layer and their format record |
| `mtp/draft-head-int2.safetensors`, `mtp/draft-head-int2.json` | 0.60 GiB | int2 draft head for speculative decoding and its format record |
| `common/ple-e4m3/shard_000.safetensors` … `shard_127.safetensors` (128 files) | 48.88 GiB | the n-gram (PLE) table, E4M3 codes with fp32 row scales, in checkpoint shard order |
| `common/bf16-mtp-vision.safetensors` | 5.69 GiB | the MTP layer's non-expert tensors and the vision encoder, bf16, unchanged |
| `common/routing-counts.json` | 0.3 MiB | per (layer, expert) routing counts from bf16 traces, used to place experts across the two cards |
| `contract-probe/probe-prompts.npz`, `contract-probe/probe-logits.npy` | 2.18 GiB | optional: 20 prompts and this container's served-arithmetic logits, for verifying a runtime against the emulation |
| `manifest.json` | 0.8 MiB | per-tensor format index, calibration provenance, checks, per-file sha256 |
| `FORMAT.md`, `QUALITY.md`, `FILES.md` | | format specification, accuracy summary, file list |
| `SHA256SUMS` | | sha256 of every file above; `mtp/` and `contract-probe/` also carry their own `SHA256SUMS` |

```bash
sha256sum -c SHA256SUMS
```

Total: 108.4 GiB (116,431,249,004 bytes). The tensor names are the checkpoint names; every quantised tensor
adds a suffix (`.e4m3`, `.row_scale`, `.gu_codes`, `.gu_scales`, `.dn_codes`, `.dn_scales`, …). Experts are stored
expert-major in one file per layer, so a rank can read only its own experts.

## Limitations

- The weights are lossy: KL 0.060 against bf16 (0.041 with the bf16 routing) and 90.7 % top-1 agreement on the
  evaluation corpus; on long documents the loss grows with position (table above). The INT4 repository is the
  choice for maximum accuracy; an INT3-hq variant we measured (not released) closes part of the gap at 3.236 bits per weight.
- The accuracy evaluation is KL- and top-1-based against bf16 plus the long-v1 document set and the MTP acceptance
  measurement; task benchmarks (math, code, knowledge), multilingual and tool-calling quality were not measured
  for this release.
- Two R9700 cards only; the format is tied to the pinned checkpoint revision `de4b8e4d` and to Paiton's rotation
  and activation rules. Transformers, stock vLLM and llama.cpp cannot load it.
- The benchmark numbers were measured with the benchmark configuration described above; the released launcher
  reproduce them: the published image re-measured 216.3 / 565.1 / 8,161 tok/s on this machine (run `rc8img-0937`).

## License and attribution

The base model [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is released by Qwen
under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE); these quantised weights are a derivative of it and are
distributed under the same license. The quantisation method builds on GPTQ. Calibration data: a 2-million-token mixed-domain corpus of our own (prose, code and assistant conversations).

Performance was measured with [BetterBench](https://github.com/GGZ14/BetterBench) 0.6.0 by GGZ14, through
the OpenAI-compatible server with the benchmark's default settings.

[Paiton](https://eliovp.com/products/paiton) · [Public Paiton plugin](https://github.com/Eliovp-BV/paiton-vllm-plugin)

## Compute used for this quantisation

The Qwen3.8-Flash-Next quantisation round ran on a shared AMD Instinct MI355X box from 7 Oct 2026 12:16 UTC to
8 Oct 2026 23:14 UTC (35.0 h of calendar time), with at most 6 GPUs at a time and every GPU capped at 600 W. It
used **167.1 GPU-hours** in 367 GPU jobs, which is 34.8 wall-clock hours of GPU occupancy (overlaps counted once)
at a mean of 4.8 GPUs busy. Per category: bf16 references, routing traces and served baselines 2.5 GPU-h; full
GPTQ quantisations of the shipped and candidate containers (INT4/INT5 and 3-bit expert families), each with
calibration, export, MTP experts, contract probe and KL evaluation, 54.2 GPU-h over 35 jobs, about 1.6-3.9 GPU-h
per full cut; KL, served-KL gate and long-context (8K-32K) evaluations of those cuts 8.6 GPU-h; MTP draft-layer
work (expert quantisation, int2 draft head, self-distillation, acceptance-aware training and acceptance
measurements) 14.9 GPU-h; research arms that shaped the recipe (format screens, trunk and contract studies,
sensitivity and error-budget runs, Hessian variants, scale refinement, fp8 KV/prefill and low-bit trunk screens)
84.4 GPU-h; a parked bulk-prefill approximation study 0.8 GPU-h and a parked INT6 trunk export 1.6 GPU-h. The
MXFP4 reference measurement in the accuracy section added 1.9 GPU-hours on the same box (0.5 GPU-h for the
standard MXFP4 rows, the rest for the control, the MTP arm and a secondary rotated-basis arm), for a total of
169.0 GPU-hours. CPU-only steps (packaging, checksums, byte-identity checks, corpus building) add well under one
CPU-hour.
