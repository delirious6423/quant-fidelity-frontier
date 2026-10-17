---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: Avifenesh
pipeline_tag: text-generation
tags:
  - nvfp4
  - gguf
  - memra
  - speculative-decoding
  - mtp
  - conversational
  - blackwell
  - qwen3
model-index:
  - name: Qwen3.8-27B-NVFP4-MTP-GGUF
    results:
      - task:
          type: text-generation
          name: Speculative decode, MTP K=3 with the masked (FR-Spec) draft head
        dataset:
          name: memra held-out agentic prompt set (own-generated ranks corpus, held out)
          type: memra-heldout-agentic
        metrics:
          - name: decode p50 tok/s, RTX PRO 6000 Blackwell
            type: throughput
            value: 137.9
          - name: decode p50 tok/s, RTX 5090 Laptop
            type: throughput
            value: 73.6
          - name: draft acceptance rate, K=3
            type: acceptance-rate
            value: 0.74
        source:
          name: memra v0.86.2 run-spec harness, medians of 5 interleaved reps
          url: https://github.com/avifenesh/memra/blob/06687c44cee617dcaaedf3ba1c47c3ad1e9626d1/crates/memra-engine/src/bin/run_spec.rs
---

# Qwen3.8-27B — NVFP4 GGUF with MTP head (memra serving artifact)

> **Serving artifact of the tiyuvta lab.** This exact file is what the lab's engine
> [memra](https://github.com/avifenesh/memra) serves for Qwen3.8-27B. The lab's public work is
> making self-deployment of open-source frontier models cheaper, easier and faster on hardware
> you can afford — [tiyuvta.ai/services](https://tiyuvta.ai/services/?c=hf-qwen38). You can test
> the lab's live deployment (DeepSeek-V4.1-Flash) on your own requests before talking to us:
> [tiyuvta.ai/services/try](https://tiyuvta.ai/services/try/?c=hf-qwen38).
>
> The hosted `qwen/qwen3.8-27b` API endpoint previously advertised here was **retired on
> 2026-09-18**; the measurements below are dated engine/service receipts, not a live offer.

NVFP4 (4-bit e2m1, per-16 FP8-e4m3 scales) GGUF of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), quantized from the
BF16 release with `llama-quantize` (NVFP4 ftype; token embeddings and output
head at Q5_K; norms F32). The **MTP (multi-token-prediction) head ships in the
file** (`blk.64`, `nextn_predict_layers=1`) — speculative decode works out of
the box on engines that read it.

Built as the serving artifact for [**memra**](https://github.com/avifenesh/memra),
a from-scratch Rust + CUDA inference engine for RTX Blackwell (sm_120a) with
exactness gates: speculative, graphed, and batched serving are gated
byte-identical to plain decode per request. This artifact is what memra's
Qwen3.8 endpoint serves at native 262,144-token context.

- Engine: https://github.com/avifenesh/memra (MIT, crates.io: `memra-server`)
- Context: 262,144 native (text→text serving; the upstream VL tower is not
  included in this artifact)
- Vocabulary: 248,320; chat template embedded (tool calling + thinking blocks)
- Which models memra runs, and the recommended artifact + flags per family:
  [docs/MODELS.md](https://github.com/avifenesh/memra/blob/main/docs/MODELS.md)

## Historical service measurement (2026-08-22, memra v0.101.0 — endpoint retired 2026-09-18)

These exact files served `qwen/qwen3.8-27b` behind `api.tiyuvta.ai` until 2026-09-18
(262,144-token context; the `frspec-sxc32768` draft head above is the always-on
speculative-decoding default). Measured through the public endpoint on the
serving build, single stream, greedy, streamed, medians per output length:

| output tokens | decode tok/s |
|---|---|
| 128 | 136 |
| 512 | 259 |
| 2048 | 166 |

Turn-8 first-token time of an 8-turn agentic conversation: 1.07 s at a
38k-token prompt, 95% prefix-cache hit. Speed varies with output length and
load; these are dated measurements of a service that has since been retired, not commitments or a current offer.

## Provenance

| | |
|---|---|
| Base | `Qwen/Qwen3.8-27B` (BF16, Apache-2.0) |
| Conversion | unsloth BF16 GGUF split (866 tensors, MTP block included) |
| Quantization | `llama-quantize` NVFP4, embd/output Q5_K |
| Verified | memra `kernel-check` / `run-gen` argmax / `run-spec` K=1..8 batteries; serve-surface canaries vs the official `Qwen/Qwen3.8-27B-FP8` reference |

## What ships here, file by file

| file | size | what it is |
|---|---|---|
| `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` | 15.7 GB | **The trunk.** NVFP4 weights, Q5_K embeddings + full 248,320-row LM head, embedded MTP (NextN) block at `blk.64`. Serves on its own; every other file in this repo is optional speed. |
| `mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf` | 1.24 GB | Standalone **masked** MTP draft head, **agentic** ranks — the serving default. |
| `mtp-Qwen3.8-27B-NVFP4-frspec-prose32768.gguf` | 1.24 GB | Same head, **prose** ranks. |
| `mtp-Qwen3.8-27B-NVFP4-frspec-mixed32768.gguf` | 1.24 GB | Same head, **mixed** ranks. |
| `q38-ranks-sxc32768.gguf.txt` | 186 KB | Agentic **ranks list, plain text: one token id per line, most-frequent first, 32,768 lines.** Drives the load-time trim (`MEMRA_FRSPEC_TRIM`) — this is the file that pairs with safetensors trunks. |
| `q38-ranks-prose-32768.txt` | 186 KB | Prose ranks, same format. |
| `q38-ranks-mixed-32768.txt` | 187 KB | Mixed ranks, same format. |
| `q38-ranks-sxc32768.gguf` / `q38-ranks-prose-32768.gguf` | 131 KB | The same ranks as a one-tensor GGUF container (`d2t`, i32 `[32768]`). Interchangeable with the `.txt` anywhere a ranks file is accepted. |

## Where the masked head lives, and what the mask does

Each `mtp-…frspec-*.gguf` is a 19-tensor file:

| tensor | shape / type | role |
|---|---|---|
| `output.weight` | `[5120, 32768]` NVFP4 | **the masked LM head** — 32,768 rows gathered from the trunk's 248,320-row head, in rank order |
| `d2t` | i32 `[32768]` | **the mask itself**: `d2t[i]` = full-vocab token id of masked row `i` |
| `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight` + `blk.64.attn_*` / `blk.64.ffn_*` | — | the MTP (NextN) block, extracted byte-verbatim from the trunk |
| `token_embd.weight` | `[5120, 248320]` Q5_K | full-vocab embeddings (drafting reads the trunk's ids) |

"Masked head" means: the draft proposes tokens only from the top-32,768 ids
ranked by how often **this model itself emits them** (FR-Spec-style `d2t`
trim — see [ggml-org/llama.cpp#25187](https://github.com/ggml-org/llama.cpp/issues/25187)).
The ranked distribution is 100% model-generated (163k own-generated tokens over
real agentic session prompts for the default flavor; external text was used as
prompts only). `topN = 32768` is what fixes the masked head's shape — a
different topN is a different artifact.

**The mask can never change output.** Verification runs on the target's full
vocabulary, so a trim moves draft acceptance (speed) and nothing else. What it
buys: the draft's per-step head read drops 7.6×, measured **+5.1% end-to-end on
RTX PRO 6000** (137.9 vs 131.2 decode p50) and **+6.4% on RTX 5090 Laptop**
(73.6 vs 69.2) against the full embedded head, even though the full head
accepts slightly more (0.76 vs 0.74). History: untrimmed 66.7% acceptance @
117.1 tok/s → trimmed 63.6% @ 121.7 tok/s (+3.9% e2e) on the first build.

## How to run it (memra)

Install memra per its [README](https://github.com/avifenesh/memra#install)
(release installer or `cargo build --release`). Three ways to attach the mask,
all producing byte-identical output to plain decode:

### 1. GGUF trunk + pre-trimmed masked head (recommended)

```bash
MEMRA_COMPAT=openai \
MEMRA_MODELS="q38=Qwen3.8-27B-NVFP4-Q5K-mtp.gguf+mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf" \
memra-server
```

The `+draft` suffix attaches the masked head to that model. Paths accept
`hf:owner/repo[:file-substring]` specs, which download on first use:

```bash
MEMRA_MODELS="q38=hf:tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF:Q5K-mtp+hf:tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF:frspec-sxc32768" \
memra-server
```

One-off CLI (no server) — `run-spec` sweeps the draft and **asserts the
speculative stream is token-identical to plain decode**:

```bash
MEMRA_MTP_DRAFT=mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf \
run-spec Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
```

### 2. Ranks file only — the trim happens at load

`MEMRA_FRSPEC_TRIM=<ranks>` takes either the `.txt` or the ranks `.gguf` and
self-trims at load: the engine gathers the 32,768 ranked rows from the trunk's
**own** `output.weight` bytes (byte-level row gather, zero requant). No
separate draft file, no cross-file quantization mismatch. Supported since
memra v0.84.

```bash
MEMRA_FRSPEC_TRIM=q38-ranks-sxc32768.gguf.txt \
MEMRA_MODELS="q38=Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
memra-server
```

### 3. Safetensors trunk + the `.txt` — no GGUF anywhere

The same `.txt` drives the trim on a Hugging Face **safetensors checkpoint
directory** (this is why the plain-text form exists). Point `MEMRA_MODELS` at
the checkpoint dir; the official FP8 checkpoint loads bit-exact and its own
`mtp.safetensors` head drafts out of the box, so the ranks are the only extra
file:

```bash
MEMRA_FRSPEC_TRIM=q38-ranks-sxc32768.gguf.txt \
MEMRA_MODELS="q38=/models/Qwen3.8-27B-FP8" \
memra-server
```

The current memra number on one RTX PRO 6000 is **250 tok/s** wall
(DFlash2, 512-token digits, TTFT included). 140 tok/s / 75 tok/s (5090 Laptop)
was the 2026-08-15 MTP-era p50, safetensors **with this masked-ranks trim**.
`MEMRA_FULL_PREC=1` disables the trim by design (the exactness ceiling wants
the natural full head).

### The attach is a log line, not the absence of an error

A wrong path or wrong flag does not fail — the trunk's embedded full head
drafts instead, and everything still works, just without the artifact you
chose. Check the boot log:

| you used | the line that proves it |
|---|---|
| `+draft` in `MEMRA_MODELS` | `[worker] q38: regime draft attached (…frspec-sxc32768.gguf)` |
| `MEMRA_MTP_DRAFT` | `[mtp-draft] loading external MTP draft: …` then `[mtp-draft] external draft head: blk.64, source=output.weight, head_vocab=32768 (trimmed, d2t map)` |
| `MEMRA_FRSPEC_TRIM` | `[frspec-trim] self-trimmed head: 32768 rows of main output.weight (Q5_K)` |

### Which flavor

| flavor | ranks file | pre-trimmed head | corpus |
|---|---|---|---|
| **agentic** (serving default) | `q38-ranks-sxc32768.gguf.txt` | `mtp-…frspec-sxc32768.gguf` | 163k own-generated tokens over real agentic sessions |
| **prose** | `q38-ranks-prose-32768.txt` | `mtp-…frspec-prose32768.gguf` | 154k own-generated tokens over essay/story/letter prompts, ~15% non-English |
| **mixed** | `q38-ranks-mixed-32768.txt` | `mtp-…frspec-mixed32768.gguf` | 50/50 normalized count-blend of both streams (same rank law) |

Short generic probes measure the three within noise of each other (acceptance
0.40 prose-text / 0.59–0.62 code); the differences live in domain tail tokens.
Pick by your traffic, don't inherit the default blindly.

## Other runtimes

The trunk is a standard GGUF: it loads wherever this model family loads, and
its embedded MTP block is present for engines that read NextN heads. The
**masked** draft files carry the `d2t` tensor layout; in mainline llama.cpp,
`d2t` is currently wired for the EAGLE3 draft architecture rather than the
Qwen MTP path (tracked in
[ggml-org/llama.cpp#25187](https://github.com/ggml-org/llama.cpp/issues/25187)),
so outside memra, draft from the trunk's embedded full head instead of these
masked files.

## Build your own ranks (and your own masked head)

Rank files are vocab + distribution artifacts of the **exact serving model**:
derive fresh ranks from the model's own generations for every model and every
requant. The tools ship in the memra repo:

```bash
git clone https://github.com/avifenesh/memra && cd memra && cargo build --release

# 1. Ranks from the model's OWN generations. Corpus text is prompts only —
#    the counted distribution is 100% model-generated. Prompts: .txt/.md file
#    = one prompt, directory = recursed, .jsonl = one per line,
#    hfds:owner/name = a HF dataset; none = the built-in mixed pack
#    (--preset code|chat|agentic|mixed).
./target/release/frspec-owngen <model.gguf|hf_dir|hf:spec> myranks.gguf 32768 my-prompts/
# -> myranks.gguf (d2t container) + myranks.gguf.txt (one id per line)
```

Serve immediately with `MEMRA_FRSPEC_TRIM=myranks.gguf.txt` (that's the whole
safetensors story), or bake a portable pre-trimmed head:

```bash
# 2. Extract the MTP block byte-verbatim, trim the head to your ranks,
#    requantize (NVFP4 head + Q4_K_M block — the measured-best order).
#    Needs python3 + numpy, llama.cpp's gguf-py (MEMRA_GGUFPY=<path>) and
#    llama-quantize (MEMRA_QUANTIZE=<path>).
tools/make-trimmed-draft.sh model.gguf myranks.gguf.txt my-draft.gguf 32768
# serve with MEMRA_MTP_DRAFT=my-draft.gguf
```

The measured laws, learned at cost
([docs/DRAFT-REGIME.md](https://github.com/avifenesh/memra/blob/main/docs/DRAFT-REGIME.md)):

- **Per model, per requant.** Foreign ranks measured −12 acceptance pts on an
  identical tokenizer. A finetune's distribution moved, so its ranks must too.
- **Chat template ON if you serve chat** (the default; `--raw` is for
  pure-continuation serving). A raw-derived rank set once left a chat cell
  with 10.9% structurally-unproposable tokens (−15 acceptance pts).
- **Corpus floor: ≥ 4× topN own-generated tokens** (131,072 for a 32,768
  head). The tool warns below it; ranks past the distribution head are noise.
- **Validate before trusting:** `frspec-owngen … --validate` A/Bs trimmed vs
  untrimmed end-to-end and prints a GOOD/WASH/BAD verdict. Acceptance explains
  a result; end-to-end tok/s decides it.

## What pairs with what

| you serve | trunk | files from this repo |
|---|---|---|
| GGUF (this repo) | `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` | one masked head (`+draft`) **or** one ranks file (`MEMRA_FRSPEC_TRIM`) |
| safetensors | official [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) checkpoint dir (carries its own MTP head) | one ranks `.txt` (`MEMRA_FRSPEC_TRIM`) |

The ranks encode this family's 248,320-token vocabulary — they transfer only
across trunks with the identical tokenizer, and by the per-model law you
should still re-derive for a requant or finetune. A DSpark block drafter for
the same target, trained rather than extracted, is published separately at
[tiyuvta/Qwen3.8-27B-DSpark-Agentic](https://huggingface.co/tiyuvta/Qwen3.8-27B-DSpark-Agentic).

## Measured (K=3, held-out prompts)

Untrimmed 66.7% acceptance @ 117.1 tok/s → trimmed 63.6% @ **121.7 tok/s
(+3.9% e2e)**. Re-measured on v0.86.2 (2026-08-16, interleaved ×5 vs the full
embedded head): **+5.1% on RTX PRO 6000** (137.9 vs 131.2 decode p50) and
**+6.4% on RTX 5090 Laptop** (73.6 vs 69.2, dead-flat reps) — the full head
accepts slightly more (0.76 vs 0.74) but the 7.6× smaller head read per draft
step wins end to end. Verification is lossless: the target verifies every
drafted token; the trim affects proposal coverage only, never output
correctness.

Built with `frspec-owngen` + `tools/make-trimmed-draft.sh` from the
[memra repo](https://github.com/avifenesh/memra) — the same recipe documented
above.
