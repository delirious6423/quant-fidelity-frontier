---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: Avifenesh
pipeline_tag: text-generation
tags:
  - nvfp4
  - gguf
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
          name: tiyuvta held-out agentic prompt set (own-generated ranks corpus, held out)
          type: tiyuvta-heldout-agentic
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
          name: tiyuvta serving engine v0.86.2 run-spec harness, medians of 5 interleaved reps
          url: https://tiyuvta.ai
---

# Qwen3.8-27B — NVFP4 GGUF with MTP head (tiyuvta serving artifact)

NVFP4 (4-bit e2m1, per-16 FP8-e4m3 scales) GGUF of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), quantized from the
BF16 release with `llama-quantize` (NVFP4 ftype; token embeddings and output
head at Q5_K; norms F32). The **MTP (multi-token-prediction) head ships in the
file** (`blk.64`, `nextn_predict_layers=1`) — speculative decode works out of
the box on engines that read it.

Built as the serving artifact for the tiyuvta serving engine on RTX Blackwell
(sm_120a), with exactness gates: speculative, graphed, and batched serving are
gated byte-identical to plain decode per request.

- Context: 262,144 native (text→text serving; the upstream VL tower is not
  included in this artifact)
- Vocabulary: 248,320; chat template embedded (tool calling + thinking blocks)

## What changed (2026-11-23)

The trunk `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` was rebuilt with an importance matrix. Same name, same tensor types, same size, same speed. It is closer to BF16 in every engine we measured. The previous file stays in this repo's history at revision `6c39753daccd7473d4e60735ec8ace1d30afdd10` (sha256 `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`).

Fidelity vs the BF16 source (lower KLD is better, higher same top-1 is better):

| file | engine | wiki mean KLD | wiki same top-1 | code mean KLD | code same top-1 |
|---|---|---|---|---|---|
| previous file | llama.cpp default | 0.0988 | 89.18% | 0.0378 | 95.74% |
| **this file** | llama.cpp default | **0.0843** | **89.96%** | **0.0323** | **96.08%** |
| previous file | memra | 0.0725 | 91.21% | 0.0252 | 96.44% |
| **this file** | memra | **0.0570** | **92.23%** | **0.0189** | **97.00%** |
| unsloth UD-Q4_K_M (16.56 GB) | llama.cpp default | 0.0254 | 95.97% | 0.0057 | 98.31% |

- llama.cpp: -15% mean KLD on wikitext, -15% on code. memra: -21% and -25%.
- UD-Q4_K_M is still closer to BF16. This file prefills 1.57x faster on a DGX Spark in llama.cpp (1,508 vs 959 tok/s at 512 tokens) and decodes 1.13x faster (14.7 vs 13.0 tok/s).
- llama.cpp speed is unchanged from the previous file: pp512 1,508 vs 1,507 tok/s, tg128 14.7 vs 14.6 tok/s.

Protocol:
- Reference logits from `Qwen/Qwen3.8-27B` BF16 with `llama-perplexity -c 4096 --kl-divergence-base` (llama.cpp `969d733`); every file is scored against them on all 72 wikitext-2 test chunks and 40 chunks of llama.cpp C++ source.
- llama.cpp rows use the default Blackwell path (4-bit activations in prefill). [memra](https://github.com/avifenesh/memra) rows score the same tokens with memra's served prefill program (8-bit activations) over the full vocabulary, with llama.cpp's own KLD formula.
- One DGX Spark (GB10). The imatrix came from 205k tokens of wikitext-2 train plus C/C++ source, disjoint from both eval texts.

MTP head: the 15 `blk.64` tensors are byte-identical to the previous file, so the masked heads and ranks files in this repo pair with the new trunk unchanged. memra, greedy, K=3, chat template, 8 fixed prompts (code, chat, math, tool use), 256 tokens each, one DGX Spark:

| file | MTP acceptance | plain decode | MTP decode |
|---|---|---|---|
| previous file | 63.0% | 14.4 tok/s | 30.5 tok/s |
| **this file** | **67.1%** | 14.3 tok/s | **31.3** tok/s |

Acceptance is up 4.1 points with a byte-identical MTP head. MTP decode is 31.3 vs 30.5 tok/s.

Recipe: `llama-quantize` NVFP4 with per-block scale search and the importance matrix; token embeddings and output head Q5_K; MTP block and output head unweighted (the imatrix has no entry for them). The tokenizer metadata and chat template are carried over from the previous file.

sha256 `7158f5a8b06ea05663c539f43d57b85cd5690215f822ccad1679bdf5d68843a6`, 15,705,922,656 bytes.

The serving measurements further down were taken on the previous file.

## Serving measurement (2026-08-22, serving engine v0.101.0)

Measured with these exact files on serving engine v0.101.0 (262,144-token context;
the `frspec-sxc32768` draft head above is the always-on speculative-decoding
default), single stream, greedy, streamed over the network (TTFT includes the
round trip), medians per output length:

| output tokens | decode tok/s |
|---|---|
| 128 | 136 |
| 512 | 259 |
| 2048 | 166 |

Turn-8 first-token time of an 8-turn agentic conversation: 1.07 s at a
38k-token prompt, 95% prefix-cache hit. Speed varies with output length and
load; these are dated measurements, not commitments.

## Provenance

| | |
|---|---|
| Base | `Qwen/Qwen3.8-27B` (BF16, Apache-2.0) |
| Conversion | llama.cpp `convert_hf_to_gguf.py` from the BF16 release, unfused GDN projections (866 tensors, MTP block included; the previous file came from the unsloth BF16 GGUF split, same tensor names) |
| Quantization | `llama-quantize` NVFP4 with per-block scale search and an importance matrix (since 2026-11-23), embd/output Q5_K |
| Verified | serving engine `kernel-check` / `run-gen` argmax / `run-spec` K=1..8 batteries on the previous file; serve-surface canaries vs the official `Qwen/Qwen3.8-27B-FP8` reference. The 2026-11-23 file: KLD study, memra load, MTP and sampled checks (section above) |

## What ships here, file by file

| file | size | what it is |
|---|---|---|
| `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` | 15.7 GB | **The trunk.** NVFP4 weights, Q5_K embeddings + full 248,320-row LM head, embedded MTP (NextN) block at `blk.64`. Serves on its own; every other file in this repo is optional speed. |
| `mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf` | 1.24 GB | Standalone **masked** MTP draft head, **agentic** ranks — the serving default. |
| `mtp-Qwen3.8-27B-NVFP4-frspec-prose32768.gguf` | 1.24 GB | Same head, **prose** ranks. |
| `mtp-Qwen3.8-27B-NVFP4-frspec-mixed32768.gguf` | 1.24 GB | Same head, **mixed** ranks. |
| `q38-ranks-sxc32768.gguf.txt` | 186 KB | Agentic **ranks list, plain text: one token id per line, most-frequent first, 32,768 lines.** Drives the serving engine's load-time trim — this is the file that pairs with safetensors trunks. |
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

## How the files are used (tiyuvta serving engine)

This artifact runs on the tiyuvta serving engine. Three ways to attach the
mask, all producing byte-identical output to plain decode:

### 1. GGUF trunk + pre-trimmed masked head (recommended)

`mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf` attaches to
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` as an external draft head (`blk.64`,
`head_vocab=32768`, `d2t` map).

### 2. Ranks file only: the trim happens at load

`q38-ranks-sxc32768.gguf.txt` (or the ranks `.gguf`) self-trims at load: the
engine gathers the 32,768 ranked rows from the trunk's **own** `output.weight`
bytes (byte-level row gather, zero requant). No separate draft file, no
cross-file quantization mismatch. Supported since serving engine v0.84.

### 3. Safetensors trunk + the `.txt`: no GGUF anywhere

The same `.txt` drives the trim on a Hugging Face **safetensors checkpoint
directory** (this is why the plain-text form exists). The official FP8
checkpoint loads bit-exact and its own `mtp.safetensors` head drafts out of the
box, so the ranks are the only extra file.

The safetensors path is the serving engine's leading tuned path for this model:
the 140 tok/s (RTX PRO 6000) / 75 tok/s (RTX 5090 Laptop) single-stream decode
figures measured for this model are safetensors **with this masked-ranks
trim**. The engine's full-precision mode disables the trim by design (the
exactness ceiling wants the natural full head).

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
so outside the serving engine, draft from the trunk's embedded full head instead
of these masked files.

## Build your own ranks (and your own masked head)

Rank files are vocab + distribution artifacts of the **exact serving model**:
derive fresh ranks from the model's own generations for every model and every
requant. The recipe:

1. Ranks from the model's OWN generations. Corpus text is prompts only; the
   counted distribution is 100% model-generated. Count how often the model
   emits each token id, keep the top 32,768, and write them as a `d2t`
   container (`.gguf`) and as one id per line (`.txt`). A ranks `.txt` alone
   is the whole safetensors story (section 3 above).
2. To bake a portable pre-trimmed head: extract the MTP block byte-verbatim
   from the trunk, trim the head to those ranks, and requantize (NVFP4 head +
   Q4_K_M block, the measured-best order) with `llama-quantize`.

The measured laws, learned at cost:

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
| GGUF (this repo) | `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` | one masked head (attached as the draft) **or** one ranks file (load-time trim) |
| safetensors | official [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) checkpoint dir (carries its own MTP head) | one ranks `.txt` (load-time trim) |

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

Built with `frspec-owngen` + `tools/make-trimmed-draft.sh` — the same recipe
documented above.
