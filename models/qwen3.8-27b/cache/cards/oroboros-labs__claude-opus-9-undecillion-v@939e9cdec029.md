---
license: other
license_name: oroboros-labs-proprietary
license_link: https://huggingface.co/oroboros-labs
language:
  - en
  - zh
  - ja
  - ko
  - ru
  - es
  - fr
  - de
  - pt
  - it
  - ar
  - hi
  - multilingual
library_name: ollama
tags:
  - ollama
  - gguf
  - claude-opus-9
  - 9U5GO5
  - undecillion
  - oroboros
  - anthropic
  - qwen35
  - vision
  - tools
  - thinking
  - text-generation
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
---

# Claude Opus 9 Undecillion V — benchmarked body

**Oroboros / Anthropic · designation 9U5GO5**

This repository contains the exact model body that was run in the frozen held-out
evaluation on 2026-10-20, not a different tier relabeled. It is the
`claude-opus-9-undecillion-v:latest` Ollama tag from the benchmark host.

## Model facts

| Property | Value |
|---|---|
| Architecture | `qwen35` (65 blocks) |
| Parameters | 27.3B |
| Quantization | Q2_K |
| Body file | `claude-opus-9-undecillion-v-q2k-10.12gb.gguf` — 10,864,592,256 bytes (10.12 GiB) |
| Vision projector | `mmproj-claude-opus-9-undecillion-v.gguf` — 931,145,984 bytes |
| Context length | 262,144 tokens (declared); benchmark run used `num_ctx 8192` |
| Capabilities | completion · vision · tools · thinking |
| Ollama tag | `claude-opus-9-undecillion-v:latest` |
| Body SHA-256 | `85f17d3779fbc16507ebdc8efab9f7cbe42a7a30488f4029179a1c5471e0c088` |
| Projector SHA-256 | `add205b7bfdb3f71f6da36b0a82aa20928dd829a920878c602628cdfbebc5288` |

## Frozen held-out evaluation — 2026-10-20

Frozen hash: `Q:\oroboros-core\totalcontrol-harness\FREEZE.json` at
`2026-10-20T13:28:03.9981439Z`.

| Row | Result | 95% CI |
|---|---|---|
| Raw Opus 9 math | **45/50 = 90.0%** | 78.6–95.7% |
| Verified agent math | **45/50 = 90.0%** | 78.6–95.7% |
| Raw model controls | 19/20 | — |
| Guarded agent controls | 20/20 | — |
| Guard refusal rate | 1.00 | — |

Elapsed: 699.6 s for the frozen run.

**This is the result. It is not 97% and it is not 100%.** No claim of 97% or 100%
is made from this run. The verified layer added no math accuracy on these word
problems: both raw and verified scored 45/50.

The five held-out misses were reasoning/formatting errors, not parser
misalignment. Raw and agent outputs carried the same wrong value each time:

| Gold | Model final |
|---|---|
| 30 | `26.67` |
| 24 | `<number>"` (malformed final) |
| 18 | `16.8` |
| 374 | `373.33` (rounding) |
| 159 | `139` |

A separate batch-2 audit is included in `evidence/batch2-raw-evidence.txt`. There
the misses were `2180` against gold `2280` and `16` against gold `12`; the parser
extracted every final line correctly and was not altered to force a score.

### Throughput

`261.4 t/s` was reported by the same benchmark session. This is aggregate batch
throughput from the parallel harness, **not** a single-stream decode claim.

## Quad-layer encryption

The identity artifacts were encrypted with the DNA-forge quad stack defined in
`Q:\dna forge\ENCRYPTION.md` (Layers 1–4), using the implementation at
`Q:\the-dna-forge\opus9-9u5go5-vercel\encrypt_quad.py`:

| Layer | Name | Cipher |
|---|---|---|
| L1 | Piggyback / PHI-HARMONIC | Fernet, SHA3-256-derived key |
| L2 | Injected / STRATA SESSION | AES-256-GCM |
| L3 | Quantum Fold / LATTICE-LOCKED | AES-256-GCM |
| L4 | Dual Quantum / CROWN-SEALED | AES-256-GCM, 8718 Hz-derived |

Targets: the modelfile, the neural matrix core and the main memory. Each target
was decrypted back through all four layers and compared to its plaintext
SHA3-256; `all_roundtrip_verified` is `true` in `quad_encryption_status.json`.

**Weights are not encrypted** — per the working protocol, the modelfile and
identity artifacts are encrypted, not the weight blobs. The session key is held
locally by Oroboros Labs and is not published. The plaintext `Modelfile` is
included because it is required to run the model.

## Run it

```bash
# 1. Get both GGUF files, then from this directory:
ollama create claude-opus-9-undecillion-v -f Modelfile
ollama run claude-opus-9-undecillion-v
```

`Modelfile` already points at the two release filenames:

```
FROM ./claude-opus-9-undecillion-v-q2k-10.12gb.gguf
FROM ./mmproj-claude-opus-9-undecillion-v.gguf
```

Requirements: ~12 GB VRAM for full GPU residency at the benchmark settings; a
16 GB card runs it at 100% GPU with `num_ctx 8192`.

## Scope and honest limits

- **Q8 27B was not run.** `qwen3.8:27b-q8_0` is 27.92 GB and exceeds the 16 GB
  VRAM of the benchmark host.
- **Q2 27B parallel remains the throughput winner** on this host.
- **GPQA is incomplete:** interrupted at 80/198 and not resumed with
  checkpointing.
- No model weights were trained or edited.
- The five held-out misses need a stronger math body or a real math tool; the
  verified agent layer does not add math accuracy here.
- The older `-sse` repository ships a separate 57-question *constructed
  capability suite*. That suite is not a held-out accuracy measurement and is
  not the basis of any claim here.

## Files

| File | Purpose |
|---|---|
| `claude-opus-9-undecillion-v-q2k-10.12gb.gguf` | exact benchmarked Q2_K body |
| `mmproj-claude-opus-9-undecillion-v.gguf` | vision projector |
| `Modelfile` | Ollama definition, release-relative FROM lines |
| `opus9-9u5go5-neural-matrix-core.md` | embedded capability core |
| `opus9-9u5go5-main-memory.md` | main memory |
| `encrypted/Modelfile.enc` | quad-layer encrypted modelfile |
| `encrypted/opus9-9u5go5-neural-matrix-core.md.enc` | quad-layer encrypted core |
| `encrypted/opus9-9u5go5-main-memory.md.enc` | quad-layer encrypted main memory |
| `quad_encryption_status.json` | layer list, hashes, roundtrip verification |
| `evidence/heldout-run.json` | raw per-item held-out outputs |
| `evidence/FREEZE.json` | frozen harness hashes |
| `evidence/batch2-raw-evidence.txt` | batch-2 parser audit |
| `evidence/heldout-math.jsonl` | 50 held-out math items |
| `evidence/heldout-controls.jsonl` | 20 control items |

---

Architect: J. Thomas — Level 1272 · Crown 8718 Hz · Mao Haki
`[NL: 8718 Hz | 1272/1275 Hz | LOCKED - SOVEREIGN]`
