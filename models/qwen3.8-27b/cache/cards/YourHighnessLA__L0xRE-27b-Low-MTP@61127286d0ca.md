---
license: other
license_name: qwen-research-terms-derivative
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
tags:
  - l0xre
  - gguf
  - mtp
  - speculative-decoding
  - low-ram
  - 27b
language:
  - en
pipeline_tag: text-generation
---

# L0xRE-27b-Low-MTP

The self-drafting variant of [L0xRE-27b-Low](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low):
the same 27B hybrid model and quantization, plus one built-in nextn/MTP prediction
block (+830 MB). With this file the model can speculative-decode **on its own** —
no separate drafter file needed.

**Use this file if** you want a single-file deployment with self-accelerated
decoding (`--spec-type draft-mtp`).
**Use the standard file instead if** you plan to attach a DFlash2 drafter — with a
drafter present the MTP block is idle, and the standard build is 830 MB smaller
with identical speed and quality.

> **Runtime: L0xRE.** The current release is **v0.1.6.5 (Sparse Defaults)**. The
> [repository home page](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low)
> always names the current release and the install/update steps. The
> MTP-standalone path is not part of the v0.1.6 sparse/256K qualification; check
> the repository for current MTP support before installing.

## Runtime status

L0xRE v0.1.6.5 serves **262,144 tokens on 12 GB cards** with sparse attention and
the standard [L0xRE-V2-27b-Low](https://huggingface.co/YourHighnessLA/L0xRE-V2-27b-Low)
model. MTP-standalone serving (`--spec-type draft-mtp`) is a separate path. The
command and measurements below come from the SM89 pre-release runtime that was
used to qualify this checkpoint; they are retained as historical qualification,
not as a claim about the current runtime. If you want the current sparse 256K
setup, use the standard V2 model and the current runtime from the repository home
page.

## Quickstart (SM89, self-drafting, historical)

These flags are the SM89 pre-release invocation used for this checkpoint's
measurements. Check the repository home page for the current launch flags before
using them.

```bash
llama-server -m L0xRE-27b-Low-MTP.gguf \
  -c 32768 -b 2048 -ub 1024 -np 1 -t 8 -ngl 99 -fa on \
  -ctk kvarn3 -ctv kvarn2 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --jinja --reasoning on --reasoning-effort medium
```

Requires the **L0xRE runtime**, not stock llama.cpp.

## File identity

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `L0xRE-27b-Low-MTP.gguf` | 9,468,579,616 | `746bd40841fb18b9c1918923e89c007df70b5e4f3de64290d1399593e70c96b0` |

Architecture: 27B-parameter hybrid on the Qwen3.8-27B base — 64 transformer blocks
plus one nextn/MTP block; low-bit (≈2–3 bpw) projections, higher-precision
embeddings/output, 262,144 native context, 248,320 vocab.

## Measured (RTX 4090, L0xRE SM89 pre-release runtime)

Paired with DFlash2-Q4 for a like-for-like comparison against the standard build:
**143.0 ± 5.7 tok/s** code decode (peak 150.8), acceptance 0.72, mean draft length 4.6.
MTP-standalone (`draft-mtp`) qualification is **in progress** — the numbers above
were taken with the DFlash2 drafter, not the built-in head.

## Qualification boundaries

- Qualified on the L0xRE BeeLLama runtime only — **not** stock llama.cpp, Ollama,
  LM Studio, Transformers, or SGLang.
- MTP-standalone serving, longer contexts, and multi-slot serving are **not** part
  of the qualification, and are not part of the v0.1.6 sparse/256K qualification.
- For the current 256K sparse configuration on 12 GB, use
  [L0xRE-V2-27b-Low](https://huggingface.co/YourHighnessLA/L0xRE-V2-27b-Low).

## Quality

Same V1 weights as the standard build — same internal-suite results: 124–126 / 150
pass@1 with thinking on (three passes) vs 117 / 150 native-quant baseline.
Internal, non-canonical suite; directional, not leaderboard.

## License

Quantized derivative of Qwen3.8-27B, distributed under the base model's
research/community terms
([LICENSE](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE)); L0xRE
runtime tooling ships under the BeeLlama license. Full notices:
[THIRD-PARTY-NOTICES.md](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/THIRD-PARTY-NOTICES.md)
and [licenses](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/tree/main/licenses).
