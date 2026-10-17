---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- gguf
- llama.cpp
- imatrix
- qwen3.8
- rtx-5090
quantized_by: trevest
---

# Qwen3.8-27B Tetris — 4.41 bpw per-tensor mix for a 24 GB card

VRAM tetris: depth-banded per-tensor precision, packed until a 27B VLM with
**163,840 tokens of context** and its in-checkpoint MTP draft head clicks into a
single RTX 5090 Mobile (24 GB) with real headroom. Built and serve-verified on an
Olares One; quantized from unsloth's BF16 convert with a code- and tool-call-weighted
importance matrix (included).

## Numbers (all measured, not estimated)

| | size | bpw | held-out ppl* | vs Q8_0 |
|---|---|---|---|---|
| Q8_0 reference | 27.05 GiB | 8.50 | 6.9887 | — |
| unsloth UD-Q4_K_XL | 16.69 GiB | 5.24 | 7.0057 | +0.24% |
| **Tetris (this repo)** | **14.03 GiB** | **4.41** | **7.1453** | **+2.24%** |

\* 120x512-token chunks of wikitext-2-test + code files held out of the calibration set.
Q8_0 measured +0.13% vs BF16 on matched chunks, so treat the Q8_0 column as ~BF16.

Serve-verified on the target card (llama.cpp b10448, `--spec-type draft-mtp`, q8_0 KV)
at ctx 163840 with zero OOM lines, including a **110,697-token prompt** (prefill 833 t/s,
generation at depth 45 t/s). ctx 180224 serves but leaves ~90 MiB of margin — not shipped.

**Draft-depth is quant-dependent — re-swept on this file** (temp 0, code/structured/prose):

| n-max | code | struct | prose | mean t/s | acceptance | VRAM |
|---|---|---|---|---|---|---|
| 2 | 72.7 | 78.5 | 64.6 | 72.0 | 0.770 | 22,856 |
| **3** | **75.5** | 83.1 | **65.7** | **74.8** | 0.704 | **23,038** |
| 4 | 74.4 | **84.2** | 64.4 | 74.3 | 0.628 | 23,188 |
| 5 | 69.9 | 83.2 | 59.5 | 70.9 | 0.563 | 23,336 |

The BF16-trained draft head accepts ~4pp fewer drafts against 4.4-bpw targets than against
the 5.2-bpw UD quant (which peaked at n-max 4) — use `--spec-draft-n-max 3` here.

## The mix

The 248k vocab and the hybrid Gated-DeltaNet architecture make stock k-quant recipes
misallocate; this mix spends bits where this architecture actually needs them:

- **FFN, depth-banded** (62.8% of params — the only real budget lever):
  layers 0-3/60-63 `q5_k` up/gate + `q6_k` down; 4-15/48-59 `iq3_s` + `q4_k`;
  deep middle 16-47 `iq3_xxs` + `iq3_s` (down always one step above up/gate)
- **Full-attention K/V projections** (16 layers): `q8_0` — what long-context retrieval reads
- **GDN decay gates** `ssm_alpha`/`ssm_beta`: **F32** — error compounds over sequence length
- **Whole MTP layer** (`blk.64`, incl. `nextn.eh_proj`): `q6_k` — the draft head gets no
  imatrix coverage, keep it k-quant and high
- `output.weight` `q6_k` / `token_embd` `q4_k`; GDN fused `attn_qkv`, `attn_gate`,
  `ssm_out`, full-attn `attn_q`/`attn_output`: `q4_k`-`q5_k`

Files: `Qwen3.8-27B-Tetris-4.4bpw.gguf` (the model), `imatrix-qwen38-27b.gguf`
(importance matrix: 40% bartowski datav3, 48% real code, 12% ChatML tool-call
transcripts, computed with `--parse-special`), `recipe-tetris.types` (the exact
`--tensor-type-file` used), `mmproj`: use `mmproj-F16.gguf` from
[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).

## Reproduce

```bash
llama-quantize --imatrix imatrix-qwen38-27b.gguf \
  --tensor-type-file recipe-tetris.types \
  --token-embedding-type q4_k --output-tensor-type q6_k \
  Qwen3.8-27B-BF16-00001-of-00002.gguf out.gguf Q5_K_M 24
```

Patterns are unanchored lowercased regex, first match wins — `blk\.64\.` leads the file
so the MTP layer can't be claimed by a band. Verify with `--dry-run` (every override is
logged) before spending the time.

## Serving (24 GB card)

```
llama-server -m Qwen3.8-27B-Tetris-4.4bpw.gguf --mmproj mmproj-F16.gguf \
  -ngl 999 --ctx-size 163840 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 --batch-size 1024 --ubatch-size 512 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --temp 1.0 --top-p 0.95 --top-k 20
```

Boot-fit is not serve-fit on this architecture: the MTP/graph workspace allocates on the
first request, so prove any higher context with a served (ideally deep) completion, not
a green health check.
