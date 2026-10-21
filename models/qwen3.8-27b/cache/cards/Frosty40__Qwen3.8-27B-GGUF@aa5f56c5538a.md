---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- gguf
- llama.cpp
- qwen3.8
- qwen3.5-arch
- vision
- intel-arc
- speculative-decoding
- mtp
---

# Qwen3.8-27B — GGUF (Q6_K, 8-bit attention projections) with vision & native-MTP speculative decoding

Quantized GGUF conversion of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(dense 27B hybrid: 48 Gated-DeltaNet linear-attention layers + 16 full-attention layers,
native vision-language, 262K context), tuned and validated end-to-end on an Intel Arc Pro B70
via llama.cpp SYCL.

## Files

| file | size | what |
|---|---|---|
| **`Qwen3.8-27B-Q6_K-QKV8-MTP.gguf`** | 23.2 GB | **recommended.** Q6_K backbone with the attention projections (`attn_k`, `attn_v`, `attn_qkv`) at Q8_0 and the output tensor at Q6_K, plus the native MTP head at `blk.64` (q8_0). Mean KL-divergence vs bf16 **−28% (code) / −32% (agentic)** vs the OQ8 file below at +0.32 GiB; decode +45–95% from MTP (see below) |
| `Qwen3.8-27B-Q6_K-QKV8.gguf` | 22.7 GB | same backbone without the MTP head (for builds without NextN/MTP support) |
| `Qwen3.8-27B-Q6_K-OQ8-MTP.gguf` | 21.3 GB | previous recommendation (2026-08-15): uniform Q6_K + Q8_0 output tensor + MTP head. Keep if you need the smallest file / the largest context |
| `Qwen3.8-27B-Q6_K-OQ8.gguf` | 20.9 GB | same as above without the MTP head |
| `mmproj-Qwen3.8-27B-F16.gguf` | 0.9 GB | vision projector (f16), for `--mmproj` |

Conversion notes: converted from the bf16 release; tokenizer and chat template byte-identical
to upstream; sha256 receipts of every input shard were taken before conversion. The MTP head in
the `-MTP` file was grafted from the official standalone release
[EigenLabs/Qwen3.8-27B-MTP-bf16](https://huggingface.co/EigenLabs/Qwen3.8-27B-MTP-bf16)
(15 bf16 tensors → `blk.64.*` / `nextn.*` q8_0, GGUF `qwen35.nextn_predict_layers = 1`);
the head's final norm is stored as `nextn.shared_head_norm`.
sha256 `Qwen3.8-27B-Q6_K-QKV8-MTP.gguf`: `0e9a210bd50b2c057add16bb2a7388a3ac8d9c47de040b8b6c397a1417ec28fb`;
`Qwen3.8-27B-Q6_K-QKV8.gguf`: `6bd5285874c123c3a70d7517af1ae075d1d3bcb3e1da7ca03302434a3df90bb4`;
`Qwen3.8-27B-Q6_K-OQ8-MTP.gguf`: `2e5c352028a6751c92964bdd4efdfb8f93d4db6cb602508ca2328f4aa6eb4c80`.

## Quality (measured, not vibes)

KL-divergence against the full bf16 model, 512-token windows × 8, three corpora (wikitext-2 prose,
real code, real agent transcripts). Lower is better; "tail" is the 99.9th-percentile per-token KLD
(the rare bad token that breaks a program); "same-top" is greedy-token agreement with bf16.

| file | prose mean / same-top | code mean / tail / same-top | agentic mean / tail / same-top |
|---|---|---|---|
| **Q6_K-QKV8(-MTP)** | 0.00163 / 97.75% | **0.00084 / 0.051 / 99.80%** | **0.00094 / 0.053 / 99.27%** |
| Q6_K-OQ8(-MTP) | 0.00170 / 97.79% | 0.00117 / 0.090 / 99.46% | 0.00137 / 0.111 / 99.36% |
| Q8_0 (reference ceiling, not published) | — | 0.00014 / 0.010 / 99.85% | 0.00019 / 0.022 / 99.76% |

Same-top differences of a few tenths of a point are inside the instrument's noise (~8K tokens); the
mean/tail reductions are not. The `-MTP` files carry byte-identical backbone tensors to their
plain twins, so quality is identical by construction; speculative decoding is output-preserving
(draft tokens are verified against the target and only exact matches are emitted). Greedy outputs
are token-identical to CPU-bf16 on spot prompts. Vision verified through llama.cpp's mtmd.

Why 8-bit attention: at 96K+ context the k/v/qkv projections feed every attention read of the KV
cache, and code/agent tokens are low-entropy — one wrong token is a broken program — so the tail
matters more than the mean. The rest of the network stays Q6_K because decode on a 32 GB card is
memory-bandwidth-bound (measured to 0.6% of the byte roofline): every byte costs speed and context.

## Requirements

- Both files need llama.cpp with **Qwen3.5-family (`qwen35`) architecture support** (b10355+).
- The `-MTP` file additionally needs **NextN/MTP layer support** for the qwen35 arch
  (`qwen35.nextn_predict_layers`, `blk.N.nextn.*` tensors) and a server exposing
  `--spec-type draft-mtp`. If your build lacks these, use the plain `Q6_K-OQ8` file.
- The model uses interleaved MRoPE and hybrid recurrent+attention memory — context shifting is
  unavailable by design (clients should budget `max_tokens`; long-chat prefix reuse works via
  context checkpoints).

## Example (llama-server, vision + native-MTP speculative decoding)

```bash
llama-server -m Qwen3.8-27B-Q6_K-QKV8-MTP.gguf --mmproj mmproj-Qwen3.8-27B-F16.gguf \
  -ngl 99 -fa on -c 102400 -np 1 -ub 4096 -b 8192 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --jinja --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 \
  --reasoning on --reasoning-budget -1
```

`--spec-draft-n-max 3` matches the head's trained block size — larger drafts measured slower
(chained acceptance decays past the trained depth). Thinking is on by default
(`reasoning_effort` ∈ xhigh/medium/low via `chat_template_kwargs`); for non-thinking use
`{"enable_thinking": false}` with the card's non-thinking sampler (temp 0.7, top-p 0.8,
presence 1.5). Keep the KV cache at f16 — quantized KV measured a large decode regression at
depth on this architecture. On a 32 GB card `-c 102400` (+ vision) is the largest context that
stays fully resident for the QKV8 file (the OQ8 file fits 106496); past that point llama.cpp
silently spills to host memory and prefill, then decode, collapse — measure decode at depth
before raising `-c`.

## Measured on Intel Arc Pro B70 (32 GB, llama.cpp SYCL, these exact files)

Serial (no speculation — both files measure identically):

| metric | value |
|---|---|
| decode tg128 @ depth 0 / 16K / 64K | 21.6 / 19.6 / 14.9 t/s |
| prefill (warmed) 4K / 16K | 997 / 959 t/s |
| sustained fill @ 61K prompt | 735 t/s |
| max context fully resident (32 GB card, +vision) | 102,400 (QKV8) / 106,496 (OQ8) |
| prefill at 96K depth (cold, QKV8 @ -c 102400) | 630 t/s |

With native-MTP speculative decoding (`-MTP` file, production server, greedy, 256-token gens):

| workload | serial | **with MTP** | acceptance |
|---|---|---|---|
| free prose | 21.6 t/s | **31.2 t/s (+45%)** | 55% |
| repetitive code | 22.3 | **40.0 (+80%)** | 81% |
| verbatim quoting | 21.8 | **38.8 (+78%)** | 78% |
| decode @ 61K depth | 15.5 | **29.3 (+89%)** | 78% |
| decode @ 93K depth | ~12.5 | **23.9 (+91%)** | 71% |

The gains grow with depth because each accepted draft amortizes the full weight+KV read.
Warm TTFT (64-token prompt): 0.59 s. Decode power ~168 W (bandwidth-bound). Numbers are
B70-specific; other backends will differ. All numbers were produced with a receipted gate
discipline (golden-token equivalence, KLD thresholds, paired A/B/A benches).

## Provenance

Converted, quantized, and validated 2026-08-14; native-MTP graft added and gated 2026-08-15;
QKV8 files added 2026-08-17 (`llama-quantize Q6_K --tensor-type 'attn_k\.weight=q8_0'
--tensor-type 'attn_v\.weight=q8_0' --tensor-type 'attn_qkv\.weight=q8_0'`, from the bf16
release). OQ8 files: `Q6_K --output-tensor-type q8_0`. No imatrix (at 6-bit the measured quality
margin made it unnecessary). Every quality number above was re-measured on the same instrument
after a SYCL-backend decode-path fix (a stale activation memo) that had previously made any
non-Q6_K weights decode incorrectly on the tuning rig — the published files were validated with
that fix in place. Uploaded by
[Frosty40](https://huggingface.co/Frosty40).
