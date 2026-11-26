---
library_name: gguf
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
quantized_by: JohnAKuntz
tags:
  - gguf
  - llama.cpp
  - gemma4
  - mixture-of-experts
  - imatrix
  - mixed-precision
  - long-context
  - raspberry-pi-5
---

# Gemma-4-26B-A4B-it — PI64K

A mixed-precision, importance-matrix-driven GGUF of
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it),
built so the model runs its **full 65,536-token context inside 16 GB of RAM** —
a Raspberry Pi 5 with 16 GB, specifically — instead of swapping.

| | |
|---|---|
| file | `gemma-4-26B-A4B-it-PI64K.gguf` |
| size | 10.57 GiB (11,346,286,272 bytes) |
| bpw | **3.592** measured (25.23 B text params) |
| tensors | 658 |
| context | 65,536 tokens (base model supports 262,144) |
| modality | text only — the vision tower is a separate `mmproj`, not included |
| sha256 | `fd66cb3817db0dc6cdd0564d70c607cc62c4b232448d61e3002d0b730f4e605e` |

## Why a mixed-precision quant instead of a uniform one

The text model is 25.24 B parameters, of which the routed experts are **90.5 %**
(22.84 B across 128 experts × 30 layers):

| component | params | share |
|---|---:|---:|
| routed experts | 22.84 B | 90.5 % |
| attention (q/k/v/o) | 1.11 B | 4.4 % |
| token embedding (tied to output head) | 0.74 B | 2.9 % |
| shared expert (dense, always active) | 0.54 B | 2.1 % |
| router + norms | 0.01 B | — |

Uniform 4-bit does not fit: Q4_0 is 14.62 GB of weights, and the measured runtime
overhead at 64 K is ~4.4 GiB (KV cache + compute buffers), so it peaks near
19 GiB. Uniform 3-bit fits but spends precision on the attention and embedding
tensors that are always active and cheap in bytes. So precision follows
importance: the routed experts carry almost all of the compression, everything
else stays wide.

Context is cheap on this architecture — 25 of 30 layers are sliding-window
(1024) and only 5 use full attention, and `attention_k_eq_v` is set — so
**the weights are the binding constraint, not the KV cache.** (In practice
llama.cpp stores K and V separately and sizes a unified cache per slot, so the
measured 64 K cache is ~3.3 GiB / 45.7 KiB per token, not the ~0.4 GiB the
sliding-window math predicts. Use `--parallel 1`: the default 4 slots multiply
that cache by four and push the model past 18 GiB.)

## Quant recipe

Calibration: 150 chunks × 512 tokens (76,800 tokens) — 35 % Wikipedia prose,
25 % source code, 30 % chat rendered through this model's real
`chat_template.jinja`, 10 % other. Importance matrix computed from the Q8_0
model with `llama-imatrix`. Allocation by per-layer importance (layer 0
dominates at 1.00 vs ~0.47 for the rest; then 3, 22, 23, 6, 4), applied with
`llama-quantize --tensor-type-file` using anchored regexes.

Actual tensor types in this file, measured from the GGUF:

| class | tensors | type | params | GiB | bpw |
|---|---:|---|---:|---:|---:|
| routed experts · `ffn_gate_up_exps` (26 layers) | 26 | `IQ2_S` | 13.20 B | 3.94 | 2.56 |
| routed experts · `ffn_gate_up_exps` (layers 0, 1, 3, 22) | 4 | `IQ3_XXS` | 2.03 B | 0.72 | 3.06 |
| routed experts · `ffn_down_exps` (all 30 layers) | 30 | `IQ4_NL` | 7.61 B | 3.99 | 4.50 |
| attention q/k/v/o | 115 | `Q6_K` | 1.11 B | 0.85 | 6.56 |
| shared expert `ffn_gate`/`ffn_up` | 60 | `Q6_K` | 0.36 B | 0.27 | 6.56 |
| shared expert `ffn_down` | 30 | `Q8_0` | 0.18 B | 0.18 | 8.50 |
| token embedding (tied) | 1 | `Q6_K` | 0.74 B | 0.56 | 6.56 |
| router (`ffn_gate_inp`) | 30 | `F32` | 0.01 B | 0.04 | 32.00 |
| norms / scales | 362 | `F32` | 0.01 B | 0.00 | 32.00 |

The attention row counts 115 tensors rather than 120 because the 5 full-attention
layers share K and V (`attention_k_eq_v`), so each of those carries one tensor
instead of two.

One honest note on that table: the plan called for `IQ2_S` on
`ffn_down_exps` too, but `llama-quantize` reported *"60 of 658 tensor(s)
required fallback quantization"* — the 704-wide expert rows are not a multiple
of `IQ2_S`'s 256-element block, so those 30 tensors were automatically upgraded
to `IQ4_NL`. That is why the file is 3.59 bpw rather than ~3.4, and it is the
better outcome: the down-projection kept 4.5 bpw.

Build: llama.cpp `a8ad690` (build 1.1.0-dev), converted from the official bf16
safetensors.

## Run it

```bash
llama-server \
  -m gemma-4-26B-A4B-it-PI64K.gguf \
  --ctx-size 65536 \
  --parallel 1 \
  --threads 4 --threads-batch 4 \
  --batch-size 512 --ubatch-size 256 \
  --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cont-batching --jinja \
  --load-mode mmap+mlock \
  --host 0.0.0.0 --port 8080
```

Measured peak RSS on a 16 GB Raspberry Pi 5 at 65,536 context, `-np 1`,
q4_0 KV, ubatch 256: **15.02 GiB** of ~15.6 GiB usable — ~0.6 GiB of headroom.
That runs and serves, but there is no room for anything else on the board.

Memory on this build decomposes as roughly: weights 10.57 GiB + KV 3.3 GiB +
compute buffers ~1.0 GiB + OS ~0.3 GiB. The measured law across four builds is
`peak_GiB ≈ weights_GiB + 4.4` at 64 K. Lowering ubatch to 128 did **not**
reclaim any of it.

If 64 K is not required, this file is heavier than it needs to be: at 32 K the
KV cache halves and a 13 GB build at noticeably higher expert precision
(`IQ3_S` / `IQ3_XXS`) fits instead.

## Evaluation

MMLU-style multiple choice, 60 questions, greedy decoding, thinking off —
4 subjects × 15 questions (college mathematics, college physics, high-school
world history, professional law), fixed seed, identical prompts for both models.

| model | size | accuracy | unparsed |
|---|---:|---:|---:|
| this quant (PI64K) | 10.57 GiB | 43/60 = **71.7 %** | 0 |
| official Q4_0 (control) | 13.62 GiB | 45/60 = **75.0 %** | 5 |

| subject | PI64K | Q4_0 |
|---|---:|---:|
| college mathematics | 10/15 | 8/15 |
| college physics | 11/15 | 11/15 |
| high-school world history | 13/15 | 14/15 |
| professional law | 9/15 | 12/15 |

At n=60 the standard error on a proportion near 0.72 is ±5.8 points, so the
3.3-point gap is **inside the noise floor of this test**. The accurate statement
is *"no measurable degradation detected at this sample size"* — not "no
degradation". Two things cut against a real regression: PI64K scored higher on
mathematics, and the Q4_0 control had 5 unparsed answers (it emitted reasoning
instead of a letter), which counts against it in a way that flatters the
comparison. Anyone deploying this on knowledge-heavy work should run a larger
eval of their own.

`llama-perplexity` could not be used on this model: llama.cpp at this commit
does not reset sliding-window state between chunks, so PPL climbs
173 → 295 → 418 → 709 as the run progresses. That is a known harness bug, not a
property of the weights; the model is healthy (tokenizer IDs match HF exactly,
factual Q&A is correct, generations parse cleanly).

## Limitations

- **Text only.** Vision and audio inputs need the separate `mmproj` file, which
  is not in this repo. Load it as `--mmproj` if you need image understanding.
- **Aggressive on the experts.** 13.2 B of the routed-expert parameters are at
  `IQ2_S`, 2.56 bpw. The expected failure mode is long-tail factual knowledge
  and rare-expert paths — the classes a 60-question MCQ cannot probe. If your
  workload is knowledge-recall heavy, prefer a wider build and less context.
- **Tight memory fit.** 0.6 GiB headroom means no co-tenant processes, no
  second context slot, and no safety margin for a different llama.cpp build's
  compute buffers.
- GPT-4-style chat/tool behaviour depends on the `--jinja` chat template shipped
  in the file; thinking mode and tool-call markup are inherited from the base
  model.
- Quantized weights: some degradation at very long context and on code/math
  prompts is expected relative to bf16.

## Provenance

Single-file build from the bf16 conversion of `google/gemma-4-26B-A4B-it`,
calibrated with a custom importance matrix (76,800 tokens of Wikipedia, source
code and template-rendered chat), quantized with llama.cpp `a8ad690`, and
validated end-to-end with `llama-server` at 65,536 context on a 16 GB Raspberry
Pi 5. Base model © Google, released under
[Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license); these
quantization modifications are distributed under the same license.
