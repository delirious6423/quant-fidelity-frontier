---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - nvfp4
  - fp4
  - w4a4
  - vllm
  - quantized
  - blackwell
  - gb10
extra_gated_prompt: >-
  This checkpoint is released under the Apache 2.0 licence of its base model,
  Qwen/Qwen3.8-27B. Access is gated so that use is acknowledged, not restricted.
  Please tell us who you are and what you intend to use it for.
extra_gated_fields:
  Name: text
  Organisation or affiliation: text
  Intended use: text
  I accept the base model licence terms: checkbox
---

# Qwen3.8-27B — NVFP4 (W4A4)

A 4-bit NVFP4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
for NVIDIA Blackwell-class GPUs, with **4-bit activations** on the NVFP4 linears so the
model reaches the FlashInfer CUTLASS FP4 GEMM path rather than a weight-only kernel.

**23.4 GB** on disk, down from ~54.7 GB at BF16.

Multimodal input, tool calling and the MTP draft head are all preserved — see
[What is in the checkpoint](#what-is-in-the-checkpoint).

> **If you want the weight-only variant instead**, see
> [Qwen3.8-27B-NVFP4-W4A16](https://huggingface.co/JasonW2025/Qwen3.8-27B-NVFP4-W4A16).
> A [comparison](#which-of-the-two-should-i-use) is below.

---

## Requirements

| | |
|---|---|
| **GPU** | Blackwell-class with FP4 tensor cores. Validated on **NVIDIA GB10** (compute capability **12.1**, sm_121, 121 GB unified memory) |
| **vLLM** | `0.26.1rc1.dev468+g6b5bec7be.d20260807` or newer |
| **Quantization flag** | `--quantization modelopt_fp4` (required) |
| **KV cache** | `--kv-cache-dtype fp8` — the checkpoint ships 32 calibrated KV scales |
| **Memory** | ~23.4 GB weights, plus KV cache and activations |

The FP4 GEMM path needs a device the FlashInfer CUTLASS NVFP4 kernel supports. On an
unsupported device vLLM will fall back to a slower kernel or refuse to load.

---

## Serving

```bash
vllm serve JasonW2025/Qwen3.8-27B-NVFP4-W4A4 --served-model-name qwen38-w4a4 --port 8000 --quantization modelopt_fp4 --kv-cache-dtype fp8 --max-model-len 32768 --gpu-memory-utilization 0.60 --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --trust-remote-code
```

### Confirm you got the fast path

The activation scales exist so this checkpoint lands on CUTLASS. Check the startup log:

```
Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM
Selected FlashInferFP8ScaledMMLinearKernel for ModelOptFp8LinearMethod
```

If you instead see `MarlinNvFp4LinearKernel`, something has forced the weight-only path —
most often a global `--linear-backend` override. Marlin has nowhere to put activation
scales and will silently ignore them, so you would be running W4A16 numerics on a W4A4
checkpoint.

### Speculative decoding — measured, and worth it

The MTP draft head ships with the model (BF16). Add:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Measured on the same box, in one session, n=3 per arm, zero variance:

| | quality | responsiveness | deployability | median turn | decode |
|---|---|---|---|---|---|
| off | 93 | 18 | 70 | 8117 ms | **11.19 tok/s** |
| **MTP n=3** | 90 | **41** | **75** | **3790 ms** | **23.65 tok/s** |

Decode throughput **2.11×**, turn time **2.14×**, deployability +5, for 3 points of quality.
(Decode measured single-stream: 512 generated tokens with `ignore_eos`, first call
discarded as warmup, median of 3.)

**Read that −3 carefully — most of it is a ceiling effect, not lost capability.** At
temperature 0 speculative decoding is lossless by construction: the greedy rejection
sampler always emits the target model's argmax. In practice the verify step runs a
batched multi-token forward where plain decode runs one token, so GEMM reduction order
differs, logits shift in the last bits, and the argmax flips where two tokens are nearly
tied. Those flips are symmetric — but 87 % of scenarios in this suite already pass, and a
passing scenario can only move down. Four scenarios changed and all four were already
passing, which is exactly what symmetric noise predicts near a ceiling (p ≈ 0.57).

If you need reproducibility rather than throughput, see [Reproducibility](#reproducibility)
below — and note that vLLM's batch-invariant mode is **not available for this model**.

### Benchmarking note

For reproducible evaluation add `--no-enable-prefix-caching`. With prefix caching on,
identical temperature-0 requests can flip pass/fail between runs.

### Reproducibility

**Do not use `VLLM_BATCH_INVARIANT=1` with this model — the engine refuses to start.**

Batch-invariant mode is the usual answer to "make temperature-0 output bit-exact". We tested
it on this checkpoint. It fails during engine startup:

```
RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN.
```

The cause is architectural, not a packaging problem. Qwen3.8-27B is a hybrid whose attention
stack is mostly **Gated DeltaNet**, and in this vLLM build *no* linear-attention or SSM
backend supports batch invariance — `MAMBA1`, `MAMBA2`, `SHORT_CONV`, `LINEAR` and
`GDN_ATTN` all report `supports_batch_invariance() == False`. The NVFP4 half of the mode
works exactly as documented (the startup log does select `CutlassNvFp4LinearKernel` before
it dies), but the run ends before that matters. **No serve flag changes this.**

What you can actually do:

| goal | how |
|---|---|
| Stable scores within one serve session | `--no-enable-prefix-caching` and `temperature 0`. We measure **zero** within-session variance on the full-69 suite this way. |
| Closest to reproducible token streams | Also drop `--speculative-config`. The verify step pushes N+1 positions through one batched forward where plain decode pushes 1, which moves logits in the last bits. |
| Bit-exact across *separate* serve sessions | **Not achievable on this stack today.** Floating-point reduction order differs between engine sessions and flips the argmax on near-ties. A seed does not help — at temperature 0 there is no RNG to fix. |

---

## Measured performance

All figures are **full-69 agentic tool-evaluation** (`tool-eval-bench 2.5.1`), thinking
off, temperature 0, seed 1234, concurrency 1, prefix caching off, 300 s request timeout.
`deployability = 0.7 × quality + 0.3 × responsiveness`, and responsiveness keys on median
**turn** time.

| serve session | quality | responsiveness | deployability | n |
|---|---|---|---|---|
| A | **93** | 19 | **71** | 2 |
| B | **93** | 18 | **70** | 3 |

Quality is **93**, reproduced across two independent serve sessions, with zero variance
within each. Median turn time 7.9–8.1 s.

**On comparing scores.** Responsiveness moves by a point between serve sessions (19 vs 18).
That is not sampling noise a seed can remove — at temperature 0 decoding is greedy, so
there is no RNG to fix. Floating-point reduction order differs between engine sessions,
which flips the argmax wherever two candidate tokens are nearly tied, and a multi-turn
agentic trace amplifies one flipped token. Do not read a 1–2 point difference against
another model as meaningful.

### General capability

| benchmark | result | setting |
|---|---|---|
| GSM8K | **97.5 %** (195/200) | 8-shot |
| MMLU | **84.8 %** (11,906/14,042) | 5-shot, full test set |

MMLU by group: STEM 84.5 %, other 82.3 %.

MMLU is the **complete 14,042-question test set**, all 57 subjects, not a sample:

| category | accuracy |
|---|---|
| Social Sciences | 90.9 % |
| Other | 86.0 % |
| STEM | 83.4 % |
| Humanities | 80.9 % |

An earlier revision of this card reported 84.0 % from the first 500 questions. That slice is
the alphabetical head of the test set and contains **5 of the 57 subjects** — no humanities,
no social sciences, i.e. neither the weakest nor the strongest category. It has been replaced
by the full run rather than annotated.

Those MMLU figures come from the **same 14,042 questions answered by both checkpoints**, so
they can be compared question by question rather than score to score. They disagree on 355
questions, but the disagreement splits almost evenly — 169 where this one is right, 186 where
the other is — giving McNemar chi-squared 0.81 against the 3.84 needed for significance. A net
edge of about 37 questions would be required; the observed net edge is 17.

**On general knowledge the two are equivalent, and that is now measured at the maximum
resolution MMLU can offer rather than assumed from a tie.**

### Fidelity to the unquantized model

Per-token negative log-likelihood against the **BF16 base**, scored on 11,138 identical
tokens of fixed text (`/v1/completions`, `echo=true, max_tokens=0`, speculation off — it
corrupts echo-logprobs). Lower is closer to the original model.

| arm | mean NLL | Δ vs BF16 |
|---|---|---|
| BF16 base | 2.047242 | — |
| **W4A16** | 2.052558 | **+0.00532** |
| W4A4 | 2.061643 | +0.01440 |

**The measured noise floor is 0.00161** — the same W4A4 weights re-served in a fresh session.
The gap between the two variants is **0.00909, i.e. 5.6× that floor**, so it is a real
difference and not session drift.

**W4A16 is roughly 2.7× closer to the unquantized model.** That is the expected direction:
W4A16 keeps 16-bit activations, while W4A4 quantizes them to FP4. On this axis — the most
sensitive one we have — the W4A16 is the higher-fidelity serve.

---

## What is in the checkpoint

| | |
|---|---|
| Tensors | 2201 across 3 shards |
| Quantized linears | NVFP4, group size 16, **4-bit activations** |
| Remaining linears | FP8 |
| KV cache | FP8 with **32 calibrated scales** (2 per full-attention layer) |
| MTP draft head | 15 tensors, BF16 — kept unquantized |
| Vision tower | 333 tensors — multimodal input works |

### Files

Only what is needed to serve is published:

```
config.json                        model-00001-of-00003.safetensors
generation_config.json             model-00002-of-00003.safetensors
chat_template.jinja                model-00003-of-00003.safetensors
tokenizer.json                     model.safetensors.index.json
tokenizer_config.json              preprocessor_config.json
vocab.json                         processor_config.json
                                   video_preprocessor_config.json
```

The vision and video processor configs are **not** optional — the checkpoint contains a
real vision tower and the processor must be able to construct itself.

---

## Which of the two should I use?

| | **W4A4** (this one) | [W4A16](https://huggingface.co/JasonW2025/Qwen3.8-27B-NVFP4-W4A16) |
|---|---|---|
| Activations | 4-bit | 16-bit |
| GEMM kernel | FlashInfer CUTLASS FP4 | Marlin (weight-only) |
| Quality (full-69) | **93** | 91 |
| Responsiveness | **19** | 16 |
| Deployability | **70–71** | 68 |
| GSM8K (8-shot) | 97.5 % | **98.5 %** |
| MMLU (5-shot, all 14,042) | 84.8 % | **84.9 %** |
| Fidelity to BF16 (ΔNLL, lower better) | +0.0144 | **+0.0053** |
| Decode, MTP n=3 | 23.65 tok/s | 23.68 tok/s |
| Size | 23.4 GB | 23.4 GB |

**These two are much closer than "start with one" suggests, and the evidence is split.**

| axis | favours | how strongly |
|---|---|---|
| Agentic suite (69 scenarios) | **W4A4**, by 2–3 points | reproduced across 5 serve sessions; our measured session spread on this model is 0–1 point |
| Fidelity to BF16 (ΔNLL) | W4A16 | 5.6× the measured noise floor |
| GSM8K | W4A16, by 2 questions | inside noise |
| MMLU (all 14,042, paired) | tie — 355 discordant, split 169/186, McNemar 0.81 | — |

**Both separations are real, and they point in opposite directions.** This checkpoint wins
the agentic suite reproducibly; W4A16 keeps 16-bit activations and lands ~2.7× closer to the
unquantized model. The artifact that is further from the base model is the better agentic one
— quantizing activations is evidently not simple damage. Same size, same decode speed.

**Pick W4A4 for agentic tool-calling** — the 2–3 point edge held across five serve sessions,
so it is a real effect rather than a lucky run. **Pick W4A16** if you want the closest match
to the unquantized model, or your stack cannot use the CUTLASS FP4 path.

**Note the decode row: the two are the same speed.** Single-stream decode on this hardware
is memory-bandwidth-bound, not compute-bound — both checkpoints are 23.4 GB and read the
same bytes per token, so the GEMM kernel does not move raw throughput (11.19 vs 11.27 tok/s
with speculation off). W4A4's advantage is a shorter end-to-end turn on the agentic suite;
it is not faster at emitting tokens, and it is **not** more faithful to the unquantized model
— see the ΔNLL row. If you see a claim that the CUTLASS path is inherently faster to decode,
that is not what we measure here.

---

## Limitations

- Validated on GB10 (sm_121). Other Blackwell parts should work but are untested here.
- Benchmarked with thinking **off**. Behaviour with extended reasoning enabled is not
  characterised.
- Scores come from an agentic tool-calling suite and two knowledge benchmarks. They do not
  characterise multimodal, long-context, or multilingual performance.
- The ~2-point cross-session variance described above applies to any comparison you run.

## Licence

Apache 2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
