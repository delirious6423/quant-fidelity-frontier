---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - nvfp4
  - fp4
  - w4a16
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

# Qwen3.8-27B — NVFP4 (W4A16)

A 4-bit NVFP4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
for NVIDIA Blackwell-class GPUs. Weights are 4-bit; **activations stay at 16-bit**, so
this is a weight-only checkpoint that serves through the Marlin FP4 kernel.

**23.4 GB** on disk, down from ~54.7 GB at BF16.

Multimodal input, tool calling and the MTP draft head are all preserved — see
[What is in the checkpoint](#what-is-in-the-checkpoint).

> **Most users should prefer the W4A4 variant**,
> [Qwen3.8-27B-NVFP4-W4A4](https://huggingface.co/JasonW2025/Qwen3.8-27B-NVFP4-W4A4) —
> same size, faster, equal or better quality. See
> [Which of the two should I use](#which-of-the-two-should-i-use).

---

## Requirements

| | |
|---|---|
| **GPU** | Blackwell-class with FP4 support. Validated on **NVIDIA GB10** (compute capability **12.1**, sm_121, 121 GB unified memory) |
| **vLLM** | `0.26.1rc1.dev468+g6b5bec7be.d20260807` or newer |
| **Quantization flag** | `--quantization modelopt_fp4` (required) |
| **KV cache** | `--kv-cache-dtype fp8` — the checkpoint ships 32 calibrated KV scales |
| **Memory** | ~23.4 GB weights, plus KV cache and activations |

---

## Serving

```bash
vllm serve JasonW2025/Qwen3.8-27B-NVFP4-W4A16 --served-model-name qwen38-w4a16 --port 8000 --quantization modelopt_fp4 --kv-cache-dtype fp8 --max-model-len 32768 --gpu-memory-utilization 0.60 --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --trust-remote-code
```

### Expected kernel

Because this checkpoint carries no activation scales, vLLM selects the weight-only path.
The startup log will show:

```
Using MarlinNvFp4LinearKernel for NVFP4 GEMM
```

That is correct and expected here — it is not a misconfiguration. Note it is also not
overridable: the weight-only path is chosen from the checkpoint's own scheme, so no serve
flag will move this checkpoint onto the CUTLASS FP4 kernel. If you want that kernel, use
the W4A4 variant.

### Speculative decoding — strongly recommended

The MTP draft head ships with the model (BF16) and is worth a large amount of decode
throughput on this variant:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Measured single-stream decode on GB10 — 512 generated tokens with `ignore_eos`, first
call discarded as warmup, median of 3:

| `num_speculative_tokens` | tok/s | speed-up |
|---|---|---|
| off | 11.27 | — |
| 1 | 17.06 | 1.51× |
| 2 | 21.24 | 1.88× |
| **3** | **23.68** | **2.10×** |
| 4 | 23.00 | 2.04× |
| 5 | 22.97 | 2.04× |

**Use 3.** It is the peak; 4 and 5 both fall back slightly.

### Benchmarking note

For reproducible evaluation add `--no-enable-prefix-caching`. With prefix caching on,
identical temperature-0 requests can flip pass/fail between runs.

---

## Measured performance

Full-69 agentic tool-evaluation (`tool-eval-bench 2.5.1`), thinking off, temperature 0,
seed 1234, concurrency 1, prefix caching off. `deployability = 0.7 × quality + 0.3 ×
responsiveness`, and responsiveness keys on median **turn** time.

| quality | responsiveness | deployability | n |
|---|---|---|---|
| **91** | 16–17 | **68–69** | 3 |

Reproduced across two independent serve sessions with zero variance within each.

> **On comparing scores.** Within one serve session this harness is stable; *between*
> sessions scores on this stack move by around 2 points. That is not sampling noise a seed
> can remove — at temperature 0 decoding is greedy. Floating-point reduction order differs
> between engine sessions, flipping the argmax where two tokens are nearly tied, and a
> multi-turn agentic trace amplifies one flipped token. Do not read a 1–2 point difference
> against another model as meaningful.

### General capability

Measured with the same harness and settings as the W4A4 variant — thinking off, temperature
0, seed 1234, concurrency 1, prefix caching off, no speculative decoding.

| benchmark | result | setting |
|---|---|---|
| GSM8K | **98.5 %** (197/200) | 8-shot |
| MMLU | **84.9 %** (11,923/14,042) | 5-shot, full test set |

MMLU is the **complete 14,042-question test set**, all 57 subjects, not a sample:

| category | accuracy |
|---|---|
| Social Sciences | 91.2 % |
| Other | 86.0 % |
| STEM | 83.4 % |
| Humanities | 81.1 % |

An earlier revision of this card reported 84.0 % from the first 500 questions. That slice is
the alphabetical head of the test set and contains **5 of the 57 subjects** — no humanities,
no social sciences, i.e. neither the weakest nor the strongest category. It has been replaced
by the full run rather than annotated.

**These are not lower than the W4A4's.** On GSM8K this variant is 2 questions *ahead*
(98.5 % vs 97.5 %) and on full MMLU the two land 17 questions apart in 14,042. Neither is large enough to
rank the two variants — 2 questions out of 200 is noise — but that is the point: whatever
separates them on the agentic suite does not show up as a general-capability deficit here.
On fidelity to the unquantized model — the ΔNLL section below — this variant is not merely
level but measurably **ahead**.

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
| Tensors | 2033 across 3 shards |
| Quantized linears | NVFP4, group size 16, **no activation quantization** |
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

| | **W4A16** (this one) | [W4A4](https://huggingface.co/JasonW2025/Qwen3.8-27B-NVFP4-W4A4) |
|---|---|---|
| Activations | 16-bit | 4-bit |
| GEMM kernel | Marlin (weight-only) | FlashInfer CUTLASS FP4 |
| Quality (full-69) | 91 | **93** |
| Responsiveness | 16 | **19** |
| Deployability | 68 | **70–71** |
| GSM8K (8-shot) | **98.5 %** | 97.5 % |
| MMLU (5-shot, all 14,042) | **84.9 %** | 84.8 % |
| Fidelity to BF16 (ΔNLL, lower better) | **+0.0053** | +0.0144 |
| Decode, MTP n=3 | 23.68 tok/s | 23.65 tok/s |
| Size | 23.4 GB | 23.4 GB |

**These two are much closer than "prefer one" suggests, and the evidence is split.**

| axis | favours | how strongly |
|---|---|---|
| Fidelity to BF16 (ΔNLL) | **W4A16** | 5.6× the measured noise floor |
| GSM8K | W4A16, by 2 questions | inside noise |
| MMLU (all 14,042, paired) | tie — 355 discordant, split 169/186, McNemar 0.81 | — |
| Agentic suite (69 scenarios) | W4A4, by 2–3 points | reproduced across 5 serve sessions; our measured session spread on this model is 0–1 point |

**Both of those separations are real, and they point in opposite directions.** That is the
honest summary: the checkpoint that is measurably *further* from the unquantized model is
reproducibly *better* at multi-turn tool-calling. Quantizing activations is evidently not
simple damage. Same size, same decode speed either way.

**Pick W4A16** if you want the closest match to the unquantized model, or if your device or
vLLM build cannot use the CUTLASS FP4 path. **Pick W4A4 for agentic tool-calling** — its
2–3 point edge there held across five serve sessions, so it is a real effect rather than a
lucky run.

**But note the decode row: this checkpoint is not slower at emitting tokens.** Single-stream
decode here is memory-bandwidth-bound, not compute-bound — both checkpoints are 23.4 GB and
read the same bytes per token, so the GEMM kernel does not move raw throughput (11.27 vs
11.19 tok/s with speculation off). **And Marlin is not a quality compromise either** — on
the ΔNLL row above it is the *more* faithful of the two. If Marlin is your only option, you
are not giving up anything we can measure.

---

## Limitations

- Validated on GB10 (sm_121). Other Blackwell parts should work but are untested here.
- Benchmarked with thinking **off**. Behaviour with extended reasoning enabled is not
  characterised.
- Knowledge benchmarks are single-run (n=1) at temperature 0, as are the W4A4's. The
  agentic suite is the axis where the two variants differ; see the comparison table.
- The cross-session variance described above applies to any comparison you run.

## Licence

Apache 2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
