---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- rapid-mlx
- qwen
- qwen3.8
- mixture-of-experts
- quantized
- 4-bit
- apple-silicon
---

## Run it on your Mac

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
rapid-mlx serve qwen3.8-flash-next-4bit
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}]}'
```

Use it from OpenCode / Claude Code / Codex: `rapid-mlx agents <name> --setup` · [Agent setup](https://rapidmlx.com/docs/agents)

[Download the Desktop app](https://rapidmlx.com/download) · 128 GB unified memory admission floor; 192 GB recommended. Experimental, text-only.

# Qwen3.8-Flash-Next 4-bit for MLX (Rapid-MLX)


A 4-bit MLX quantization of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
for Apple Silicon, built and served by [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX).

- **Architecture (upstream):** ~180B total parameters (about 125B non-embedding plus a
  51B n-gram / PLE embedding), ~6B activated per token; 512 experts (10 routed + 1 shared);
  hybrid Gated DeltaNet + sparse attention; 48 layers; 262,144-token native context.
- **About the "30B params" badge:** the Hub counts safetensors elements, and this repository
  packs eight 4-bit weights into each `U32` element — 22.49B `U32` × 8 = 179.9B quantized
  weights, plus 7.36B `BF16` elements (quantization scales/biases and unquantized tensors).
  The badge is an artifact of the packed format, not the model size.
- **This artifact:** mixed 4-bit affine quantization (contract below), 28 safetensors shards,
  ~105 GB download, **~103–105 GB of MLX active unified memory while serving**.
- **Status:** experimental and text-only in Rapid-MLX 0.13.2. Native multi-token prediction
  (MTP) is available as an explicit opt-in; the vision tower is not enabled.

## Run with Rapid-MLX

```bash
# install or upgrade to the release that adds native MTP and the latest prefill path
curl -fsSL https://rapidmlx.com/install.sh | bash
# alternative: python3 -m pip install -U 'rapid-mlx>=0.13.2'

# ordinary decoding (downloads this repo on first use)
rapid-mlx serve qwen3.8-flash-next-4bit --host 127.0.0.1 --port 8464
```

For long-output workloads such as coding agents, opt into the checkpoint's native one-layer
MTP head:

```bash
rapid-mlx serve qwen3.8-flash-next-4bit \
  --host 127.0.0.1 --port 8464 \
  --speculative-config '{"method":"mtp"}'
```

MTP remains off unless requested. Normal opt-in serving keeps Rapid-MLX's expected-value
controller enabled, allowing it to park speculation when the current workload does not benefit.
It is most useful when generation time dominates; long-prefill, short-answer workloads should
usually use ordinary decoding.

The server exposes OpenAI-compatible and Anthropic-compatible endpoints:

```bash
curl -s http://127.0.0.1:8464/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-4bit",
       "messages":[{"role":"user","content":"Explain KV caching in two sentences."}],
       "max_tokens":256}'
```

The Rapid-MLX Desktop catalog admits this model on Macs with at least 128 GB of unified memory.

## Hardware

| Tier | Guidance |
| --- | --- |
| 256 GB | Measured here; the full 32K ordinary and MTP grids completed without out-of-memory. |
| 192 GB | **Recommended.** Weights are ~99 GB before context cache, MTP, and allocator headroom. |
| 128 GB | Admission floor only. **Not physically tested**; expect tight headroom. |

## Performance on Mac Studio (M3 Ultra, 256 GB)

These are Rapid-MLX 0.13.2 release-qualification measurements with this repository pinned at
revision `dcf657e4`. The machine was quiet with no other model resident. Each row is the median
of three batch-size-one runs with a cold prefix cache and 256 requested decode tokens.

### Faster long-context prefill

The final batched compressed-key cache was compared with the preceding optimized QSA path so
that the table isolates this 0.13.2 optimization. Attention math, cache precision, model weights,
2,048-token prefill chunks, prompts, and hardware were unchanged.

| Target (reported) prompt tokens | Previous TTFT | 0.13.2 batched TTFT | TTFT reduction | 0.13.2 prefill | 0.13.2 decode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 (92) | 0.385 s | 0.346 s | 10.1% | 266.3 tok/s | 25.67 tok/s |
| 2,048 (2,012) | 3.346 s | 2.262 s | **32.4%** | **889.4 tok/s** | 24.27 tok/s |
| 8,192 (8,156) | 13.689 s | 9.236 s | **32.5%** | **883.1 tok/s** | 23.40 tok/s |
| 32,768 (32,732) | 62.851 s | 44.659 s | **28.9%** | **732.9 tok/s** | 21.72 tok/s |

MLX active memory remained approximately 102.8–103.8 GB. The allocator's largest historical
peak during model loading and long-context work was 148.1 GB; process RSS is not a reliable
unified-memory sizing figure.

### Opt-in native MTP decode

The native one-layer MTP path raises sustained generation by 36–42% in its fixed-K1
qualification run. The target model verifies every proposed token. Both sides used the same
engine, checkpoint, prompts, and fresh-process isolation; only MTP was changed.

| Target (reported) prompt tokens | Serial decode | Native MTP decode | Decode speedup | Serial TTFT | MTP TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 (92) | 25.17 tok/s | **34.85 tok/s** | **1.38x (+38.5%)** | 0.393 s | 0.381 s |
| 2,048 (2,012) | 23.64 tok/s | **33.53 tok/s** | **1.42x (+41.8%)** | 3.346 s | 3.515 s |
| 8,192 (8,156) | 22.82 tok/s | **32.20 tok/s** | **1.41x (+41.1%)** | 13.643 s | 14.581 s |
| 32,768 (32,732) | 21.16 tok/s | **28.82 tok/s** | **1.36x (+36.2%)** | 62.844 s | 67.707 s |

The benchmark deliberately used
`--speculative-config '{"method":"mtp","disable_auto_k":true}'` to keep K=1 active and isolate
the MTP path; the recommended command above leaves adaptive control on. Fixed K=1 accepted
1,409 of 1,844 proposals (76.41%). MTP added 5–8% TTFT at 2K–32K and up to 6.6 GB of active
memory, which is why it is an opt-in decode optimization rather than the default.

### Warm prompt reuse

The 0.13.2 semantic prefix cache keys the exact rendered prompt and preserves the model's
recurrent state. In the measured warm repeat, a 5,288-token prompt reused 5,273 tokens and
completed in **0.539 s instead of 6.497 s**; native MTP continued proposing after the cache hit.

Prefill rate is reported prompt tokens divided by TTFT (including request and first-token
overhead); decode rate excludes TTFT. Full environments, exact commands, methodology, and
correctness evidence:

- [Long-context prefill and correctness](https://github.com/raullenchai/Rapid-MLX/blob/v0.13.2/docs/benchmarks/qwen38-flash-next-m3-ultra.md)
- [Native MTP qualification](https://github.com/raullenchai/Rapid-MLX/blob/v0.13.2/docs/benchmarks/qwen38-flash-next-mtp-m3-ultra.md)
- [Rapid-MLX 0.13.2 release notes](https://github.com/raullenchai/Rapid-MLX/blob/v0.13.2/docs/release-notes/v0.13.2.md)

## Correctness battery

A deterministic 45-case battery (English and Chinese; checkable math and reasoning; five
JSON-schema responses; automatic and forced tool calls on both API protocols; code generation
including an executable multi-file project; 8K and 32K needle recall; multi-turn and system
behaviour; stop sequences) completed with **45/45 effective functional outcomes** on the
optimized 0.13.2 path after documented harness adjudications. All 12 thinking cases pass at a
normal reasoning budget (`max_tokens` 4096 or the default OpenAI path); all tool, structured
output, protocol, project, and long-context cases pass. The native MTP run matched the ordinary
decode baseline on all 45 functional outcomes. Details and scoring caveats are in the benchmark
documents linked above.

## Sampled standard evals (identical harness, 4-bit vs 4-bit sibling)

Measured on 2026-08-27 with the EleutherAI evaluation harness (`lm_eval` 0.4.12) driving the
Rapid-MLX 0.13.1 OpenAI-compatible endpoint on the same Mac Studio, for this artifact and for
the 4-bit dense sibling `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` (revision `aa985c29`, speculative
decoding off). Settings were identical for both models: thinking **off** (`--no-thinking`;
no logged sample contains a `<think>` block), temperature 0, seed 1234, one request at a
time, chat template applied, the first N examples of each task.

| Task | N | Flash-Next-4bit | Qwen3.8-27B-4bit |
| --- | ---: | ---: | ---: |
| MMLU-Redux 2.0, generative 0-shot (4 per subject × 57) | 228 | **86.8** ± 2.1 | 83.3 ± 2.3 |
| HumanEval instruct, pass@1 | 100 | 96.0 ± 2.0 | **98.0** ± 1.4 |
| GSM8K 0-shot CoT, flexible-extract | 100 | **81.0** ± 3.9 | 80.0 ± 4.0 |
| GSM8K, same samples, answer-aware re-score (supplementary) | 100 | **96.0** | 94.0 |
| IFEval prompt-level strict / loose | 100 | **84.0** / **88.0** | 82.0 / 84.0 |
| IFEval instruction-level strict / loose | 163 | **89.0** / **92.6** | 88.3 / 90.2 |

Reading: on this sample the 4-bit Flash-Next scores on par with the 4-bit dense 27B — a little
ahead on knowledge, math and instruction following, two problems behind on HumanEval. Every
difference is within the sampling error at N = 100, so the supportable claim is "on par", not
"better".

Details:

- MMLU-Redux: Flash-Next answers 37 of 57 subjects 4/4 (27B: 33). Both are weakest on
  college mathematics (1/4), abstract algebra, econometrics, professional accounting and
  global facts (2/4).
- HumanEval: Flash-Next misses problems 32, 76, 84, 93; the 27B misses 32 and 93. The stock
  `humaneval_instruct` task scores 0 over chat completions because its inherited
  completion-style stop strings (`\ndef`, `\n#`, …) cut a chat reply at the first line, so a
  local variant with the same dataset, prompt and pass@1 metric (`until: []`, fenced-code-block
  extraction) was used for both models.
- GSM8K: the harness `flexible-extract` filter takes the *last* number in the reply; both
  models answer in bold and then restate context (`**$64** for the 16 glasses` → 16). 15 of
  Flash-Next's 19 harness misses are this extraction artifact. The supplementary row re-scores
  the identical saved samples preferring the last bold number; the harness number stays the
  headline. `strict-match` is 0.0 for both (it expects the literal "The answer is N.").
- IFEval, instruction-level strict by category (Flash / 27B): change_case 18/19 · 19/19,
  combination 10/11 · 9/11, detectable_content 8/9 · 8/9, detectable_format 28/29 · 27/29,
  keywords 32/39 · 31/39, language 3/4 · 4/4, length_constraints 26/29 · 25/29,
  punctuation 10/12 · 11/12, startend 10/11 · 10/11.

Caveats:

1. Sampled (first N examples, N = 100; MMLU-Redux 4 per subject = 228). ± is the harness
   standard error; 1–4 point gaps are not significant.
2. Non-thinking mode, temperature 0, single greedy run. Qwen's published numbers are bf16 with
   thinking enabled and are not comparable.
3. Two harness adaptations, applied identically to both models (chat-safe HumanEval variant;
   GSM8K answer-aware re-score reported beside, never instead of, the harness number).
4. Both models are 4-bit. There is no bf16 baseline in this comparison (the bf16 checkpoint is
   335 GiB and does not fit the 256 GB machine), so quantization fidelity versus bf16 is not
   measured here; the comparison isolates Flash-Next-4bit versus the dense 27B-4bit.
5. Wall clock on the M3 Ultra: Flash-Next 68.5 min, 27B 44.5 min for the full set.

Reproducibility: every command, the harness results JSON, the per-example samples, the
re-score script and the chat-safe HumanEval task are in the Rapid-MLX repository under
[`docs/benchmarks/qwen38-flash-next-sampled-evals/`](https://github.com/raullenchai/Rapid-MLX/tree/main/docs/benchmarks/qwen38-flash-next-sampled-evals).

## Upstream evaluations

Qwen's published benchmark results for Qwen3.8-Flash-Next are on the
[upstream model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Those numbers were
measured by Qwen on the bf16 model with thinking enabled; they have **not** been measured on
this quantization and should not be attributed to it.

## Quantization contract

A mixed affine quantization chosen to preserve the routing- and PLE-sensitive tensors:

- PLE embedding shards: 4-bit, group size 32
- MoE routing gate tensors: 8-bit, group size 64
- Remaining quantized tensors: 4-bit, group size 64

The complete tensor-level mapping is recorded in `config.json`.

## Provenance and integrity

- Source model: `Qwen/Qwen3.8-Flash-Next`, source revision `f5d08274`
- Converter: Rapid-MLX `scripts/qwen38_streaming_convert.py`, converter revision `7ad61c17`
- Weights measured above: this repository at commit `dcf657e4acda2aae72da99cde65b6c491cd96998`
  (later commits only touch this card and the LICENSE file)
- 28 safetensors shards with `model.safetensors.index.json`; `SHA256SUMS.txt` lists the
  SHA-256 of every published file

## Status

This checkpoint is experimental and text-only. Native MTP is optional; the vision tower is not
enabled in Rapid-MLX 0.13.2. Validate memory headroom and output quality on your own prompts
before relying on it in production.

## License

Built with Qwen. The weights in this repository are a derivative of `Qwen/Qwen3.8-Flash-Next`
and are distributed under the **Qwen Community License 1.0** (see [`LICENSE`](LICENSE)).

Copyright (c) 2026 Qwen. Permission is hereby granted, free of charge, to any person obtaining
a copy of this software, including the model weights, parameters, configuration files,
inference code and associated documentation files, to deal in the software without
restriction, subject to the conditions in the LICENSE file: the copyright notice and permission
notice must be included in all copies or substantial portions of the software; products above
the user/revenue thresholds named in the license must display the model name prominently; and
Model-as-a-Service or AI Work Assistant businesses need a separate license from Qwen for
commercial use.
