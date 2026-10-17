---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
tags:
- quantization
- trellis-codec
- paw
- gguf
---

# PAW-27B

_If this model is useful to you, support the work by following [@nawariokra](https://x.com/nawariokra) on X._


A **~2.17 bpw** trellis-coded quantization of `Qwen/Qwen3.8-27B` that runs on
**one 24 GB GPU** with a **256k context** at 38–78 tok/s.

| | |
|---|---|
| size | 7.814 GB (this file: `PAW-27B.gguf`) |
| context | 262,144 tokens on a single RTX 3090 |
| speed | 78.4 tok/s short · 47.8 tok/s at 191k code context (MTP drafter) |
| code | **87.2% HumanEval+**, 75.9% MBPP+ |
| general | **58.6% MMLU-Pro**, 94% GSM8K |

## Requirements — read this first

This is **not** a standard GGUF. It uses the PAW trellis format (`paw-dense`
arch) and requires the [`llama-paw`](https://github.com/irawanw/llama-paw)
fork. **Stock llama.cpp cannot read this model.**

## How it compares

Against two same-family IQ2XXS GGUFs, measured by us on the same machine with
byte-identical harnesses:

| | size | MMLU-Pro (500) | GSM8K (100) | HumanEval+ | MBPP+ | IFBench-64 strict |
|---|---:|---:|---:|---:|---:|---:|
| **PAW-27B** | 7.814 GB | **58.6%** | **94%** | **87.2%** | **75.9%** | **23.44%** |
| Unsloth IQ2XXS | 7.266 GB | 46.2% | 93% | 84.1% | 72.8% | 14.06% |
| AtomicChat AD-IQ2XXS | 8.977 GB | 53.0% | **94%** | 73.8% | 71.7% | 20.31% |

PAW-27B leads MMLU-Pro, HumanEval+, MBPP+ and strict IFBench. Unsloth's file is
0.548 GB smaller and gives up 12.4 points of MMLU-Pro. AtomicChat is *larger*
than PAW-27B and does not recover the coding gap.

## Honest summary

**These are peer comparisons, not retention numbers.** No HumanEval, MBPP or
GSM8K figure exists for `Qwen/Qwen3.8-27B` at any precision, so there is no
published denominator to claim retention against on those axes, and we are not
going to invent one. What the table shows is that at roughly the same file
size this quantization is ahead of the two obvious alternatives — not how much
of the unquantized parent survives.

The references that *do* exist for this parent are IFBench (79.5 unquantized,
official Qwen) and LiveCodeBench v6 (90.3). The IFBench figure here is a
64-item defect sentinel, **not** the 300-item benchmark, so it is directional
only and not comparable to the official 79.5.

## Serving

See [SERVING.md](SERVING.md) for the full configuration. Short version:

- requires `llama-paw`, not stock llama.cpp
- speculative decoding with the parent's own MTP head
  (`--spec-type draft-mtp`) — **22.7 → 78.4 tok/s, lossless**
- `-ub 2048` is a hard ceiling at `-c 262144`, not a preference
- KV at q8_0 is free; `-fa on` is required for it
- **for agent/tool use, do not send reasoning back across turns** — see the
  loop limitation below

## Limitations

- **Agent clients loop unless reasoning is dropped from history.** The model
  copies its own prior `<think>` block verbatim and then repeats the action it
  went with. Serve with `--no-reasoning-preserve` *and* configure the client
  not to return `reasoning_content`. Details and measurements in
  [SERVING.md](SERVING.md).
- **Unbounded thinking returns empty answers.** Use
  `--reasoning-budget 2048`, not `-1`.
- **Rare-glyph copy defect.** The model cannot emit `➞` (U+279E) and aborts
  generation on prompts containing it; in this fork the server returns HTTP
  500. Eight HumanEval prompts contain it and it reproduces deterministically
  under greedy decoding. This is a model defect, not a harness artifact, and
  the affected tasks are scored as failures in the numbers above.
- **No parent denominator** on HumanEval/MBPP/GSM8K (see above).
- IFBench here is a 64-item sentinel, not the full benchmark.
- Long-context *quality* is unmeasured; only speed and footprint were measured
  at 191k–256k.

## Integrity

```
sha256 f5ec4b409de07f43bbe7bb5d9b3215181c53f5a26876821f732839a2ea3ab8cf  PAW-27B.gguf
```

## License

Apache 2.0, inherited from the base model `Qwen/Qwen3.8-27B`.
