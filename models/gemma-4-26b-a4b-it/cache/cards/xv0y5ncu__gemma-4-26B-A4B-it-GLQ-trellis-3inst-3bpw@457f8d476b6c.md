---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
tags:
- glq
- quantized
- trellis
- tcq
- moe
---

# gemma-4-26B-A4B-it — GLQ trellis (3INST) 3 bpw

[gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) quantized with
[GLQ](https://github.com/cnygaard/glq) using the QTIP-style trellis (TCQ) codebook at
**3 bits/weight**, lookup-free 3INST variant, uniform rate across all layers.

```bash
pip install "glq>=0.8.8" vllm "transformers>=5.13.1,<5.15"
vllm serve xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-3bpw --quantization glq
```

**Pin transformers below 5.15** — 5.15.0 moved gemma-4 to a per-layer config and
`config.head_dim` now raises, which vLLM hits before loading a single weight.

## Why this exists

The existing `gemma-4-26B-A4B-it-GLQ-3bpw` uses the E8-shell codebook and is *dominated*:
its stage-2 indices land in int16, so it occupies ~14.98 GB — no smaller than the same
model at 4 bpw — while scoring worse. This build is **12 GB**, genuinely below the 4 bpw
trellis checkpoint's 13.95 GB.

| | this build | trellis 4 bpw | shell 3 bpw |
| :-- | :-- | :-- | :-- |
| on disk | **11.9 GB** | 13.95 GB | ~14.98 GB |
| weights loaded (vLLM) | **10.41 GiB** | 14.4 GiB | — |
| avg weight SQNR | 16.38 dB | 22.11 dB | — |

Lower SQNR than 4 bpw is expected — it has a quarter fewer bits.

## Quality — AIME 2026

| precision | weights loaded | AIME 2026 (avg@8) | source |
| :-- | :-- | :-- | :-- |
| bf16 | 48.5 GiB | 88.3% | Google's published figure |
| NVFP4 (`nvidia/Gemma-4-26B-A4B-NVFP4`) | 18.2 GiB | 86.25% | measured |
| GLQ trellis **4 bpw** | 14.4 GiB | 86.25% | measured |
| **this build, GLQ 3 bpw** | **10.4 GiB** | **82.08%** | measured |

avg@8 over the 30 problems, 32k thinking budget, the model's own sampling defaults
(temp 1.0 / top_p 0.95 / top_k 64), vLLM 0.27.1, RTX PRO 6000 Blackwell. All three measured
arms ran on the same GPU with identical settings; the bf16 row is a *published* number from a
different harness, not a paired arm.

|  | 3 bpw | 4 bpw | NVFP4 |
| :-- | :-- | :-- | :-- |
| avg@8 | 197/240 | 207/240 | 207/240 |
| pass@8 | 27/30 | 29/30 | 28/30 |
| solved on all 8 samples | 18/30 | 23/30 | 21/30 |
| mean generation (tokens) | **18,707** | 15,026 | 15,117 |
| **truncated at 32k** | **74** | 29 | 24 |
| no answer | 16 | 5 | 10 |

**The −4.2 points against 4 bpw is inside the noise band.** With 30 problems the standard
error of the mean is 5.4 pt, so this is not a measurable quality difference on this evidence.

**What *is* outside noise is the truncation count — 74 against 29.** At 3 bpw the model
reasons noticeably longer (18.7k tokens vs 15.0k) and hits the 32k ceiling two and a half
times as often. A truncated answer scores as wrong however sound the reasoning was, and the
no-answer count moves the same way (16 vs 5). So part of the gap is budget exhaustion rather
than degraded reasoning, and **this score probably understates the checkpoint**. A rerun at a
64k budget would separate the two; it has not been done.

## Speed and footprint

Measured with `benchmarks/run_model.py` on one RTX PRO 6000 Blackwell, vLLM 0.27.1,
`max_model_len 2048`, FULL CUDA graphs captured:

|  | |
| :-- | :-- |
| weights (vLLM's own "Model loading took" line) | **10.41 GiB** |
| load | 5.7 s |
| KV capacity | 317,931 tokens |
| decode, B=1 | 89.9 tok/s, TTFT 22 ms |
| decode, B=32 | 1518 tok/s aggregate, 47.4/stream, TTFT 481 ms |

**Footprint is directly comparable to the 4 bpw build** — both figures come from vLLM's own
load line: **10.41 GiB against 14.4 GiB**, a 28% reduction. The throughput figures are *not*
comparable to that card's, which were produced by `vllm bench sweep serve` with different
request shapes; a matched re-measurement has not been run.

## Runtime support

| runtime | status |
|---|---|
| **vLLM** with glq >= 0.8.8 | **works** — the numbers above |
| **HF transformers** | **broken — not a fault of this checkpoint** |

The HF path produces incoherent output for gemma-4 MoE under *every* GLQ codebook: the
published, vLLM-validated checkpoints of this same base model fail identically. Nothing
outside GLQ's quantizer reassembles gemma-4's stacked `Gemma4TextExperts` tensors. Use vLLM.


## How it was produced

```bash
GLQ_TRELLIS_VARIANT=3inst GLQ_QUANT_EXPERT_WORKERS=1 glq-quantize \
    --model google/gemma-4-26B-A4B-it \
    --codebook trellis --bpw 3 --nsamples 128 --seqlen 2048 --streaming
```

`GLQ_QUANT_EXPERT_WORKERS=1` is **required at this bit-rate today**: the default 8-thread
expert pool races CUDA-graph capture and aborts with a device-side assert. At 3 bpw the
Viterbi state space doubles (2^(16-K)), which shifts the graph-vs-eager chunk threshold so
both paths run concurrently — the hazard `glq/trellis.py` documents. Single-worker avoids
it; the proper fix is `capture_error_mode="thread_local"` on the capture.

## License

Apache 2.0, inherited from the base model.
